"""Get Danceverse video descriptions by comparing groups of videos."""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Callable, Optional, TypeVar

from dotenv import load_dotenv
from google import genai
from google.genai import types


PROMPTS_DIR = Path(__file__).resolve().parent / "config" / "prompts"
PROMPT_VERSION_PATTERN = re.compile(r"^(?P<name>.+)_v(?P<version>\d+)\.txt$")
NO_AUDIO_PROMPT_VERSION_PATTERN = re.compile(
    r"^(?P<name>.+)_v(?P<version>\d+)_no_audio\.txt$"
)
COMPARISON_PROMPT_NAME = "comparison"
COMPARISON_EXEMPLAR_PROMPT_NAME = "comparison_exemplar"
COMPARISON_MODES = ("random", "exemplar", "monster")
GEMINI_UPLOAD_DISPLAY_NAME = "dance_video.mp4"
DEFAULT_EXEMPLAR_DIR = Path("data_upload/exemplars")
EXEMPLAR_PROMPT_REFERENCE_COUNT = 3
MONSTER_RANDOM_VARIANTS = ("random1", "random2")
MONSTER_EXEMPLAR_VARIANTS = ("exemplar1", "exemplar2")
MONSTER_VARIANTS = (*MONSTER_RANDOM_VARIANTS, *MONSTER_EXEMPLAR_VARIANTS)
COMPARISON_SCHEMA_FIELDS = [
    "freeform_description",
    "body_emphasis",
    "energy",
    "foundation_cues",
    "musicality",
    "tempo_feel",
    "texture",
    "salient_differences",
]
NO_AUDIO_COMPARISON_SCHEMA_FIELDS = [
    field for field in COMPARISON_SCHEMA_FIELDS if field != "musicality"
]
TRANSIENT_API_ATTEMPTS = 3
TRANSIENT_API_BACKOFF_SECONDS = 10
INVALID_JSON_ATTEMPTS = 4
INVALID_JSON_BACKOFF_SECONDS = 5
GEMINI_REQUEST_TIMEOUT_MS = 120_000

T = TypeVar("T")

def is_transient_api_error(error: Exception) -> bool:
    """
    Return true for API failures that are worth retrying at this script layer.
    """
    status_code = getattr(error, "status_code", None)
    if status_code in {429, 500, 502, 503, 504}:
        return True

    text = str(error)
    return any(
        marker in text
        for marker in [
            "429",
            "500 INTERNAL",
            "502",
            "503",
            "504",
            "Internal error encountered",
            "No route to host",
            "ReadTimeout",
            "ReadError",
            "nodename nor servname provided",
            "timed out",
            "timeout",
        ]
    )


def json_generation_config() -> dict:
    """
    Return common Gemini options for JSON responses.
    """
    return {
        "response_mime_type": "application/json",
        "http_options": {"timeout": GEMINI_REQUEST_TIMEOUT_MS},
    }


def retry_transient_api_call(label: str, operation: Callable[[], T]) -> T:
    """
    Retry transient Gemini API failures after the SDK has exhausted its own retry.
    """
    last_error: Optional[Exception] = None
    for attempt in range(1, TRANSIENT_API_ATTEMPTS + 1):
        try:
            return operation()
        except Exception as error:
            if not is_transient_api_error(error):
                raise

            last_error = error
            if attempt == TRANSIENT_API_ATTEMPTS:
                break

            sleep_seconds = TRANSIENT_API_BACKOFF_SECONDS * attempt
            print(
                f"Warning: transient Gemini error during {label} "
                f"(attempt {attempt}/{TRANSIENT_API_ATTEMPTS}): {error}. "
                f"Retrying in {sleep_seconds}s.",
                file=sys.stderr,
            )
            time.sleep(sleep_seconds)

    raise RuntimeError(
        f"Gemini API failed during {label} after "
        f"{TRANSIENT_API_ATTEMPTS} attempts."
    ) from last_error


def resolve_project_path(path: Path) -> Path:
    """
    Resolve relative paths from the project root so the script works anywhere.
    """
    if path.is_absolute():
        return path

    project_root = Path(__file__).resolve().parent
    return (project_root / path).resolve()


def find_video_files(videos_dir: Path) -> list[Path]:
    """
    Recursively collect .mp4 files in deterministic order.
    """
    return sorted(path for path in videos_dir.rglob("*.mp4") if path.is_file())


def containing_root(path: Path, roots: list[Path]) -> Optional[Path]:
    """
    Return the most specific configured root containing path.
    """
    containing_roots = []
    for root in roots:
        try:
            path.relative_to(root)
        except ValueError:
            continue
        containing_roots.append(root)

    if not containing_roots:
        return None

    return max(containing_roots, key=lambda root: len(root.parts))


def build_video_id_map(video_paths: list[Path], video_roots: list[Path]) -> dict[Path, str]:
    """
    Prefer filename stems, but use relative-path stems when duplicate names exist.
    """
    stem_counts: dict[str, int] = {}
    for video_path in video_paths:
        stem_counts[video_path.stem] = stem_counts.get(video_path.stem, 0) + 1

    video_ids: dict[Path, str] = {}
    for video_path in video_paths:
        if stem_counts[video_path.stem] == 1:
            video_id = video_path.stem
        else:
            root = containing_root(video_path, video_roots)
            if root is None:
                video_id = video_path.stem
            else:
                relative_stem = video_path.relative_to(root).with_suffix("")
                video_id = "__".join([root.name, *relative_stem.parts])

        video_ids[video_path] = video_id

    return video_ids


def partition_videos(
    video_paths: list[Path],
    group_size: int,
    seed: int,
) -> list[list[Path]]:
    """
    Shuffle deterministically and partition into mostly-3 groups.
    """
    shuffled = list(video_paths)
    random.Random(seed).shuffle(shuffled)

    if group_size != 3:
        groups = [
            shuffled[index : index + group_size]
            for index in range(0, len(shuffled), group_size)
        ]
        if any(len(group) not in {3, 4} for group in groups):
            raise ValueError(
                "--group-size must produce only 3- or 4-video groups. "
                "Use the default --group-size 3 for the prototype pipeline."
            )
        return groups

    total_videos = len(shuffled)
    if total_videos < 3:
        raise ValueError("At least 3 videos are required for comparison groups.")

    remainder = total_videos % 3
    if remainder == 0:
        group_lengths = [3] * (total_videos // 3)
    elif remainder == 1:
        if total_videos < 4:
            raise ValueError("At least 4 videos are required to make a remainder-1 group.")
        group_lengths = [4] + [3] * ((total_videos - 4) // 3)
    else:
        if total_videos < 8:
            raise ValueError(
                "At least 8 videos are required to avoid a remainder-2 group."
            )
        group_lengths = [4, 4] + [3] * ((total_videos - 8) // 3)

    groups: list[list[Path]] = []
    start = 0
    for group_length in group_lengths:
        groups.append(shuffled[start : start + group_length])
        start += group_length

    return groups


def path_label(path: Path, root: Path) -> str:
    """
    Return a stable local path label for logs and metadata.
    """
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def build_video_source_labels(video_paths: list[Path], roots: list[Path]) -> dict[Path, str]:
    """
    Build local source labels without exposing them to Gemini.
    """
    labels = {}
    for path in video_paths:
        root = containing_root(path, roots)
        if root is None:
            labels[path] = str(path)
        else:
            labels[path] = f"{root.name}/{path_label(path, root)}"
    return labels


def select_exemplar_videos(
    video_paths: list[Path],
    video_ids: dict[Path, str],
    exemplar_prefixes: Optional[list[str]],
) -> list[Path]:
    """
    Pick exemplar videos either by requested prefixes or by sorted directory order.
    """
    if not exemplar_prefixes:
        return sorted(video_paths, key=lambda path: video_ids[path])

    exemplars = []
    for prefix in exemplar_prefixes:
        matches = [
            path
            for path in video_paths
            if video_ids[path].lower().startswith(prefix.lower())
        ]
        if not matches:
            raise ValueError(f"No exemplar video matched prefix {prefix!r}.")

        exemplars.append(sorted(matches, key=lambda path: video_ids[path])[0])

    return exemplars


def exemplar_comparison_groups(
    video_paths: list[Path],
    exemplars: list[Path],
) -> list[list[Path]]:
    """
    Build one group per target video, using shared exemplar context videos.
    """
    groups = []
    for target_path in video_paths:
        groups.append([target_path, *exemplars])

    return groups


def validate_exemplar_set(exemplars: list[Path]) -> None:
    """
    Ensure exemplar mode matches comparison_exemplar_v0.txt.
    """
    if len(exemplars) != EXEMPLAR_PROMPT_REFERENCE_COUNT:
        raise ValueError(
            "Exemplar comparison requires exactly "
            f"{EXEMPLAR_PROMPT_REFERENCE_COUNT} exemplar videos for "
            "comparison_exemplar_v0.txt."
        )


def resolve_comparison_mode(args: argparse.Namespace) -> str:
    """
    Resolve the new explicit mode flag, honoring the legacy exemplar flag.
    """
    if args.comparison_mode:
        if args.compare_with_exemplar is not None:
            legacy_mode = "exemplar" if args.compare_with_exemplar else "random"
            if legacy_mode != args.comparison_mode:
                raise ValueError(
                    "--comparison-mode conflicts with deprecated "
                    "--compare-with-exemplar."
                )
        return args.comparison_mode

    if args.compare_with_exemplar is not None:
        return "exemplar" if args.compare_with_exemplar else "random"

    return "random"


def prompt_name_for_comparison_mode(comparison_mode: str) -> str:
    """
    Return the prompt family for a comparison mode.
    """
    if comparison_mode in {"exemplar", "monster_exemplar"}:
        return COMPARISON_EXEMPLAR_PROMPT_NAME
    if comparison_mode in {"random", "monster_random"}:
        return COMPARISON_PROMPT_NAME

    raise ValueError(f"Unsupported comparison mode: {comparison_mode}")


def monster_seed(base_seed: int, round_number: int, random_variant_index: int) -> int:
    """
    Return a deterministic seed for a monster-mode random variant.
    """
    return base_seed + (round_number - 1) * len(MONSTER_RANDOM_VARIANTS) + random_variant_index


def pretty_video_count(num_videos: int) -> str:
    """
    Return the word Gemini should see in the comparison prompt.
    """
    count_words = {
        3: "THREE",
        4: "FOUR",
        5: "FIVE",
    }
    try:
        return count_words[num_videos]
    except KeyError as error:
        raise ValueError("Comparative prompts currently support 3, 4, or 5 videos.") from error


def top_level_schema_shape(
    video_keys: list[str],
    fields: Optional[list[str]] = None,
) -> str:
    """
    Render the exact top-level JSON shape expected from Gemini.
    """
    objects = []
    for index, video_key in enumerate(video_keys):
        comma = "," if index < len(video_keys) - 1 else ""
        if fields is None:
            objects.append(f'  "{video_key}": "..."{comma}')
            continue

        field_lines = []
        for field_index, field in enumerate(fields):
            field_comma = "," if field_index < len(fields) - 1 else ""
            field_lines.append(f'    "{field}": "..."{field_comma}')
        field_body = "\n".join(field_lines)
        objects.append(f'  "{video_key}": {{\n{field_body}\n  }}{comma}')

    return "{\n" + "\n".join(objects) + "\n}"


def prompt_uses_field_schema(template: str) -> bool:
    """
    Return true when the prompt requests nested per-video description fields.
    """
    return "Each video key should contain these fields:" in template


def create_downsampled_video(video_path: Path, fps: float, no_audio: bool = False) -> Path:
    """
    Create a temporary low-FPS MP4, optionally stripping the audio stream.
    """
    if fps <= 0:
        raise ValueError("--fps must be greater than 0.")

    temp_file = tempfile.NamedTemporaryFile(
        prefix="dance_video_",
        suffix=".mp4",
        delete=False,
    )
    output_path = Path(temp_file.name)
    temp_file.close()

    command = [
        "ffmpeg",
        "-y",
        "-i",
        str(video_path),
        "-vf",
        f"fps={fps:g}",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "23",
    ]
    if no_audio:
        command.extend(["-an"])
    else:
        command.extend(["-c:a", "copy"])
    command.append(str(output_path))

    try:
        subprocess.run(command, capture_output=True, text=True, check=True)
    except FileNotFoundError as error:
        output_path.unlink(missing_ok=True)
        raise RuntimeError(
            "ffmpeg is required to process videos before upload. "
            "Install ffmpeg and try again."
        ) from error
    except subprocess.CalledProcessError as error:
        output_path.unlink(missing_ok=True)
        details = error.stderr.strip() or error.stdout.strip()
        message = "ffmpeg failed while processing the video. ffmpeg is required."
        if details:
            message = f"{message}\n\nffmpeg output:\n{details}"
        raise RuntimeError(message) from error

    return output_path


def upload_video(client: genai.Client, video_path: Path) -> types.File:
    """
    Upload a local video file to Gemini and return the uploaded file object.
    """
    return client.files.upload(
        file=video_path,
        config={
            "display_name": GEMINI_UPLOAD_DISPLAY_NAME,
            "mime_type": "video/mp4",
        },
    )


def wait_for_uploaded_video(
    client: genai.Client,
    uploaded_video: types.File,
    timeout_seconds: int = 300,
    poll_seconds: int = 5,
) -> types.File:
    """
    Wait until Gemini has finished processing the uploaded video.
    """
    if not uploaded_video.name:
        raise ValueError("Gemini did not return a file name for the uploaded video.")

    deadline = time.monotonic() + timeout_seconds
    current_video = uploaded_video

    while current_video.state == types.FileState.PROCESSING:
        if time.monotonic() >= deadline:
            raise TimeoutError(
                f"Timed out waiting for Gemini to process: {uploaded_video.name}"
            )

        time.sleep(poll_seconds)
        current_video = client.files.get(name=uploaded_video.name)

    if current_video.state == types.FileState.FAILED:
        raise ValueError(f"Gemini failed to process uploaded video: {uploaded_video.name}")

    return current_video


def read_prompt_template(path: Path) -> str:
    """
    Read a prompt template from disk.
    """
    try:
        return path.read_text(encoding="utf-8").strip()
    except FileNotFoundError as error:
        raise FileNotFoundError(f"Prompt template not found: {path}") from error


def latest_prompt_template_path(prompt_name: str, no_audio: bool = False) -> Path:
    """
    Return the highest versioned prompt path, falling back to the unversioned file.

    For example, comparison_v3.txt is preferred over comparison_v2.txt and
    comparison.txt. This keeps prompt iteration in config/prompts explicit while
    letting the pipeline use the latest version by default.
    """
    suffix = "_no_audio" if no_audio else ""
    pattern = NO_AUDIO_PROMPT_VERSION_PATTERN if no_audio else PROMPT_VERSION_PATTERN
    candidates: list[tuple[int, Path]] = []
    for path in PROMPTS_DIR.glob(f"{prompt_name}_v*{suffix}.txt"):
        match = pattern.match(path.name)
        if not match or match.group("name") != prompt_name:
            continue
        candidates.append((int(match.group("version")), path))

    if candidates:
        return max(candidates, key=lambda item: item[0])[1]

    return PROMPTS_DIR / f"{prompt_name}{suffix}.txt"


def build_comparison_prompt(
    num_videos: int,
    prompt_name: str,
    response_video_count: Optional[int] = None,
    no_audio: bool = False,
) -> str:
    """
    Build a prompt for comparing uploaded videos.
    """
    count_word = pretty_video_count(num_videos)
    output_count = response_video_count or num_videos
    video_keys = [f"video_{index}" for index in range(1, output_count + 1)]
    key_list = ", ".join(video_keys)

    template = read_prompt_template(latest_prompt_template_path(prompt_name, no_audio=no_audio))
    schema_fields = (
        NO_AUDIO_COMPARISON_SCHEMA_FIELDS if no_audio else COMPARISON_SCHEMA_FIELDS
    )
    fields = schema_fields if prompt_uses_field_schema(template) else None
    schema_shape = top_level_schema_shape(video_keys, fields=fields)

    return template.format(
        count_word=count_word,
        key_list=key_list,
        schema_shape=schema_shape,
    )


def write_json(path: Path, data: Any) -> None:
    """
    Save pretty JSON, creating parent folders first.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def read_json(path: Path) -> Any:
    """
    Read a JSON file into a dictionary.
    """
    return json.loads(path.read_text(encoding="utf-8"))


def clean_json_response(response_text: str) -> str:
    """
    Remove Markdown code fences so model output can be parsed as JSON.
    """
    cleaned = response_text.strip()

    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        lines = lines[1:]

        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]

        cleaned = "\n".join(lines).strip()

    return cleaned


def parse_gemini_json(response_text: str) -> dict:
    """
    Clean the model output and parse it into a Python dictionary.
    """
    cleaned_text = clean_json_response(response_text)

    try:
        return json.loads(cleaned_text)
    except json.JSONDecodeError as error:
        raise ValueError(
            "Gemini did not return valid JSON that could be parsed."
        ) from error


def parse_response_json(response_text: str, raw_response_path: Path) -> dict:
    """
    Parse Gemini JSON, saving raw and cleaned text when parsing fails.
    """
    try:
        return parse_gemini_json(response_text)
    except ValueError as error:
        raw_response_path.parent.mkdir(parents=True, exist_ok=True)
        raw_response_path.write_text(response_text, encoding="utf-8")

        cleaned_path = raw_response_path.with_name(
            f"{raw_response_path.stem}_cleaned{raw_response_path.suffix}"
        )
        cleaned_path.write_text(clean_json_response(response_text), encoding="utf-8")

        raise ValueError(
            "Gemini did not return valid JSON. Saved the raw response to "
            f"{raw_response_path} and the cleaned response to {cleaned_path}."
        ) from error


def read_recovered_response(raw_response_path: Path) -> Optional[dict]:
    """
    Read a cleaned raw response left by a previous failed parse, if it is valid JSON.
    """
    cleaned_path = raw_response_path.with_name(
        f"{raw_response_path.stem}_cleaned{raw_response_path.suffix}"
    )
    if not cleaned_path.exists():
        return None

    try:
        return parse_gemini_json(cleaned_path.read_text(encoding="utf-8"))
    except ValueError:
        return None


def delete_uploaded_file(client: genai.Client, uploaded_file: types.File) -> None:
    """
    Best-effort cleanup for Gemini Files API uploads.
    """
    if not uploaded_file.name:
        return

    try:
        client.files.delete(name=uploaded_file.name)
    except Exception as cleanup_error:
        print(
            f"Warning: could not delete uploaded Gemini file {uploaded_file.name}: "
            f"{cleanup_error}",
            file=sys.stderr,
        )


def upload_processed_videos(
    client: genai.Client,
    video_paths: list[Path],
    fps: float,
    no_audio: bool = False,
) -> tuple[list[types.File], list[Path]]:
    """
    Downsample and upload videos, returning uploaded File objects and temp paths.
    """
    processed_paths: list[Path] = []
    uploaded_files: list[types.File] = []

    try:
        for video_path in video_paths:
            processed_path = create_downsampled_video(video_path, fps, no_audio=no_audio)
            processed_paths.append(processed_path)

            uploaded_file = retry_transient_api_call(
                f"uploading {video_path.name}",
                lambda: upload_video(client, processed_path),
            )
            uploaded_file = retry_transient_api_call(
                f"processing uploaded {video_path.name}",
                lambda: wait_for_uploaded_video(client, uploaded_file),
            )
            uploaded_files.append(uploaded_file)
    except Exception:
        for uploaded_file in uploaded_files:
            delete_uploaded_file(client, uploaded_file)
        for processed_path in processed_paths:
            processed_path.unlink(missing_ok=True)
        raise

    return uploaded_files, processed_paths


def cleanup_uploads_and_temp_files(
    client: genai.Client,
    uploaded_files: list[types.File],
    processed_paths: list[Path],
) -> None:
    """
    Remove Gemini uploads and local temporary videos.
    """
    for uploaded_file in uploaded_files:
        delete_uploaded_file(client, uploaded_file)

    for processed_path in processed_paths:
        processed_path.unlink(missing_ok=True)


def fallback_fps_values(fps: float) -> list[float]:
    """
    Return lower-FPS fallbacks for groups that repeatedly produce empty responses.
    """
    candidates = [fps, min(fps, 5.0), min(fps, 2.0)]
    values: list[float] = []
    for candidate in candidates:
        if candidate > 0 and all(abs(candidate - value) > 0.001 for value in values):
            values.append(candidate)
    return values


def group_context(
    group: list[Path],
    video_ids: dict[Path, str],
    video_source_labels: dict[Path, str],
    round_name: str,
    group_index: int,
    seed: Optional[int],
    write_video_keys: Optional[list[str]] = None,
    grouping_mode: str = "random",
    comparison_attempt: Optional[int] = None,
    description_variant: Optional[str] = None,
    no_audio: bool = False,
) -> dict:
    """
    Build stable metadata for one comparison group.
    """
    videos = {}
    for index, video_path in enumerate(group, start=1):
        videos[f"video_{index}"] = {
            "video_id": video_ids[video_path],
            "filepath": video_source_labels[video_path],
        }

    return {
        "round": round_name,
        "group_index": group_index,
        "seed": seed,
        "grouping_mode": grouping_mode,
        "comparison_attempt": comparison_attempt,
        "description_variant": description_variant,
        "no_audio": no_audio,
        "write_video_keys": write_video_keys or list(videos),
        "videos": videos,
    }


def round_suffix(round_name: str) -> str:
    """
    Return the compact filename suffix for a comparison round.
    """
    if round_name.startswith("round"):
        return f"r{round_name.removeprefix('round')}"
    return round_name


def round_description_path(
    round_dir: Path,
    video_id: str,
    round_name: str,
    description_variant: Optional[str] = None,
) -> Path:
    """
    Return the per-video comparison description path for a round.
    """
    suffix = round_suffix(round_name)
    if description_variant:
        return round_dir / f"{video_id}.{suffix}.{description_variant}.json"

    return round_dir / f"{video_id}.{suffix}.json"


def write_round_descriptions(
    round_dir: Path,
    round_name: str,
    group_result: dict,
    context: dict,
) -> dict[str, Any]:
    """
    Split one group response into per-video round description JSON files.
    """
    descriptions_by_video = {}

    for video_key in context.get("write_video_keys", context["videos"].keys()):
        video_info = context["videos"][video_key]
        if video_key not in group_result:
            raise ValueError(
                f"Missing {video_key} in {round_name} group "
                f"{context['group_index']:03d} result."
            )

        video_id = video_info["video_id"]
        description = group_result[video_key]
        write_json(
            round_description_path(
                round_dir,
                video_id,
                round_name,
                context.get("description_variant"),
            ),
            description,
        )
        descriptions_by_video[video_id] = description

    return descriptions_by_video


def write_group_contexts(round_dir: Path, round_name: str, group_contexts: list[dict]) -> None:
    """
    Persist group assignments as the round progresses.
    """
    write_json(round_dir / "_groups.json", {"round": round_name, "groups": group_contexts})


def run_comparison_group(
    client: genai.Client,
    group: list[Path],
    model: str,
    fps: float,
    raw_response_path: Path,
    prompt_name: str,
    response_video_count: Optional[int] = None,
    no_audio: bool = False,
) -> dict:
    """
    Upload one group and ask Gemini for comparison descriptions.
    """
    prompt = build_comparison_prompt(
        len(group),
        prompt_name=prompt_name,
        response_video_count=response_video_count,
        no_audio=no_audio,
    )
    group_label = ", ".join(path.name for path in group)
    last_parse_error: Optional[Exception] = None

    for fps_attempt in fallback_fps_values(fps):
        uploaded_files: list[types.File] = []
        processed_paths: list[Path] = []

        try:
            uploaded_files, processed_paths = upload_processed_videos(
                client,
                group,
                fps_attempt,
                no_audio=no_audio,
            )
            if abs(fps_attempt - fps) > 0.001:
                print(
                    f"Retrying comparison group for {group_label} at "
                    f"{fps_attempt:g} fps.",
                    file=sys.stderr,
                )

            for attempt in range(1, INVALID_JSON_ATTEMPTS + 1):
                response = retry_transient_api_call(
                    f"generating comparison group for {group_label}",
                    lambda: client.models.generate_content(
                        model=model,
                        contents=[prompt, *uploaded_files],
                        config=json_generation_config(),
                    ),
                )

                try:
                    return parse_response_json(response.text or "", raw_response_path)
                except ValueError as error:
                    if not (response.text or ""):
                        debug_path = raw_response_path.with_name(
                            f"{raw_response_path.stem}_response_debug"
                            f"{raw_response_path.suffix}"
                        )
                        debug_path.write_text(repr(response), encoding="utf-8")

                    last_parse_error = error
                    if attempt == INVALID_JSON_ATTEMPTS:
                        break

                    sleep_seconds = INVALID_JSON_BACKOFF_SECONDS * attempt
                    print(
                        f"Warning: invalid/empty JSON during comparison group for "
                        f"{group_label} (attempt {attempt}/{INVALID_JSON_ATTEMPTS}). "
                        f"Retrying in {sleep_seconds}s.",
                        file=sys.stderr,
                    )
                    time.sleep(sleep_seconds)
        finally:
            cleanup_uploads_and_temp_files(client, uploaded_files, processed_paths)

        print(
            f"Warning: comparison group for {group_label} did not return valid JSON "
            f"at {fps_attempt:g} fps.",
            file=sys.stderr,
        )

    raise ValueError(
        "Gemini did not return valid JSON after FPS fallbacks for "
        f"{group_label}."
    ) from last_parse_error


def rescue_anchor_videos(
    target: Path,
    failed_group: list[Path],
    video_paths: list[Path],
    count: int = 2,
) -> list[Path]:
    """
    Pick stable comparison anchors when an exact random group fails repeatedly.
    """
    anchors = [path for path in video_paths if path not in failed_group and path != target]
    if len(anchors) < count:
        anchors.extend(path for path in failed_group if path != target)
    return anchors[:count]


def run_random_group_rescue(
    client: genai.Client,
    group: list[Path],
    video_paths: list[Path],
    video_ids: dict[Path, str],
    video_source_labels: dict[Path, str],
    round_dir: Path,
    round_name: str,
    group_index: int,
    seed: Optional[int],
    comparison_attempt: int,
    description_variant: Optional[str],
    raw_stem: str,
    model: str,
    fps: float,
    prompt_name: str,
    group_contexts: list[dict],
    no_audio: bool = False,
) -> dict[str, Any]:
    """
    Recover a failed random group by describing each target with stable anchors.
    """
    descriptions_by_video: dict[str, Any] = {}

    for target_index, target in enumerate(group, start=1):
        rescue_group = [target, *rescue_anchor_videos(target, group, video_paths)]
        rescue_raw_path = (
            round_dir / f"{raw_stem}.rescue_{target_index}_raw_response.txt"
        )
        rescue_context = group_context(
            rescue_group,
            video_ids,
            video_source_labels,
            round_name,
            group_index,
            seed,
            write_video_keys=["video_1"],
            grouping_mode="random_rescue",
            comparison_attempt=comparison_attempt,
            description_variant=description_variant,
            no_audio=no_audio,
        )
        group_contexts.append(rescue_context)

        print(
            f"RESCUE {round_name} group {group_index:03d} video {target_index}: "
            + ", ".join(video_ids[path] for path in rescue_group)
        )
        result = run_comparison_group(
            client,
            rescue_group,
            model,
            fps,
            rescue_raw_path,
            prompt_name=prompt_name,
            response_video_count=None,
            no_audio=no_audio,
        )
        descriptions_by_video.update(
            write_round_descriptions(
                round_dir,
                round_name,
                result,
                rescue_context,
            )
        )
        write_group_contexts(round_dir, round_name, group_contexts)

    return descriptions_by_video


def run_round(
    client: genai.Client,
    round_name: str,
    seed: int,
    comparison_attempt: int,
    video_paths: list[Path],
    video_ids: dict[Path, str],
    video_source_labels: dict[Path, str],
    output_dir: Path,
    group_size: int,
    fps: float,
    model: str,
    overwrite: bool,
    comparison_mode: str,
    exemplars: list[Path],
    description_variant: Optional[str] = None,
    group_contexts: Optional[list[dict]] = None,
    no_audio: bool = False,
) -> dict[str, Any]:
    """
    Run one comparison-description round and return descriptions by video_id.
    """
    exemplar_ids: list[str] = []
    if comparison_mode in {"exemplar", "monster_exemplar"}:
        validate_exemplar_set(exemplars)
        exemplar_ids = [video_ids[path] for path in exemplars]
        groups = exemplar_comparison_groups(video_paths, exemplars)
        grouping_mode = "exemplar"
        prompt_name = prompt_name_for_comparison_mode(comparison_mode)
        write_video_keys = ["video_1"]
        response_video_count = 1
        context_seed = None
    elif comparison_mode in {"random", "monster_random"}:
        groups = partition_videos(video_paths, group_size, seed)
        grouping_mode = "random"
        prompt_name = prompt_name_for_comparison_mode(comparison_mode)
        write_video_keys = None
        response_video_count = None
        context_seed = seed
    else:
        raise ValueError(f"Unsupported comparison mode: {comparison_mode}")

    round_dir = output_dir / round_name
    round_dir.mkdir(parents=True, exist_ok=True)

    descriptions_by_video: dict[str, dict] = {}
    if group_contexts is None:
        group_contexts = []

    variant_label = f" {description_variant}" if description_variant else ""
    if comparison_mode in {"exemplar", "monster_exemplar"}:
        print(
            f"\n{round_name}{variant_label}: {len(groups)} exemplar attempts "
            f"using {', '.join(exemplar_ids)}"
        )
    else:
        print(f"\n{round_name}{variant_label}: {len(groups)} groups with seed {seed}")

    for group_index, group in enumerate(groups):
        raw_stem = f"group_{group_index:03d}"
        if description_variant:
            raw_stem = f"{raw_stem}.{description_variant}"
        raw_response_path = round_dir / f"{raw_stem}_raw_response.txt"
        context = group_context(
            group,
            video_ids,
            video_source_labels,
            round_name,
            group_index,
            context_seed,
            write_video_keys=write_video_keys,
            grouping_mode=grouping_mode,
            comparison_attempt=comparison_attempt,
            description_variant=description_variant,
            no_audio=no_audio,
        )
        group_contexts.append(context)

        existing_paths = [
            round_description_path(
                round_dir,
                context["videos"][video_key]["video_id"],
                round_name,
                description_variant,
            )
            for video_key in context.get("write_video_keys", context["videos"].keys())
        ]
        if all(path.exists() for path in existing_paths) and not overwrite:
            print(
                f"SKIP {round_name}{variant_label} group {group_index:03d}: "
                "per-video files exist"
            )
            for video_key in context.get("write_video_keys", context["videos"].keys()):
                video_info = context["videos"][video_key]
                video_id = video_info["video_id"]
                descriptions_by_video[video_id] = read_json(
                    round_description_path(
                        round_dir,
                        video_id,
                        round_name,
                        description_variant,
                    )
                )
            write_group_contexts(round_dir, round_name, group_contexts)
            continue

        if not overwrite:
            recovered_result = read_recovered_response(raw_response_path)
            if recovered_result is not None:
                print(
                    f"RECOVER {round_name} group {group_index:03d}: "
                    f"{raw_response_path.with_name(raw_response_path.stem + '_cleaned' + raw_response_path.suffix)}"
                )
                descriptions_by_video.update(
                    write_round_descriptions(
                        round_dir,
                        round_name,
                        recovered_result,
                        context,
                    )
                )
                write_group_contexts(round_dir, round_name, group_contexts)
                continue

        print(
            f"PROCESS {round_name}{variant_label} group {group_index:03d}: "
            + ", ".join(video_ids[path] for path in group)
        )
        try:
            result = run_comparison_group(
                client,
                group,
                model,
                fps,
                raw_response_path,
                prompt_name=prompt_name,
                response_video_count=response_video_count,
                no_audio=no_audio,
            )
        except Exception as error:
            if grouping_mode == "random":
                print(
                    f"Warning: failed exact random group {group_index:03d}; "
                    "trying per-video rescue comparisons.",
                    file=sys.stderr,
                )
                try:
                    written_descriptions = run_random_group_rescue(
                        client,
                        group,
                        video_paths,
                        video_ids,
                        video_source_labels,
                        round_dir,
                        round_name,
                        group_index,
                        context_seed,
                        comparison_attempt,
                        description_variant,
                        raw_stem,
                        model,
                        fps,
                        prompt_name,
                        group_contexts,
                        no_audio=no_audio,
                    )
                except Exception as rescue_error:
                    video_list = ", ".join(video_ids[path] for path in group)
                    raise RuntimeError(
                        f"Failed while rescuing {round_name}{variant_label} "
                        f"group {group_index:03d} "
                        f"({video_list})."
                    ) from rescue_error

                descriptions_by_video.update(written_descriptions)
                write_group_contexts(round_dir, round_name, group_contexts)
                print(
                    f"WROTE {round_name}{variant_label} group "
                    f"{group_index:03d} via rescue: "
                    f"{len(written_descriptions)} per-video files"
                )
                continue

            video_list = ", ".join(video_ids[path] for path in group)
            raise RuntimeError(
                f"Failed while processing {round_name}{variant_label} "
                f"group {group_index:03d} "
                f"({video_list})."
            ) from error
        written_descriptions = write_round_descriptions(
            round_dir,
            round_name,
            result,
            context,
        )
        descriptions_by_video.update(written_descriptions)
        write_group_contexts(round_dir, round_name, group_contexts)
        print(
            f"WROTE {round_name}{variant_label} group {group_index:03d}: "
            f"{len(written_descriptions)} per-video files"
        )

    write_group_contexts(round_dir, round_name, group_contexts)
    return descriptions_by_video


def build_master_index(
    video_paths: list[Path],
    video_ids: dict[Path, str],
    video_source_labels: dict[Path, str],
    output_dir: Path,
    round_names: list[str],
    description_variants: Optional[list[str]] = None,
) -> dict:
    """
    Build a compact index from video_id to source and comparison artifacts.
    """
    index = {}
    for video_path in video_paths:
        video_id = video_ids[video_path]
        video_index = {
            "filepath": video_source_labels[video_path],
            "comparison_description_paths": {},
        }
        for round_name in round_names:
            round_dir = output_dir / round_name
            if description_variants:
                video_index["comparison_description_paths"][round_name] = {
                    variant: str(
                        round_description_path(
                            round_dir,
                            video_id,
                            round_name,
                            variant,
                        ).relative_to(output_dir)
                    )
                    for variant in description_variants
                }
            else:
                video_index["comparison_description_paths"][round_name] = str(
                    round_description_path(round_dir, video_id, round_name).relative_to(
                        output_dir
                    )
                )

        index[video_id] = video_index

    return index


def parse_args() -> argparse.Namespace:
    """
    Parse command-line options for the comparison-only pipeline.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Get Danceverse video descriptions through comparison rounds."
        )
    )
    parser.add_argument(
        "--videos-dir",
        type=Path,
        action="append",
        default=None,
        help=(
            "Directory of target .mp4 videos. May be passed more than once; "
            "all videos are pooled before comparison groups are made. "
            "Defaults to data_upload/compressed."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/descriptions"),
    )
    parser.add_argument("--group-size", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fps", type=float, default=10)
    parser.add_argument("--model", default="gemini-2.5-pro")
    parser.add_argument(
        "--no-audio",
        action="store_true",
        help=(
            "Use *_no_audio prompt templates, omit musicality from the output "
            "schema, strip audio during preprocessing, and default to "
            "data_upload/compressed_no_audio when --videos-dir is omitted."
        ),
    )
    parser.add_argument(
        "--comparison-rounds",
        type=int,
        default=2,
        help="Number of comparison rounds to run. No synthesis is run.",
    )
    parser.add_argument(
        "--comparison-mode",
        choices=COMPARISON_MODES,
        default=None,
        help=(
            "Comparison grouping mode. 'random' uses latest comparison_v*.txt; "
            "'exemplar' uses latest comparison_exemplar_v*.txt; 'monster' "
            "runs random1/random2/exemplar1/exemplar2 per round."
        ),
    )
    parser.add_argument(
        "--compare-with-exemplar",
        type=int,
        choices=[0, 1],
        default=None,
        help=(
            "Deprecated alias for --comparison-mode exemplar when 1, random when 0."
        ),
    )
    parser.add_argument(
        "--exemplar-dir",
        type=Path,
        default=DEFAULT_EXEMPLAR_DIR,
        help=(
            "Directory of exemplar-only .mp4 files used when "
            "--comparison-mode exemplar or monster."
        ),
    )
    parser.add_argument(
        "--exemplar-prefixes",
        nargs="+",
        default=None,
        help=(
            "Optional video-id prefixes to choose exemplars from --exemplar-dir. "
            "If omitted, all videos in --exemplar-dir are used."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Regenerate existing comparison-round JSON files.",
    )
    return parser.parse_args()


def main() -> None:
    """
    Run comparison rounds only.
    """
    args = parse_args()
    comparison_mode = resolve_comparison_mode(args)
    default_videos_dir = (
        Path("data_upload/compressed_no_audio")
        if args.no_audio
        else Path("data_upload/compressed")
    )
    videos_dirs = [
        resolve_project_path(path)
        for path in (args.videos_dir or [default_videos_dir])
    ]
    exemplar_dir = resolve_project_path(args.exemplar_dir)
    output_dir = resolve_project_path(args.output_dir)

    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key or api_key == "PASTE_KEY_HERE":
        raise ValueError("Set GEMINI_API_KEY in .env before running this script.")

    if args.group_size < 3:
        raise ValueError("--group-size must be at least 3.")
    if args.comparison_rounds < 1:
        raise ValueError("--comparison-rounds must be at least 1.")

    for videos_dir in videos_dirs:
        if not videos_dir.exists():
            raise FileNotFoundError(f"Video folder not found: {videos_dir}")

    target_video_paths = []
    for videos_dir in videos_dirs:
        target_video_paths.extend(find_video_files(videos_dir))
    target_video_paths = sorted(set(target_video_paths))
    if not target_video_paths:
        print(
            "No .mp4 videos found in "
            + ", ".join(str(videos_dir) for videos_dir in videos_dirs)
            + "."
        )
        return

    target_video_ids = build_video_id_map(target_video_paths, videos_dirs)
    target_source_labels = build_video_source_labels(target_video_paths, videos_dirs)

    exemplars: list[Path] = []
    exemplar_video_ids: dict[Path, str] = {}
    exemplar_source_labels: dict[Path, str] = {}
    if comparison_mode in {"exemplar", "monster"}:
        if not exemplar_dir.exists():
            raise FileNotFoundError(f"Exemplar folder not found: {exemplar_dir}")

        exemplar_video_paths = find_video_files(exemplar_dir)
        if not exemplar_video_paths:
            raise ValueError(f"No .mp4 exemplar videos found in {exemplar_dir}.")

        exemplar_video_ids = build_video_id_map(exemplar_video_paths, [exemplar_dir])
        exemplar_source_labels = build_video_source_labels(
            exemplar_video_paths,
            [exemplar_dir],
        )
        exemplars = select_exemplar_videos(
            exemplar_video_paths,
            exemplar_video_ids,
            args.exemplar_prefixes,
        )
        validate_exemplar_set(exemplars)

        exemplar_set = set(exemplars)
        target_video_paths = [
            video_path for video_path in target_video_paths if video_path not in exemplar_set
        ]
        target_video_ids = {
            video_path: video_id
            for video_path, video_id in target_video_ids.items()
            if video_path in target_video_paths
        }
        target_source_labels = {
            video_path: label
            for video_path, label in target_source_labels.items()
            if video_path in target_video_paths
        }

    video_ids = {**target_video_ids, **exemplar_video_ids}
    video_source_labels = {**target_source_labels, **exemplar_source_labels}
    if comparison_mode == "monster":
        random_prompt_path = latest_prompt_template_path(
            COMPARISON_PROMPT_NAME,
            no_audio=args.no_audio,
        )
        exemplar_prompt_path = latest_prompt_template_path(
            COMPARISON_EXEMPLAR_PROMPT_NAME,
            no_audio=args.no_audio,
        )
    else:
        comparison_prompt_name = prompt_name_for_comparison_mode(comparison_mode)
        comparison_prompt_path = latest_prompt_template_path(
            comparison_prompt_name,
            no_audio=args.no_audio,
        )

    print(
        f"Found {len(target_video_paths)} target videos in "
        + ", ".join(str(videos_dir) for videos_dir in videos_dirs)
    )
    if comparison_mode in {"exemplar", "monster"}:
        print(f"Found {len(exemplars)} exemplar videos in {exemplar_dir}")
        print("Exemplars: " + ", ".join(video_ids[path] for path in exemplars))
    print(f"Output directory: {output_dir}")
    print(f"Comparison mode: {comparison_mode}")
    print(f"No audio mode: {args.no_audio}")
    if comparison_mode == "monster":
        print(f"Random comparison prompt: {random_prompt_path.name}")
        print(f"Exemplar comparison prompt: {exemplar_prompt_path.name}")
        print("Description variants: " + ", ".join(MONSTER_VARIANTS))
    else:
        print(f"Comparison prompt: {comparison_prompt_path.name}")

    with genai.Client(api_key=api_key) as client:
        round_names = []
        for round_number in range(1, args.comparison_rounds + 1):
            round_name = f"round{round_number}"
            round_names.append(round_name)
            if comparison_mode == "monster":
                group_contexts: list[dict] = []
                for variant_index, variant in enumerate(MONSTER_RANDOM_VARIANTS):
                    run_round(
                        client=client,
                        round_name=round_name,
                        seed=monster_seed(args.seed, round_number, variant_index),
                        comparison_attempt=round_number,
                        video_paths=target_video_paths,
                        video_ids=video_ids,
                        video_source_labels=video_source_labels,
                        output_dir=output_dir,
                        group_size=args.group_size,
                        fps=args.fps,
                        model=args.model,
                        overwrite=args.overwrite,
                        comparison_mode="monster_random",
                        exemplars=exemplars,
                        description_variant=variant,
                        group_contexts=group_contexts,
                        no_audio=args.no_audio,
                    )
                for variant in MONSTER_EXEMPLAR_VARIANTS:
                    run_round(
                        client=client,
                        round_name=round_name,
                        seed=args.seed,
                        comparison_attempt=round_number,
                        video_paths=target_video_paths,
                        video_ids=video_ids,
                        video_source_labels=video_source_labels,
                        output_dir=output_dir,
                        group_size=args.group_size,
                        fps=args.fps,
                        model=args.model,
                        overwrite=args.overwrite,
                        comparison_mode="monster_exemplar",
                        exemplars=exemplars,
                        description_variant=variant,
                        group_contexts=group_contexts,
                        no_audio=args.no_audio,
                    )
            else:
                run_round(
                    client=client,
                    round_name=round_name,
                    seed=args.seed + round_number - 1,
                    comparison_attempt=round_number,
                    video_paths=target_video_paths,
                    video_ids=video_ids,
                    video_source_labels=video_source_labels,
                    output_dir=output_dir,
                    group_size=args.group_size,
                    fps=args.fps,
                    model=args.model,
                    overwrite=args.overwrite,
                    comparison_mode=comparison_mode,
                    exemplars=exemplars,
                    no_audio=args.no_audio,
                )

    index = build_master_index(
        video_paths=target_video_paths,
        video_ids=video_ids,
        video_source_labels=video_source_labels,
        output_dir=output_dir,
        round_names=round_names,
        description_variants=list(MONSTER_VARIANTS) if comparison_mode == "monster" else None,
    )
    index_path = output_dir / "index.json"
    write_json(index_path, index)
    print(f"Wrote index: {index_path}")

    print("\nDescription pipeline complete.")


if __name__ == "__main__":
    main()
