"""Get temporal movement labels for DanceVerse videos."""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from dotenv import load_dotenv
from google import genai
from google.genai import types

from get_descriptions import (
    INVALID_JSON_ATTEMPTS,
    INVALID_JSON_BACKOFF_SECONDS,
    build_video_id_map,
    build_video_source_labels,
    fallback_fps_values,
    find_video_files,
    group_context,
    json_generation_config,
    latest_prompt_template_path,
    parse_response_json,
    partition_videos,
    pretty_video_count,
    read_prompt_template,
    resolve_project_path,
    round_suffix,
    retry_transient_api_call,
    upload_video,
    wait_for_uploaded_video,
    write_json,
)


PROMPT_NAME = "movement_timeline"
MAX_ANALYSIS_SECONDS = 30.0
DEFAULT_INTERVAL_SECONDS = 1.0
DEFAULT_SECONDARY_LABEL_WEIGHT = 0.75
MOVEMENT_LABELS = [
    "upper_body",
    "lower_body",
    "full_body",
    "traveling",
    "low_level",
    "jump",
    "turn",
    "still",
    "mixed",
    "unclear",
]
QUALITY_LABELS = [
    "flowy",
    "groovy",
    "sharp",
    "intricate",
    "powerful",
    "laid_back",
    "neutral",
    "still",
    "unclear",
]
SPEED_LABELS = ["fast", "moderate", "slow", "still", "unclear"]
VOCABS = {
    "movement": MOVEMENT_LABELS,
    "quality": QUALITY_LABELS,
    "speed": SPEED_LABELS,
}
LABELS_PER_INTERVAL = len(MOVEMENT_LABELS) + len(QUALITY_LABELS) + len(SPEED_LABELS)
MULTI_LABEL_FIELDS = {"movement", "quality"}


def ffprobe_duration(video_path: Path) -> float:
    """Return video duration in seconds using ffprobe."""
    command = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(video_path),
    ]
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=True)
    except FileNotFoundError as error:
        raise RuntimeError("ffprobe is required to read video durations.") from error
    except subprocess.CalledProcessError as error:
        details = error.stderr.strip() or error.stdout.strip()
        raise RuntimeError(f"ffprobe failed for {video_path}: {details}") from error

    try:
        duration = float(result.stdout.strip())
    except ValueError as error:
        raise RuntimeError(f"Could not parse duration for {video_path}.") from error

    if duration <= 0:
        raise ValueError(f"Video duration must be positive for {video_path}.")
    return duration


def create_timeline_video(video_path: Path, fps: float, duration_limit: float) -> Path:
    """Create a temporary low-FPS, no-audio MP4 clipped to the requested duration."""
    if fps <= 0:
        raise ValueError("--fps must be greater than 0.")
    if duration_limit <= 0:
        raise ValueError("duration_limit must be greater than 0.")

    temp_file = tempfile.NamedTemporaryFile(
        prefix="dance_timeline_",
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
        "-t",
        f"{duration_limit:.3f}",
        "-vf",
        f"fps={fps:g}",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "23",
        "-an",
        str(output_path),
    ]

    try:
        subprocess.run(command, capture_output=True, text=True, check=True)
    except FileNotFoundError as error:
        output_path.unlink(missing_ok=True)
        raise RuntimeError("ffmpeg is required to process videos before upload.") from error
    except subprocess.CalledProcessError as error:
        output_path.unlink(missing_ok=True)
        details = error.stderr.strip() or error.stdout.strip()
        message = f"ffmpeg failed while processing {video_path}."
        if details:
            message = f"{message}\n\nffmpeg output:\n{details}"
        raise RuntimeError(message) from error

    return output_path


def interval_endpoint(value: float) -> float:
    """Round interval endpoints stably for prompt and validation."""
    return round(value + 0.0, 3)


def final_timeline_positions(interval_seconds: float) -> int:
    """Return the fixed number of looped positions for the interval length."""
    if interval_seconds <= 0:
        raise ValueError("--interval-seconds must be greater than 0.")
    return int(math.ceil(MAX_ANALYSIS_SECONDS / interval_seconds))


def timeline_vector_dimensions(interval_seconds: float) -> int:
    """Return flattened vector dimensions for the interval length."""
    return final_timeline_positions(interval_seconds) * LABELS_PER_INTERVAL


def intervals_for_duration(
    duration: float,
    interval_seconds: float = DEFAULT_INTERVAL_SECONDS,
) -> tuple[float, list[dict[str, float]]]:
    """Return consecutive intervals over the first <=30 seconds."""
    if interval_seconds <= 0:
        raise ValueError("--interval-seconds must be greater than 0.")
    analyzed_duration = interval_endpoint(min(duration, MAX_ANALYSIS_SECONDS))
    interval_count = int(math.ceil(analyzed_duration / interval_seconds))
    intervals = []
    for index in range(interval_count):
        start = interval_endpoint(float(index) * interval_seconds)
        end = interval_endpoint(min(float(index + 1) * interval_seconds, analyzed_duration))
        if end > start:
            intervals.append({"start": start, "end": end})
    return analyzed_duration, intervals


def top_level_timeline_schema(video_keys: list[str], label_mode: str = "single") -> str:
    """Render the exact top-level timeline JSON shape expected from Gemini."""
    movement_value = '["..."]' if label_mode == "multi" else '"..."'
    quality_value = '["..."]' if label_mode == "multi" else '"..."'
    objects = []
    for index, video_key in enumerate(video_keys):
        comma = "," if index < len(video_keys) - 1 else ""
        objects.append(
            f'  "{video_key}": {{\n'
            '    "timeline": [\n'
            "      {\n"
            '        "start": 0.0,\n'
            '        "end": 1.0,\n'
            f'        "movement": {movement_value},\n'
            f'        "quality": {quality_value},\n'
            '        "speed": "..."\n'
            "      }\n"
            f"    ]\n"
            f"  }}{comma}"
        )
    return "{\n" + "\n".join(objects) + "\n}"


def format_interval_seconds(interval_seconds: float) -> str:
    """Format interval length for model-facing prompt text."""
    if abs(interval_seconds - round(interval_seconds)) <= 0.001:
        return str(int(round(interval_seconds)))
    return f"{interval_seconds:g}"


def build_timeline_prompt(
    intervals_by_key: dict[str, list[dict[str, float]]],
    interval_seconds: float = DEFAULT_INTERVAL_SECONDS,
    prompt_path: Optional[Path] = None,
    label_mode: str = "single",
) -> str:
    """Build the movement-timeline prompt."""
    video_keys = list(intervals_by_key)
    template = read_prompt_template(prompt_path or latest_prompt_template_path(PROMPT_NAME))
    if abs(interval_seconds - DEFAULT_INTERVAL_SECONDS) > 0.001:
        seconds_text = format_interval_seconds(interval_seconds)
        template = template.replace(
            "consecutive 1-second intervals",
            f"consecutive {seconds_text}-second intervals",
        )
        template = template.replace(
            "consecutive 1 second intervals",
            f"consecutive {seconds_text} second intervals",
        )
    replacements = {
        "{count_word}": pretty_video_count(len(video_keys)),
        "{key_list}": ", ".join(video_keys),
        "{interval_lists}": json.dumps(intervals_by_key, indent=2),
        "{schema_shape}": top_level_timeline_schema(video_keys, label_mode=label_mode),
    }
    for placeholder, value in replacements.items():
        template = template.replace(placeholder, value)
    return template


def upload_timeline_videos(
    client: genai.Client,
    video_paths: list[Path],
    fps: float,
    analyzed_durations: dict[Path, float],
) -> tuple[list[types.File], list[Path]]:
    """Clip/downsample/no-audio process and upload videos for timeline analysis."""
    processed_paths: list[Path] = []
    uploaded_files: list[types.File] = []
    try:
        for video_path in video_paths:
            processed_path = create_timeline_video(
                video_path,
                fps=fps,
                duration_limit=analyzed_durations[video_path],
            )
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
        cleanup_timeline_temp_files(processed_paths)
        raise

    return uploaded_files, processed_paths


def cleanup_timeline_temp_files(processed_paths: list[Path]) -> None:
    """Remove local temporary timeline videos.

    Gemini file deletion can hang for minutes during API turbulence. These
    experimental uploads expire remotely, so the timeline runner keeps forward
    progress by only removing local temp files here.
    """
    for processed_path in processed_paths:
        processed_path.unlink(missing_ok=True)


def same_timestamp(actual: Any, expected: float) -> bool:
    """Compare model timestamps to requested timestamps with tiny tolerance."""
    return isinstance(actual, (int, float)) and abs(float(actual) - expected) <= 0.001


def validate_timeline_response(
    result: dict,
    intervals_by_key: dict[str, list[dict[str, float]]],
    label_mode: str = "single",
) -> list[str]:
    """Return validation errors for a movement-timeline response."""
    errors: list[str] = []
    expected_keys = set(intervals_by_key)
    actual_keys = set(result) if isinstance(result, dict) else set()
    if actual_keys != expected_keys:
        errors.append(
            "Top-level keys must be exactly "
            f"{sorted(expected_keys)}, got {sorted(actual_keys)}."
        )
        if not isinstance(result, dict):
            return errors

    for video_key, expected_intervals in intervals_by_key.items():
        video_result = result.get(video_key)
        if not isinstance(video_result, dict):
            errors.append(f"{video_key} must be an object.")
            continue
        timeline = video_result.get("timeline")
        if not isinstance(timeline, list):
            errors.append(f"{video_key}.timeline must be a list.")
            continue
        if len(timeline) != len(expected_intervals):
            errors.append(
                f"{video_key}.timeline must contain {len(expected_intervals)} intervals, "
                f"got {len(timeline)}."
            )
            continue

        seen = set()
        for index, (item, expected) in enumerate(zip(timeline, expected_intervals)):
            if not isinstance(item, dict):
                errors.append(f"{video_key}.timeline[{index}] must be an object.")
                continue
            start = item.get("start")
            end = item.get("end")
            if not same_timestamp(start, expected["start"]):
                errors.append(
                    f"{video_key}.timeline[{index}].start must be {expected['start']}, "
                    f"got {start!r}."
                )
            if not same_timestamp(end, expected["end"]):
                errors.append(
                    f"{video_key}.timeline[{index}].end must be {expected['end']}, "
                    f"got {end!r}."
                )
            pair = (float(start), float(end)) if isinstance(start, (int, float)) and isinstance(end, (int, float)) else None
            if pair in seen:
                errors.append(f"{video_key}.timeline has duplicated interval {pair}.")
            if pair is not None:
                seen.add(pair)

            for field, labels in VOCABS.items():
                value = item.get(field)
                if label_mode == "multi" and field in MULTI_LABEL_FIELDS:
                    if not isinstance(value, list):
                        errors.append(
                            f"{video_key}.timeline[{index}].{field} must be a JSON array "
                            f"containing one or two labels, got {value!r}."
                        )
                        continue
                    if not 1 <= len(value) <= 2:
                        errors.append(
                            f"{video_key}.timeline[{index}].{field} must contain one or "
                            f"two labels, got {value!r}."
                        )
                        continue
                    if len(set(value)) != len(value):
                        errors.append(
                            f"{video_key}.timeline[{index}].{field} must not repeat labels, "
                            f"got {value!r}."
                        )
                    invalid = [label for label in value if label not in labels]
                    if invalid:
                        errors.append(
                            f"{video_key}.timeline[{index}].{field} labels must be from "
                            f"{labels}, got invalid labels {invalid!r}."
                        )
                    if any(label in {"mixed", "unclear"} for label in value) and len(value) > 1:
                        errors.append(
                            f"{video_key}.timeline[{index}].{field} cannot combine mixed "
                            f"or unclear with another label, got {value!r}."
                        )
                    if field == "quality" and any(label in {"neutral", "still", "unclear"} for label in value) and len(value) > 1:
                        errors.append(
                            f"{video_key}.timeline[{index}].quality cannot combine neutral, "
                            f"still, or unclear with another label, got {value!r}."
                        )
                    if field == "movement" and "full_body" in value and (
                        "upper_body" in value or "lower_body" in value
                    ):
                        errors.append(
                            f"{video_key}.timeline[{index}].movement cannot combine full_body "
                            f"with upper_body or lower_body, got {value!r}."
                        )
                    continue

                if value not in labels:
                    errors.append(
                        f"{video_key}.timeline[{index}].{field} must be one of "
                        f"{labels}, got {value!r}."
                    )
    return errors


def correction_prompt(base_prompt: str, errors: list[str]) -> str:
    """Ask Gemini to correct only the invalid JSON structure/labels."""
    error_text = "\n".join(f"- {error}" for error in errors[:80])
    return (
        f"{base_prompt}\n\n"
        "Your previous response failed validation for these reasons:\n"
        f"{error_text}\n\n"
        "Return a corrected JSON object only."
    )


def one_hot(label: str, labels: list[str]) -> list[int]:
    """Encode one label in a fixed vocab order."""
    return [1 if candidate == label else 0 for candidate in labels]


def multi_hot(
    value: Any,
    labels: list[str],
    secondary_label_weight: float = DEFAULT_SECONDARY_LABEL_WEIGHT,
) -> list[float]:
    """Encode one or more ordered labels in a fixed vocab order."""
    selected = [value] if isinstance(value, str) else list(value)
    weights = {}
    for index, label in enumerate(selected):
        weights[label] = 1.0 if index == 0 else secondary_label_weight
    return [weights.get(candidate, 0.0) for candidate in labels]


def encode_interval(
    item: dict,
    secondary_label_weight: float = DEFAULT_SECONDARY_LABEL_WEIGHT,
) -> list[float]:
    """Encode one timeline interval as movement + quality + speed vectors."""
    return (
        multi_hot(item["movement"], MOVEMENT_LABELS, secondary_label_weight)
        + multi_hot(item["quality"], QUALITY_LABELS, secondary_label_weight)
        + one_hot(item["speed"], SPEED_LABELS)
    )


def encode_timeline(
    timeline: list[dict],
    interval_seconds: float = DEFAULT_INTERVAL_SECONDS,
    secondary_label_weight: float = DEFAULT_SECONDARY_LABEL_WEIGHT,
) -> tuple[list[list[float]], list[list[float]], list[float]]:
    """Encode real intervals, loop to 30 positions, and flatten interval-major."""
    encoded_real = [
        encode_interval(item, secondary_label_weight)
        for item in timeline
    ]
    final_positions = final_timeline_positions(interval_seconds)
    encoded_looped = [
        encoded_real[index % len(encoded_real)]
        for index in range(final_positions)
    ]
    flat = [value for interval in encoded_looped for value in interval]
    expected_dimensions = timeline_vector_dimensions(interval_seconds)
    if len(flat) != expected_dimensions:
        raise ValueError(f"Timeline vector must be {expected_dimensions} dims.")
    return encoded_real, encoded_looped, flat


def anchor_scores(
    timeline: list[dict],
    interval_seconds: float = DEFAULT_INTERVAL_SECONDS,
    secondary_label_weight: float = DEFAULT_SECONDARY_LABEL_WEIGHT,
) -> dict[str, dict[str, float]]:
    """Return weighted label activation rates across the looped timeline."""
    final_positions = final_timeline_positions(interval_seconds)
    looped = [timeline[index % len(timeline)] for index in range(final_positions)]
    scores = {}
    for field, labels in VOCABS.items():
        counts = {label: 0 for label in labels}
        for item in looped:
            value = item[field]
            selected = [value] if isinstance(value, str) else value
            for index, label in enumerate(selected):
                counts[label] += 1.0 if index == 0 else secondary_label_weight
        scores[field] = {
            label: counts[label] / final_positions
            for label in labels
        }
    return scores


def format_label_value(value: Any) -> str:
    """Render string or list labels compactly for embedding text."""
    if isinstance(value, list):
        return "+".join(value)
    return value


def timeline_text(timeline: list[dict]) -> str:
    """Render a compact text representation that get_embeddings.py can embed."""
    parts = []
    for item in timeline:
        parts.append(
            f"{item['start']}-{item['end']}: "
            f"movement {format_label_value(item['movement'])}, "
            f"quality {format_label_value(item['quality'])}, "
            f"speed {item['speed']}"
        )
    return " ".join(parts)


def timeline_output_path(
    round_dir: Path,
    video_id: str,
    round_name: str,
    variant: Optional[str] = None,
    output_layout: str = "timeline",
) -> Path:
    """Return the per-video timeline output path."""
    if output_layout == "descriptions":
        if variant is None:
            raise ValueError("Description-layout timeline outputs require a variant.")
        return round_dir / f"{video_id}.{round_suffix(round_name)}.{variant}.json"

    variant_suffix = f".{variant}" if variant else ""
    return round_dir / f"{video_id}.{round_name}{variant_suffix}.timeline.json"


def write_timeline_files(
    round_dir: Path,
    round_name: str,
    result: dict,
    context: dict,
    prompt_path: Path,
    model: str,
    durations: dict[str, float],
    analyzed_durations: dict[str, float],
    intervals_by_key: dict[str, list[dict[str, float]]],
    interval_seconds: float = DEFAULT_INTERVAL_SECONDS,
    label_mode: str = "single",
    secondary_label_weight: float = DEFAULT_SECONDARY_LABEL_WEIGHT,
    variant: Optional[str] = None,
    output_layout: str = "timeline",
) -> dict[str, dict]:
    """Split a valid group response into per-video timeline JSON files."""
    written = {}
    created_at = datetime.now(timezone.utc).isoformat()
    for video_key in context["write_video_keys"]:
        video_info = context["videos"][video_key]
        video_id = video_info["video_id"]
        timeline = result[video_key]["timeline"]
        final_positions = final_timeline_positions(interval_seconds)
        encoded_real, encoded_looped, vector = encode_timeline(
            timeline,
            interval_seconds,
            secondary_label_weight=secondary_label_weight,
        )
        record = {
            "video_id": video_id,
            "round": round_name,
            "source_filepath": video_info["filepath"],
            "model": model,
            "created_at": created_at,
            "prompt_name": PROMPT_NAME,
            "prompt_path": str(prompt_path),
            "timeline_label_mode": label_mode,
            "secondary_label_weight": secondary_label_weight,
            "description_variant": variant,
            "freeform_description": timeline_text(timeline),
            "movement_vocab": MOVEMENT_LABELS,
            "quality_vocab": QUALITY_LABELS,
            "speed_vocab": SPEED_LABELS,
            "video_duration": durations[video_key],
            "analyzed_duration": analyzed_durations[video_key],
            "interval_seconds": interval_seconds,
            "final_timeline_positions": final_positions,
            "requested_intervals": intervals_by_key[video_key],
            "num_real_intervals": len(timeline),
            "looping_applied": len(timeline) < final_positions,
            "timeline": timeline,
            "encoded_real_timeline": encoded_real,
            "encoded_looped_timeline": encoded_looped,
            "timeline_vector": vector,
            "timeline_vector_dimensions": len(vector),
            "anchor_scores": anchor_scores(
                timeline,
                interval_seconds,
                secondary_label_weight=secondary_label_weight,
            ),
        }
        output_path = timeline_output_path(
            round_dir,
            video_id,
            round_name,
            variant=variant,
            output_layout=output_layout,
        )
        write_json(output_path, record)
        written[video_id] = record
    return written


def run_timeline_group(
    client: genai.Client,
    group: list[Path],
    model: str,
    fps: float,
    raw_response_path: Path,
    durations_by_path: dict[Path, float],
    interval_seconds: float = DEFAULT_INTERVAL_SECONDS,
    prompt_path: Optional[Path] = None,
    label_mode: str = "single",
) -> tuple[dict, dict[str, float], dict[str, float], dict[str, list[dict[str, float]]]]:
    """Upload one group and ask Gemini for movement timelines."""
    durations: dict[str, float] = {}
    analyzed_durations: dict[str, float] = {}
    intervals_by_key: dict[str, list[dict[str, float]]] = {}
    analyzed_durations_by_path: dict[Path, float] = {}
    for index, video_path in enumerate(group, start=1):
        video_key = f"video_{index}"
        duration = durations_by_path[video_path]
        analyzed_duration, intervals = intervals_for_duration(duration, interval_seconds)
        durations[video_key] = duration
        analyzed_durations[video_key] = analyzed_duration
        analyzed_durations_by_path[video_path] = analyzed_duration
        intervals_by_key[video_key] = intervals

    base_prompt = build_timeline_prompt(
        intervals_by_key,
        interval_seconds,
        prompt_path=prompt_path,
        label_mode=label_mode,
    )
    group_label = ", ".join(path.name for path in group)
    last_error: Optional[Exception] = None

    for fps_attempt in fallback_fps_values(fps):
        uploaded_files: list[types.File] = []
        processed_paths: list[Path] = []
        try:
            uploaded_files, processed_paths = upload_timeline_videos(
                client,
                group,
                fps_attempt,
                analyzed_durations_by_path,
            )
            prompt = base_prompt
            for attempt in range(1, INVALID_JSON_ATTEMPTS + 1):
                response = retry_transient_api_call(
                    f"generating movement timeline for {group_label}",
                    lambda: client.models.generate_content(
                        model=model,
                        contents=[prompt, *uploaded_files],
                        config=json_generation_config(),
                    ),
                )
                raw_response_path.parent.mkdir(parents=True, exist_ok=True)
                raw_response_path.write_text(response.text or "", encoding="utf-8")
                try:
                    result = parse_response_json(response.text or "", raw_response_path)
                    errors = validate_timeline_response(
                        result,
                        intervals_by_key,
                        label_mode=label_mode,
                    )
                    if not errors:
                        return result, durations, analyzed_durations, intervals_by_key
                    raise ValueError("\n".join(errors))
                except ValueError as error:
                    last_error = error
                    if attempt == INVALID_JSON_ATTEMPTS:
                        break
                    prompt = correction_prompt(base_prompt, str(error).splitlines())
                    sleep_seconds = INVALID_JSON_BACKOFF_SECONDS * attempt
                    print(
                        f"Warning: invalid timeline JSON for {group_label} "
                        f"(attempt {attempt}/{INVALID_JSON_ATTEMPTS}). "
                        f"Retrying in {sleep_seconds}s.",
                        file=sys.stderr,
                    )
                    time.sleep(sleep_seconds)
        finally:
            cleanup_timeline_temp_files(processed_paths)

        print(
            f"Warning: movement timeline group for {group_label} failed at "
            f"{fps_attempt:g} fps.",
            file=sys.stderr,
        )

    raise ValueError(f"Could not get a valid timeline for {group_label}.") from last_error


def parse_args() -> argparse.Namespace:
    """Parse command-line options."""
    parser = argparse.ArgumentParser(
        description="Get temporal movement timeline labels for no-audio videos."
    )
    parser.add_argument(
        "--videos-dir",
        type=Path,
        action="append",
        default=None,
        help=(
            "Directory of .mp4 videos. May be passed more than once. "
            "Defaults to data_upload/compressed_no_audio."
        ),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("data/movement_timelines"))
    parser.add_argument("--group-size", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fps", type=float, default=10)
    parser.add_argument("--model", default="gemini-2.5-pro")
    parser.add_argument("--timeline-rounds", type=int, default=1)
    parser.add_argument(
        "--interval-seconds",
        type=float,
        default=DEFAULT_INTERVAL_SECONDS,
        help="Interval length for movement timeline labels. Defaults to 1 second.",
    )
    parser.add_argument(
        "--prompt-path",
        type=Path,
        default=None,
        help=(
            "Prompt template to use. Defaults to the latest movement_timeline_v*.txt. "
            "Use config/prompts/movement_timeline_v0.txt with "
            "--timeline-label-mode single to run the old pipeline."
        ),
    )
    parser.add_argument(
        "--timeline-label-mode",
        choices=["single", "multi"],
        default="multi",
        help=(
            "Label format expected from the prompt. 'multi' supports v1 arrays for "
            "movement/quality; 'single' preserves the old v0 string format."
        ),
    )
    parser.add_argument(
        "--secondary-label-weight",
        type=float,
        default=DEFAULT_SECONDARY_LABEL_WEIGHT,
        help=(
            "Weight for the second movement/quality label in multi-label timelines. "
            "The first label is always weighted 1.0. Defaults to 0.75."
        ),
    )
    parser.add_argument(
        "--timeline-variants",
        type=int,
        default=1,
        help="Number of independent timeline variants per round.",
    )
    parser.add_argument(
        "--output-layout",
        choices=["timeline", "descriptions"],
        default="timeline",
        help=(
            "'timeline' writes video.roundN.timeline.json files. "
            "'descriptions' writes video.rN.timelineK.json files that "
            "get_embeddings.py treats as description variants."
        ),
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Run movement-timeline extraction."""
    args = parse_args()
    if args.group_size < 3:
        raise ValueError("--group-size must be at least 3.")
    if args.timeline_rounds < 1:
        raise ValueError("--timeline-rounds must be at least 1.")
    if args.timeline_variants < 1:
        raise ValueError("--timeline-variants must be at least 1.")
    if args.interval_seconds <= 0:
        raise ValueError("--interval-seconds must be greater than 0.")
    if args.secondary_label_weight < 0:
        raise ValueError("--secondary-label-weight must be nonnegative.")

    videos_dirs = [
        resolve_project_path(path)
        for path in (args.videos_dir or [Path("data_upload/compressed_no_audio")])
    ]
    output_dir = resolve_project_path(args.output_dir)
    prompt_path = (
        resolve_project_path(args.prompt_path)
        if args.prompt_path
        else latest_prompt_template_path(PROMPT_NAME)
    )

    for videos_dir in videos_dirs:
        if not videos_dir.exists():
            raise FileNotFoundError(f"Video folder not found: {videos_dir}")

    video_paths = []
    for videos_dir in videos_dirs:
        video_paths.extend(find_video_files(videos_dir))
    video_paths = sorted(set(video_paths))
    if not video_paths:
        print("No .mp4 videos found.")
        return

    video_ids = build_video_id_map(video_paths, videos_dirs)
    video_source_labels = build_video_source_labels(video_paths, videos_dirs)
    durations_by_path = {path: ffprobe_duration(path) for path in video_paths}

    print(f"Found {len(video_paths)} videos.")
    print(f"Output directory: {output_dir}")
    print(f"Movement timeline prompt: {prompt_path.name}")
    print(f"Timeline label mode: {args.timeline_label_mode}")
    print(f"Secondary label weight: {args.secondary_label_weight:g}")
    print(f"Interval length: {args.interval_seconds:g}s")
    print("Videos have audio stripped during preprocessing.")

    if args.dry_run:
        for video_path in video_paths:
            analyzed_duration, intervals = intervals_for_duration(
                durations_by_path[video_path],
                args.interval_seconds,
            )
            print(
                f"{video_ids[video_path]}: duration={durations_by_path[video_path]:.3f}s, "
                f"analyzed={analyzed_duration:.3f}s, intervals={len(intervals)}"
            )
        return

    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key or api_key == "PASTE_KEY_HERE":
        raise ValueError("Set GEMINI_API_KEY in .env before running this script.")

    with genai.Client(api_key=api_key) as client:
        for round_number in range(1, args.timeline_rounds + 1):
            round_name = f"round{round_number}"
            round_dir = output_dir / round_name
            for variant_number in range(1, args.timeline_variants + 1):
                variant = f"timeline{variant_number}"
                variant_seed = (
                    args.seed
                    + (round_number - 1) * args.timeline_variants
                    + variant_number
                    - 1
                )
                groups = partition_videos(video_paths, args.group_size, variant_seed)
                group_contexts = []
                for group_index, group in enumerate(groups, start=1):
                    context = group_context(
                        group,
                        video_ids=video_ids,
                        video_source_labels=video_source_labels,
                        round_name=round_name,
                        group_index=group_index,
                        seed=variant_seed,
                        grouping_mode="movement_timeline",
                        description_variant=variant,
                        no_audio=True,
                    )
                    group_contexts.append(context)
                    pending = [
                        key
                        for key, video_info in context["videos"].items()
                        if args.overwrite
                        or not timeline_output_path(
                            round_dir,
                            video_info["video_id"],
                            round_name,
                            variant=variant,
                            output_layout=args.output_layout,
                        ).exists()
                    ]
                    if not pending:
                        print(
                            f"SKIP {round_name} {variant} group {group_index:03d}: "
                            "already complete"
                        )
                        continue
                    context["write_video_keys"] = pending
                    raw_response_path = (
                        round_dir
                        / "_raw_responses"
                        / variant
                        / f"group_{group_index:03d}.txt"
                    )
                    result, durations, analyzed_durations, intervals_by_key = run_timeline_group(
                        client=client,
                        group=group,
                        model=args.model,
                        fps=args.fps,
                        raw_response_path=raw_response_path,
                        durations_by_path=durations_by_path,
                        interval_seconds=args.interval_seconds,
                        prompt_path=prompt_path,
                        label_mode=args.timeline_label_mode,
                    )
                    write_timeline_files(
                        round_dir=round_dir,
                        round_name=round_name,
                        result=result,
                        context=context,
                        prompt_path=prompt_path,
                        model=args.model,
                        durations=durations,
                        analyzed_durations=analyzed_durations,
                        intervals_by_key=intervals_by_key,
                        interval_seconds=args.interval_seconds,
                        label_mode=args.timeline_label_mode,
                        secondary_label_weight=args.secondary_label_weight,
                        variant=variant,
                        output_layout=args.output_layout,
                    )
                    write_json(
                        round_dir / f"_groups_{variant}.json",
                        {"round": round_name, "variant": variant, "groups": group_contexts},
                    )
                    print(
                        f"WROTE {round_name} {variant} group {group_index:03d}: "
                        f"{len(pending)} files"
                    )
                write_json(
                    round_dir / f"_groups_{variant}.json",
                    {"round": round_name, "variant": variant, "groups": group_contexts},
                )


if __name__ == "__main__":
    main()
