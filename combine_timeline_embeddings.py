"""Combine DanceVerse description embeddings with movement-timeline vectors."""

from __future__ import annotations

import argparse
import json
import sys
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional


FEATURE_MODES = ("description-only", "timeline-only", "combined")
TIMELINE_VECTOR_DIMENSIONS = 720


def resolve_project_path(path: Path) -> Path:
    """Resolve relative paths from the project root."""
    if path.is_absolute():
        return path
    return (Path(__file__).resolve().parent / path).resolve()


def load_json(path: Path) -> Any:
    """Read JSON from disk."""
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: dict) -> None:
    """Write pretty JSON, creating parent folders first."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def timeline_sort_key(path: Path) -> tuple[int, str]:
    """Sort timeline files by round number and path."""
    round_name = path.parent.name
    if round_name.startswith("round") and round_name.removeprefix("round").isdigit():
        return (int(round_name.removeprefix("round")), str(path))
    return (sys.maxsize, str(path))


def collect_timeline_records(timeline_dir: Path) -> tuple[dict[tuple[str, str], dict], dict[str, list[dict]]]:
    """Collect per-video movement timeline records."""
    by_video_round: dict[tuple[str, str], dict] = {}
    by_video: dict[str, list[dict]] = {}

    paths = [
        path
        for path in sorted(timeline_dir.rglob("*.json"), key=timeline_sort_key)
        if not path.name.startswith("_")
    ]
    for path in paths:
        data = load_json(path)
        video_id = data.get("video_id")
        round_name = data.get("round") or path.parent.name
        vector = data.get("timeline_vector")
        if not isinstance(video_id, str) or not video_id:
            raise ValueError(f"Missing video_id in {path}.")
        if not isinstance(round_name, str) or not round_name:
            raise ValueError(f"Missing round in {path}.")
        if not isinstance(vector, list) or len(vector) != TIMELINE_VECTOR_DIMENSIONS:
            raise ValueError(
                f"{path} must contain a {TIMELINE_VECTOR_DIMENSIONS}-dim timeline_vector."
            )

        record = {
            "video_id": video_id,
            "round": round_name,
            "source_timeline_path": str(path.relative_to(timeline_dir)),
            "timeline_vector": [float(value) for value in vector],
            "timeline_metadata": {
                "analyzed_duration": data.get("analyzed_duration"),
                "num_real_intervals": data.get("num_real_intervals"),
                "looping_applied": data.get("looping_applied"),
                "anchor_scores": data.get("anchor_scores"),
            },
        }
        by_video_round[(video_id, round_name)] = record
        by_video.setdefault(video_id, []).append(record)

    return by_video_round, by_video


def select_timeline_record(
    embedding_record: dict,
    by_video_round: dict[tuple[str, str], dict],
    by_video: dict[str, list[dict]],
) -> Optional[dict]:
    """Find the best timeline record for one embedding record."""
    video_id = embedding_record.get("video_id")
    round_name = embedding_record.get("round")
    if not isinstance(video_id, str):
        return None

    if isinstance(round_name, str):
        exact = by_video_round.get((video_id, round_name))
        if exact is not None:
            return exact

    candidates = by_video.get(video_id, [])
    if not candidates:
        return None
    return candidates[0]


def scale_vector(vector: list[float], weight: float) -> list[float]:
    """Scale a numeric vector."""
    return [float(value) * weight for value in vector]


def combine_record(
    record: dict,
    timeline_record: Optional[dict],
    feature_mode: str,
    timeline_weight: float,
    require_timeline: bool,
) -> dict:
    """Return one output embedding record."""
    output = deepcopy(record)
    description_embedding = output.get("embedding")
    if not isinstance(description_embedding, list):
        raise ValueError(f"Missing description embedding for {record.get('embedding_id')}.")

    timeline_vector = None
    if timeline_record is not None:
        timeline_vector = scale_vector(timeline_record["timeline_vector"], timeline_weight)
    elif feature_mode != "description-only" and require_timeline:
        raise ValueError(f"Missing timeline for video_id {record.get('video_id')!r}.")

    if feature_mode == "description-only":
        output["embedding"] = [float(value) for value in description_embedding]
    elif feature_mode == "timeline-only":
        if timeline_vector is None:
            raise ValueError(f"Missing timeline for video_id {record.get('video_id')!r}.")
        output["embedding"] = timeline_vector
    elif feature_mode == "combined":
        if timeline_vector is None:
            raise ValueError(f"Missing timeline for video_id {record.get('video_id')!r}.")
        output["embedding"] = [float(value) for value in description_embedding] + timeline_vector
    else:
        raise ValueError(f"Unsupported feature mode: {feature_mode}")

    output["feature_mode"] = feature_mode
    output["description_embedding_dimensions"] = len(description_embedding)
    output["timeline_vector_dimensions"] = (
        len(timeline_vector) if timeline_vector is not None else 0
    )
    output["timeline_weight"] = timeline_weight
    if timeline_record is not None:
        output["source_timeline_path"] = timeline_record["source_timeline_path"]
        output["timeline_metadata"] = timeline_record["timeline_metadata"]

    return output


def parse_args() -> argparse.Namespace:
    """Parse command-line options."""
    parser = argparse.ArgumentParser(
        description="Append movement-timeline vectors to existing DanceVerse embeddings."
    )
    parser.add_argument(
        "--embeddings-path",
        type=Path,
        required=True,
        help="Existing embeddings.json produced by get_embeddings.py.",
    )
    parser.add_argument(
        "--timeline-dir",
        type=Path,
        required=True,
        help="Directory produced by get_movement_timeline.py.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        required=True,
        help="Output embeddings JSON path.",
    )
    parser.add_argument(
        "--feature-mode",
        choices=FEATURE_MODES,
        default="combined",
    )
    parser.add_argument("--timeline-weight", type=float, default=1.0)
    parser.add_argument(
        "--allow-missing-timeline",
        action="store_true",
        help="Keep description-only records when no matching timeline exists.",
    )
    return parser.parse_args()


def main() -> None:
    """Combine embedding records with timeline vectors."""
    args = parse_args()
    embeddings_path = resolve_project_path(args.embeddings_path)
    timeline_dir = resolve_project_path(args.timeline_dir)
    output_path = resolve_project_path(args.output_path)

    source = load_json(embeddings_path)
    records = source.get("records", [])
    if not isinstance(records, list) or not records:
        raise ValueError(f"No records found in {embeddings_path}.")

    by_video_round, by_video = collect_timeline_records(timeline_dir)
    if not by_video_round:
        raise ValueError(f"No timeline JSON files found under {timeline_dir}.")

    missing = []
    combined_records = []
    require_timeline = not args.allow_missing_timeline
    for record in records:
        timeline_record = select_timeline_record(record, by_video_round, by_video)
        if timeline_record is None:
            missing.append(record.get("video_id"))
            if args.feature_mode != "description-only" and require_timeline:
                continue
        combined_records.append(
            combine_record(
                record=record,
                timeline_record=timeline_record,
                feature_mode=args.feature_mode,
                timeline_weight=args.timeline_weight,
                require_timeline=require_timeline,
            )
        )

    if missing and args.feature_mode != "description-only" and require_timeline:
        unique_missing = sorted({str(video_id) for video_id in missing})
        preview = ", ".join(unique_missing[:20])
        raise ValueError(
            f"Missing timelines for {len(unique_missing)} video ids: {preview}"
        )

    dimensions = {len(record["embedding"]) for record in combined_records}
    if len(dimensions) != 1:
        raise ValueError(f"Inconsistent output embedding dimensions: {sorted(dimensions)}")

    output = {
        **{key: value for key, value in source.items() if key != "records"},
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_embeddings_path": str(embeddings_path),
        "timeline_dir": str(timeline_dir),
        "feature_mode": args.feature_mode,
        "timeline_weight": args.timeline_weight,
        "timeline_vector_dimensions": TIMELINE_VECTOR_DIMENSIONS,
        "embedding_dimensions": dimensions.pop(),
        "records": combined_records,
    }
    write_json(output_path, output)
    print(f"Wrote {len(combined_records)} records to {output_path}")


if __name__ == "__main__":
    main()
