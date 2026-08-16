"""Embed DanceVerse round descriptions and compare round1/round2 agreement."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional, TypeVar

import numpy as np
from dotenv import load_dotenv
from google import genai


DEFAULT_EMBEDDING_MODEL = "gemini-embedding-001"

# Change this project default to choose what text gets embedded.
# Options:
# - "direct_json": embed stable pretty-printed JSON, including field names.
# - "concatenated_values": embed only field values joined into one paragraph.
DEFAULT_TEXT_MODE = "concatenated_values"

DEFAULT_OUTPUT_DIRS = {
    "direct_json": Path("data/embeddings/descriptions_direct_json_v0"),
    "concatenated_values": Path("data/embeddings/descriptions_concatenated_values_v0"),
}
DESCRIPTION_FIELD_ORDER = [
    "freeform_description",
    "body_emphasis",
    "energy",
    "foundation_cues",
    "musicality",
    "tempo_feel",
    "texture",
    "salient_differences",
    "embedding_summary",
    "salient_characteristics",
]
TRANSIENT_API_ATTEMPTS = 3
TRANSIENT_API_BACKOFF_SECONDS = 10
BATCH_SIZE = 32

T = TypeVar("T")


def load_json(path: Path) -> Any:
    """Read a JSON file into a dictionary."""
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: dict) -> None:
    """Write pretty JSON, creating parent folders first."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def stable_json_text(data: Any) -> str:
    """
    Serialize the description JSON exactly as it will be sent to the embedder.
    """
    return json.dumps(data, indent=2, sort_keys=True, ensure_ascii=False)


def concatenated_values_text(data: Any) -> str:
    """
    Join description field values into one paragraph without JSON keys.
    """
    if isinstance(data, str):
        return data.strip()

    if not isinstance(data, dict):
        return str(data).strip()

    values = []
    for field in DESCRIPTION_FIELD_ORDER:
        value = data.get(field)
        if value is None:
            continue

        text = str(value).strip()
        if text:
            values.append(text)

    return " ".join(values)


def embedding_text(data: Any, text_mode: str) -> str:
    """
    Build the exact text sent to the embedding model.
    """
    if text_mode == "direct_json":
        return stable_json_text(data)
    if text_mode == "concatenated_values":
        return concatenated_values_text(data)

    raise ValueError(f"Unsupported text mode: {text_mode}")


def resolve_project_path(path: Path) -> Path:
    """Resolve relative paths from the project root."""
    if path.is_absolute():
        return path

    project_root = Path(__file__).resolve().parent
    return (project_root / path).resolve()


def parse_video_id_and_variant_from_path(
    path: Path,
    round_suffix: str,
) -> tuple[str, Optional[str]]:
    """
    Derive video_id and optional description variant from a round JSON path.
    """
    stem = path.stem
    if stem.endswith(round_suffix):
        return stem[: -len(round_suffix)], None

    variant_marker = f"{round_suffix}."
    if variant_marker in stem:
        video_id, variant = stem.rsplit(variant_marker, 1)
        if video_id and variant:
            return video_id, variant

    return stem, None


def video_id_from_path(path: Path, round_suffix: str) -> str:
    """
    Derive video_id from a filename stem by removing the expected round suffix.
    """
    video_id, _ = parse_video_id_and_variant_from_path(path, round_suffix)
    return video_id


def round_sort_key(path: Path) -> tuple[int, str]:
    """Sort round directories numerically when named round1, round2, etc."""
    match = re.fullmatch(r"round(\d+)", path.name)
    if match:
        return (int(match.group(1)), path.name)
    return (sys.maxsize, path.name)


def round_suffix_from_name(round_name: str) -> str:
    """Return the filename suffix produced by get_descriptions.py for a round."""
    match = re.fullmatch(r"round(\d+)", round_name)
    if match:
        return f".r{match.group(1)}"
    return f".{round_name}"


def discover_round_specs(input_dir: Path) -> list[tuple[str, str, str]]:
    """Find every comparison round directory under the descriptions directory."""
    round_dirs = [
        path
        for path in input_dir.iterdir()
        if path.is_dir() and re.fullmatch(r"round\d+", path.name)
    ]
    return [
        (path.name, path.name, round_suffix_from_name(path.name))
        for path in sorted(round_dirs, key=round_sort_key)
    ]


def short_stable_suffix(text: str) -> str:
    """Return a short deterministic suffix for duplicate embedding IDs."""
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:8]


def collect_round_records(
    input_dir: Path,
    round_dir_name: str,
    round_name: str,
    round_suffix: str,
    text_mode: str,
) -> list[dict]:
    """Collect description JSON records for one round."""
    round_dir = input_dir / round_dir_name
    if not round_dir.exists():
        print(f"Warning: round directory not found: {round_dir}", file=sys.stderr)
        return []

    records = []
    seen_record_keys: dict[tuple[str, Optional[str]], int] = {}
    duplicate_counts: dict[tuple[str, Optional[str]], int] = {}

    for path in sorted(round_dir.rglob("*.json")):
        if "_raw_responses" in path.parts or path.name == "_groups.json":
            continue

        relative_path = path.relative_to(input_dir)
        video_id, variant = parse_video_id_and_variant_from_path(path, round_suffix)
        if not path.stem.endswith(round_suffix) and variant is None:
            print(
                f"Warning: {relative_path} does not end with expected suffix "
                f"{round_suffix!r}; using {path.stem!r} as video_id.",
                file=sys.stderr,
            )

        description_json = load_json(path)
        text = embedding_text(description_json, text_mode)
        embedding_id = f"{video_id}__{round_name}"
        if variant:
            embedding_id = f"{embedding_id}__{variant}"

        record_key = (video_id, variant)
        seen_record_keys[record_key] = seen_record_keys.get(record_key, 0) + 1
        if seen_record_keys[record_key] > 1:
            duplicate_counts[record_key] = duplicate_counts.get(record_key, 1) + 1
            suffix = short_stable_suffix(str(relative_path))
            embedding_id = f"{embedding_id}__{suffix}"
            print(
                f"Warning: duplicate record {record_key!r} in {round_name}; "
                f"using embedding_id {embedding_id!r}.",
                file=sys.stderr,
            )

        records.append(
            {
                "embedding_id": embedding_id,
                "video_id": video_id,
                "round": round_name,
                "variant": variant or "default",
                "source_json_path": str(relative_path),
                "text_mode": text_mode,
                "text": text,
            }
        )

    return records


def collect_records(
    input_dir: Path,
    round_specs: list[tuple[str, str, str]],
    text_mode: str,
) -> list[dict]:
    """Collect description records from every configured comparison round."""
    records = []
    for round_dir, round_name, round_suffix in round_specs:
        records.extend(
            collect_round_records(
                input_dir,
                round_dir,
                round_name,
                round_suffix,
                text_mode,
            )
        )
    return records


def load_existing_embeddings(path: Path) -> dict:
    """Return existing embedded records keyed by embedding_id."""
    if not path.exists():
        return {}

    data = load_json(path)
    existing = {}
    for record in data.get("records", []):
        embedding = record.get("embedding")
        if isinstance(record.get("embedding_id"), str) and embedding is not None:
            existing[record["embedding_id"]] = record

    return existing


def is_transient_api_error(error: Exception) -> bool:
    """Return true for API failures worth retrying."""
    status_code = getattr(error, "status_code", None)
    if status_code in {429, 500, 502, 503, 504}:
        return True

    text = str(error)
    return any(
        marker in text
        for marker in [
            "429",
            "500",
            "502",
            "503",
            "504",
            "RESOURCE_EXHAUSTED",
            "Internal error",
            "Service Unavailable",
            "timeout",
            "timed out",
        ]
    )


def retry_transient_api_call(label: str, operation: Callable[[], T]) -> T:
    """Retry transient embedding API failures with exponential backoff."""
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

            sleep_seconds = TRANSIENT_API_BACKOFF_SECONDS * (2 ** (attempt - 1))
            print(
                f"Warning: transient API error during {label} "
                f"(attempt {attempt}/{TRANSIENT_API_ATTEMPTS}): {error}. "
                f"Retrying in {sleep_seconds}s.",
                file=sys.stderr,
            )
            time.sleep(sleep_seconds)

    raise RuntimeError(
        f"Embedding API failed during {label} after "
        f"{TRANSIENT_API_ATTEMPTS} attempts."
    ) from last_error


def vector_from_candidate(candidate) -> Optional[list[float]]:
    """Extract one vector from a possible SDK embedding object/dict/list."""
    if candidate is None:
        return None

    if isinstance(candidate, list):
        if all(isinstance(value, (int, float)) for value in candidate):
            return [float(value) for value in candidate]
        return None

    if isinstance(candidate, dict):
        for key in ("values", "embedding"):
            vector = vector_from_candidate(candidate.get(key))
            if vector is not None:
                return vector
        return None

    for attr in ("values", "embedding"):
        if hasattr(candidate, attr):
            vector = vector_from_candidate(getattr(candidate, attr))
            if vector is not None:
                return vector

    return None


def extract_embedding_vectors(response) -> list[list[float]]:
    """
    Extract embedding vectors from GenAI SDK responses across response shapes.
    """
    candidates = []

    if hasattr(response, "embeddings"):
        embeddings = getattr(response, "embeddings")
        if embeddings is not None:
            candidates.extend(list(embeddings))

    if hasattr(response, "embedding"):
        candidates.append(getattr(response, "embedding"))

    if isinstance(response, dict):
        for key in ("embeddings", "embedding"):
            value = response.get(key)
            if isinstance(value, list):
                candidates.extend(value)
            elif value is not None:
                candidates.append(value)

    vector = vector_from_candidate(response)
    if vector is not None:
        return [vector]

    vectors = []
    for candidate in candidates:
        vector = vector_from_candidate(candidate)
        if vector is not None:
            vectors.append(vector)

    if not vectors:
        raise ValueError(f"Could not extract embedding vectors from response: {response!r}")

    return vectors


def embed_texts(client, texts: list[str], model: str) -> list[list[float]]:
    """
    Embed texts, trying SDK batch support first and falling back to one at a time.
    """
    if not texts:
        return []

    try:
        response = retry_transient_api_call(
            f"embedding batch of {len(texts)} texts",
            lambda: client.models.embed_content(model=model, contents=texts),
        )
        vectors = extract_embedding_vectors(response)
        if len(vectors) != len(texts):
            raise ValueError(
                f"Batch returned {len(vectors)} embeddings for {len(texts)} texts."
            )
        return vectors
    except Exception as error:
        print(
            f"Warning: batch embedding failed; falling back to single-text calls: {error}",
            file=sys.stderr,
        )

    vectors = []
    for index, text in enumerate(texts, start=1):
        response = retry_transient_api_call(
            f"embedding text {index}/{len(texts)}",
            lambda: client.models.embed_content(model=model, contents=text),
        )
        extracted = extract_embedding_vectors(response)
        if len(extracted) != 1:
            raise ValueError(
                f"Expected one embedding for one text, got {len(extracted)}."
            )
        vectors.append(extracted[0])

    return vectors


def validate_embeddings(records: list[dict]) -> int:
    """Validate embedding vectors and return dimensionality."""
    dimensions = set()
    for record in records:
        embedding = record.get("embedding")
        if not isinstance(embedding, list) or not embedding:
            raise ValueError(f"Missing embedding for {record['embedding_id']}.")
        if not all(isinstance(value, (int, float)) for value in embedding):
            raise ValueError(f"Non-numeric embedding in {record['embedding_id']}.")
        record["embedding"] = [float(value) for value in embedding]
        dimensions.add(len(record["embedding"]))

    if len(dimensions) != 1:
        raise ValueError(f"Inconsistent embedding dimensions: {sorted(dimensions)}")

    return dimensions.pop()


def normalize_matrix(matrix: np.ndarray) -> np.ndarray:
    """
    Normalize vectors so cosine similarity becomes a dot product.

    Cosine similarity compares whether two vectors point in similar directions,
    regardless of their raw length.
    """
    matrix = np.nan_to_num(
        np.asarray(matrix, dtype=np.float64),
        copy=False,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


def cosine_similarity_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Return pairwise cosine similarities between two embedding matrices."""
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        similarities = normalize_matrix(a) @ normalize_matrix(b).T
    return np.nan_to_num(similarities, nan=0.0, posinf=0.0, neginf=0.0)


def records_matrix(records: list[dict]) -> np.ndarray:
    """Convert record embeddings to a numpy matrix."""
    return np.asarray([record["embedding"] for record in records], dtype=float)


def vector_distribution_stats(values: np.ndarray) -> dict[str, Optional[float]]:
    """Return compact descriptive stats for a vector of similarity values."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return {
            "mean": None,
            "median": None,
            "std": None,
            "min": None,
            "p05": None,
            "p25": None,
            "p75": None,
            "p95": None,
            "max": None,
        }

    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "p05": float(np.percentile(values, 5)),
        "p25": float(np.percentile(values, 25)),
        "p75": float(np.percentile(values, 75)),
        "p95": float(np.percentile(values, 95)),
        "max": float(np.max(values)),
    }


def embedding_spread_stats(records: list[dict]) -> dict:
    """
    Quantify whether embeddings are spread out or collapsing together.

    High cross-video cosine similarities suggest the fixed description structure
    may be making different videos look too similar. For unit-normalized vectors,
    a centroid norm near 1.0 also means the vectors point in nearly the same
    direction; lower values mean more spread.
    """
    matrix = records_matrix(records)
    normalized = normalize_matrix(matrix)
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        sim = normalized @ normalized.T
    sim = np.nan_to_num(sim, nan=0.0, posinf=0.0, neginf=0.0)
    num_records = len(records)

    off_diagonal_mask = ~np.eye(num_records, dtype=bool)
    video_ids = np.asarray([record["video_id"] for record in records])
    cross_video_mask = off_diagonal_mask & (video_ids[:, None] != video_ids[None, :])
    same_video_mask = off_diagonal_mask & (video_ids[:, None] == video_ids[None, :])

    centroid = np.mean(normalized, axis=0)
    centroid_norm = float(np.linalg.norm(centroid))
    centroid_direction = normalize_matrix(centroid.reshape(1, -1))
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        centroid_similarities = normalized @ centroid_direction.T
    centroid_similarities = np.nan_to_num(
        centroid_similarities,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    centered = normalized - np.mean(normalized, axis=0)
    singular_values = np.linalg.svd(centered, compute_uv=False)
    variance = singular_values**2
    total_variance = float(np.sum(variance))
    first_pc_variance_percent = (
        100.0 * float(variance[0]) / total_variance if total_variance else None
    )

    return {
        "all_pairwise": vector_distribution_stats(sim[off_diagonal_mask]),
        "cross_video_pairwise": vector_distribution_stats(sim[cross_video_mask]),
        "same_video_pairwise": vector_distribution_stats(sim[same_video_mask]),
        "centroid_norm": centroid_norm,
        "similarity_to_centroid": vector_distribution_stats(centroid_similarities.ravel()),
        "first_pc_variance_percent": first_pc_variance_percent,
    }


def percentile_from_rank(rank: int, num_candidates: int) -> float:
    """Return how high a match ranked, as a percentile where 100 is best."""
    if num_candidates <= 1:
        return 100.0
    return 100.0 * (num_candidates - rank) / (num_candidates - 1)


def write_same_video_similarity(records: list[dict], output_path: Path) -> dict:
    """
    Write round1-vs-round2 same-video diagnostics and return summary stats.

    This script checks whether two independently generated descriptions of the
    same video land near each other in embedding space.
    """
    round1_records = [record for record in records if record["round"] == "round1"]
    round2_records = [record for record in records if record["round"] == "round2"]
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    similarities = []
    best_true_match_ranks = []

    if round1_records and round2_records:
        sim = cosine_similarity_matrix(
            records_matrix(round1_records),
            records_matrix(round2_records),
        )
        for r1_index, r1_record in enumerate(round1_records):
            order = np.argsort(-sim[r1_index])
            ranked_indices = [int(index) for index in order]
            matching_indices = [
                index
                for index, r2_record in enumerate(round2_records)
                if r2_record["video_id"] == r1_record["video_id"]
            ]
            matching_ranks = []

            for r2_index in matching_indices:
                rank = ranked_indices.index(r2_index) + 1
                matching_ranks.append(rank)
                cosine = float(sim[r1_index, r2_index])
                similarities.append(cosine)

                rows.append(
                    {
                        "video_id": r1_record["video_id"],
                        "round1_embedding_id": r1_record["embedding_id"],
                        "round2_embedding_id": round2_records[r2_index]["embedding_id"],
                        "cosine_similarity": f"{cosine:.10f}",
                        "rank_of_true_round2_match": rank,
                        "num_round2_candidates": len(round2_records),
                        "true_match_percentile": f"{percentile_from_rank(rank, len(round2_records)):.6f}",
                    }
                )
            if matching_ranks:
                best_true_match_ranks.append(min(matching_ranks))

    fieldnames = [
        "video_id",
        "round1_embedding_id",
        "round2_embedding_id",
        "cosine_similarity",
        "rank_of_true_round2_match",
        "num_round2_candidates",
        "true_match_percentile",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return {
        "same_video_similarities": similarities,
        "round1_true_match_ranks": best_true_match_ranks,
        "num_same_video_pairs": len(rows),
    }


def write_nearest_neighbors(
    records: list[dict],
    output_path: Path,
    top_k: int = 10,
) -> None:
    """Write top-k nearest neighbors for every embedding record."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sim = cosine_similarity_matrix(records_matrix(records), records_matrix(records))

    fieldnames = [
        "query_embedding_id",
        "query_video_id",
        "query_round",
        "neighbor_rank",
        "neighbor_embedding_id",
        "neighbor_video_id",
        "neighbor_round",
        "same_video",
        "cosine_similarity",
    ]
    rows = []
    for query_index, query_record in enumerate(records):
        order = np.argsort(-sim[query_index])
        neighbor_rank = 0
        for neighbor_index in order:
            neighbor_index = int(neighbor_index)
            if neighbor_index == query_index:
                continue

            neighbor = records[neighbor_index]
            neighbor_rank += 1
            rows.append(
                {
                    "query_embedding_id": query_record["embedding_id"],
                    "query_video_id": query_record["video_id"],
                    "query_round": query_record["round"],
                    "neighbor_rank": neighbor_rank,
                    "neighbor_embedding_id": neighbor["embedding_id"],
                    "neighbor_video_id": neighbor["video_id"],
                    "neighbor_round": neighbor["round"],
                    "same_video": query_record["video_id"] == neighbor["video_id"],
                    "cosine_similarity": f"{float(sim[query_index, neighbor_index]):.10f}",
                }
            )
            if neighbor_rank == top_k:
                break

    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_readable_neighbors(
    records: list[dict],
    output_path: Path,
    top_k: int = 10,
) -> None:
    """
    Write a human-readable nearest-neighbor list for each description embedding.

    This preserves round1 and round2 separately so the diagnostic question stays
    visible: does a video's round1 description land near its round2 description?
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sim = cosine_similarity_matrix(records_matrix(records), records_matrix(records))

    lines = [
        "DanceVerse Nearest Description Embeddings",
        "",
        "Each line keeps round1 and round2 separate. A same-video neighbor is marked with *.",
        "",
    ]
    for query_index, query in enumerate(records):
        order = np.argsort(-sim[query_index])
        neighbors = []
        for neighbor_index in order:
            neighbor_index = int(neighbor_index)
            if neighbor_index == query_index:
                continue

            neighbor = records[neighbor_index]
            same_video_marker = "*" if neighbor["video_id"] == query["video_id"] else ""
            neighbors.append(
                f"{neighbor['embedding_id']}{same_video_marker} "
                f"({float(sim[query_index, neighbor_index]):.4f})"
            )
            if len(neighbors) == top_k:
                break

        lines.append(f"{query['embedding_id']}: {', '.join(neighbors)}")

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def format_float(value: Optional[float]) -> str:
    """Format optional floats for the text summary."""
    if value is None or math.isnan(value):
        return "n/a"
    return f"{value:.6f}"


def format_distribution(stats: dict[str, Optional[float]]) -> list[str]:
    """Format a similarity distribution for summary.txt."""
    return [
        f"mean: {format_float(stats['mean'])}",
        f"median: {format_float(stats['median'])}",
        f"std: {format_float(stats['std'])}",
        f"min: {format_float(stats['min'])}",
        f"p05: {format_float(stats['p05'])}",
        f"p25: {format_float(stats['p25'])}",
        f"p75: {format_float(stats['p75'])}",
        f"p95: {format_float(stats['p95'])}",
        f"max: {format_float(stats['max'])}",
    ]


def write_summary(
    records: list[dict],
    output_path: Path,
    model: str,
    text_mode: str,
    input_dir: Path,
    dimensionality: int,
    same_video_stats: dict,
) -> None:
    """Write a compact text report for the embedding diagnostic."""
    rounds = sorted({record["round"] for record in records})
    record_counts_by_round = {
        round_name: sum(1 for record in records if record["round"] == round_name)
        for round_name in rounds
    }
    round1_records = [record for record in records if record["round"] == "round1"]
    round2_records = [record for record in records if record["round"] == "round2"]
    round1_video_ids = {record["video_id"] for record in round1_records}
    round2_video_ids = {record["video_id"] for record in round2_records}
    all_video_ids = round1_video_ids | round2_video_ids
    videos_with_both = round1_video_ids & round2_video_ids
    missing_round1 = sorted(round2_video_ids - round1_video_ids)
    missing_round2 = sorted(round1_video_ids - round2_video_ids)

    similarities = np.asarray(same_video_stats["same_video_similarities"], dtype=float)
    if len(similarities):
        mean_similarity = float(np.mean(similarities))
        median_similarity = float(np.median(similarities))
        min_similarity = float(np.min(similarities))
        max_similarity = float(np.max(similarities))
    else:
        mean_similarity = median_similarity = min_similarity = max_similarity = None

    true_match_ranks = same_video_stats["round1_true_match_ranks"]
    rank1_percent = (
        100.0 * sum(rank == 1 for rank in true_match_ranks) / len(true_match_ranks)
        if true_match_ranks
        else 0.0
    )
    rank5_percent = (
        100.0 * sum(rank <= 5 for rank in true_match_ranks) / len(true_match_ranks)
        if true_match_ranks
        else 0.0
    )
    spread_stats = embedding_spread_stats(records)
    cross_video_stats = spread_stats["cross_video_pairwise"]

    lines = [
        "DanceVerse Embedding Diagnostic",
        "",
        "============================================================",
        "Quick Summary",
        "============================================================",
        "",
        "1. same-video similarity",
        f"mean: {format_float(mean_similarity)}",
        f"min: {format_float(min_similarity)}",
        f"max: {format_float(max_similarity)}",
        f"true round2 match rank 1: {rank1_percent:.2f}%",
        f"true round2 match rank <= 5: {rank5_percent:.2f}%",
        "",
        "2. cross-video similarity",
        f"mean: {format_float(cross_video_stats['mean'])}",
        f"min: {format_float(cross_video_stats['min'])}",
        f"max: {format_float(cross_video_stats['max'])}",
        "",
        "============================================================",
        "Detailed Report",
        "============================================================",
        "",
        f"embedding model: {model}",
        f"text mode: {text_mode}",
        f"input dir: {input_dir}",
        f"number of records: {len(records)}",
        "records by round:",
        *(f"- {round_name}: {count}" for round_name, count in record_counts_by_round.items()),
        f"number of unique videos: {len(all_video_ids)}",
        f"number of videos with both round1 and round2: {len(videos_with_both)}",
        f"embedding dimensionality: {dimensionality}",
        "",
        "same-video round1-vs-round2 cosine similarity:",
        f"mean: {format_float(mean_similarity)}",
        f"median: {format_float(median_similarity)}",
        f"min: {format_float(min_similarity)}",
        f"max: {format_float(max_similarity)}",
        "",
        f"true round2 match rank 1: {rank1_percent:.2f}%",
        f"true round2 match rank <= 5: {rank5_percent:.2f}%",
        "",
        "overall embedding spread:",
        "all pairwise cosine similarities, excluding self-pairs:",
        *format_distribution(spread_stats["all_pairwise"]),
        "",
        "cross-video pairwise cosine similarities, excluding same-video pairs:",
        *format_distribution(spread_stats["cross_video_pairwise"]),
        "",
        "same-video pairwise cosine similarities, across any available rounds:",
        *format_distribution(spread_stats["same_video_pairwise"]),
        "",
        "centroid/collapse diagnostics:",
        f"centroid norm: {format_float(spread_stats['centroid_norm'])}",
        "similarity to centroid:",
        *format_distribution(spread_stats["similarity_to_centroid"]),
        (
            "first principal component variance: "
            f"{format_float(spread_stats['first_pc_variance_percent'])}%"
        ),
        "",
        "videos missing round1:",
        *(f"- {video_id}" for video_id in missing_round1),
        "none" if not missing_round1 else "",
        "",
        "videos missing round2:",
        *(f"- {video_id}" for video_id in missing_round2),
        "none" if not missing_round2 else "",
        "",
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(line for line in lines if line != ""), encoding="utf-8")


def merge_existing_embeddings(records: list[dict], existing_records: dict) -> list[dict]:
    """Copy existing embeddings onto matching records."""
    merged = []
    for record in records:
        existing = existing_records.get(record["embedding_id"])
        if existing is not None and "embedding" in existing:
            merged_record = dict(record)
            merged_record["embedding"] = existing["embedding"]
            merged.append(merged_record)
        else:
            merged.append(record)
    return merged


def save_embeddings_file(
    output_path: Path,
    records: list[dict],
    model: str,
    text_mode: str,
    input_dir: Path,
    created_at: str,
) -> None:
    """Save embeddings and metadata incrementally."""
    write_json(
        output_path,
        {
            "model": model,
            "created_at": created_at,
            "input_dir": str(input_dir),
            "text_mode": text_mode,
            "records": records,
        },
    )


def embed_missing_records(
    client,
    records: list[dict],
    model: str,
    text_mode: str,
    output_path: Path,
    input_dir: Path,
    created_at: str,
) -> None:
    """
    Embed missing records in batches and save after every successful batch.

    An embedding is a numeric vector representing the semantic content of text.
    Here the text is the raw, stable JSON serialization of each description.
    """
    missing_indices = [
        index for index, record in enumerate(records) if "embedding" not in record
    ]
    if not missing_indices:
        print("All embeddings already exist; reusing cached output.")
        return

    print(f"Embedding {len(missing_indices)} missing records.")
    for start in range(0, len(missing_indices), BATCH_SIZE):
        batch_indices = missing_indices[start : start + BATCH_SIZE]
        texts = [records[index]["text"] for index in batch_indices]
        vectors = embed_texts(client, texts, model)
        for record_index, vector in zip(batch_indices, vectors):
            records[record_index]["embedding"] = vector

        save_embeddings_file(output_path, records, model, text_mode, input_dir, created_at)
        print(
            f"Saved progress: {min(start + BATCH_SIZE, len(missing_indices))}/"
            f"{len(missing_indices)} newly embedded records."
        )


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    default_model = os.getenv("EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)
    parser = argparse.ArgumentParser(
        description=(
            "Embed raw DanceVerse comparison-round JSON files and check "
            "round agreement."
        )
    )
    parser.add_argument("--input-dir", type=Path, default=Path("data/descriptions"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Defaults to a text-mode-specific folder under data/embeddings.",
    )
    parser.add_argument(
        "--text-mode",
        choices=sorted(DEFAULT_OUTPUT_DIRS),
        default=DEFAULT_TEXT_MODE,
        help="direct_json embeds stable JSON; concatenated_values embeds only JSON values joined as a paragraph.",
    )
    parser.add_argument(
        "--round-dir",
        action="append",
        default=None,
        help=(
            "Comparison round directory to include. May be passed more than once. "
            "Defaults to all roundN directories under --input-dir."
        ),
    )
    parser.add_argument("--model", default=default_model)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-records", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    """Run the direct-JSON embedding diagnostic."""
    load_dotenv()
    args = parse_args()
    input_dir = resolve_project_path(args.input_dir)
    output_dir = resolve_project_path(args.output_dir or DEFAULT_OUTPUT_DIRS[args.text_mode])
    embeddings_path = output_dir / "embeddings.json"
    if args.round_dir:
        round_specs = [
            (round_dir, round_dir, round_suffix_from_name(round_dir))
            for round_dir in args.round_dir
        ]
    else:
        round_specs = discover_round_specs(input_dir)

    if not round_specs:
        raise ValueError(f"No comparison round directories found under {input_dir}.")

    records = collect_records(
        input_dir=input_dir,
        round_specs=round_specs,
        text_mode=args.text_mode,
    )
    if args.max_records is not None:
        records = records[: args.max_records]

    print(f"Collected {len(records)} records from {input_dir}.")
    if not records:
        raise ValueError("No JSON description records found.")

    if args.dry_run:
        print(json.dumps(records, indent=2, ensure_ascii=False))
        return

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key or api_key == "PASTE_KEY_HERE":
        raise ValueError("Set GEMINI_API_KEY in .env before running this script.")

    created_at = datetime.now(timezone.utc).isoformat()
    if embeddings_path.exists() and not args.overwrite:
        records = merge_existing_embeddings(records, load_existing_embeddings(embeddings_path))

    with genai.Client(api_key=api_key) as client:
        embed_missing_records(
            client=client,
            records=records,
            model=args.model,
            text_mode=args.text_mode,
            output_path=embeddings_path,
            input_dir=input_dir,
            created_at=created_at,
        )

    dimensionality = validate_embeddings(records)
    save_embeddings_file(
        embeddings_path,
        records,
        args.model,
        args.text_mode,
        input_dir,
        created_at,
    )

    same_video_stats = write_same_video_similarity(
        records,
        output_dir / "same_video_similarity.csv",
    )
    write_nearest_neighbors(records, output_dir / "nearest_neighbors.csv", top_k=10)
    write_readable_neighbors(records, output_dir / "description_neighbors.txt", top_k=10)
    write_summary(
        records=records,
        output_path=output_dir / "summary.txt",
        model=args.model,
        text_mode=args.text_mode,
        input_dir=input_dir,
        dimensionality=dimensionality,
        same_video_stats=same_video_stats,
    )
    print(f"Wrote outputs to {output_dir}")


if __name__ == "__main__":
    main()
