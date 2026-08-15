# DanceVerse Prototype

Prototype pipeline for clustering and visualizing dance videos based on choreographic style and movement qualities.

DanceVerse is an exploratory Python project for analyzing short dance video clips with multimodal LLMs and embedding models. The pipeline generates no-audio choreography descriptions, extracts interval-based movement timelines, converts outputs into vectors, and combines those vectors for downstream clustering or visualization.

## What This Includes

- Comparative choreography description generation from video clips
- No-audio prompting, so the model focuses on visible movement rather than music
- Interval-based movement timeline labeling
- Description embedding generation
- Feature combination for description embeddings plus movement timeline vectors

## What Is Not Included

This public prototype intentionally excludes local/private artifacts:

- Raw videos
- Generated descriptions, embeddings, plots, logs, and trial folders
- API keys or `.env` files
- Large local data directories

## Repository Structure

```text
.
├── get_descriptions.py
├── get_movement_timeline.py
├── get_embeddings.py
├── combine_timeline_embeddings.py
├── config/
│   └── prompts/
│       ├── comparison_v5_no_audio.txt
│       └── movement_timeline_v1.txt
├── .env.example
├── .gitignore
├── requirements.txt
└── README.md
```

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Then add your Gemini API key to `.env`:

```text
GEMINI_API_KEY=your_api_key_here
```

Do not commit `.env`.

## Basic Pipeline

1. Prepare short `.mp4` dance clips locally.
2. Run `get_descriptions.py` to generate comparative choreography descriptions.
3. Run `get_movement_timeline.py` to generate interval-level movement labels.
4. Run `get_embeddings.py` to embed the descriptions.
5. Run `combine_timeline_embeddings.py` to concatenate description embeddings with movement timeline features.
6. Use the resulting vectors for clustering or projection plots.

## Status

This is a prototype. The code is organized around experimentation rather than a polished application interface, and the data used for development is not included in this repository.
