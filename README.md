# DanceVerse Prototype
_Alison Kim, Richard Guo_

DanceVerse is an interactive tool that maps choreography video datasets based on stylistic and movement qualities. The pipeline turns dance clips into text descriptions, creates per-interval movement labels, and converts them into embeddings for clustering and visualization. 

Click on the figure below to check out the interactive plot! Each dot represents a dance video that was described, embedded, then plotted using UMAP.

[![DanceVerse interactive scatter plot](danceverse-prototype.png)](https://ajkim000.github.io/danceverse-prototype/danceverse-prototype.html)

Future work will include a larger video dataset and interactive controls for exploring dance qualities a viewer may be seeking.

## Setup

This project uses Python scripts plus the Gemini API. Generated data, local videos, and API keys are not included in this repo.

```bash
git clone https://github.com/ajkim000/danceverse-prototype.git
cd danceverse-prototype
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Then add your Gemini API key to `.env`:

```text
GEMINI_API_KEY=your_api_key_here
```

You also need `ffmpeg` and `ffprobe` installed locally for video preprocessing. On macOS:

```bash
brew install ffmpeg
```

## Basic Pipeline

1. Prepare short `.mp4` dance clips locally.

   For example: `data/videos/compressed_no_audio/personal/`

2. Generate comparative choreography descriptions.

   ```bash
   python get_descriptions.py \
     --videos-dir data/videos/compressed_no_audio/personal \
     --output-dir data/descriptions/example_trial \
     --no-audio
   ```

3. Generate movement timelines.

   ```bash
   python get_movement_timeline.py \
     --videos-dir data/videos/compressed_no_audio/personal \
     --output-dir data/movement_timelines/example_trial \
     --timeline-rounds 2 \
     --timeline-variants 2
   ```

4. Embed the text descriptions.

   ```bash
   python get_embeddings.py \
     --input-dir data/descriptions/example_trial \
     --output-dir data/embeddings/example_trial
   ```

5. Combine description embeddings with movement timeline vectors.

   ```bash
   python combine_timeline_embeddings.py \
     --embeddings-path data/embeddings/example_trial/embeddings.json \
     --timeline-dir data/movement_timelines/example_trial \
     --output-path data/embeddings/example_trial/combined_embeddings.json
   ```

6. Use the resulting vectors for clustering or projection plots.
