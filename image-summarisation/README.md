# Summarise 1,000 Images for $0.05

**Vision-language batch inference turns image captioning from a manual task into an automated pipeline**

Writing alt-text, social media captions, or content descriptions for a large image library is the kind of work that is either done manually (expensive, slow) or not done at all. Vision-language models can generate reasonable image summaries, but the economics only work at scale when inference is cheap enough to run against thousands of images without thinking about the cost. We ran Qwen3-VL-30B against 1,000 Unsplash photographs via the Doubleword Batch API and generated social media-style summaries for $0.05 total, roughly $0.00005 per image.

To run this yourself, sign up at [app.doubleword.ai](https://app.doubleword.ai) and generate an API key.

## What we did

We took 1,000 photographs from the [Unsplash Lite dataset](https://github.com/unsplash/datasets), a freely available collection of high-quality stock photos with metadata (descriptions, photographer names). Each image was resized to 720p, sent to a vision-language model along with its existing caption and photographer credit, and the model was asked to produce a concise social media-style summary.

The prompt is straightforward: given the image, the existing caption, and the photographer username, write a summary suitable for a social media post. The model sees the actual image pixels, not just the metadata, so the summaries reflect visual content that the original captions may not mention.

## Data

The [Unsplash Lite dataset](https://github.com/unsplash/datasets) contains 25,000 photographs contributed by Unsplash photographers. We used the first 1,000 images from `photos.csv000`. Each record includes a photo URL, a text description, and the photographer's username. Images are fetched from Unsplash's CDN and resized to 1280x720 before encoding as base64 for the vision model.

## Baseline

Without batch inference, the alternatives for captioning 1,000 images are:

- **Manual captioning**: A human writing social media captions at 2 minutes per image would spend roughly 33 hours on 1,000 images.
- **Real-time API calls**: Running the same model via real-time inference costs ~$0.16 per 1M input tokens versus $0.05 for batch, making the batch route about 3x cheaper.
- **Using existing metadata only**: The Unsplash dataset already includes descriptions, but these are photographer-submitted and inconsistent in tone, length, and detail. Many are missing entirely.

## Results

The batch processed 1,000 images and returned summaries for each. With Qwen3-VL-30B via the Doubleword Batch API, the total cost was approximately $0.05.

### Cost comparison

| Provider | Model | Mode | Input (per 1M tokens) | Output (per 1M tokens) | Estimated cost (1,000 images) |
|----------|-------|------|-----------------------|------------------------|-------------------------------|
| Doubleword | Qwen3-VL-30B | 24h Batch | $0.05 | $0.20 | ~$0.05 |
| Doubleword | Qwen3-VL-30B | Realtime | $0.16 | $0.80 | ~$0.16 |
| Doubleword | Qwen3-VL-235B | 24h Batch | $0.20 | $0.60 | ~$0.20 |
| OpenAI | GPT-5-mini | Batch | $0.075 | $0.30 | ~$0.08 |
| OpenAI | GPT-5.2 | Batch | $0.875 | $7.00 | ~$0.88 |

Prices: [Doubleword model pricing](https://docs.doubleword.ai/batches/model-pricing), [OpenAI pricing](https://platform.openai.com/docs/pricing).

The cost difference grows linearly. At 100,000 images, the gap between Qwen3-30B batch ($5) and GPT-5.2 batch ($88) becomes significant. For a task like captioning an entire stock photo library, the model choice determines whether the project is worth running at all.

### Limitations

**No ground truth evaluation.** Image summarisation is subjective, so we have no accuracy metric. The summaries look reasonable on manual inspection, but we haven't run a systematic quality assessment comparing models or measuring factual consistency against image content.

**Single prompt.** We used one prompt template for all images. Different use cases (alt-text for accessibility, product descriptions, social media posts) would benefit from tailored prompts, and quality may vary across them.

**Unsplash bias.** The dataset consists of curated, high-quality stock photos. Results on user-generated content, screenshots, documents, or other image types may differ.

## Replication

### Setup

```bash
cd image-summarisation && uv sync
```

Download the [Unsplash Lite dataset](https://github.com/unsplash/datasets) and extract it so that `unsplash-research-dataset-lite-latest/photos.csv000` exists in this directory.

### Running summarisation

Set your API key and submit a batch:

```bash
export DOUBLEWORD_API_KEY="your-key"
uv run image-summarisation run -i unsplash-research-dataset-lite-latest/photos.csv000 -m 30b -n 1000
```

The `-m 30b` flag selects Qwen3-VL-30B. Use `-m 235b` for the larger model, or any model alias from the standard set.

Check status and download results:

```bash
uv run image-summarisation status
```

Combine results into a CSV:

```bash
uv run image-summarisation analyze -i unsplash-research-dataset-lite-latest/photos.csv000
```

This produces `results/summaries.csv` with columns for the image URL, original description, photographer, and the generated summary.

### Preview without submitting

Use `--dry-run` to prepare the batch file without uploading:

```bash
uv run image-summarisation run -i unsplash-research-dataset-lite-latest/photos.csv000 --dry-run
```

## Conclusion

At $0.05 for 1,000 images, batch vision inference is cheap enough to treat image captioning as an automated pipeline rather than a manual task. The Doubleword Batch API with Qwen3-VL-30B handles the workload at a cost low enough that you can caption your entire image library speculatively, without needing to justify the expense per image. For organisations sitting on large uncaptioned image collections (media libraries, e-commerce catalogues, content archives), this is the difference between having metadata and not.
