# Configuration Reference

Complete reference for all pipeline configuration options.

---

## Table of Contents

1. [Configuration File Location](#configuration-file-location)
2. [Embedding Configuration](#embedding-configuration)
3. [Firm-Level Topic Model](#firm-level-topic-model)
4. [Theme-Level Topic Model](#theme-level-topic-model)
5. [Theme Validation](#theme-validation)
6. [LLM Configuration](#llm-configuration)
7. [Environment Variables](#environment-variables)
8. [Configuration Precedence](#configuration-precedence)

---

## Configuration File Location

**Primary configuration:** `cloud/config/production.yaml`

This is the single source of truth for all pipeline settings. The pipeline loads this file automatically when running `scripts/run_unified_pipeline.py`.

**Alternative configs:**
- `cloud/config/local.yaml` — CPU mode for local development
- `cloud/config/default.yaml` — Fallback defaults

---

## Embedding Configuration

Controls the sentence embedding model used for vectorization.

```yaml
embedding:
  model: "all-mpnet-base-v2"
  dimension: 768
  device: "cuda"
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | string | `"all-mpnet-base-v2"` | HuggingFace model name for sentence embeddings |
| `dimension` | int | `768` | Output embedding dimension (must match model) |
| `device` | string | `"cpu"` | Compute device: `"cuda"` for GPU, `"cpu"` for CPU |

### Supported Models

| Model | Dimension | Quality | Speed | Notes |
|-------|-----------|---------|-------|-------|
| `all-mpnet-base-v2` | 768 | Good | Fast | Default, recommended for most use cases |
| `all-MiniLM-L6-v2` | 384 | Moderate | Very Fast | Lightweight alternative |
| `Alibaba-NLP/gte-Qwen2-1.5B-instruct` | 1536 | Better | Medium | Higher quality embeddings |
| `Qwen/Qwen3-Embedding-8B` | 4096 | Best | Slow | SOTA quality, requires GPU |

**Important:** When changing the model, you must:
1. Update `dimension` to match the model's output
2. Set `EMBEDDING_DIMENSION` environment variable before running
3. Recreate database tables (dimensions are fixed at table creation)

---

## Firm-Level Topic Model

Configuration for per-firm topic clustering. Optimized for ~300 sentences producing ~25 topics.

```yaml
firm_topic_model:
  umap:
    n_neighbors: 15
    n_components: 10
    min_dist: 0.0
    metric: "cosine"
    random_state: 42

  hdbscan:
    min_cluster_size: 6
    min_samples: 2
    metric: "euclidean"
    cluster_selection_method: "leaf"

  vectorizer:
    ngram_range: [1, 2]
    min_df: 2

  representation:
    mmr_diversity: 0.3
    pos_model: "en_core_web_sm"
```

### UMAP Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `n_neighbors` | int | `15` | Size of local neighborhood for manifold approximation. Lower = more local structure. |
| `n_components` | int | `10` | Target dimensionality after reduction |
| `min_dist` | float | `0.0` | Minimum distance between points in embedded space. `0.0` = tight clusters. |
| `metric` | string | `"cosine"` | Distance metric: `"cosine"`, `"euclidean"`, `"manhattan"` |
| `random_state` | int | `42` | Random seed for reproducibility |

### HDBSCAN Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `min_cluster_size` | int | `6` | Minimum number of points to form a cluster |
| `min_samples` | int | `2` | Number of samples in neighborhood for core points |
| `metric` | string | `"euclidean"` | Distance metric (on UMAP output) |
| `cluster_selection_method` | string | `"leaf"` | `"leaf"` = smaller clusters, `"eom"` = larger clusters |

### Vectorizer Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `ngram_range` | [int, int] | `[1, 2]` | N-gram range for topic representation |
| `min_df` | int | `2` | Minimum document frequency for terms |

### Representation Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `mmr_diversity` | float | `0.3` | Diversity factor for MMR representation (0-1) |
| `pos_model` | string | `"en_core_web_sm"` | SpaCy model for POS-based representation |

---

## Theme-Level Topic Model

Configuration for cross-firm theme clustering. Optimized for 750+ topics producing ~100 themes.

```yaml
theme_topic_model:
  umap:
    n_neighbors: 30
    n_components: 15
    min_dist: 0.05
    metric: "cosine"
    random_state: 42

  hdbscan:
    min_cluster_size: 10
    min_samples: 3
    metric: "euclidean"
    cluster_selection_method: "eom"

  vectorizer:
    ngram_range: [1, 2]
    min_df: 3

  representation:
    mmr_diversity: 0.4
    pos_model: "en_core_web_sm"
```

### Key Differences from Firm-Level

| Parameter | Firm-Level | Theme-Level | Rationale |
|-----------|------------|-------------|-----------|
| `n_neighbors` | 15 | 30 | Themes need more global structure |
| `n_components` | 10 | 15 | More dimensions for complex topic space |
| `min_dist` | 0.0 | 0.05 | Slight separation for visualization |
| `min_cluster_size` | 6 | 10 | Themes should be more substantial |
| `cluster_selection_method` | leaf | eom | Prefer larger, more stable themes |

---

## Theme Validation

Rules for validating cross-firm themes.

```yaml
validation:
  min_firms: 2
  max_firm_dominance: 0.4
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `min_firms` | int | `2` | Minimum firms required for a valid theme |
| `max_firm_dominance` | float | `0.4` | Maximum fraction of theme topics from single firm |

### Validation Logic

A theme is **valid** if:
1. It contains topics from at least `min_firms` different companies
2. No single firm contributes more than `max_firm_dominance` (40%) of topics

**Why these rules?**
- Ensures themes are truly cross-firm, not single-company artifacts
- Prevents one firm's topics from dominating a theme

---

## LLM Configuration

Settings for the xAI/Grok LLM used for topic and theme summarization.

```yaml
llm:
  model: "grok-4-1-fast-reasoning"
  max_concurrent: 50
  timeout: 30
  max_retries: 3
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | string | `"grok-4-1-fast-reasoning"` | xAI model name |
| `max_concurrent` | int | `50` | Maximum concurrent API requests |
| `timeout` | int | `30` | Request timeout in seconds |
| `max_retries` | int | `3` | Retry attempts on transient failures |

### API Key

The LLM requires an API key set via environment variable:

```bash
export XAI_API_KEY="your-api-key"
```

Or in `.env` file:
```
XAI_API_KEY=your-api-key
```

**Without API key:** Pipeline uses keyword-based fallbacks instead of LLM summaries.

---

## Environment Variables

Environment variables override config file settings.

| Variable | Description | Example |
|----------|-------------|---------|
| `DATABASE_URL` | PostgreSQL connection string | `postgresql://ftm:pass@host:5432/ftm` |
| `XAI_API_KEY` | xAI API key for LLM | `xai-xxx` |
| `CONFIG_PATH` | Override config file location | `/path/to/config.yaml` |
| `EMBEDDING_DIMENSION` | Override embedding dimension | `4096` |
| `DEVICE` | Override compute device | `cuda` or `cpu` |
| `TEST_MODE` | Enable test mode | `mag7` |
| `MAX_FIRMS` | Limit firms to process | `100` |

---

## Configuration Precedence

When the same setting is available in multiple places:

1. **Environment variable** (highest priority)
2. **Config file** (`production.yaml`)
3. **Code defaults** (lowest priority)

**Example:**
```bash
# Config file has device: "cuda"
# Environment variable overrides it
DEVICE=cpu python scripts/run_unified_pipeline.py
# Result: Uses CPU
```

---

## Example Configurations

### Development (Local CPU)

```yaml
embedding:
  model: "all-mpnet-base-v2"
  dimension: 768
  device: "cpu"

firm_topic_model:
  hdbscan:
    min_cluster_size: 3  # Smaller for fewer test documents
```

### Production (Cloud GPU)

```yaml
embedding:
  model: "all-mpnet-base-v2"
  dimension: 768
  device: "cuda"

firm_topic_model:
  hdbscan:
    min_cluster_size: 6

theme_topic_model:
  hdbscan:
    min_cluster_size: 10
```

### High-Quality (Large Model)

```yaml
embedding:
  model: "Qwen/Qwen3-Embedding-8B"
  dimension: 4096
  device: "cuda"

# Note: Requires EMBEDDING_DIMENSION=4096 and database recreation
```
