# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Financial Topic Modeling research project that builds an NLP pipeline for identifying cross-firm investment themes from earnings call transcripts. The project has two phases:

1. **Local_BERTopic_MVP**: A proof-of-concept implementation (legacy code - poorly structured, use for reference only)
2. **Cloud Migration** (in development): Scalable AWS infrastructure using map-reduce pattern

## Architecture

The pipeline follows a hierarchical approach:

1. **Data Ingestion**: Load earnings call transcripts from WRDS, local CSV, or cloud storage (S3/Athena)
2. **Firm-Level Topic Modeling**: Per-firm BERTopic clustering of sentences into topics
3. **Cross-Firm Theme Identification**: Re-cluster firm topics to discover universal themes
4. **Downstream Analysis**: Sentiment analysis (FinBERT) and event studies (future work)

Key architectural patterns:
- **Map phase**: Independent firm-level processing (parallelizable via AWS Batch)
- **Reduce phase**: Cross-firm aggregation (single instance, larger compute)
- **Topic model abstraction**: Designed for swappable implementations (BERTopic, LDA, neural methods)

## Development Commands

```bash
# Activate virtual environment
source venv/bin/activate

# Run the main pipeline (from Local_BERTopic_MVP directory)
python run_pipeline.py

# Run pipeline programmatically
python -c "from src.main import CrossFirmThemePipeline; p = CrossFirmThemePipeline('src/config/config.yaml')"

# Run with specific firms (from Local_BERTopic_MVP)
python -m src.main --config src/config/config.yaml --firms "Tesla, Inc." --start-date 2024-10-01 --end-date 2024-12-01 --output-dir output
```

## Key Configuration

Configuration lives in `Local_BERTopic_MVP/src/config/config.yaml`. Important settings:
- `data_ingestion.data_source`: "local", "wrds", or "cloud"
- `data_ingestion.local_csv.path`: Path to transcript CSV
- `topic_modeling.sentence_model.model_name`: Embedding model (default: all-mpnet-base-v2)
- `topic_persistence.enabled`: Cache firm-level results for faster iteration

Environment variables (in `.env`):
- `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`: For cloud data source
- `WRDS_USERNAME`, `WRDS_PASSWORD`: For WRDS data source
- `OPENAI_API_KEY`: For LLM-based topic naming (optional)

## Module Structure (Local_BERTopic_MVP/src/)

- `main.py` / `run_pipeline.py`: Pipeline orchestration entry points
- `data_ingestion/`: Data connectors (WRDS, local CSV, cloud) and transcript processing
- `topic_modeling/`: BERTopic-based firm-level topic analysis
- `theme_identification/`: Cross-firm theme clustering and validation
- `sentiment_analysis/`: FinBERT-based sentiment scoring (downstream)
- `event_study/`: Statistical event study framework (downstream)
- `config/`: YAML configuration management
- `utils/`: Logging and batch orchestration utilities

## Design Principles

Per project vision (First_Transcript.md):
- Prioritize simplicity over complexity - "The best engineers write code my mom could read"
- Use boring technology, over-document the "why", under-engineer the "how"
- Test-driven development with validation-as-you-go
- Modular design for topic model swappability
- Cloud architecture: AWS Batch (map) + SageMaker (reduce) + DynamoDB (output)

## Data Files

- `transcripts_2023-01-01_to_2023-03-31_enriched.csv`: Sample transcript dataset (~380MB)
- Output hierarchy: `themes → topics → sentences → firms`
