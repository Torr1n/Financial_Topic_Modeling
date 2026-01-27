# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Financial Topic Modeling research project that builds an NLP pipeline for identifying cross-firm investment themes from earnings call transcripts. The project has three main components:

1. **cloud/**: Production topic modeling pipeline (AWS-ready)
2. **downstream/**: Sentiment analysis, event studies, and portfolio sorts
3. **legacy/**: Old MVP code (reference only)

## Architecture

The pipeline follows a hierarchical approach:

1. **Data Ingestion**: Load earnings call transcripts from WRDS, local CSV, or cloud storage
2. **Firm-Level Topic Modeling**: Per-firm BERTopic clustering of sentences into topics
3. **Cross-Firm Theme Identification**: Re-cluster firm topics to discover universal themes
4. **Export**: Bridge script adds PERMNO identifiers from WRDS
5. **Downstream Analysis**: FinBERT sentiment, event study regressions, portfolio sorts

Key architectural patterns:
- **Map phase**: Independent firm-level processing (parallelizable via AWS Batch)
- **Reduce phase**: Cross-firm aggregation (single instance, larger compute)
- **Topic model abstraction**: Designed for swappable implementations (BERTopic, LDA, neural methods)

## Development Commands

```bash
# Activate virtual environment
source venv/bin/activate

# Run cloud pipeline
python -m cloud.src.pipeline.unified_pipeline --config cloud/config/production.yaml

# Export themes for downstream
python -m cloud.src.export.export_for_downstream \
    --db-url postgresql://user:pass@host/db \
    --output downstream/data/themes.json

# Run downstream analysis (all stages)
cd downstream && python cli.py --themes data/themes.json --output results/

# Run downstream (sentiment only - no WRDS needed)
cd downstream && python cli.py --themes data/themes.json --stages sentiment
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
