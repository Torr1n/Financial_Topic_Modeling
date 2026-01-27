"""
Downstream Analysis Module for Financial Topic Modeling.

This package provides components for analyzing thematic sentiment from earnings
call transcripts, including event studies, portfolio sorts, and sentiment analysis.

Module Architecture
-------------------
The package is organized into several functional areas:

**Event Study Components** (intentionally separate files):

- ``event_study.ThematicES``: Orchestrates thematic event study workflow.
  Pulls WRDS data (Compustat, CRSP, IBES), computes financial covariates,
  and prepares data for regression analysis. Use this for complete thematic
  event study analysis.

- ``event_study_module.EventStudy``: Core CAR/BHAR calculation engine.
  Implements multiple risk models (market-adjusted, market, FF3, FF4).
  Use this for pure statistical event study computations without covariate
  preparation. Can be used independently for any event study.

**Why Two Event Study Files?**

These files are intentionally kept separate:

1. **Separation of Concerns**: ThematicES handles business logic and data
   orchestration; EventStudy handles statistical computation.

2. **Reusability**: EventStudy can be used for any event study, not just
   thematic analysis.

3. **Testability**: EventStudy can be tested with mock data without requiring
   WRDS covariate queries.

**Other Components**:

- ``portfolio_sorts.PortfolioSorts``: Post-event portfolio return analysis
  using WRDS data.

- ``thematic_sentiment_analyzer.ThematicSentimentAnalyzer``: FinBERT-based
  sentiment scoring for earnings call sentences.

- ``wrds_connection.WRDSConnection``: Context manager for WRDS database
  connections with proper lifecycle management.

- ``utils``: Consolidated utility functions for logging, data loading,
  batched processing, and visualization.

- ``visualization``: Chart and visualization functions for regression
  summaries and portfolio analysis.

Quick Start
-----------
For thematic event study analysis::

    from src.event_study import ThematicES
    from src.wrds_connection import WRDSConnection

    events = [{"permno": 10002, "edate": "05/29/2012", "sentiment": 1}]

    with WRDSConnection() as conn:
        study = ThematicES(events, wrds_connection=conn)
        results = study.doAll()

For standalone CAR/BHAR calculation::

    from src.event_study_module import EventStudy

    events = [{"permno": 10002, "edate": "05/29/2012"}]
    es = EventStudy(output_path="/path/to/output")
    results = es.eventstudy(data=events, model='madj', output='df')

For sentiment analysis::

    from src.thematic_sentiment_analyzer import ThematicSentimentAnalyzer

    analyzer = ThematicSentimentAnalyzer()
    scores = analyzer.analyze_themes(themes_data)

See Also
--------
- CLAUDE.md: Project overview and development commands
- cli.py: Command-line interface for running analyses
- config.py: Configuration management
"""

# Public API exports
__all__ = [
    # Event study components
    "ThematicES",
    "EventStudy",
    # Portfolio analysis
    "PortfolioSorts",
    # Sentiment analysis
    "ThematicSentimentAnalyzer",
    # Infrastructure
    "WRDSConnection",
]

# Lazy imports to avoid circular dependencies and improve startup time
# Users should import directly from submodules for explicit imports:
#   from src.event_study import ThematicES
#   from src.event_study_module import EventStudy

def __getattr__(name):
    """Lazy import of public API classes."""
    if name == "ThematicES":
        from src.event_study import ThematicES
        return ThematicES
    elif name == "EventStudy":
        from src.event_study_module import EventStudy
        return EventStudy
    elif name == "PortfolioSorts":
        from src.portfolio_sorts import PortfolioSorts
        return PortfolioSorts
    elif name == "ThematicSentimentAnalyzer":
        from src.thematic_sentiment_analyzer import ThematicSentimentAnalyzer
        return ThematicSentimentAnalyzer
    elif name == "WRDSConnection":
        from src.wrds_connection import WRDSConnection
        return WRDSConnection
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
