"""
Regression tests to verify output format equivalence.

These tests use mock-based expected outputs (golden files) to verify that
the pipeline produces outputs in the correct format. Since we don't have
WRDS access, tests focus on format validation rather than data accuracy.
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

# Path to fixtures directory
FIXTURES_DIR = Path(__file__).parent / 'fixtures'


class TestSentimentOutputFormat:
    """Regression tests for sentiment analysis output format."""

    def test_sentiment_output_has_required_columns(self):
        """Sentiment CSV output should contain required columns for event study."""
        # Load sample sentiment output
        sentiment_df = pd.read_csv(FIXTURES_DIR / 'sample_sentiment.csv')

        # Verify required columns exist
        required_columns = ['permno', 'edate', 'sentiment']
        for col in required_columns:
            assert col in sentiment_df.columns, f"Missing required column: {col}"

    def test_sentiment_output_format_matches_expected(self):
        """Sentiment output format should match expected structure."""
        sentiment_df = pd.read_csv(FIXTURES_DIR / 'sample_sentiment.csv')

        # Verify data types are compatible
        assert sentiment_df['permno'].dtype in ['int64', 'float64'], "permno should be numeric"
        assert sentiment_df['sentiment'].dtype in ['float64', 'float32'], "sentiment should be float"

        # Verify sentiment values are in expected range
        assert sentiment_df['sentiment'].min() >= -1.0, "sentiment should be >= -1"
        assert sentiment_df['sentiment'].max() <= 1.0, "sentiment should be <= 1"


class TestEventStudyOutputFormat:
    """Regression tests for event study output format."""

    def test_event_study_output_has_required_columns(self):
        """Event study CSV output should contain required columns."""
        event_study_df = pd.read_csv(FIXTURES_DIR / 'expected_event_study.csv')

        # Verify required columns exist
        required_columns = ['permno', 'edate', 'sentiment', 'car']
        for col in required_columns:
            assert col in event_study_df.columns, f"Missing required column: {col}"

    def test_event_study_output_format_matches_expected(self):
        """Event study output format should match expected structure."""
        event_study_df = pd.read_csv(FIXTURES_DIR / 'expected_event_study.csv')

        # Verify CAR values are reasonable (typically small percentages)
        assert event_study_df['car'].abs().max() < 1.0, "CAR values should be < 100%"

        # Verify no missing values in key columns
        assert not event_study_df['permno'].isna().any(), "permno should not have NaN"
        assert not event_study_df['car'].isna().any(), "car should not have NaN"


class TestRegressionSummaryFormat:
    """Regression tests for regression summary output format."""

    def test_regression_summary_has_required_columns(self):
        """Regression summary CSV should contain required columns."""
        summary_df = pd.read_csv(FIXTURES_DIR / 'expected_regression_summary.csv')

        # Verify required columns exist
        required_columns = [
            'Theme_ID', 'Theme_Name', 'Sentiment_Coef', 'Std_Error',
            't_Statistic', 'p_Value', 'Significance', 'R_Squared',
            'Adj_R_Squared', 'N_Observations'
        ]
        for col in required_columns:
            assert col in summary_df.columns, f"Missing required column: {col}"

    def test_regression_summary_format_via_visualization(self, tmp_path):
        """Visualization function should produce expected regression summary format."""
        from src.visualization import create_regression_significance_summary

        # Create mock regression models
        mock_model_1 = MagicMock()
        mock_model_1.params = pd.Series({'const': 0.01, 'sentiment': 0.05})
        mock_model_1.bse = pd.Series({'const': 0.001, 'sentiment': 0.02})
        mock_model_1.tvalues = pd.Series({'const': 10.0, 'sentiment': 2.5})
        mock_model_1.pvalues = pd.Series({'const': 0.001, 'sentiment': 0.012})
        mock_model_1.rsquared = 0.15
        mock_model_1.rsquared_adj = 0.12
        mock_model_1.nobs = 100

        class MockResult:
            def __init__(self, model):
                self.model = model

        event_study_models = {'theme_001': MockResult(mock_model_1)}
        themes = {'theme_001': {'theme_name': 'Technology Innovation'}}

        # Run the function
        result_df = create_regression_significance_summary(
            event_study_models, themes, tmp_path
        )

        # Verify output format matches golden file format
        expected_df = pd.read_csv(FIXTURES_DIR / 'expected_regression_summary.csv')

        assert result_df is not None
        assert set(result_df.columns) == set(expected_df.columns), \
            "Output columns should match expected columns"

        # Verify CSV file was created
        assert (tmp_path / 'regression_significance_summary.csv').exists()
        assert (tmp_path / 'regression_significance_summary.txt').exists()


class TestPortfolioSortsFormat:
    """Regression tests for portfolio sorts output format."""

    def test_portfolio_returns_has_required_columns(self):
        """Portfolio returns CSV should contain required columns."""
        portfolio_df = pd.read_csv(FIXTURES_DIR / 'expected_portfolio_returns.csv')

        # Verify required columns exist
        required_columns = ['bucket', 'days_from_event', 'vw_return', 'cumulative_return']
        for col in required_columns:
            assert col in portfolio_df.columns, f"Missing required column: {col}"

    def test_portfolio_returns_has_all_buckets(self):
        """Portfolio returns should include Low, Medium, High buckets."""
        portfolio_df = pd.read_csv(FIXTURES_DIR / 'expected_portfolio_returns.csv')

        buckets = portfolio_df['bucket'].unique()
        expected_buckets = ['Low', 'Medium', 'High']

        for bucket in expected_buckets:
            assert bucket in buckets, f"Missing bucket: {bucket}"

    def test_combined_portfolio_analysis_format(self, tmp_path):
        """Combined portfolio analysis should produce expected format."""
        from src.visualization import create_combined_portfolio_analysis

        # Create sample portfolio results for multiple themes
        def create_theme_portfolio(theme_id):
            data = []
            for bucket in ['Low', 'Medium', 'High']:
                cumulative = 0.0
                for day in range(0, 31, 10):
                    daily_return = 0.001 if bucket == 'High' else (-0.001 if bucket == 'Low' else 0.0)
                    cumulative += daily_return
                    data.append({
                        'bucket': bucket,
                        'days_from_event': day,
                        'vw_return': daily_return,
                        'cumulative_return': cumulative
                    })
            return pd.DataFrame(data)

        portfolio_results = {
            'theme_001': create_theme_portfolio('theme_001'),
            'theme_002': create_theme_portfolio('theme_002'),
        }

        # Run the function
        combined_portfolio = create_combined_portfolio_analysis(portfolio_results, tmp_path)

        # Verify output format matches golden file format
        expected_df = pd.read_csv(FIXTURES_DIR / 'expected_portfolio_returns.csv')

        assert combined_portfolio is not None
        # Check all expected columns are present
        for col in expected_df.columns:
            assert col in combined_portfolio.columns, f"Missing column: {col}"

        # Verify CSV file was created
        assert (tmp_path / 'combined_all_themes_portfolio_returns.csv').exists()
