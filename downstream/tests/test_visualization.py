"""Tests for visualization module functions.

These tests verify that visualization functions produce expected output formats
and execute without errors using sample data fixtures.
"""

import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest


# Check if matplotlib is available and working
def _matplotlib_available():
    """Check if matplotlib can be imported without errors."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        return True
    except (ImportError, AttributeError):
        return False


MATPLOTLIB_AVAILABLE = _matplotlib_available()


class TestCreateRegressionSignificanceSummary:
    """Tests for create_regression_significance_summary function."""

    def test_generates_expected_output_format(self, tmp_path):
        """Test that function generates CSV and TXT outputs with expected columns."""
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

        mock_model_2 = MagicMock()
        mock_model_2.params = pd.Series({'const': -0.02, 'sentiment': -0.03})
        mock_model_2.bse = pd.Series({'const': 0.002, 'sentiment': 0.015})
        mock_model_2.tvalues = pd.Series({'const': -10.0, 'sentiment': -2.0})
        mock_model_2.pvalues = pd.Series({'const': 0.001, 'sentiment': 0.046})
        mock_model_2.rsquared = 0.10
        mock_model_2.rsquared_adj = 0.08
        mock_model_2.nobs = 80

        # Create mock result objects (mimicking run_pipeline.py RegressionResult)
        class MockResult:
            def __init__(self, model):
                self.model = model

        event_study_models = {
            'theme_001': MockResult(mock_model_1),
            'theme_002': MockResult(mock_model_2),
        }

        themes = {
            'theme_001': {'theme_name': 'Technology Innovation'},
            'theme_002': {'theme_name': 'Financial Performance'},
        }

        # Run the function
        result_df = create_regression_significance_summary(
            event_study_models, themes, tmp_path
        )

        # Verify DataFrame output
        assert result_df is not None
        assert len(result_df) == 2

        # Check expected columns exist
        expected_columns = [
            'Theme_ID', 'Theme_Name', 'Sentiment_Coef', 'Std_Error',
            't_Statistic', 'p_Value', 'Significance', 'R_Squared',
            'Adj_R_Squared', 'N_Observations'
        ]
        for col in expected_columns:
            assert col in result_df.columns, f"Missing column: {col}"

        # Verify sorting by p-value (ascending)
        assert result_df.iloc[0]['p_Value'] <= result_df.iloc[1]['p_Value']

        # Verify files were created
        assert (tmp_path / 'regression_significance_summary.csv').exists()
        assert (tmp_path / 'regression_significance_summary.txt').exists()

    def test_handles_empty_models(self, tmp_path):
        """Test that function handles empty model dictionary gracefully."""
        from src.visualization import create_regression_significance_summary

        result_df = create_regression_significance_summary({}, {}, tmp_path)
        assert result_df is None


class TestCreatePortfolioTimeSeriesChart:
    """Tests for create_portfolio_time_series_chart function."""

    @pytest.mark.skipif(not MATPLOTLIB_AVAILABLE, reason="matplotlib not available or incompatible")
    def test_creates_chart_without_errors(self, tmp_path):
        """Test that chart function executes without errors and creates PNG."""
        from src.visualization import create_portfolio_time_series_chart

        # Create sample portfolio data
        portfolio_data = []
        for bucket in ['Low', 'Medium', 'High']:
            cumulative = 0.0
            for day in range(0, 91, 10):
                daily_return = 0.001 if bucket == 'High' else (-0.001 if bucket == 'Low' else 0.0)
                cumulative += daily_return
                portfolio_data.append({
                    'bucket': bucket,
                    'days_from_event': day,
                    'vw_return': daily_return,
                    'cumulative_return': cumulative,
                    'n_themes': 5
                })

        combined_portfolio = pd.DataFrame(portfolio_data)

        # Run the function
        create_portfolio_time_series_chart(combined_portfolio, tmp_path)

        # Verify chart was created
        chart_path = tmp_path / 'portfolio_time_series_chart.png'
        assert chart_path.exists(), "Chart PNG file was not created"
        assert chart_path.stat().st_size > 0, "Chart file is empty"

    def test_handles_empty_portfolio(self, tmp_path):
        """Test that function handles empty portfolio data gracefully."""
        from src.visualization import create_portfolio_time_series_chart

        # Should not raise an error
        create_portfolio_time_series_chart(None, tmp_path)
        create_portfolio_time_series_chart(pd.DataFrame(), tmp_path)


class TestCreateCombinedPortfolioAnalysis:
    """Tests for create_combined_portfolio_analysis function."""

    def test_produces_expected_structure(self, tmp_path):
        """Test that combined analysis produces expected DataFrame structure."""
        from src.visualization import create_combined_portfolio_analysis

        # Create sample portfolio results for multiple themes
        def create_theme_portfolio(theme_id):
            data = []
            for bucket in ['Low', 'Medium', 'High']:
                cumulative = 0.0
                for day in range(0, 91, 10):
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
            'theme_003': create_theme_portfolio('theme_003'),
        }

        # Run the function
        combined_portfolio = create_combined_portfolio_analysis(portfolio_results, tmp_path)

        # Verify output structure
        assert combined_portfolio is not None
        assert not combined_portfolio.empty

        # Check expected columns
        expected_columns = ['bucket', 'days_from_event', 'vw_return', 'cumulative_return', 'n_themes']
        for col in expected_columns:
            assert col in combined_portfolio.columns, f"Missing column: {col}"

        # Verify all buckets are present
        buckets_in_result = combined_portfolio['bucket'].unique()
        assert 'Low' in buckets_in_result
        assert 'Medium' in buckets_in_result
        assert 'High' in buckets_in_result

        # Verify CSV was saved
        assert (tmp_path / 'combined_all_themes_portfolio_returns.csv').exists()

    def test_handles_empty_results(self, tmp_path):
        """Test that function handles empty portfolio results gracefully."""
        from src.visualization import create_combined_portfolio_analysis

        result = create_combined_portfolio_analysis({}, tmp_path)
        assert result is None
