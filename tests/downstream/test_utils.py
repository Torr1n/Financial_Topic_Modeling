"""Tests for downstream utils module."""
import sys
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd
import pytest

# Add downstream to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'downstream'))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'downstream' / 'src'))

from src.utils import create_regression_significance_summary


def test_create_regression_summary_empty():
    """Test summary handles empty input."""
    with tempfile.TemporaryDirectory() as tmpdir:
        result = create_regression_significance_summary({}, [], tmpdir)
        assert result.empty


def test_create_regression_summary_with_data():
    """Test summary with mock regression results."""
    # Create mock model
    mock_model = MagicMock()
    mock_model.params = pd.Series({'sentiment': 0.05, 'intercept': 0.01})
    mock_model.pvalues = pd.Series({'sentiment': 0.03, 'intercept': 0.001})
    mock_model.tvalues = pd.Series({'sentiment': 2.1, 'intercept': 3.5})
    mock_model.rsquared = 0.15
    mock_model.rsquared_adj = 0.12
    mock_model.nobs = 100

    themes = [{'theme_id': 'theme_001', 'theme_name': 'Test Theme'}]
    models = {'theme_001': mock_model}

    with tempfile.TemporaryDirectory() as tmpdir:
        result = create_regression_significance_summary(models, themes, tmpdir)

        assert len(result) == 1
        assert result.iloc[0]['theme_id'] == 'theme_001'
        assert result.iloc[0]['sentiment_coef'] == 0.05
        assert result.iloc[0]['significant_5pct'] == True

        # Check files created
        assert os.path.exists(os.path.join(tmpdir, 'regression_significance_summary.csv'))
        assert os.path.exists(os.path.join(tmpdir, 'regression_significance_summary.txt'))


def test_create_regression_summary_handles_list_theme_name():
    """Test that theme_name as list is handled."""
    mock_model = MagicMock()
    mock_model.params = pd.Series({'sentiment': 0.02})
    mock_model.pvalues = pd.Series({'sentiment': 0.08})
    mock_model.tvalues = pd.Series({'sentiment': 1.5})
    mock_model.rsquared = 0.10
    mock_model.rsquared_adj = 0.08
    mock_model.nobs = 50

    # theme_name as list (edge case from original code)
    themes = [{'theme_id': 'theme_002', 'theme_name': ['Theme With List Name']}]
    models = {'theme_002': mock_model}

    with tempfile.TemporaryDirectory() as tmpdir:
        result = create_regression_significance_summary(models, themes, tmpdir)

        assert len(result) == 1
        assert result.iloc[0]['theme_name'] == 'Theme With List Name'


def test_create_regression_summary_significance_levels():
    """Test significance level classification."""
    def make_mock_model(pval):
        m = MagicMock()
        m.params = pd.Series({'sentiment': 0.01})
        m.pvalues = pd.Series({'sentiment': pval})
        m.tvalues = pd.Series({'sentiment': 2.0})
        m.rsquared = 0.1
        m.rsquared_adj = 0.08
        m.nobs = 100
        return m

    themes = [
        {'theme_id': 't1', 'theme_name': 'Significant 5%'},
        {'theme_id': 't2', 'theme_name': 'Significant 10%'},
        {'theme_id': 't3', 'theme_name': 'Not Significant'},
    ]
    models = {
        't1': make_mock_model(0.03),  # < 5%
        't2': make_mock_model(0.07),  # < 10%
        't3': make_mock_model(0.15),  # > 10%
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        result = create_regression_significance_summary(models, themes, tmpdir)

        # Should be sorted by p-value
        assert result.iloc[0]['theme_id'] == 't1'
        assert result.iloc[1]['theme_id'] == 't2'
        assert result.iloc[2]['theme_id'] == 't3'

        # Check significance flags
        assert result[result['theme_id'] == 't1']['significant_5pct'].values[0] == True
        assert result[result['theme_id'] == 't2']['significant_5pct'].values[0] == False
        assert result[result['theme_id'] == 't2']['significant_10pct'].values[0] == True
        assert result[result['theme_id'] == 't3']['significant_10pct'].values[0] == False
