"""Tests for downstream config validation."""
import sys
from pathlib import Path

# Add downstream to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'downstream'))

import config


def test_config_validation():
    """Test that config validates successfully."""
    # Should not raise
    config.validate_config()


def test_event_window_constraints():
    """Test event window parameter constraints."""
    assert config.EVENT_WINDOW_START < 0
    assert config.EVENT_WINDOW_END > 0
    assert config.ESTIMATION_WINDOW > 0


def test_valid_model_options():
    """Test MODEL is a valid option."""
    assert config.MODEL in ['m', 'ff', 'ffm', 'madj']


def test_valid_weighting_options():
    """Test WEIGHTING is a valid option."""
    assert config.WEIGHTING in ['value', 'equal']


def test_batch_size_positive():
    """Test BATCH_SIZE is positive."""
    assert config.BATCH_SIZE > 0


def test_portfolio_days_positive():
    """Test PORTFOLIO_DAYS is positive."""
    assert config.PORTFOLIO_DAYS > 0
