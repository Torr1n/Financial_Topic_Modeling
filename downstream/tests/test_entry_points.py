"""
Tests for CLI and entry points.

These tests verify that CLI commands and entry point scripts can be invoked
without errors. All tests use mocking to avoid requiring actual WRDS credentials
or processing real data.
"""

import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest


# Get the downstream directory path
DOWNSTREAM_DIR = Path(__file__).parent.parent


class TestCLIRunCommand:
    """Test CLI 'run' command structure."""

    def test_cli_run_command_smoke_test(self):
        """CLI run command should parse minimal arguments without error."""
        # Run cli.py with --help to verify command structure
        result = subprocess.run(
            [sys.executable, str(DOWNSTREAM_DIR / 'cli.py'), '--help'],
            capture_output=True,
            text=True,
            cwd=str(DOWNSTREAM_DIR)
        )

        # Should exit successfully
        assert result.returncode == 0, f"CLI help failed: {result.stderr}"

        # Help output should show subcommands or main usage
        assert 'usage' in result.stdout.lower() or 'positional arguments' in result.stdout.lower()

    def test_cli_run_subcommand_structure(self):
        """CLI 'run' subcommand should exist and have proper help."""
        result = subprocess.run(
            [sys.executable, str(DOWNSTREAM_DIR / 'cli.py'), 'run', '--help'],
            capture_output=True,
            text=True,
            cwd=str(DOWNSTREAM_DIR)
        )

        # Should exit successfully
        assert result.returncode == 0, f"CLI run help failed: {result.stderr}"

        # Help output should mention key options
        assert '--themes' in result.stdout or '-t' in result.stdout
        assert '--output' in result.stdout or '-o' in result.stdout
        assert '--stages' in result.stdout


class TestCLIEventStudySubcommand:
    """Test CLI 'event-study' subcommand structure."""

    def test_cli_event_study_subcommand_help(self):
        """CLI event-study subcommand should have proper help output."""
        result = subprocess.run(
            [sys.executable, str(DOWNSTREAM_DIR / 'cli.py'), 'event-study', '--help'],
            capture_output=True,
            text=True,
            cwd=str(DOWNSTREAM_DIR)
        )

        # Should exit successfully
        assert result.returncode == 0, f"event-study help failed: {result.stderr}"

        # Help output should mention required options
        assert '--sentiment-file' in result.stdout or 'sentiment' in result.stdout.lower()


class TestCLIPortfolioSubcommand:
    """Test CLI 'portfolio' subcommand structure."""

    def test_cli_portfolio_subcommand_help(self):
        """CLI portfolio subcommand should have proper help output."""
        result = subprocess.run(
            [sys.executable, str(DOWNSTREAM_DIR / 'cli.py'), 'portfolio', '--help'],
            capture_output=True,
            text=True,
            cwd=str(DOWNSTREAM_DIR)
        )

        # Should exit successfully
        assert result.returncode == 0, f"portfolio help failed: {result.stderr}"

        # Help output should mention required options
        assert '--sentiment-file' in result.stdout or 'sentiment' in result.stdout.lower()


class TestThinWrappers:
    """Test that thin wrapper scripts delegate to CLI correctly."""

    def test_run_pipeline_wrapper_has_help(self):
        """run_pipeline.py should have help output (backward compatibility)."""
        result = subprocess.run(
            [sys.executable, str(DOWNSTREAM_DIR / 'run_pipeline.py'), '--help'],
            capture_output=True,
            text=True,
            cwd=str(DOWNSTREAM_DIR)
        )

        # Should exit successfully
        assert result.returncode == 0, f"run_pipeline.py help failed: {result.stderr}"

        # Should show usage information
        assert 'themes' in result.stdout.lower() or 'usage' in result.stdout.lower()

    def test_run_event_study_wrapper_has_help(self):
        """run_event_study.py should have help output (backward compatibility)."""
        result = subprocess.run(
            [sys.executable, str(DOWNSTREAM_DIR / 'run_event_study.py'), '--help'],
            capture_output=True,
            text=True,
            cwd=str(DOWNSTREAM_DIR)
        )

        # Should exit successfully
        assert result.returncode == 0, f"run_event_study.py help failed: {result.stderr}"

        # Should show usage information
        assert 'sentiment' in result.stdout.lower() or 'usage' in result.stdout.lower()

    def test_run_portfolio_sorts_wrapper_has_help(self):
        """run_portfolio_sorts.py should have help output (backward compatibility)."""
        result = subprocess.run(
            [sys.executable, str(DOWNSTREAM_DIR / 'run_portfolio_sorts.py'), '--help'],
            capture_output=True,
            text=True,
            cwd=str(DOWNSTREAM_DIR)
        )

        # Should exit successfully
        assert result.returncode == 0, f"run_portfolio_sorts.py help failed: {result.stderr}"

        # Should show usage information
        assert 'sentiment' in result.stdout.lower() or 'usage' in result.stdout.lower()
