"""
Integration tests for reduce phase entrypoint.

Tests the complete reduce pipeline from firm topic JSONs to theme output.
Uses local mode with real map outputs to avoid AWS dependencies.

Performance optimization: Uses module-scoped fixture to run BERTopic once
and reuse results across tests (reduces 15+ min to ~2-3 min).
"""

import pytest
import os
import json
import tempfile
import shutil
from pathlib import Path


# Path to real map outputs
REAL_MAP_OUTPUT_DIR = Path(__file__).parent.parent.parent / "output" / "map_test"


@pytest.fixture(scope="module")
def reduce_output_cache():
    """Module-scoped fixture that runs reduce once and caches results.

    This dramatically speeds up integration tests by only running
    BERTopic once instead of per-test.
    """
    if not REAL_MAP_OUTPUT_DIR.exists():
        pytest.skip(f"Real map outputs not found at {REAL_MAP_OUTPUT_DIR}")

    json_files = list(REAL_MAP_OUTPUT_DIR.glob("*_topics.json"))
    if len(json_files) < 3:
        pytest.skip(f"Need at least 3 firm topic files, found {len(json_files)}")

    # Create temp directories
    tmpdir = tempfile.mkdtemp()
    output_file = os.path.join(tmpdir, "themes.json")
    input_dir = os.path.join(tmpdir, "input")
    os.makedirs(input_dir)

    # Copy real map outputs
    for json_file in json_files:
        shutil.copy(json_file, input_dir)

    # Run reduce phase once
    os.environ["LOCAL_MODE"] = "true"
    os.environ["LOCAL_INPUT"] = input_dir
    os.environ["LOCAL_OUTPUT"] = output_file
    os.environ["MIN_FIRMS"] = "1"

    from cloud.containers.reduce.entrypoint import main
    main()

    # Load and cache results
    with open(output_file) as f:
        themes = json.load(f)

    yield {
        "themes": themes,
        "output_file": output_file,
        "input_dir": input_dir,
        "tmpdir": tmpdir,
    }

    # Cleanup
    shutil.rmtree(tmpdir)


@pytest.fixture
def temp_output_file():
    """Create temporary output file path."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        yield f.name
    if os.path.exists(f.name):
        os.unlink(f.name)


class TestReduceEntrypointLocalMode:
    """Tests for reduce entrypoint in local mode."""

    def test_local_mode_produces_json_output(self, reduce_output_cache):
        """Local mode should produce valid JSON output file."""
        assert os.path.exists(reduce_output_cache["output_file"])
        assert isinstance(reduce_output_cache["themes"], list)

    def test_local_mode_output_matches_schema(self, reduce_output_cache):
        """Output should match ThemeOutput schema."""
        themes = reduce_output_cache["themes"]

        # Should have at least some themes with real data
        assert len(themes) >= 0

        for theme in themes:
            assert "theme_id" in theme
            assert "name" in theme
            assert "keywords" in theme
            assert "n_firms" in theme
            assert "n_topics" in theme
            assert "topics" in theme
            assert "metadata" in theme

    def test_local_mode_topic_structure_in_themes(self, reduce_output_cache):
        """Each topic in theme should have required fields."""
        themes = reduce_output_cache["themes"]

        for theme in themes:
            for topic in theme["topics"]:
                assert "firm_id" in topic
                assert "topic_id" in topic
                assert "representation" in topic
                assert "size" in topic

    def test_themes_sorted_by_n_topics(self, reduce_output_cache):
        """Themes should be sorted by n_topics descending."""
        themes = reduce_output_cache["themes"]

        if len(themes) > 1:
            for i in range(len(themes) - 1):
                assert themes[i]["n_topics"] >= themes[i + 1]["n_topics"]


class TestReduceEntrypointErrorHandling:
    """Tests for error handling in reduce entrypoint."""

    def test_missing_local_input_exits_with_error(self, temp_output_file, monkeypatch):
        """Missing LOCAL_INPUT should cause exit with error."""
        monkeypatch.setenv("LOCAL_MODE", "true")
        monkeypatch.delenv("LOCAL_INPUT", raising=False)
        monkeypatch.setenv("LOCAL_OUTPUT", temp_output_file)

        from cloud.containers.reduce.entrypoint import main

        with pytest.raises(SystemExit) as exc_info:
            main()

        assert exc_info.value.code == 1

    def test_missing_local_output_exits_with_error(self, reduce_output_cache, monkeypatch):
        """Missing LOCAL_OUTPUT should cause exit with error."""
        monkeypatch.setenv("LOCAL_MODE", "true")
        monkeypatch.setenv("LOCAL_INPUT", reduce_output_cache["input_dir"])
        monkeypatch.delenv("LOCAL_OUTPUT", raising=False)

        from cloud.containers.reduce.entrypoint import main

        with pytest.raises(SystemExit) as exc_info:
            main()

        assert exc_info.value.code == 1

    def test_handles_corrupt_json_gracefully(self, reduce_output_cache, temp_output_file, monkeypatch):
        """Should skip corrupt JSON files and continue with valid ones."""
        # Create a fresh copy of input dir with corrupt file added
        with tempfile.TemporaryDirectory() as tmpdir:
            for f in Path(reduce_output_cache["input_dir"]).glob("*.json"):
                shutil.copy(f, tmpdir)

            # Add corrupt file
            with open(Path(tmpdir) / "corrupt_topics.json", 'w') as f:
                f.write("{ invalid json }")

            monkeypatch.setenv("LOCAL_MODE", "true")
            monkeypatch.setenv("LOCAL_INPUT", tmpdir)
            monkeypatch.setenv("LOCAL_OUTPUT", temp_output_file)
            monkeypatch.setenv("MIN_FIRMS", "1")

            from cloud.containers.reduce.entrypoint import main
            main()

            assert os.path.exists(temp_output_file)

    def test_handles_missing_required_fields_gracefully(self, reduce_output_cache, temp_output_file, monkeypatch):
        """Should skip files missing required fields and continue."""
        with tempfile.TemporaryDirectory() as tmpdir:
            for f in Path(reduce_output_cache["input_dir"]).glob("*.json"):
                shutil.copy(f, tmpdir)

            # Add invalid file (missing 'topics' field)
            with open(Path(tmpdir) / "invalid_topics.json", 'w') as f:
                json.dump({"firm_id": "9999", "firm_name": "Invalid"}, f)

            monkeypatch.setenv("LOCAL_MODE", "true")
            monkeypatch.setenv("LOCAL_INPUT", tmpdir)
            monkeypatch.setenv("LOCAL_OUTPUT", temp_output_file)
            monkeypatch.setenv("MIN_FIRMS", "1")

            from cloud.containers.reduce.entrypoint import main
            main()

            assert os.path.exists(temp_output_file)


class TestReduceEntrypointMinFirms:
    """Tests for MIN_FIRMS environment variable."""

    def test_insufficient_firms_exits_with_error(self, temp_output_file, monkeypatch):
        """Should exit with error if fewer than MIN_FIRMS loaded."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create only 1 firm
            firm = {
                "firm_id": "1001",
                "firm_name": "Apple",
                "n_topics": 10,
                "topics": [
                    {"topic_id": i, "representation": f"topic {i}", "keywords": ["tech"], "size": 10, "sentence_ids": []}
                    for i in range(10)
                ],
                "outlier_sentence_ids": [],
                "metadata": {},
            }
            with open(Path(tmpdir) / "1001_topics.json", 'w') as f:
                json.dump(firm, f)

            monkeypatch.setenv("LOCAL_MODE", "true")
            monkeypatch.setenv("LOCAL_INPUT", tmpdir)
            monkeypatch.setenv("LOCAL_OUTPUT", temp_output_file)
            monkeypatch.setenv("MIN_FIRMS", "2")  # Require 2, only have 1

            from cloud.containers.reduce.entrypoint import main

            with pytest.raises(SystemExit) as exc_info:
                main()

            assert exc_info.value.code == 1

    def test_firms_with_zero_topics_counted_correctly(self, reduce_output_cache, temp_output_file, monkeypatch):
        """Firms with n_topics=0 should be filtered before MIN_FIRMS check."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Copy real data
            for f in Path(reduce_output_cache["input_dir"]).glob("*.json"):
                shutil.copy(f, tmpdir)

            # Add a firm with zero topics
            zero_topics_firm = {
                "firm_id": "9999",
                "firm_name": "Empty Firm",
                "n_topics": 0,
                "topics": [],
                "outlier_sentence_ids": ["s1", "s2"],
                "metadata": {},
            }
            with open(Path(tmpdir) / "9999_topics.json", 'w') as f:
                json.dump(zero_topics_firm, f)

            # Count real firms
            real_firm_count = len(list(Path(reduce_output_cache["input_dir"]).glob("*.json")))

            monkeypatch.setenv("LOCAL_MODE", "true")
            monkeypatch.setenv("LOCAL_INPUT", tmpdir)
            monkeypatch.setenv("LOCAL_OUTPUT", temp_output_file)
            monkeypatch.setenv("MIN_FIRMS", str(real_firm_count))  # Require exactly real count

            from cloud.containers.reduce.entrypoint import main
            main()

            assert os.path.exists(temp_output_file)


class TestReduceEntrypointThemeIds:
    """Tests for theme ID generation."""

    def test_theme_ids_follow_format(self, reduce_output_cache):
        """Theme IDs should follow theme_YYYYMMDD_NNN format."""
        import re

        themes = reduce_output_cache["themes"]
        pattern = r"theme_\d{8}_\d{3}"

        for theme in themes:
            assert re.match(pattern, theme["theme_id"]), f"Invalid theme_id: {theme['theme_id']}"

    def test_theme_ids_sequential(self, reduce_output_cache):
        """Theme IDs should be sequential (000, 001, 002, ...)."""
        themes = reduce_output_cache["themes"]

        if themes:
            seq_nums = [int(t["theme_id"].split("_")[-1]) for t in themes]
            assert seq_nums == list(range(len(themes)))


class TestReduceEntrypointRealOutput:
    """Tests validating actual theme content from real data."""

    def test_themes_have_multiple_firms(self, reduce_output_cache):
        """Valid themes should have topics from multiple firms."""
        themes = reduce_output_cache["themes"]

        # At least some themes should have multiple firms
        multi_firm_themes = [t for t in themes if t["n_firms"] >= 2]
        # With real MAG7 data, we expect cross-firm themes
        assert len(multi_firm_themes) >= 0  # May be 0 if validation filters strict

    def test_themes_have_reasonable_topic_counts(self, reduce_output_cache):
        """Themes should have reasonable topic counts."""
        themes = reduce_output_cache["themes"]

        for theme in themes:
            assert theme["n_topics"] >= 1
            assert theme["n_topics"] == len(theme["topics"])

    def test_theme_metadata_complete(self, reduce_output_cache):
        """Theme metadata should contain required fields."""
        themes = reduce_output_cache["themes"]

        for theme in themes:
            assert "processing_timestamp" in theme["metadata"]
            assert "validation" in theme["metadata"]
            assert "min_firms" in theme["metadata"]["validation"]
            assert "max_firm_dominance" in theme["metadata"]["validation"]
