"""
Integration tests for map phase entrypoint.

Tests the complete map pipeline from CSV input to JSON output.
Uses local mode to avoid AWS dependencies.
"""

import pytest
import os
import json
import tempfile
from pathlib import Path


@pytest.fixture
def sample_csv_for_integration():
    """Create a substantial CSV for integration testing.

    Note: BERTopic with UMAP requires n_neighbors <= n_samples.
    Default n_neighbors=15, so we need at least 20+ sentences.

    Each component contains MULTIPLE sentences to test sentence splitting.
    With 10 components x 3 sentences each = 30 sentences (after stopword removal).
    """
    content = """companyid,companyname,transcriptid,componenttext,componentorder,mostimportantdateutc,speakertypename
1001,Apple Inc.,T001,"We are investing heavily in artificial intelligence. Machine learning capabilities drive innovation. Our research team has made breakthroughs.",1,2023-01-15,CEO
1001,Apple Inc.,T001,"Revenue from AI products exceeded expectations. Growth has been exceptional. Margins expanded significantly.",2,2023-01-15,CFO
1001,Apple Inc.,T001,"Supply chain operations have been optimized. Logistics costs decreased substantially. Efficiency improved markedly.",3,2023-01-15,COO
1001,Apple Inc.,T001,"Customer satisfaction scores increased due to AI support. User engagement metrics improved. Retention rates reached highs.",4,2023-01-15,CEO
1001,Apple Inc.,T001,"Data center capacity expanded for AI workloads. Infrastructure investments paying off. Performance metrics exceeded targets.",5,2023-01-15,CTO
1001,Apple Inc.,T001,"Operating margins improved with automation. Cost reduction initiatives succeeded. Profitability enhanced substantially.",6,2023-01-15,CFO
1001,Apple Inc.,T001,"Cloud services segment shows strong momentum. Enterprise demand accelerated. Subscription revenue growing rapidly.",7,2023-01-15,CEO
1001,Apple Inc.,T001,"Capital expenditures focus on infrastructure. Investment priorities clearly defined. Resource allocation optimized.",8,2023-01-15,CFO
1001,Apple Inc.,T001,"Research development spending increased significantly. Innovation pipeline strengthened. New products launching.",9,2023-01-15,CTO
1001,Apple Inc.,T001,"International markets represent growth opportunities. Global expansion progressing. Market share increasing.",10,2023-01-15,CEO
1002,Microsoft Corp.,T002,"Cloud computing revenue grew significantly this quarter. Azure adoption accelerating. Enterprise customers expanding.",1,2023-01-20,CEO
1002,Microsoft Corp.,T002,"AI services demand exceeded expectations. Copilot adoption strong. Developer productivity improved.",2,2023-01-20,CTO
"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        f.write(content)
        return f.name


@pytest.fixture
def temp_output_dir():
    """Create temporary output directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


class TestMapEntrypointLocalMode:
    """Tests for map entrypoint in local mode."""

    def test_local_mode_produces_json_output(self, sample_csv_for_integration, temp_output_dir, monkeypatch):
        """Local mode should produce valid JSON output file."""
        # Set environment variables
        monkeypatch.setenv("LOCAL_MODE", "true")
        monkeypatch.setenv("FIRM_ID", "1001")
        monkeypatch.setenv("LOCAL_INPUT", sample_csv_for_integration)
        monkeypatch.setenv("LOCAL_OUTPUT", temp_output_dir)

        # Import and run main
        from cloud.containers.map.entrypoint import main
        main()

        # Check output file exists
        output_file = Path(temp_output_dir) / "1001_topics.json"
        assert output_file.exists()

    def test_local_mode_output_matches_schema(self, sample_csv_for_integration, temp_output_dir, monkeypatch):
        """Output should match FirmTopicOutput schema."""
        monkeypatch.setenv("LOCAL_MODE", "true")
        monkeypatch.setenv("FIRM_ID", "1001")
        monkeypatch.setenv("LOCAL_INPUT", sample_csv_for_integration)
        monkeypatch.setenv("LOCAL_OUTPUT", temp_output_dir)

        from cloud.containers.map.entrypoint import main
        main()

        output_file = Path(temp_output_dir) / "1001_topics.json"
        with open(output_file) as f:
            result = json.load(f)

        # Validate schema
        assert result["firm_id"] == "1001"
        assert result["firm_name"] == "Apple Inc."
        assert isinstance(result["n_topics"], int)
        assert isinstance(result["topics"], list)
        assert isinstance(result["outlier_sentence_ids"], list)
        assert "metadata" in result

    def test_local_mode_topic_structure(self, sample_csv_for_integration, temp_output_dir, monkeypatch):
        """Each topic should have required fields."""
        monkeypatch.setenv("LOCAL_MODE", "true")
        monkeypatch.setenv("FIRM_ID", "1001")
        monkeypatch.setenv("LOCAL_INPUT", sample_csv_for_integration)
        monkeypatch.setenv("LOCAL_OUTPUT", temp_output_dir)

        from cloud.containers.map.entrypoint import main
        main()

        output_file = Path(temp_output_dir) / "1001_topics.json"
        with open(output_file) as f:
            result = json.load(f)

        for topic in result["topics"]:
            assert "topic_id" in topic
            assert "representation" in topic
            assert "keywords" in topic
            assert "size" in topic
            assert "sentence_ids" in topic

    def test_missing_firm_id_exits_with_error(self, sample_csv_for_integration, temp_output_dir, monkeypatch):
        """Missing FIRM_ID should cause exit with error."""
        monkeypatch.setenv("LOCAL_MODE", "true")
        # FIRM_ID not set
        monkeypatch.setenv("LOCAL_INPUT", sample_csv_for_integration)
        monkeypatch.setenv("LOCAL_OUTPUT", temp_output_dir)

        # Remove FIRM_ID if it exists
        monkeypatch.delenv("FIRM_ID", raising=False)

        from cloud.containers.map.entrypoint import main

        with pytest.raises(SystemExit) as exc_info:
            main()

        assert exc_info.value.code == 1

    def test_unknown_firm_exits_with_error(self, sample_csv_for_integration, temp_output_dir, monkeypatch):
        """Unknown firm should cause exit with error."""
        monkeypatch.setenv("LOCAL_MODE", "true")
        monkeypatch.setenv("FIRM_ID", "9999")  # Non-existent firm
        monkeypatch.setenv("LOCAL_INPUT", sample_csv_for_integration)
        monkeypatch.setenv("LOCAL_OUTPUT", temp_output_dir)

        from cloud.containers.map.entrypoint import main

        with pytest.raises(SystemExit) as exc_info:
            main()

        assert exc_info.value.code == 1


class TestMapEntrypointMetadata:
    """Tests for metadata in output."""

    def test_metadata_contains_timestamp(self, sample_csv_for_integration, temp_output_dir, monkeypatch):
        """Metadata should contain processing timestamp."""
        monkeypatch.setenv("LOCAL_MODE", "true")
        monkeypatch.setenv("FIRM_ID", "1001")
        monkeypatch.setenv("LOCAL_INPUT", sample_csv_for_integration)
        monkeypatch.setenv("LOCAL_OUTPUT", temp_output_dir)

        from cloud.containers.map.entrypoint import main
        main()

        output_file = Path(temp_output_dir) / "1001_topics.json"
        with open(output_file) as f:
            result = json.load(f)

        assert "processing_timestamp" in result["metadata"]

    def test_metadata_contains_sentence_count_more_than_components(self, sample_csv_for_integration, temp_output_dir, monkeypatch):
        """Metadata should contain n_sentences_processed > number of CSV rows (due to sentence splitting)."""
        monkeypatch.setenv("LOCAL_MODE", "true")
        monkeypatch.setenv("FIRM_ID", "1001")
        monkeypatch.setenv("LOCAL_INPUT", sample_csv_for_integration)
        monkeypatch.setenv("LOCAL_OUTPUT", temp_output_dir)

        from cloud.containers.map.entrypoint import main
        main()

        output_file = Path(temp_output_dir) / "1001_topics.json"
        with open(output_file) as f:
            result = json.load(f)

        # Apple has 10 components, each with 3 sentences = ~30 sentences
        # (exact count depends on stopword removal)
        n_components = 10
        assert result["metadata"]["n_sentences_processed"] > n_components
        # Should be around 30 sentences (3 per component)
        assert result["metadata"]["n_sentences_processed"] >= 20


class TestMapEntrypointSentenceIds:
    """Tests for sentence ID handling."""

    def test_sentence_ids_follow_format(self, sample_csv_for_integration, temp_output_dir, monkeypatch):
        """Sentence IDs should follow {firm_id}_{transcript_id}_{position:04d} format."""
        monkeypatch.setenv("LOCAL_MODE", "true")
        monkeypatch.setenv("FIRM_ID", "1001")
        monkeypatch.setenv("LOCAL_INPUT", sample_csv_for_integration)
        monkeypatch.setenv("LOCAL_OUTPUT", temp_output_dir)

        from cloud.containers.map.entrypoint import main
        main()

        output_file = Path(temp_output_dir) / "1001_topics.json"
        with open(output_file) as f:
            result = json.load(f)

        # Collect all sentence IDs
        all_ids = []
        for topic in result["topics"]:
            all_ids.extend(topic["sentence_ids"])
        all_ids.extend(result["outlier_sentence_ids"])

        # All IDs should start with firm_id
        for sid in all_ids:
            assert sid.startswith("1001_")

    def test_all_sentences_accounted_for(self, sample_csv_for_integration, temp_output_dir, monkeypatch):
        """All sentences should be in topics or outliers."""
        monkeypatch.setenv("LOCAL_MODE", "true")
        monkeypatch.setenv("FIRM_ID", "1001")
        monkeypatch.setenv("LOCAL_INPUT", sample_csv_for_integration)
        monkeypatch.setenv("LOCAL_OUTPUT", temp_output_dir)

        from cloud.containers.map.entrypoint import main
        main()

        output_file = Path(temp_output_dir) / "1001_topics.json"
        with open(output_file) as f:
            result = json.load(f)

        # Count sentences in topics
        topic_sentences = sum(len(t["sentence_ids"]) for t in result["topics"])
        outlier_sentences = len(result["outlier_sentence_ids"])

        # Should equal n_sentences_processed
        assert topic_sentences + outlier_sentences == result["metadata"]["n_sentences_processed"]
