# Phase 3 Review Package: Reduce Container

**Date:** 2024-12-01
**Phase:** 3 - Reduce Container
**Status:** Implementation Complete, Pending Approval

---

## Executive Summary

Phase 3 implements the **reduce phase** of the map-reduce pipeline, aggregating firm-level topics into cross-firm themes using **Dual-BERTopic**. The implementation follows the approved plan precisely and reuses existing patterns from Phase 2.

**Key Deliverables:**
- `ThemeAggregator` class with validation filters
- Reduce container (Dockerfile, requirements, entrypoint)
- Comprehensive unit tests (23 tests)
- Integration tests with real MAG7 data (15 tests)
- Local test script

---

## Architecture: Dual-BERTopic Approach

```
┌─────────────────────────────────────────────────────────────────┐
│                        REDUCE PHASE                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  FirmTopicOutput JSONs     ThemeAggregator      ThemeOutput     │
│  ┌─────────────────┐      ┌──────────────┐     ┌─────────────┐  │
│  │ Firm A Topics   │      │              │     │ Theme 001   │  │
│  │ - representation├──┐   │ 1. Extract   │     │ - name      │  │
│  │ - keywords      │  │   │    topic     │     │ - keywords  │  │
│  │ - size          │  │   │    reps as   │     │ - n_firms   │  │
│  └─────────────────┘  │   │    "docs"    │     │ - topics[]  │  │
│  ┌─────────────────┐  │   │              │     └─────────────┘  │
│  │ Firm B Topics   │  ├──►│ 2. Run       │────►┌─────────────┐  │
│  │ - representation│  │   │    BERTopic  │     │ Theme 002   │  │
│  │ - keywords      │  │   │    (re-embed)│     │ - name      │  │
│  │ - size          │  │   │              │     │ - keywords  │  │
│  └─────────────────┘  │   │ 3. Validate  │     │ - n_firms   │  │
│  ┌─────────────────┐  │   │    themes    │     │ - topics[]  │  │
│  │ Firm C Topics   ├──┘   │              │     └─────────────┘  │
│  │ ...             │      └──────────────┘                      │
│  └─────────────────┘                                            │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

**Key Insight:** Topic representations (strings) become "documents" for re-embedding. The same `TopicModel` interface and `BERTopicModel` class work for both firm-level and theme-level clustering.

---

## Files Created

| File | Lines | Purpose |
|------|-------|---------|
| `cloud/src/theme_aggregator.py` | 207 | ThemeAggregator class |
| `cloud/containers/reduce/Dockerfile` | 31 | Container definition |
| `cloud/containers/reduce/requirements.txt` | 18 | Dependencies |
| `cloud/containers/reduce/entrypoint.py` | 255 | Entry point |
| `cloud/scripts/local_test_reduce.sh` | 75 | Local test script |
| `tests/unit/test_theme_aggregator.py` | 550 | Unit tests (23 tests) |
| `tests/integration/test_reduce_entrypoint.py` | 312 | Integration tests (15 tests) |

## Files Modified

| File | Change |
|------|--------|
| `tests/conftest.py` | Added `sample_firm_topic_outputs` and `validate_theme_output` fixtures |

---

## ThemeAggregator Implementation

### Class Structure

```python
class ThemeAggregator:
    def __init__(self, topic_model: TopicModel, config: Dict[str, Any]):
        self.model = topic_model
        self.config = config
        self.min_firms = config.get('validation', {}).get('min_firms', 2)
        self.max_dominance = config.get('validation', {}).get('max_firm_dominance', 0.4)

    def aggregate(self, firm_results: List[Dict]) -> List[Dict]:
        """Main entry point - returns list of ThemeOutput dicts."""

    def _extract_topic_documents(self, firm_results) -> Tuple[List[str], List[Dict]]:
        """Extract topic representations as 'documents' for re-embedding."""

    def _group_into_themes(self, theme_result, topic_metadata) -> List[Dict]:
        """Group topics by theme assignment from BERTopic."""

    def _validate_themes(self, themes) -> List[Dict]:
        """Apply min_firms and max_dominance filters."""
```

### Validation Filters

1. **min_firms (default=2):** Theme must have topics from at least 2 distinct firms
2. **max_firm_dominance (default=0.4):** No single firm can have >40% of topics in a theme

### Output Schema (ThemeOutput)

```python
{
    "theme_id": "theme_20241201_001",
    "name": "AI Investment Strategy",  # From BERTopic representation
    "keywords": ["ai", "machine", "learning", ...],
    "n_firms": 5,
    "n_topics": 12,
    "topics": [
        {
            "firm_id": "1001",
            "topic_id": 0,
            "representation": "ai investment strategy",
            "size": 25
        },
        ...
    ],
    "metadata": {
        "processing_timestamp": "2024-12-01T12:00:00Z",
        "model_config": {...},
        "validation": {
            "min_firms": 2,
            "max_firm_dominance": 0.4
        }
    }
}
```

---

## Reduce Container Entrypoint

### Environment Variables

| Variable | Mode | Required | Description |
|----------|------|----------|-------------|
| `LOCAL_MODE` | Both | No | Set to "true" for local testing |
| `LOCAL_INPUT` | Local | Yes | Directory with firm topic JSONs |
| `LOCAL_OUTPUT` | Local | Yes | Path for output themes JSON |
| `S3_INPUT_BUCKET` | Cloud | Yes | Bucket with firm topic JSONs |
| `S3_INPUT_PREFIX` | Cloud | No | Key prefix (default: "firm-topics/") |
| `DYNAMODB_TABLE` | Cloud | Yes | Table name for theme writes |
| `MIN_FIRMS` | Both | No | Minimum firms to proceed (default: 1) |
| `CONFIG_PATH` | Both | No | Path to config YAML |
| `LOG_LEVEL` | Both | No | Logging level (default: INFO) |

### Processing Flow

1. Parse environment variables
2. Load firm topic JSONs (local or S3)
3. Skip corrupt/invalid files (log warning, continue)
4. Filter out firms with n_topics=0
5. Check MIN_FIRMS requirement
6. Initialize BERTopicModel and ThemeAggregator
7. Run aggregation
8. Generate theme_ids (`theme_YYYYMMDD_NNN`)
9. Save output (local JSON or DynamoDB)
10. Log summary

### Error Handling

- **Corrupt JSON:** Skip file, log warning, continue
- **Missing required fields:** Skip file, log warning, continue
- **Insufficient firms:** Exit with code 1
- **Missing env vars:** Exit with code 1

---

## Test Coverage

### Unit Tests (23 tests)

| Test Class | Tests | Coverage |
|------------|-------|----------|
| `TestThemeAggregatorInit` | 4 | Dependency injection, config extraction |
| `TestThemeAggregatorAggregate` | 3 | Core aggregation logic |
| `TestThemeAggregatorValidation` | 4 | min_firms, max_dominance, outliers |
| `TestThemeAggregatorEdgeCases` | 4 | Empty input, all outliers, single firm |
| `TestThemeOutputSchema` | 5 | Schema compliance |
| `TestThemeOutputJsonSerializable` | 1 | JSON serialization |
| `TestThemeIdGeneration` | 2 | ID format, sorting |

### Integration Tests (15 tests)

| Test Class | Tests | Coverage |
|------------|-------|----------|
| `TestReduceEntrypointLocalMode` | 4 | End-to-end local mode |
| `TestReduceEntrypointErrorHandling` | 4 | Error conditions |
| `TestReduceEntrypointMinFirms` | 2 | MIN_FIRMS validation |
| `TestReduceEntrypointThemeIds` | 2 | Theme ID generation |
| `TestReduceEntrypointRealOutput` | 3 | Real data validation |

### Performance Optimization

Integration tests use a **module-scoped fixture** that runs BERTopic once and caches results. This reduces runtime from **15+ minutes to ~2-3 minutes**.

---

## Design Decisions

### 1. Dual-BERTopic (Not Centroids)

**Decision:** Re-embed topic representations as new documents instead of carrying embeddings forward.

**Rationale:**
- Cleaner architecture (no 768-dim centroid storage)
- Model-agnostic (works with any embedding model)
- Same interface for firm and theme clustering

### 2. Module-Scoped Test Fixture

**Decision:** Run reduce phase once and cache results for all integration tests.

**Rationale:**
- BERTopic is slow (~2 min per run)
- Running per-test would take 15+ minutes
- Cached results still validate all schema/logic requirements

### 3. Real Data for Integration Tests

**Decision:** Use actual MAG7 firm outputs instead of synthetic data.

**Rationale:**
- Synthetic data caused `min_df` errors in CountVectorizer
- Real data provides more meaningful validation
- Tests are more representative of production behavior

### 4. Validation Filter Thresholds

**Decision:** `min_firms=2`, `max_firm_dominance=0.4`

**Rationale:**
- Cross-firm themes need at least 2 firms to be meaningful
- 40% dominance threshold prevents single-firm-dominated themes
- Values from approved plan, configurable via config.yaml

---

## Verification Commands

### Run Unit Tests
```bash
python -m pytest tests/unit/test_theme_aggregator.py -v
```

### Run Integration Tests
```bash
python -m pytest tests/integration/test_reduce_entrypoint.py -v
```

### Run Full Test Suite with Coverage
```bash
python -m pytest tests/unit/ tests/integration/ -v --cov=cloud/src --cov-report=term-missing
```

### Run Local End-to-End Test
```bash
./cloud/scripts/local_test_reduce.sh
```

### Verify Output
```bash
# Check themes were created
cat output/reduce_test/themes.json | python -m json.tool | head -50

# Count themes
cat output/reduce_test/themes.json | python -c "import json,sys; d=json.load(sys.stdin); print(f'Themes: {len(d)}')"
```

---

## Known Limitations

1. **S3TranscriptConnector:** Still a stub - not needed for Phase 3 but required for full cloud deployment.

2. **Integration Test Runtime:** ~2-3 minutes due to BERTopic. Acceptable but noted.

3. **Validation Strictness:** With `min_firms=2` and `max_dominance=0.4`, many themes may be filtered. This is intentional but may need tuning.

4. **Single-Threaded:** Reduce phase runs single-threaded. Acceptable for current scale.

---

## Files for Review

### Critical (Must Review)
- `cloud/src/theme_aggregator.py` - Core logic
- `cloud/containers/reduce/entrypoint.py` - Entry point
- `tests/unit/test_theme_aggregator.py` - Unit tests

### Reference
- `cloud/containers/reduce/Dockerfile` - Container definition
- `tests/integration/test_reduce_entrypoint.py` - Integration tests
- `tests/conftest.py` - Shared fixtures

---

## Checklist for Approval

- [ ] ThemeAggregator correctly clusters topics into themes
- [ ] Validation filters work (min_firms, max_firm_dominance)
- [ ] Reduce entrypoint works in local mode
- [ ] Themes written to DynamoDB (mocked) with correct schema
- [ ] All unit tests pass (23/23)
- [ ] All integration tests pass (15/15)
- [ ] Coverage ≥80%
- [ ] Can process real map output and produce valid themes

---

## Next Steps (After Approval)

**Phase 4: Terraform Infrastructure**
- Modular Terraform structure
- S3 bucket for transcripts and firm topics
- DynamoDB table with composite keys
- AWS Batch job definitions
- IAM roles and policies

**HALT:** Await approval before proceeding to Phase 4.

---

## Appendix: Sample Theme Output

From real MAG7 data run:

```json
{
  "theme_id": "theme_20241201_000",
  "name": "revenue growth, growth revenue, quarter revenue",
  "keywords": ["revenue", "growth", "quarter", "year", "percent"],
  "n_firms": 7,
  "n_topics": 18,
  "topics": [
    {"firm_id": "21835", "topic_id": 1, "representation": "revenue growth...", "size": 23},
    {"firm_id": "24937", "topic_id": 2, "representation": "quarterly revenue...", "size": 19},
    ...
  ],
  "metadata": {
    "processing_timestamp": "2024-12-01T18:30:00Z",
    "validation": {"min_firms": 2, "max_firm_dominance": 0.4}
  }
}
```

---

*Generated by Claude Code - Phase 3 Implementation*
