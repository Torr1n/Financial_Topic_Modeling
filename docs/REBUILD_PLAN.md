# Rebuild Plan

## Current Status (as of 2026-02-09)

The core pipeline and downstream analysis are built, tested, and validated on AWS (11 firms, ~350 topics, ~19 themes, ~15 min, ~$1.30). What follows are the remaining work items to bring the project to a complete, presentable state.

---

## Work Items

### 1. S3 Connector

**What:** Implement `cloud/src/connectors/s3_connector.py` so the pipeline can ingest transcripts directly from S3 instead of requiring a local CSV on the instance.

**Why:** Right now someone has to manually place the CSV on the EC2 instance. This is the biggest functional gap in making the pipeline truly cloud-native.

**Approach:**
- Follow the `LocalCSVConnector` pattern in `cloud/src/connectors/local_csv.py`
- Use boto3 to stream/download CSV from S3
- Implement the `DataConnector` interface from `cloud/src/interfaces.py`
- Add `s3` as a valid data source option in `production.yaml`
- Unit tests with mocked boto3 (moto)

**Files:**
- `cloud/src/connectors/s3_connector.py` (existing stub, needs implementation)
- `cloud/config/production.yaml` (add S3 config section)
- `tests/unit/test_connectors.py` (add S3 tests)

**Status:** Not started

---

### 2. Test Runs

**What:** Run the pipeline end-to-end multiple times on real data. Evaluate output quality and tune hyperparameters if needed.

**Why:** The pipeline has only been validated on 11 MAG7 firms. Need to verify it generalizes, check topic/theme quality, and confirm nothing breaks at larger scale.

**Approach:**
- Run with current production.yaml settings on available transcript data
- Review output: Are themes coherent? Topic counts reasonable? Any firms failing?
- Tune UMAP/HDBSCAN hyperparameters in production.yaml if output quality is off
- Document findings (what worked, what was changed, final settings)

**Dependencies:** S3 connector (if data lives in S3) or manual data placement on instance

**Status:** Not started — waiting on data prep clarification from owner

---

### 3. Matplotlib Figures

**What:** Create matplotlib figures that tell the full story from topic discovery through financial impact.

**Why:** Currently only ONE actual matplotlib figure exists (portfolio cumulative returns in `downstream/src/visualization.py`). Everything else is text tables and CSVs. Need a complete set of figures for presenting research results.

**Existing (polish):**
- Portfolio cumulative returns by sentiment tercile (line chart, already in visualization.py)

**New — Cloud Pipeline Output (topics/themes):**

| Figure | Type | Data Source | Story It Tells |
|--------|------|-------------|----------------|
| Theme overview | Horizontal bar chart | themes table (n_topics, n_firms) | "Here are the cross-firm themes we discovered" |
| Firm-theme heatmap | Heatmap/matrix | topics table (firm_id × theme_id) | "Which firms discuss which themes" — the core cross-firm narrative |
| Topic count distribution | Histogram | topics table grouped by firm | "Pipeline produces consistent topic counts across firms" |

**New — Downstream Output (sentiment/event study):**

| Figure | Type | Data Source | Story It Tells |
|--------|------|-------------|----------------|
| Sentiment by theme | Box/violin plot | sentiment scores per theme | "Some themes are discussed positively, others negatively" |
| CAR event study | Line plot with CI bands | CAR values around event window [-10, +10] | "Abnormal returns around earnings calls by sentiment" |
| Regression forest plot | Horizontal dot plot with CI | sentiment coefficients per theme | "Which themes have statistically significant sentiment-return relationships" |

**Approach:**
- Cloud pipeline figures: new module `cloud/src/visualization.py` (queries DB directly)
- Downstream figures: extend `downstream/src/visualization.py`
- Consistent styling across all figures (shared color palette, font sizes, figsize conventions)
- All figures saved as 300 DPI PNGs

**Files:**
- `cloud/src/visualization.py` (new — theme overview, heatmap, topic distribution)
- `downstream/src/visualization.py` (extend — sentiment boxes, CAR plot, forest plot; polish existing portfolio chart)

**Status:** Not started

---

### 4. Data Prep (TBD)

**What:** Get data into the format/location the owner needs.

**Why:** Owner mentioned needing data ready in some form. Could mean:
- Exporting pipeline output for downstream consumption (the `export_for_downstream.py` bridge)
- Organizing raw transcript CSVs in S3
- Formatting final results for a paper/presentation

**Approach:** TBD after owner clarifies what "data ready" means.

**Status:** Not started — blocked on owner input

---

## Suggested Order

```
S3 Connector  →  Data Prep (if needed)  →  Test Runs  →  Figures
     1                  2                       3            4
```

S3 connector unblocks cloud-native test runs. Test runs produce real output. Figures visualize that real output. Data prep slots in wherever the owner needs it.

## Open Questions for Owner

1. What does "get data ready" mean specifically? Export format? S3 organization? Something else?
2. How many firms / what time range for the test runs?
3. Any specific hyperparameter concerns from previous runs?
