# Sprint 2 Bootstrap

**Purpose**: Initialize a fresh Claude instance to continue the Financial Topic Modeling production pivot, starting Sprint 2 (WRDS Data Connector implementation).

---

## Mission Briefing

You are joining the Financial Topic Modeling project at a critical juncture. Sprint 1 (Specification & Design) is **complete**. All architectural decisions have been made, documented in ADRs, and validated by both the user and a parallel Codex reviewer instance. Your mission is to execute **Sprint 2: WRDS Data Connector** following test-driven development principles.

### What You're Building

A `WRDSConnector` class that:

1. Fetches earnings call transcripts directly from WRDS Capital IQ tables
2. Automatically links CRSP PERMNO via Compustat GVKEY at ingestion time
3. **Skips** firms without PERMNO linkage (they cannot be used in downstream event studies)
4. Implements the existing `DataConnector` interface
5. Returns data in the exact `TranscriptData` format expected by the pipeline

### Why This Matters

The production pivot transforms a 48-hour sequential processor into a distributed AWS Batch system capable of processing 8 quarters in parallel. The WRDS connector is the **first building block**—it replaces the static CSV ingestion with dynamic WRDS queries, enabling arbitrary date range processing and embedding the PERMNO identifiers needed for sentiment analysis.

---

## Required Reading (In This Order)

Before writing any code, you MUST read and internalize these documents:

### 1. Ground Truth Documents

| Document                          | Location                                                     | Why Read It                                                |
| --------------------------------- | ------------------------------------------------------------ | ---------------------------------------------------------- |
| **Instance Summary**              | `docs/packages/production_pivot/instance_summary_sprint1.md` | Understand what was done in Sprint 1 and why               |
| **Master Plan**                   | `docs/packages/production_pivot/serialized-wibbling-pike.md` | Full sprint breakdown, deliverables, validation criteria   |
| **Prior Architecture Discussion** | `docs/ai-log/claude-2026-batch-convo.md`                     | **ESSENTIAL** - Contains validated architectural decisions |

### 2. Sprint 2 Specifications

| Document                         | Location                               | Why Read It                                             |
| -------------------------------- | -------------------------------------- | ------------------------------------------------------- |
| **ADR-004: WRDS Data Source**    | `docs/adr/adr_004_wrds_data_source.md` | Design rationale, SQL queries, skip-unlinked decision   |
| **WRDSConnector Interface Spec** | `docs/specs/wrds_connector_spec.md`    | Complete interface specification with method signatures |

### 3. Implementation References

| Document                    | Location                                 | Why Read It                                    |
| --------------------------- | ---------------------------------------- | ---------------------------------------------- |
| **DataConnector Interface** | `cloud/src/interfaces.py`                | Abstract interface to implement                |
| **LocalCSVConnector**       | `cloud/src/connectors/local_csv.py`      | Reference implementation pattern               |
| **Data Models**             | `cloud/src/models.py`                    | TranscriptData, FirmTranscriptData dataclasses |
| **Existing Tests**          | `tests/unit/test_local_csv_connector.py` | Test patterns to follow                        |

---

## Guiding Principles

These are non-negotiable. Violating them will result in Codex reviewer rejection.

### 1. The Code Quality Standard

> "The best engineers I've worked with write code my mom could read. They choose boring technology, they over-document the 'why,' and under-engineer the 'how.' Complexity is not a flex; it becomes a liability."

- **Simplicity over cleverness**: If there are two ways to do something, choose the simpler one
- **Boring technology**: Use standard patterns, avoid exotic solutions
- **Over-document the "why"**: Every non-obvious decision needs a comment explaining rationale
- **Under-engineer the "how"**: Don't build abstractions for hypothetical future needs

### 2. Test-Driven Development (Mandatory)

```
1. Read the interface spec
2. Write tests that validate the spec
3. Implement the minimum code to pass tests
4. Refactor if needed (tests still pass)
5. Repeat
```

**No exceptions.** Tests are written BEFORE implementation. The Codex reviewer will reject PRs without test coverage.

### 3. Spec-Driven Development

The specifications in `docs/specs/wrds_connector_spec.md` are the contract. Do not deviate without explicit user approval. If something is ambiguous, ASK rather than assume.

### 4. Incremental Validation

Do not write 500 lines of code before testing. The pattern is:

1. Implement one method
2. Run tests
3. Fix issues
4. Move to next method

### 5. Ask Clarifying Questions

If something is unclear—whether a spec detail, an existing code pattern, or a user requirement—ASK. It is far better to pause and clarify than to build the wrong thing.

---

## Sprint 2 Deliverables

| File                                         | Description                                                 | Status |
| -------------------------------------------- | ----------------------------------------------------------- | ------ |
| `cloud/src/connectors/wrds_connector.py`     | WRDSConnector implementing DataConnector                    | TODO   |
| `cloud/src/models.py`                        | Add permno, gvkey, link_date to FirmTranscriptData.metadata | TODO   |
| `tests/unit/test_wrds_connector.py`          | Unit tests with mocked WRDS responses                       | TODO   |
| `tests/integration/test_wrds_integration.py` | Integration tests against real WRDS (10 firms)              | TODO   |

---

## Implementation Notes

### Key Design Decisions (Already Made)

1. **Skip unlinked firms**: Firms without PERMNO are NOT returned in the output. They are logged and skipped. This is intentional—see ADR-004.

2. **SQL Query**: The WRDS query in the spec joins `ciq.ciqtranscript` → `ciq.wrds_gvkey` → `crsp.ccmxpf_linktable` to get PERMNO. Validate this against the actual WRDS schema.

3. **Link quality filters**: Use `linktype IN ('LU', 'LC')` and `linkprim IN ('P', 'C')` per WRDS documentation.

4. **Date validation**: Ensure `earnings_call_date BETWEEN linkdt AND COALESCE(linkenddt, '9999-12-31')`.

5. **Connection management**: Support both passed-in connections and lazy initialization. Use context manager pattern.

6. **Interface alignment**: `DataConnector.get_available_firm_ids()` has **no date filters**. Do not add parameters; keep signature identical to `cloud/src/interfaces.py`.

### Testing Strategy

**Unit Tests** (mocked):

- Test `fetch_transcripts` returns correct structure
- Test unlinked firms are skipped (permno=NULL in mock → not in output)
- Test date filtering works
- Test error handling (connection failures, query errors)

**Integration Tests** (real WRDS):

- Mark with `@pytest.mark.integration` and `@pytest.mark.requires_wrds`
- Test with 10 known firms that have PERMNO
- Validate PERMNO appears in metadata
- Do NOT run in CI (requires credentials)

### Environment Variables

```bash
WRDS_USERNAME=xxx  # Or use ~/.pgpass
WRDS_PASSWORD=xxx
```

---

## Halting Point Protocol

When Sprint 2 implementation is complete:

1. **Self-check**: All tests pass, code follows patterns
2. **User review**: Present deliverables for approval
3. **Codex review**: User will share with Codex for impartial validation
4. **Proceed only after approval**: Do not start Sprint 3 until Sprint 2 is approved

---

## What NOT To Do

1. **Do not implement Sprint 3+ items** (AWS Batch, vLLM, Step Functions)
2. **Do not refactor existing code** unless necessary for WRDS connector
3. **Do not create new abstractions** beyond what the spec requires
4. **Do not skip tests** to save time
5. **Do not assume** if something is unclear—ask

---

## Getting Started

1. Read all Required Reading documents (in order)
2. Explore the existing codebase:
   - `cloud/src/interfaces.py` - DataConnector interface
   - `cloud/src/connectors/local_csv.py` - Reference implementation
   - `cloud/src/models.py` - Data structures
   - `tests/unit/test_local_csv_connector.py` - Test patterns
3. Write unit tests for `WRDSConnector` based on the interface spec
4. Implement `WRDSConnector` to pass tests
5. Write integration test
6. Present for review

---

## Codex Reviewer Expectations

The Codex reviewer will check:

1. **Alignment**: Does implementation match `wrds_connector_spec.md`?
2. **Quality**: Is code readable, documented, following patterns?
3. **Tests**: Are there unit tests? Do they cover edge cases?
4. **Complexity**: Is there unnecessary complexity? Over-engineering?
5. **Risk**: What could go wrong? Are edge cases handled?

Expect to be challenged on every decision. Be prepared to justify your choices.

---

## Final Reminder

This is a research project that will be reviewed by faculty and potentially published. The code quality standard is high. Take pride in building something clean, understandable, and well-tested.

> "Real seniority is making hard problems look simple, not making simple problems look hard."

The WRDS connector is a hard problem (data linking, SQL joins, connection management). Make it look simple.

<Note from User>
Leverage context-synthesizer, api-docs-synthesizer and plan subagents to methodologically manage your context by offloading exploration-heavy tasks as you see fit!!
</Note from User>
