# Changelog

From v1.0 on, results change only to fix bugs, and every fix is listed here with the numbers it
changes (roadmap 4.7). Dissertation and paper numbers come from the v1.0 tag.

## v1.0.0 (frozen; tag created after the Phase 4 merge)

### Tool (Phase 4)
- `shortcutcheck run`: release gate; exit code 0 pass, 1 a relation exceeds the threshold, 2 bad
  input; JSON report with rates, Wilson intervals, flipped apps and MR4 per metadata field.
- pytest plugin (`@pytest.mark.shortcutcheck`, `shortcutcheck_report` and `shortcutcheck`
  fixtures, `--shortcutcheck-reports`), CI demo and workflow template.
- `shortcutcheck dashboard`: Streamlit view of a gate report.
- `shortcutcheck expert-eval`: questionnaire analysis for the expert evaluation.
- Built-in Dada defaults, so the gate runs in projects without this repository's config.

### Research (Phases 1-3 and addenda)
- Pre-registered H1-H3 (`PREREGISTRATION.md`) and addenda A-F, each committed before the data it
  tests were opened; every test-data read logged in `logs/test_access.jsonl`.
- Results: H1 partially supported; H2 reversed (replicated under Dada's protocol, addendum B);
  H3 no signal; fix proof of concept GO (C); H4 supported, H5a-c hold (D); AZL replication partial
  (E); proposed mechanism not supported (F).

### Quality
- CI on Linux and Windows with lint, strict type checking and tests; coverage floor 80% enforced.

## Fixes after v1.0

None yet.
