# Questionnaire

Participant code: ______ (the researcher fills this in)

## A. Background

1. Current role: ML engineer / data scientist / security analyst / application security /
   researcher / other: ______
2. Years of professional experience in that area: <1 / 1–3 / 4–7 / 8+
3. How often do you work with CI pipelines? never / sometimes / weekly / daily
4. Familiarity with malware detection: none / some / experienced

## B. Ratings

1 = strongly disagree, 2 = disagree, 3 = neutral, 4 = agree, 5 = strongly agree.

| Code | Statement | 1 | 2 | 3 | 4 | 5 |
| --- | --- | --- | --- | --- | --- | --- |
| U1 | ShortcutCheck would help me decide whether a detector is safe to release. | | | | | |
| U2 | The per-field breakdown tells me something I would not easily find out otherwise. | | | | | |
| U3 | I would use ShortcutCheck on detectors I work with. | | | | | |
| C1 | I understood what a violation rate means. | | | | | |
| C2 | I understood why the shortcut-prone model failed. | | | | | |
| C3 | The dashboard made it clear which apps changed prediction and why. | | | | | |
| I1 | The gate would fit into our CI pipeline with little effort. | | | | | |
| I2 | A failing gate would be a reasonable reason to block a merge. | | | | | |
| I3 | The run time is acceptable for a CI check. | | | | | |

Constructs: usefulness (U1–U3), understandability (C1–C3), CI fit (I1–I3).

## C. Open questions

1. What did you find most useful?
2. What was confusing or missing?
3. What would stop you from adding the gate to a pipeline?
4. Which threshold would you choose, and why?
5. Any other comments?
