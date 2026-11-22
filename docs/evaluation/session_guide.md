# Session guide (researcher script)

Before the session: `cd examples/ci-demo && python make_demo.py`, then
`shortcutcheck run models/shortcut.joblib data/apps.csv --train data/train.csv --report shortcut.json`
and the same for `robust`. Record `git describe --tags` in the session log.

## 1. Consent (5 min)
Go through the information sheet; answer questions; collect the signed consent form. Start any
recording only if box 6 is initialled.

## 2. Background (2 min)
Questionnaire part A.

## 3. Introduction (5 min, read aloud)
"Malware detectors can learn properties such as build year instead of behaviour. ShortcutCheck
changes only an app's metadata and counts how often the detector's decision changes. A high rate
means the decision depends on metadata. It needs no labels for the apps it tests, and it can run
as a test in CI. Today we are evaluating the tool, not you; please think aloud."

## 4. Task 1: the gate (8 min)
Ask the participant to run, from `examples/ci-demo`:

```
shortcutcheck run models/robust.joblib data/apps.csv --train data/train.csv
shortcutcheck run models/shortcut.joblib data/apps.csv --train data/train.csv
```

Question: "What does each result tell you? Would you release each model?"
**Answer key (correct if the participant says):** the robust model passes (no prediction changes
when metadata change); the shortcut-prone model fails because its decisions change with build
year (MR1 and MR4 above the threshold), so it should not be released as is. Note: correct /
partly correct / incorrect.

## 5. Task 2: the dashboard (8 min)
`shortcutcheck dashboard shortcut.json`
Questions: "Which metadata field does this model rely on?" (answer key: `year`). "Pick one app
that changed prediction: what changed?" (answer key: its year, possibly with size and SDK
fields, and its malware probability crossed 0.5). Note correct / partly / incorrect.

## 6. Task 3: CI (4 min)
Show `examples/ci-demo/github-workflow-example.yml` and the demo workflow run where the
shortcut-prone model failed CI. Question: "Would you add this to your pipeline? What would you
change?"

## 7. Questionnaire (8 min)
Parts B and C. Thank the participant; remind them of the withdrawal date.

## After the session
Enter the answers in `responses.csv` (copy of `responses_template.csv`) under the participant
code; file the consent form separately; delete any recording within two weeks.
