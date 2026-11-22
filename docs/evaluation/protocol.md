# Expert evaluation protocol (roadmap 4.6)

**Status: draft for ethics review. No session may take place before the protocol and the consent
form are approved (roadmap 4.6 Check).**

## Aim

To find out whether practitioners find ShortcutCheck useful, understandable and suitable for a
continuous-integration (CI) workflow, and to collect their suggestions. This evaluates the tool;
it does not test the participants.

## Research questions

- **EQ1 Usefulness:** do practitioners consider the gate and the dashboard useful for deciding
  whether a malware detector is safe to release?
- **EQ2 Understandability:** can they interpret the report (violation rate, interval, per-field
  breakdown, flipped apps) correctly without help?
- **EQ3 CI fit:** would they add the gate to a CI pipeline, and what would stop them?

Success criterion (NFR8): median usefulness rating of at least 4 on the 5-point scale.

## Participants

- 5 to 8 practitioners with at least one year of experience in machine learning engineering,
  data science, or security (malware analysis, security operations, application security).
- Recruited through the researcher's and supervisor's professional networks and the university's
  industry contacts, by an invitation e-mail with the information sheet attached.
- Exclusions: under 18; current students in the researcher's own cohort; anyone the researcher
  manages or is managed by.
- No payment. Participation is voluntary and can be withdrawn (see the information sheet).

## Materials

- ShortcutCheck installed on the researcher's laptop (version tag recorded in the session log).
- The CI demo (`examples/ci-demo`): synthetic data and two detectors, one robust and one
  shortcut-prone. **No real malware, no application packages, no personal data.**
- The dashboard opened on the two demo reports.
- `docs/evaluation/session_guide.md` (script), `questionnaire.md`, `responses_template.csv`.

## Procedure (about 40 minutes, online or in person)

1. Welcome, information sheet, questions, consent (5 min). Nothing is recorded before consent.
2. Background questions (2 min): role, years of experience, familiarity with CI and with
   malware detection (no names in the response file; a participant code such as P03 is used).
3. Introduction to ShortcutCheck (5 min, scripted).
4. Task 1 (8 min): run the gate on the robust model and on the shortcut-prone model; explain in
   their own words what each result means. The researcher notes whether the interpretation is
   correct against the answer key in the session guide.
5. Task 2 (8 min): open the shortcut-prone model's report in the dashboard; identify which
   metadata field the model relies on and inspect one flipped app.
6. Task 3 (4 min): read the CI workflow template; say whether and how they would use it.
7. Questionnaire (8 min): Likert items and open questions.

Think-aloud comments are written down by the researcher. Audio or screen recording only if the
participant ticks the optional box on the consent form; recordings are deleted after notes are
checked, within two weeks.

## Data and analysis

- Responses are stored as `docs/evaluation/responses.csv` with participant codes only; consent
  forms are stored separately from responses, in the university's secure storage, and are never
  committed to the repository.
- `scripts/expert_eval_analysis.py` computes medians and interquartile ranges per construct,
  the share of correct interpretations per task, and the NFR8 result. Open answers are coded
  thematically (two passes, the second blind to participant code).
- Results are reported in aggregate only; no quotation is attributed to an identifiable person.
- Data are kept until the degree result is confirmed and then deleted, unless the participant
  agreed to their anonymous responses being kept for publication.

## Risks

Minimal: the tasks use synthetic data and a laptop the researcher controls. The main risk is a
participant feeling evaluated; the briefing states that the tool, not the participant, is being
evaluated, and that any task can be skipped.

## Researcher and supervisor

- Researcher: Nisula Dasanayake, Informatics Institute of Technology (University of Westminster
  degree), [researcher e-mail].
- Supervisor: [supervisor name, e-mail].
