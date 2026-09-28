# Experiment evidence contracts

The experiment workflow contracts are pure evidence builders and validators.
They do not start or stop work, call model providers, allocate resources, route
requests, admit models, approve adoption, or control releases.

## Run identity and reconciliation — #30

`nexus_learning.experiment_run_identity` binds one logical experiment
generation to its subject commit/tree, harness, cohort, frozen policy, requested
and configured execution identity, producer/effect identity, observed state,
timestamps, optional #27 evidence-origin provenance, and terminal result receipt.
Missing optional observations stay explicit as `None`.

`classify_run_observation` reports the next observation/reconciliation need.
`RUNNING` requires observing the same run, while `OUTCOME_UNKNOWN` requires
reconciliation with the owning effect plane. Neither state authorizes retry or a
new effect. A source/harness/cohort/policy binding change under the same numeric
generation is reported as an identity mismatch; the caller must create and
record a new generation explicitly.

## Cohort preflight and staged stop gate — #31

`nexus_learning.experiment_preflight` binds normalized calibration/holdout
cases, source-group independence, label distribution and constant-answer
baseline, ground-truth provenance, leakage observations, incumbent identity,
hard gates, comparator compatibility, and resource/headroom observations.
Missing evidence becomes `DEFER`; duplicate identities, forbidden overlap,
known leakage, and frozen-gate resource failures become `FAIL`.

Comparator observations cover sampling fields, reasoning mode, context/prefill,
cache semantics, concurrency/admission, and material runtime settings. Frozen
compatibility amendments are required when a requested field is rejected.
Pairwise comparator validation rejects unobserved or core-method differences
before receipts are combined. Explicitly observed cache/runtime differences cap
the comparison scope at
`WHOLE_RUNTIME_STACK_ONLY`; the workflow may evaluate that scope only when all
non-stack dimensions are explicitly matched and every dimension was observed.
This scope does not claim isolated cache, runtime, or scheduler parity.

Resource receipts retain each measured concurrency level, headroom observation,
admission state, errors, and recovery. A pre-registered stop rule records higher
levels as skipped after a resource failure. Missing telemetry is explicit and
defers readiness unless resource evidence is declared inapplicable with a
reason.

`build_staged_gate_evidence` advances only through preflight, smoke, calibration,
and then holdout eligibility. A pre-registered hard-gate failure yields a
terminal `STOP` before holdout; later evidence cannot rewrite that result.
`DEFER` remains resumable: incomplete preflight may be superseded only by
complete evidence with the same experiment, cohort, frozen policy, and gates;
an incomplete stage observation may be superseded by a new bound observation,
while the earlier deferred receipt remains in history. Deferred evidence cannot
close a campaign.
Necessary-condition negative-stop evidence is validated through #26. Quality
gates remain upstream of the existing #22 economics comparison.

## Campaign closeout and reopen evidence — #32

`nexus_learning.campaign_closeout` projects completed #30 run receipts and #31
preflight/stage bindings into a campaign summary. Missing receipts or
nonterminal runs/gates produce an explicit `INCOMPLETE` closeout. Completed
negative dispositions remain terminal records, and `next_gate="NONE_DEFINED"`
is valid.

Each experiment records its highest safe claim and `not_proven` list. Missing
formal correctness certification, unopened holdout, mismatched comparator
configuration, and resource limits observed only through a lower concurrency
are carried into `not_proven` automatically where applicable.

Reopen evaluation requires a matching, named material-delta trigger or an exact
owner-decision reference and hash. It requires a higher experiment generation
and returns the previous closeout and receipt hashes so the historical record
remains unchanged. Material-delta evidence hashes bind the canonical `changes`
mapping, and dot-separated trigger paths traverse nested mappings. It never
schedules or starts the new experiment.
