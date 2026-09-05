# Decision log

## Active core goal (supersedes historical proposals D-001–D-013)

Authoritative settings and preregistered gates: [EVALUATION_PROTOCOL.md](EVALUATION_PROTOCOL.md)
and `configs/core_protocol.json`. Historical entries below are not current commands.

- D-015 **Frozen choice**: literal user target `Qwen/Qwen3-4B` at
  `1cfa9a7208912126459214e8b04321603b3df60c`, not Base. Exact-revision card is
  Apache-2.0; no new access terms accepted. No attempt to match Table1.
- D-016 **Frozen choice**: custom causal likelihood scorer, batch1/zero-shot;
  WT2 prefix-conditioned token PPL on64 windows; HellaSwag/ARC-C256 seeded examples
  each; all metrics retained, primary meanNLL/acc_norm declared before new scores.
- D-017 **Frozen choice**: grouped128 signed int8 RTN with top-bit midpoint
  reconstruction at4/6/8; fixed quantizers and future adaptive share this family.
  No quantizer search for paper matching. All new results under `results/core-v1`.
- D-018 **Scope**: no loading mechanism, 8B models, PTB, full table, dynamic
  batching, async loading or custom performance kernels. Older proposals deferred.
- D-019 **Gate**: two complete fresh runs per fixed mode; exact metrics/gates
  in protocol. No router work until stable and internally sensible.
- D-020 **Planned, not trained**: causal context-only fixed8 feature prepass,
  train-only local distillation-error MLP and equal per-type precision quotas;
  exact training choices must be locked after integration before fitting.
- D-021 **Safety repair**: preflight now rejects any selected compute process,
  nonzero utilization, recheck change or failed process query, not just low memory.
  No other user's process is signaled. Mock safety tests cover refusal paths.

## Historical proposal log

Statuses:

- **Reported**: stated in the paper or an official source.
- **Measured**: produced by this project and linked to raw output.
- **Inferred**: supported by evidence but not directly stated.
- **Proposed**: a choice for the next experiment.
- **Open**: no defensible choice yet.

| ID | Status | Question | Current decision | What would make us revise it |
| --- | --- | --- | --- | --- |
| D-001 | Proposed | Which model first? | `Qwen/Qwen3-4B-Base` only | Author model IDs, a baseline mismatch, or unavailable access |
| D-002 | Proposed | Which language? | Python with PyTorch for the functional version | Author code, unsupported kernels, or evidence another stack is required |
| D-003 | Inferred | Base or instruction model? | Start with base checkpoints because the paper uses perplexity and benchmark-style base-model scores | Exact author checkpoint IDs or baseline results favoring instruction models |
| D-004 | Proposed | Qwen3-4B revision? | `906bfd4`, the latest visible commit before submission in the official history | Author revision, content hash mismatch, or a controlled revision comparison |
| D-005 | Proposed | Evaluation harness revision? | EleutherAI `lm-eval` commit `d021bf8` (v0.4.9.1), released before submission | Author command, task incompatibility, or comparison showing another revision matches coherently |
| D-006 | Proposed | Number of examples and shots? | Full available splits and zero-shot as the first candidate | Paper supplement, author response, or score pattern incompatible with zero-shot |
| D-007 | Proposed | Which metric? | Record every returned metric; predeclare common metrics before comparison | Never revise after viewing a match without documenting a new independent reason |
| D-008 | Open | Static quantizer? | Compare plausible weight-only and library quantizers; choose no winner yet | A method matches accuracy, memory, and latency together or authors identify it |
| D-009 | Proposed | Bit-plane signed format? | Test sign-magnitude first; also test affine unsigned and two's complement | Exact author rule or clear numerical/system advantage under the same static weights |
| D-010 | Proposed | Candidate precisions? | 4, 6, and 8 bits because the figure shows low/mid/high and the table includes 4/8 baselines | Author values or routing experiments showing the middle value is wrong |
| D-011 | Open | Router training data? | Use no benchmark evaluation examples; select a public calibration corpus only after license and size review | Author data or evidence of domain-specific routing requirements |
| D-012 | Proposed | Router objective? | Teacher-student agreement plus an explicit expected-bit cost | Author loss or a different loss that demonstrably avoids all-8-bit collapse without hidden tuning |
| D-013 | Reported | Loader style to match first? | Synchronous on-demand transfer | The paper explicitly attributes latency to sequential transfer; an asynchronous version is a separate extension |
| D-014 | Measured by user | What shared-server guard is required? | Activate `~/.venv`, select one GPU with at least 20,000 MiB free, recheck immediately before launch, and never interrupt other processes | A lab administrator supplies a different reservation policy or measured model peak requires a predeclared threshold change |

## Experiment entry template

Copy this section for each material run.

```text
Date/time:
Decision or claim tested:
Source of the idea: paper / measured result / inference / proposed next step
Command:
Environment and hardware record:
Inputs and immutable revisions:
Expected outcome and acceptance rule, written before the run:
Observed raw artifact paths:
What is now known:
What remains unknown:
Continue, pause, revise, or stop:
Reason:
Next smallest distinguishing test:
```
