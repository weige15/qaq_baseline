# QAQ reproduction claim report

## Scope and status

- Date:
- Model(s) attempted:
- Overall status: exact / functional / approximate / blocked
- Why this label is justified:

## Environment

- Model IDs and immutable revisions:
- Dataset IDs and immutable revisions:
- Evaluation code revision:
- Python, PyTorch, Transformers, CUDA, and driver versions:
- GPU, CPU, RAM, storage, and interconnect:
- Random seeds:

## Claim matrix

| Paper claim | Paper evidence | Our method | Raw artifact | Result | Status | Remaining uncertainty |
| --- | --- | --- | --- | --- | --- | --- |
| Static FP16 quality | Table 1 |  |  |  |  |  |
| Static 8-bit quality | Table 1 |  |  |  |  |  |
| Static 4-bit quality | Table 1 |  |  |  |  |  |
| QAQ matches static 8-bit quality | Table 1 |  |  |  |  |  |
| Router adapts to queries | Method text |  |  |  |  |  |
| Bit-plane representation supports selected precisions | Equations 1 and 4 |  |  |  |  |  |
| On-demand mode lowers GPU memory | Table 1 |  |  |  |  |  |
| On-demand mode adds latency | Table 1 |  |  |  |  |  |

## Metrics and predeclared rules

- HellaSwag metric(s):
- PIQA metric(s):
- ARC metric(s):
- Perplexity text preparation and formula:
- Accuracy tolerance:
- Perplexity tolerance:
- Memory comparison rule:
- Latency comparison rule:
- Routing non-collapse rule:

## Routing evidence

- Candidate precisions:
- Mean selected bits:
- Distribution by block:
- Distribution by task/query group:
- Fraction selecting only the maximum precision:
- Precision-cost setting and how it was selected:

## System evidence

- Resident GPU state by mode:
- CPU state and pinned buffers:
- Bytes transferred per query:
- Cache capacity and eviction rule:
- Warm-up and measurement repeats:
- Allocated versus reserved peak memory:
- Median latency and spread:

## Confirmed findings

Only include facts directly supported by raw artifacts.

## Approximate or substitute evidence

Name every choice that differs from or is absent in the paper.

## Failed attempts

Include the hypothesis, command, result, and why it was rejected.

## Blocked claims

For each blocker, state the missing input and what would unlock progress.

## Final decision

- Continue, pause, revise, or stop:
- Evidence supporting that decision:
- Next smallest useful experiment, if any:

