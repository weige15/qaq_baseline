# Router training and genuine-use gate — passed

Preregistration: `c052ca3`; `ROUTER_PROTOCOL.md`, `configs/router_protocol.json`
and tracked `configs/router_data_lock.json`. No final-score-driven search.

-192 distinct WT2 train articles and32 validation articles; train/development
  exclude final examples by normalized title/text and exact32-token-span checks.
  Manifest records606/59 eligible articles and25 rejected records.
-Local FP16 teacher block NMSE collected at fixed8 student inputs.224 contexts,
  first128 tokens only; all72x3 targets/raw features saved in
  `results/core-v1/router-collect/cache.pt`. Four smoke examples' features and
  targets match the full collection exactly. Peak13,049,010,688 bytes,431.52s.
-**A1**, the first frozen attempt, passed; **A2/A3 were not run**.72 independent
  MLPs,68→32→3,166,104 total parameters, CPU FP32, seed31416,300 full-batch epochs.
  Standardized log-NMSE train loss1.516051→0.001761; dev1.544935→0.985984.
  The train/dev gap is substantial; noncollapse is not evidence of benefit.
-Dev:31 unique profiles among32 queries; largest profile frequency6.25%;16
  attention and19 FFN blocks vary. Quotas force12 each4/6/8 per block type; they
  do not explain which blocks change across queries. Constant-feature ablation
  yields exactly one profile.72 constant length coordinates are masked.
-Real-model gate recomputed all32 dev features exactly and reproduced all fit
  profiles.304 distinct executed projection/precision combinations exactly
  match requested reconstruction. On3 dev contexts, adaptive last-token logits
  are finite and differ from fixed4 and fixed8. Raw logits are saved.
-Real scoring API receives only the same128-token prefix despite changed WT2
  suffix, synthetic MC options and gold. Profiles remain identical and an exact
  scoring repeat passes. Integrated fixed4/fixed8 endpoint smoke samples exactly
  reproduce this project's earlier fresh fixed baseline losses.

Evidence: `results/core-v1/router-training/{selection.json,static_policy.json,A1/}`,
`router-verify/{results.json,dev_routes.json,raw_logits.pt,causality-*.jsonl}`,
per-job `command.json`, source snapshots and corresponding `logs/*.log`.
All GPU stages guarded/serial on a process-free RTX3090 (physical GPU6); no
other processes interrupted.18 CPU tests pass. No runtime packages changed.

**Final follow-through:** all six locked comparison runs are now complete and
exact-repeat verified. A1 narrowly passes the preregistered static comparison,
but random and even fixed4 have better WT2 perplexity. No further attempt was
trained. See `REPLICATION_REPORT.md` for all comparisons and the qualified
conclusion; this stage report alone does not establish final quality benefit.
