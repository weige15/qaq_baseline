# Questions for the QAQ authors

These questions are ordered by how much uncertainty they remove. A code release or complete run script would supersede many of them.

1. Can you share the implementation, an anonymized archive, or exact pseudocode for quantization, routing, and loading?
2. What exact model repository IDs and immutable revisions produced Table 1? Were they base or instruction-tuned checkpoints?
3. Which software versions and evaluation commands produced each task, including the selected metric (`acc` or `acc_norm`), shot count, dataset revision, batch size, context length, and WikiText-2/PTB text preparation?
4. What quantizer produced the static 8-bit and 4-bit baselines: signed format, scale and zero point, per-tensor/channel/group rule, group size, calibration data, clipping, packing, and excluded modules?
5. How are negative weights and scales represented in equation (1), and does `W_j^(b)` in equation (4) mean one plane or a reconstruction from the top `b` planes?
6. What are the candidate low, mid, and high bit widths, and is routing per transformer layer, per attention/FFN sub-block, or per linear module?
7. What hidden feature enters the router, how is it pooled, what is the MLP shape, and are router weights shared across blocks?
8. What is the complete router loss, including knowledge-distillation temperature, any precision or latency penalty, training corpus, number of examples, optimizer, learning rate, schedule, epochs, and random seeds?
9. What precision distribution did the trained router select per model and task? Can you share mean bits and the fraction of 4/6/8-bit routes?
10. How are selected planes materialized for matrix multiplication: packed custom kernel, integer reconstruction, or dequantization to floating point?
11. In on-demand mode, what remains resident on GPU, when are weights evicted, what cache capacity and replacement rule are used, and are CPU buffers pinned?
12. What GPU, CPU, RAM, interconnect, CUDA, driver, warm-up, repeat count, synchronization, and peak-memory API produced the latency and memory columns?
13. Why do static 4-bit and static 8-bit have identical reported GPU memory, and why do QAQ quality cells equal static 8-bit at printed precision except for the 0.02 Qwen3-4B WikiText-2 difference?
14. For LLaMA-3.1-8B HellaSwag, does FP16 report `acc_norm` while the quantized rows report `acc`, or is the 78.90 to 59.99 change from another cause?

## Short request draft

Subject: Reproduction details for QAQ (NeurIPS 2025 MLForSys)

Hello QAQ authors,

I am preparing an evidence-based reproduction of your QAQ paper. The paper gives the high-level router, bit-plane, and on-demand loading design, but I could not identify several settings needed to compare against Table 1. Would you be willing to share code or run scripts? If that is not possible, the most helpful details would be the exact model revisions, quantizer and signed bit-plane format, candidate precisions, router architecture and full loss/training setup, evaluation commands and metric variants, routing distributions, and hardware/timing/memory procedure. I am especially checking the identical static 4/8-bit memory values and whether the LLaMA HellaSwag rows use the same `acc` or `acc_norm` metric.

I will label any reconstruction choices separately from your reported method. Thank you for any details you can share.
