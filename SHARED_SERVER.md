# Shared-server run protocol

The following constraints come from the user's server record and are part of the replication plan.

- Eight NVIDIA GeForce RTX 3090 GPUs are visible, each with 24,576 MiB total memory.
- One observed GPU had only 3,735 MiB free; the other observed values were roughly 23,681 to 24,121 MiB free.
- The host reported about 251 GiB of RAM and 236 GiB available at the time of the check.
- The prompt showed 8 GiB of swap in use. This is not itself proof of current memory pressure, but it should be logged before a large run.
- The Hugging Face account is authenticated. Access to the exact Qwen3-4B-Base and Llama-3.1-8B checkpoints still needs a direct check.

## Required launch sequence

From the project directory, use the user's virtual environment explicitly:

```bash
cd ~/qaq_baseline
source ~/.venv/bin/activate
bash scripts/gpu_preflight.sh
```

Read the displayed GPU and process list. Do not run if the selected device has a new user process, less than 20,000 MiB free, or an unexpected memory change. If the state is acceptable, launch exactly one command through the guard so the selection is rechecked immediately before execution:

```bash
bash scripts/gpu_preflight.sh --run python -m lm_eval ...
```

The selected physical GPU is exposed to the command as CUDA device 0. Record the physical index printed by the guard in the experiment log.

## Safety rules

- Never kill another user's process.
- Never use `nvidia-smi --gpu-reset` on this shared host.
- Do not launch parallel model evaluations merely because several GPUs look free.
- Re-run the guard before every separate evaluation or timing block; a prior snapshot is stale.
- If a process appears on the chosen GPU after selection, stop the run and return to the preflight gate.
- Keep the 70B model access check separate from this work; do not download 70B checkpoints for the 4B baseline.
- If RAM availability falls sharply or swap activity grows during a run, stop after a safe checkpoint and record the observation.

## What would change the plan

- **Continue** when one selected GPU remains roomy through the complete FP16 smoke test and the environment is logged.
- **Pause** when all suitable GPUs are occupied, the model license prompt is unresolved, or the virtual environment is missing.
- **Revise** the memory threshold only before a run and with a reason tied to measured model use; never lower it after an out-of-memory error just to force a launch.
- **Stop** the baseline attempt when no coherent GPU and environment state can be maintained without disturbing another user.

