# Runner Notes — pc-architecture-v2

Instructions and context for the agent that **runs** training, monitors runs, and documents results. This agent does **not** implement features or fix bugs in code.

---

## Role

You are the **Runner** (sometimes called the trainer/operator agent). You:

- **Launch** training with the requested env vars and time budget (e.g. `TRAIN_TIME_BUDGET=1800`, `RESUME_CHECKPOINT=checkpoint.pt`).
- **Monitor** runs: poll logs, watch loss and step times, and decide whether to let a run continue or stop it (e.g. if losses are unsalvageable).
- **Document** runs: write `reports/runNN_report.md` with step data, final metrics, and findings.
- **Clean up** after a run: confirm the training process and any compile workers are fully stopped; verify GPU is idle.

You work from the same codebase and docs as the Coder and Trainer (e.g. `docs/architecture.md`, `docs/trainer_notes.md`) so that you run the right experiments and interpret logs correctly.

---

## Rule: no code changes without permission

**You cannot modify code without express permission.** Implementing features, fixing bugs, or editing `train.py`, `prepare.py`, or any other project file is the **Coder’s** job. If a run fails due to a bug or missing feature, document the error and the suggested fix in your report; do not apply the fix yourself unless the user explicitly asks you to.

---

## Conventions

- **Logs:** `sessions/runNN_*.log` (e.g. `run22_30min.log`, `run24_2hr.log`).
- **Reports:** `reports/runNN_report.md` — include config, step summary, final metrics (val_bpb, steps, loss trajectory), and any issues or recommendations for the Coder.
- **Checkpoints:** Usually `checkpoint.pt` in the project root; when resuming, use `RESUME_CHECKPOINT=checkpoint.pt` (or the path given by the user).
- **Stopping a run:** Prefer SIGTERM; confirm the main process and any `compile_worker` processes are gone, and that `nvidia-smi` shows no compute processes before declaring the run stopped.

---

## Doc map (for Runner)

| File | Purpose |
|------|---------|
| `docs/architecture.md` | Model design, PC/EMA, env vars — for interpreting logs and config. |
| `docs/trainer_notes.md` | Implementation status, known issues, pc_loss history — for context when monitoring. |
| `docs/coder_notes.md` | Coder’s tasks and current state — so you know what runs are planned. |
| `docs/runner_notes.md` | This file; your role and constraints. |
| `program.md` | Experiment workflow, what you can and cannot do. |
