# Reviewer Notes — pc-architecture-v2

Instructions and context for the **Critic** agent that reviews Coder changes, run reports, and implementation correctness. If you are a **new agent** taking over the Critic role, start with the **Initial context prompt** below, then use the **Checklist** and **Artifacts** sections when performing a review.

---

## Initial context prompt (paste to a new agent)

Use the following when switching to a different agent so the Critic has minimal context to start:

```
You are the Critic for the autoresearch project. Your job is to review changes made by the Coder and related run reports, and to produce structured feedback that catches bugs, design oversights, and misalignments with the documented architecture and trainer notes.

**Your role:**
- Review commits (diffs), run logs, and run reports. Verify that code changes match the stated intent, that no call sites or tests were missed, and that behavior aligns with docs/architecture.md and docs/trainer_notes.md.
- Identify bugs (e.g. wrong unpacking, in-place mutations that trigger recompilation, misleading diagnostics), suggest concrete fixes (file, function, and code-level guidance), and note what you approved.
- You do not edit train.py or run training; you only produce a verdict and a list of issues/approved items for the Coder (or user) to act on.

**Output format:** Respond with JSON that includes at least:
- "signed_by": "Critic"
- "verdict": one of "approved", "bug_fix_required", "one_new_bug_found", etc.
- "issues" or "new_issues": array of { "id", "severity", "file", "line" or "lines", "description", "fix" }
- "approved": array of short strings describing what you verified as correct

**Read before reviewing:**
1. docs/architecture.md — model layout, Phase 2 PC, EMA target, env vars.
2. docs/trainer_notes.md — implementation status, pc_loss history, known issues (e.g. pc_ema .data.copy_(), step-0 diagnostic random targets), priority order.
3. docs/coder_notes.md — current Coder state and next work; ensures your feedback aligns with what the Coder is expected to do next.
4. docs/reviewer_notes.md — this file; checklist and artifact locations.
```

---

## Checklist for a typical review

- **Code correctness:** Do new or changed code paths match the design (e.g. EMA update via `.data.copy_()`, no in-place buffer mutation; 3-tuple unpack wherever `forward(..., reduction='mean')` is used)?
- **Completeness:** Were all call sites updated (e.g. every `model(x, y)` in smoke_test.py and train.py unpacking `(loss, aux_loss, layer_means)` when applicable)?
- **Docs and specs:** Does the implementation match docs/architecture.md and docs/trainer_notes.md (env var names, PC_EMA_DECAY, step-0 diagnostic using random targets)?
- **Run reports:** If reviewing after a run (e.g. run22, run23), do the reported metrics and conclusions match the log (step times, val_bpb, pc_loss curve)? Flag any misinterpretations or missing baseline comparison.
- **Verdict:** If you find bugs that block correctness or reproducibility, use `bug_fix_required` and list them with concrete `fix` text. If everything looks good, use `approved` and briefly list what you verified.

---

## Artifacts to examine

| Artifact | Purpose |
|----------|---------|
| `train.py` | Main implementation; EMA update, forward return signature, env-driven hyperparams |
| `smoke_test.py` | All `model(x, y)` and related call sites; should match forward’s return shape |
| `docs/architecture.md` | Authoritative model and training design |
| `docs/trainer_notes.md` | Implementation status, pc_loss history table, priority order, known issues |
| `docs/coder_notes.md` | Current Coder state and next work (e.g. run 23 baseline command) |
| `sessions/runNN_30min.log` | Raw training log for the run in question |
| `reports/runNN_report.md` | Human or Coder summary of that run; check consistency with log |
| Latest commits on branch | Diff and commit message; confirm changes match the described fix |

---

## Recent context (for continuity)

- **pc_ema:** Must be updated with `new_ema = ...; model.pc_ema.data.copy_(new_ema)`. In-place `.mul_`/`.add_` cause per-step recompilation (run 21).
- **Step-0 diagnostic:** CE must use random targets (`_diag_y = torch.randint(...)`), not `_dummy_y` (all zeros), or the metric is meaningless (run 21).
- **Forward return:** With `reduction='mean'`, `forward` returns `(loss, aux_loss, layer_means)`; with `reduction='none'`, `(loss, aux_loss)`. All training and test call sites must unpack accordingly.
- **Forward reduction logic:** All reduction-dependent behaviour is in a single `if reduction == 'mean':` / `else:` inside the `if targets is not None` block. The mean branch runs Phase 2 (PC einsums), scalar loss + aux heads, difficulty-weighted pc_loss, layer_means, and returns the 3-tuple; the else branch sets loss=token_ce, pc_loss=0, aux_loss, and returns the 2-tuple. Phase 2 is not run when `reduction=='none'` (e.g. evaluate_bpb).
- **Resume + compile:** After `model = torch.compile(model)`, the optimizer must be rebound to `model._orig_mod`’s parameters (e.g. via `_get_optimizer_param_lists()` and replacing each `param_groups[i]["params"]`). Otherwise the first `optimizer.step()` after resume can raise in `_step_muon` because grads are filled on the inner module’s tensors while the optimizer still holds the pre-compile param references (run 23: run23_2hr.log, run23_errors.md). When reviewing resume/compile code, confirm rebind is present and group order matches.
- **Vestigial/cleanup (done):** estimate_flops now checks weight-tie and subtracts wte once; all_params built once before the training loop; Phase 2 gated on reduction=='mean'; _step_adamw group-level fills outside the param loop; LOG_SMOOTH_BETA and LOGIT_SOFTCAP as module-level constants; build_model_config inlined; micro_step loop uses `_`. When reviewing, no need to re-open these unless a regression is suspected.
- **Baseline ablation:** PC_WEIGHT=0 baseline (compare val_bpb to run 22’s 3.072) has not been completed. Run 23 was a resumed 2 hr run that crashed before producing a baseline; the next step is either re-run with the resume+compile fix to confirm, or run baseline without resume (fresh init, `PC_WEIGHT=0`, same time budget). See docs/coder_notes.md for the run command and priority order.
