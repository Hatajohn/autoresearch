# Critic Notes

Open items for the Coder. The Critic adds entries here; the Coder resolves them.

<!-- format: - [ ] <issue> | <file>:<location> -->
- [ ] LOG_MIN_WINDOW accepts any integer with no validation — a value like 1 or an odd number silently produces non-standard window sizes. Add an assert or doc comment in the override branch. | train.py:_compute_window_sizes(), LOG branch
