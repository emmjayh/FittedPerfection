# Working on FittedPerfection

Read [docs/SPEC.md](docs/SPEC.md) first, including its "Changes for this build" section.

Rules from the spec (§2):
- No AI in the fit. The `fit` stage runs sealed (`fittedperfection.seal.sealed()`): numpy,
  scipy and this package only.
- The target bodies are fixed. Never pose, reshape or tilt one to suit a garment.
- Respect the design. A fit that passes the numbers but looks wrong has failed.
- Never overwrite sources, labels, bindings or results; the runner sets old folders aside.
- Nothing is accepted automatically: `fit_reviewed` and `game_ready` stay false.
- No mannequin-skin placement.

Rules for this build:
- Nothing from the old prototype: no code, styles, configs, venvs or results. Write new.
- Downloads only from official sources, and only those the owner approved.
- Held-out outfits (`outfits/test-fits/`) are not used while developing.

Process (§3):
- One branch per change; never two people or agents editing the fitter at once.
- One fit configuration (`fittedperfection/settings.py`). A change is adopted or removed, never a flag.
- Fix causes, not symptoms: no later step that repairs what an earlier step broke.
- One change at a time, run on all four bodies, compared front and side against the source
  and the previous version (`python -m fittedperfection view`) before it is kept.
- Long batches run from a frozen copy (`python -m fittedperfection freeze`).
- List every module a stage uses in its `Stage(code=...)` entry so its hash invalidates correctly.
