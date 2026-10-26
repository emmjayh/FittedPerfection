# FittedPerfection

Fits a generated 3D outfit (Trellis/Pixal3D) onto four fixed MPFB bodies, each skinned to
its own rig. The fit is local, deterministic maths. Meta's SAM 3.1 labels the outfit's parts
beforehand, and Muse Wonder reviews the results afterwards. Every fit is a draft until a
person reviews it. Full spec: [docs/SPEC.md](docs/SPEC.md).

**Status:** build step 1 (foundations) is done. The seven stages are declared but not built
yet, so a run stops at `intake`.

## Setup

```powershell
python -m venv .venv
.venv/Scripts/pip install -r requirements.txt
copy config.example.json config.json      # Blender path, bodies folder
copy secrets.example.env secrets.env      # paste your Meta Model API key (MODEL_API_KEY)
```

`config.json`, `secrets.env`, `runs/`, `frozen/` and the outfit archives stay out of git.

## Commands

```powershell
.venv/Scripts/python -m fittedperfection outfits                 # list the test outfits
.venv/Scripts/python -m fittedperfection run moonshorts          # run (or resume) one outfit
.venv/Scripts/python -m fittedperfection run moonshorts --rerun-from fit
.venv/Scripts/python -m fittedperfection run ARCHIVE.zip --fresh # stage mode: first fit only, with provenance
.venv/Scripts/python -m fittedperfection run purpleshorts --test-fit   # held-out outfits need --test-fit
.venv/Scripts/python -m fittedperfection status moonshorts
.venv/Scripts/python -m fittedperfection freeze                  # frozen copy of committed code for long batches
.venv/Scripts/python -m fittedperfection view                    # 3D review viewer on http://127.0.0.1:8765/
.venv/Scripts/python -m pytest
```

## Test outfits

`outfits/manifest.json` lists 10 generator archives by hash. The 6 in `outfits/dev/` are for
development. The 4 in `outfits/test-fits/` are held out: they only run with `--test-fit`.

## How a run works

`runs/<name>/` holds one outfit. Each stage writes one folder:

| Stage | Folder | Runs on |
|---|---|---|
| intake | `full/` | Blender |
| views | `analysis/views/` | Blender |
| vision | `analysis/vision/` | SAM 3.1, Meta Model API |
| label | `labels/` | numpy |
| bind | `bind/` | numpy |
| fit | `fit/<body>/` | numpy/scipy, sealed: no network, no other programs |
| review | `review/` | Blender renders, Muse Wonder (Meta Model API) |

Each folder has a `stage.json` with the hashes of everything the stage read: its code, its
settings section, model pins, tools, the garment or bodies, and earlier stages' outputs.
- An unchanged stage is reused.
- A changed one sets its old folder aside as `<folder>.stale-N` and writes the new result to
  `<folder>.partial`, which is renamed into place only when the stage finishes.
- Nothing is deleted or overwritten.
- Every run says `fit_reviewed: false` and `game_ready: false`.

There is one fit configuration, `fittedperfection/settings.py`, with no profiles or flags.
The Meta models are pinned in `fittedperfection/models.py`.

## Viewer

Pick an outfit, a fit version and one or all four bodies. You can compare two versions side by
side; show the body as skin, X-ray or hidden; and use the front, 3/4, side and back cameras.
The source renders sit alongside. three.js 0.186.1 is bundled under `viewer/vendor/three/`
(see its `SOURCE.txt`).
