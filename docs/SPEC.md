# FittedPerfection build spec

Spec for a fresh build of FittedPerfection. It describes the methods to use, drawn from the earlier
prototype (0.11.0-dev), its docs and its trial history (written 2026-10-26).

## Changes for this build (2026-10-26)

These decisions override the original text below where they differ.

- **No prototype code.** Nothing from the prototype is copied or adapted: no code, styles,
  configs, venvs or results. Everything is written new.
- **Generators:** outfits may come from Trellis, Pixal3D or Rodin.
- **Test outfits** are the 10 generator archives in `outfits/` (see `outfits/manifest.json`):
  6 in `dev/` for development, 4 in `test-fits/` held out for test fits only.
- **SAM 3.1 runs through the Meta Model API**, not locally. It needs no venv, local weights or
  PyTorch. The vision stage needs the network; its requests and responses are cached by
  hash, so the stage demo runs offline after vision when a cached result exists.
  - Model `sam-3.1` at `https://api.meta.ai/v1/responses`, key `MODEL_API_KEY`.
  - One phrase per request. The orbit is sent as an MP4 turntable so part ids track across
    frames; top and bottom go as still images.
  - Results vary slightly between calls, so every response is saved and reused while the
    request is unchanged.
  - Masks are decoded with Meta's `meta-sam-parser` (PyPI, pinned).
- **Bodies are remade fresh with MPFB**, 2 female and 2 male: `body_01_medium` (F),
  `body_02_tall_slender` (F), `body_03_short_heavy` (M), `body_04_broad_muscular` (M).
  MPFB default rig, A-pose, one shared topology.
- **Labels are checked visually only.** The step 2 regression check against the
  prototype's labels is dropped.
- **Hackathon use only.** Commercial licence checks (§4a) are out of scope for now.

---

## 1. What FittedPerfection is

FittedPerfection fits a generated 3D outfit onto fixed human bodies:

- **Input:** a single textured mesh of an outfit on a mannequin, from Trellis/Pixal3D.
- **Output:** the outfit on each target body, skinned to that body's own rig.
- **Targets:** four MPFB/MakeHuman bodies:
  - `body_01_medium`
  - `body_02_tall_slender`
  - `body_03_short_heavy`
  - `body_04_broad_muscular`

  All four use the MPFB default rig in an A-pose, and they share one mesh topology.

**Purpose:** a Meta hackathon (video plus live demo; SAM is a Meta model), and later a
benchmark or sellable tool.

## 2. Non-negotiable rules

1. **No AI in the fit.** The fit is local, deterministic numerical code, and network access
   and child processes are blocked while it runs. Meta models prepare the inputs (SAM 3.1
   labels) and check the results (Muse Wonder). None of them runs inside the fit or changes
   a fit. See §4a.
2. **The body is fixed.** Never pose, reshape or tilt the target body to suit a garment. Every
   garment piece, shoes included, conforms to the body.
3. **Respect the design.** The fitted outfit must still visibly be the same design: puffs,
   flares, hem lengths, chunky soles, hardware. Close to the body, with no blown-up limbs and
   no shrink-wrapping. A fit that passes numbers but looks wrong has failed.
4. **Never overwrite.** Sources, labels, bindings and previous results are kept. A rerun
   writes a new folder or sets the old one aside.
5. **Nothing is accepted automatically.** Every result is a draft until a human reviews it
   from the front and side against the source.
6. **No mannequin-skin placement.** Mannequin skin is evidence (where legs are) and is
   deleted from the deliverable, but the body is not fitted to it. The user rejected that
   approach.

## 3. Process rules (lessons from the prototype)

- **Git from day one.** One repo and one branch per change. Never two people or agents
  editing the fitter at once.
- **One fit configuration.** No stack of opt-in feature flags and candidate profiles. A change
  is either adopted or deleted.
- **Fix causes, not symptoms.** Never add a later step to repair what an earlier step broke.
  If a step creates spikes, fix that step.
- **One change at a time, judged visually.** Every change runs the regression set on all four
  bodies. It is compared front and side against the source and against the previous version
  before it is kept.
- **Freeze code for runs.** Long batches run from a frozen copy, so edits can't invalidate
  them midway.

## 4. Pipeline overview

```
archive.zip ─► 1 intake ─► 2 views ─► 3 vision ─► 4 label ─► 5 bind ─► 6 fit ─► 7 audit/review

1 intake, 2 views ... Blender
3 vision ............ SAM 3.1 (Meta Model API)
4 label, 5 bind ..... numpy
6 fit ............... numpy/scipy, no models, network blocked
7 audit/review ...... numpy, Blender renders, Muse Wonder check (API)
```

- Each stage writes one folder plus a JSON record of its input hashes.
- A stage is reused when its inputs are unchanged.
- `--rerun-from STAGE` sets that stage and later ones aside (`*.stale-N`) and reruns them.
- Stage mode (`--fresh`) refuses any garment whose bytes were seen before. It writes a
  provenance card (sha256, prior-run check, time, machine, versions) and a self-refreshing
  `live.html` for the stage demo.

## 4a. Meta models

Two Meta models, each with one job. Both are new to this build, so each needs its own
regression check before it's trusted.

| Model | Job | Stage | Runs |
|---|---|---|---|
| SAM 3.1 (Mar 2026, drop-in successor to SAM 3) | Part segmentation from the fixed text vocabulary; multi-object tracking keeps each part's identity across views | 3 vision → 4 label | Meta Model API (cloud, needs network) |
| Muse Wonder (Oct 2026, `muse-wonder-1.0`) | Visual reviewer: compares the source render with the fitted renders and reports skin showing, design changes and defects; second opinion on mannequin-skin labels | 7 review (and label review) | Meta Model API (cloud, needs network) |

Rules:

- **SAM 3.1 produces evidence, never geometry.** Its masks feed the labels. The deterministic
  code decides, and it refuses when evidence disagrees.
- **Muse Wonder only reviews.** It reviews the skin labels and the fitted results, always
  outside the fit's network/process guard. It never edits a mesh, a label or a setting, and
  never accepts a fit. Its findings are saved with the model id, the prompt hash and the
  images it saw. Human review stays the decision.
- The stage demo must still work offline after vision (with a cached vision result) up to
  the review step; Muse Wonder needs a network connection.
- **Versions are pinned.** API model ids are recorded in every run, like code hashes. A model
  change invalidates the stages that use it.
- **Licences:** check each model's terms before any commercial use (SAM-family models use
  Meta's own licence, not Apache/MIT).

## 5. Stage details

### 5.1 Intake

- Unpack the generator archive (OBJ or FBX plus `texture_diffuse.png`) in Blender.
- Join the parts, apply transforms and export GLB. Refuse to overwrite.
- Keep two meshes:
  - **Full** (`full/GARMENT.glb`): the original resolution (0.5–1M faces) and the deliverable.
  - **Cage:** a decimated proxy of about 40,000 faces. All labelling and fitting maths runs
    on the cage.
- Fix one canonical frame: Z up, garment forward = −Y, garment's own left = +X. Record the basis.
- Determine orientation from evidence (toe direction), never by guessing. If it's ambiguous, stop.

### 5.2 Views

- Render the cage from 6 calibrated orthographic views (front, back, left, right, top, bottom).
- For each view save the colour image and a **face-ID buffer**: which cage face is visible at
  each pixel. The face-ID buffer is what makes lifting masks to 3D exact.

### 5.3 Vision (2D): SAM 3.1

- Run SAM 3.1 through the Meta Model API with text prompts from one fixed, versioned
  vocabulary. No per-outfit prompts.
- The vocabulary (v5) maps phrases to classes:

| Class | Phrases |
|---|---|
| torso | shirt, jacket, top, hoodie, vest, coat, sweater, dress |
| sleeve / cuff / collar / hood | sleeve / sleeve cuff / collar / hood |
| waistband | waistband, belt |
| pants | pants, shorts, jeans |
| pant_leg | pant leg, sock, stocking, tights, leggings |
| skirt | skirt |
| footwear | boot, shoe, sneaker |
| hardware | buckle, metal clasp, chain |
| attachment | pocket, strap, zipper |
| hanging | coat tail, cape |
| skin | skin, bare skin, bare leg, bare back, bare shoulders |

- **Length prompts are evidence only:** "shorts", "knee-length shorts", "capri pants", "long
  pants". Their result is a legwear length class used when nothing below the hem shows where
  the leg ends.
- **Rejected prompts** include 'mannequin', 'bare arm' and 'exposed skin'; they over-select.
- **Cross-view identity:** feed the views as an ordered orbit sequence and use SAM 3.1's
  multi-object tracking, so the same part (left sleeve, right boot) keeps one identity in
  every view. This replaces the prototype's point-prompt propagation.

### 5.4 Label (2D → one label per 3D face). Core method; keep as is.

1. **Lift:** for each view and mask, every visible face under the mask gets evidence for that class.
2. **Specificity:** where masks overlap, the smaller mask wins. A sleeve beats "jacket", a
   buckle beats "boot".
3. **Fuse across views:** average evidence over the views where the face is visible. A view
   that sees the face without the label counts against it.
4. **Graph smoothing:** solve a harmonic/graph problem over the cage's face adjacency. Edges
   are weakened across UV seams and creases over about 40°, so labels spread along fabric
   and stop at real boundaries.
5. **Structural rules** (deterministic, after smoothing):
   - **Branch split:** pants split into `pant_leg.left/right` where the leg tubes separate.
   - **Open shell:** torso cloth that stops enclosing the body becomes `hanging` (coat tails).
   - **Host:** each connected attachment piece takes the label of what it's sewn onto.
   - **Sides:** sided classes become `.left` / `.right` (+X is the garment's left).
6. **Skin check:** a face is mannequin skin only when SAM 3.1's skin prompts and the rendered
   colour agree (colour matches the confident skin tone). A skin face whose colour clearly
   differs reverts to its best garment label. This keeps straps and stockings painted on skin.
7. **Skin review (Muse Wonder):** the label review image goes to Muse Wonder for a second
   opinion: "is anything marked as skin actually clothing, or the reverse?". Its findings are
   flagged for the human; it never changes labels itself.
8. **Covered leg skin:** skin inside a leg band mostly covered by legwear or footwear becomes
   that garment.
9. **Coherent limbs:** a limb gets one consistent label along its length.

Outputs: per-face labels, a review image, a JSON record.

### 5.5 Bind (measure the garment)

From the labels, measure "where the body must be inside this outfit":

- **Tube centrelines** for each sleeve and leg: section centroids along the tube, tolerant of gaps.
- **Sleeves:** use the largest connected sleeve piece. Ignore fragments under 3% of torso
  area. A cuff counts only if it's near the sleeve.
- **Legs:** evidence = pant legs + bare-leg skin + boot shafts below the crotch.
- **Crotch height:** where the leg tubes branch. Anthropometric prior 0.59 × shoulder
  height, ±0.07.
- **Shoulder line:** from the torso top centre. Sleeveless garments use a `shoulder_top` landmark.
- **Feet:** footwear below the crotch; toe direction from the sole's long axis plus forward
  evidence. If ambiguous, refuse.
- **Knees:** a knee read from a single mid-leg point is unreliable (frills and boot tops drag
  it). Use a robust estimate (e.g. narrowest inner-layer section near the expected knee
  height) or none, and record which.
- **Legwear length class** (from the length prompts) → hem fraction: shorts .30, knee .60,
  cropped .78, full 1.0.
- Record everything with hashes. The binding is reused by every body.

### 5.6 Fit

**Step A: build the stand-in ("donor") body inside the garment.** Reshape the medium MPFB
body so it sits inside the source outfit:

- Scale and translate on the shoulder centre, scaled by shoulder-to-crotch span.
- Leg length from crotch-to-floor, clamped to 0.85–1.2. The figure stands on the floor; don't
  use hem position, because hems are design: baggy hems pool, cropped hems stop high.
- Centre the trunk horizontally inside the torso and pants cloth.
- Pose the limbs along the bind centrelines with linear-blend skinning.
  - Arms aim from the body's own shoulder joint to the cuff, which handles dropped-shoulder sleeves.
  - Legs aim hip → knee → ankle.
  - Knee bends read from a single mid-leg point were unreliable, because frills drag it. Use
    a robust knee estimate or none.
- **Girth:** scale each region to sit just inside the innermost cloth layer, capped to 0.85–1.15.

**Step B: un-pose the garment.** Move garment points into the stand-in's rest pose using a
soft blend of nearby same-region body transforms. Rigid pieces use one transform each.

**Step C: transfer to each target.** This is what made unseen outfits work.

- The donor and the targets share topology, so each body vertex has an exact move
  `target − donor`.
- Each garment point moves by a smooth, distance-weighted average of the moves of its k≈32
  nearest same-region body vertices (Gaussian, σ = max(2·d₁, 0.03·height)).
- There are no per-anchor rotations: those fanned cloth out into spikes.
- **Regions:** trunk, arm.L/R (including hands), leg.L/R (including feet), and hips = trunk +
  both legs around the crotch, so trouser seats don't split.
- **Rigid groups** (hardware, shoe soles) are placed as exact copies of the source piece:
  similarity fit source → field, with rotation clamped at 25°. Physical (UV-split) twin
  vertices always move together.

**Step D: one coordinated cloth solve.** Replaces the prototype's chain of repair passes. A
single ARAP-style system that balances, in the same solve:

- **source-shape energy:** keep local edges and rotations like the source, which preserves design;
- **soft placement targets** from Step C;
- **body contact:** vertex and triangle springs keeping cloth outside the skin plus an ease of
  about 0.6% of height;
- **hard anchors** only for rigid parts.

Do not follow it with independent per-vertex pushes or despike passes.

**Step E: shoes.** Open problem, see §7. Current rule: shoes are rigid copies placed by the
field, and boot shafts above the ankle are warped radially to the target calf. Never press
the shoe top onto the foot.

**Step F: skinning.** Weights from the target's own rig, transferred from the nearest body
triangles. At most 4 normalized influences; twins and rigid groups share weights.

**Step G: full-resolution reconstruction.**

- Map each full-resolution vertex to its nearest cage triangle (barycentric coordinates plus offset).
- Reconstruct it on the fitted cage, using a smooth per-vertex rotation field (not
  per-triangle frames).
- Full-resolution faces owned by mannequin skin are dropped.
- Neighbouring full-resolution vertices can bind to different cage layers (trouser vs boot).
  The coordinated solve, or a welded harmonic repair, must keep them together.

### 5.7 Audit and review

**Automated diagnostics, not acceptance:**

- Sampled penetration (cloth inside body); minimum clearance.
- **Edge stretch versus the source:** count faces stretched more than 3× and more than 10×.
  This metric found the spike bugs.
- **Global silhouette:** width profile versus the source, normalized by height.
  Uncalibrated: it reported 47–61% on fits that looked right. Calibrate against human labels
  before using it as a gate.
- **Coverage:** rays from the body outward, compared with what the source covered. Finds
  holes such as missing stockings.
- **Synthetic poses:** elbow 30/60° and knee 15/30°, checking cloth clearance.
- Hashes of every input, the code and outputs.

**Muse Wonder review (automated reviewer, never acceptance):**

- Inputs: the source render beside the four-angle renders of each body, plus the part labels.
- Asks for structured findings: skin showing where the design covers it; design changes
  (sleeve and hem length, volume, shoe shape, hardware); holes, spikes and floating pieces,
  each with the view and approximate location.
- Output saved as `review/wonder.json` with model id, prompt hash, image hashes and time, and
  shown in the viewer next to the fit.
- Findings are a checklist for the human, not a verdict. Measure their usefulness against
  human review before relying on them.

**Human review:**

- Blender renders: front, three-quarter, side and back per body, with the body in a flat skin
  tone so penetrations show.
- The 3D web viewer: outfit/body/version selector, X-ray body, front/side/back buttons and the
  source image alongside.

## 6. Data contracts

- Run folder `runs/<name>/`:
  - `run.json`: stages, status, hashes, provenance
  - `full/`, `analysis/views/`, `labels/`, `bind/`
  - `fit/<body>/`: `DRAFT.glb`, `fit.report.json`, `cage-fit.npz`
  - `comparison.png`, `live.html`
- Every report carries `fit_reviewed: false`, `game_ready: false`, the code and input hashes,
  the settings, and the gates. A fit with failing gates is `rejected_draft` but is still
  written, so it can be inspected.
- **Qualification run:** frozen code, settings, suite and bodies; a case × body matrix;
  resumable; an `index.html` progress page. Development cases are kept separate from
  never-seen holdouts.

## 7. Open problems (don't repeat these mistakes)

1. **Shoes.** The flat MPFB foot versus shoes designed for heels or platforms:
   - Pressing the shoe top onto the foot destroyed boots (pointy slippers, split soles). Rejected.
   - Tilting the body's foot into the heel is forbidden (body is fixed).
   - Proposed: keep the shoe rigid; scale it up uniformly only if the foot is longer or wider
     than the inside; align heel and centre; for heels, tilt the shoe flat at the ball of the
     foot and shorten only the heel block or platform. Check from the side.
   - The "ambiguous disconnected footwear components" rule refuses whenever a second shoe
     piece exceeds 10% of the first. That wrongly blocks 4 of 9 outfits (bluetie, tank top,
     pirate cloak; floaty fails a related shaft check). Pieces inside the same foot's
     footprint belong to that shoe.
2. **Stockings and tights** that end up beside the leg (knee-bend error) or patchy (moonshorts
   white stockings). Placement must get the lower leg right.
3. **Silhouette metric is uncalibrated.** Don't gate on it until it matches human judgement.
4. **Animation:** at elbow 60° / knee 30°, sleeves penetrate. Static fits only so far.
5. **Body proportions:** MPFB medium is much heavier than the generator's slim mannequin, so
   outfits widen. Decide how much widening is acceptable per design.

## 8. Suggested build order

1. Repo + frozen runner + viewer (the viewer helps from day one).
2. Intake → views → SAM 3.1 → label → bind. Labels checked visually (see Changes).
3. Fit Steps A–C (donor, un-pose, transfer) with rigid copies. Review.
4. Step D (coordinated solve) as the only clearance mechanism. Review.
5. Steps F–G (skinning, full-res). Review with the stretch metric.
6. Shoes (§7.1) as their own module, judged from the side.
7. Muse Wonder review step (skin labels and fitted results). Compare its findings with human
   review on the regression set before relying on it.
8. Qualification harness: development set + never-seen holdouts, frozen.
