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
- **An unclear facing asks the owner** (§5.1's "If it's ambiguous, stop" becomes "stop and ask"). When
  intake's facing is provisional and the footwear can't confirm it (bluetiebaggypants: sneakers under
  pooled trousers, soles reaching as far behind the ankle as in front), bind stops and the app shows
  intake's front and side views: "Is this the outfit's front?" The answer is recorded by garment
  (`confirmations/<sha256>.json`, or `confirm-facing NAME --front|--back` from a terminal) and counts as
  evidence. Without one the outfit is refused as before; the fitter never guesses. Shoes that point away
  from the front still refuse outright. A "back" answer is recorded, but turning an outfit round isn't
  supported yet.
- **A garment part in the skin tone** (a white top on a white mannequin) can't be told from skin
  by colour. Skin recovery never takes faces from it. A skin face whose mirror image (the same
  spot on the other side) lies in such a part takes that part, sides swapped, since mannequins are
  symmetric.
- **Hackathon use only.** Commercial licence checks (§4a) are out of scope for now.
- **Donor leg length is clamped to 0.85–1.6, not 0.85–1.2** (§5.6 A). Generator figures often
  have much longer legs than the MPFB bodies (2 of 5 dev outfits need 1.35–1.57). At 1.2 the
  donor stands on the floor with its trunk 15–20 cm low, and the top transfers above the
  bodies' heads. Bind already keeps the crotch within ±0.07 × shoulder height of its prior,
  which works out to about 0.85–1.6 for the donor, so the clamp only catches what bind would
  have refused.
- **Body shoulders are measured the way bind measured the garment's:** sleeve roots, or
  `shoulder_top` for sleeveless garments.
- **Donor legs aim over the rear quarter of the sole** (heel to toe, from bind). The leg
  centreline ends at the front of the shoe, and a tall boot's "ankle" is its shaft top; aiming at
  either tilted the legs, so tights and boots sat off-centre on every body.
- **Donor neck length (new in Step A):** generator figures are headless with long neck stubs, and their
  neck cloth runs to the top of the stub. The donor's neck lengthens (never shortens, clamped to 1.0–3.0)
  so the top of the garment's neck cloth (torso, collar and a hood's neck) sits 55% of the way up the neck
  (`neck_top_share`). Without it, collars landed on the bodies' chins. Putting the cloth top right under the
  head instead left every neckline at the jaw.
- **Donor arm length (new in Step A):** a cuff sits at the wrist. Where a sleeve has a labelled cuff,
  the donor's arm stretches along itself (0.7–1.5) until its wrist meets the cuff's centre, so sleeves
  map elbow to elbow and wrist to wrist. Without a cuff the arm keeps its length. The generator
  figures' arms differed by up to 17 cm, which put cuffs past the fingertips (blackshirtsuspenderjeans)
  or above the wrists (blackpantsgreentop).
- **Donor trunk girth fits depth and width separately** (replacing §5.6 A's single girth factor for the
  trunk, and its front-to-back centring). Each 3 cm trunk slice measures the innermost cloth in narrow
  sectors at the front, back and sides. Depth and width scale separately (0.85–1.15), and the slice moves
  front to back (up to 6 cm) so its front and back margins match, carrying the limbs with it (legs by the
  lowest slice, each arm by the slice at its shoulder). A side with no cloth (a back opening, a
  keyhole) takes depth and offset from the nearest covered slices. One radial factor, with the trunk
  centred by an average, left the donor's back up to 6 cm outside a busty dress with a thin back
  (hallowewenbaddie): openings started inside the target body, and the wrap spread them as it pushed
  them out.
- **Printed designs stay stiff in the wrap** (Step D's "source-shape energy", strengthened where the
  design is printed):
  - **Detection (label stage):** a print is a part of a garment piece whose colour stands out from
    that piece's median colour, near a sharp colour edge. Baked-in shading changes gradually, so it
    doesn't count.
  - **Effect:** the wrap makes edges inside a print 20× stiffer (`print_stiffness`), so the plain
    fabric takes up the squeeze of wrapping.
  - **Why:** pulling a loose top in shortens it around the body but not top to bottom, which sheared
    baddietanknoarms' barcode. Its worst shear drops from 2.06× to 1.46× on body_01, and the fit of
    plain fabric is unchanged.
- **Step D is a wrap that keeps the shape** (owner's direction: "a lot closer to the skin,
  essentially as a wrap, but without losing the entire shape"). Height above the skin splits into
  bulk (smoothed looseness) and relief (folds, pockets, ruffles). The wrap keeps all the relief and
  30% of the looseness (`wrap_keep`), never closer than ease. Skirts, hanging cloth, and cloth next to
  the head or hands keep their looseness. Cloth standing well off the body hangs rather than fits
  (wide trouser legs, puff sleeves, a flared peplum): from 6 cm of looseness it keeps more, and all of
  it from 16 cm (`wrap_hanging_from_m`, `wrap_hanging_full_m`), so a wide cut stays wide while snug
  cloth still wraps close. One ARAP solve balances the shape of the Step C result, the wrap targets,
  point-and-triangle contact springs and rigid anchors.
  The audit's coverage check searches the same 15 cm on the fit as on the source, since hanging cloth
  keeps its looseness on the bodies too.
- **Step E shoes, no body warping** (owner's decisions, replacing §7.1's heel flattening):
  - **Size:** each shoe is scaled uniformly to each body's foot. The foot it was made for is
    taken as 15% of the source figure's height; the donor's foot gives where it sits.
  - **Placement:** a rigid move onto the foot with the footbed on the sole, measured from the
    sole's edge at the ball. Pitch is kept, so heels stay heels.
  - **Raised model:** the whole fit goes up so the soles stand on the floor
    (`root_offset_m`, a single translation for the game). The body itself is unchanged.
  - **Ghost removal:** body faces inside or poking through a shoe are listed as hidden
    (`hidden_body_faces.npy`). `BODY.glb` is the body as shown.
  - **Boot shafts:** footwear rising at least 15 cm above the ankle line follows the leg rather
    than the rigid shoe. In the wrap it keeps its looseness and is stiff, like a print, so a boot
    keeps its structured shape; contact still keeps the calf inside. Lower shoes stay fully rigid.
  - **Cloth meeting a shoe** (a stocking ending inside it, a trouser cuff over it) moves with the
    shoe where they touch, fading back to its own placement within 8 cm (`shoe_join_blend_m`).
    The shoe goes to the foot and the cloth to the leg; without this a gap opens between them.
    Wrapping shafts 70% in like cloth made moonshorts' boots look melted.
  - **Shoes fused in the source:** faces joining the left and right shoe (the generator fuses shoes
    that touch) are dropped. Each shoe goes to its own foot, so those faces stretched into a bar
    between the feet (hallowewenbaddie).
- **Steps F–G deliverable:** `fit/<body>/OUTFIT.glb` is the full-resolution outfit with its
  texture, skinned to the body's own skeleton (4 influences), with a root node carrying
  `root_offset_m`. `BODY.glb` is the shown body (covered faces hidden) on the same skeleton.
  - Full-resolution vertices bind to cage faces of their own label, so a trouser hem never binds
    to the boot under it.
  - Hanging cloth above the knees takes pelvis-led weights, so tails follow the hips.
  - `preview --pose` checks the skinning in the §5.7 test pose (elbows 60°, knees 30°).

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
