# Reusable prompt

Paste everything between the lines into a new Claude Code session that has the `justinwelcome/test`
repository attached, attach your video, and fill in the PARAMETERS block. Blank lines use the defaults.

---

I'm attaching a video. Run my biomechanics overlay pipeline on it, exactly as defined in the `biomech/`
folder of my repo `justinwelcome/test`. If `biomech/` isn't on the checked-out branch, get it from
branch `ccr-38e59a3d-5bikxl`. Read `biomech/README.md` first and follow it.

Steps:
1. Set up a Python venv in your scratchpad and `pip install -r biomech/requirements.txt`. CPU is fine.
2. Run `pipeline.py identify` on the video. Look at `identify.jpg` yourself and match each person in my
   parameters to their #number, using my descriptions. Pick a frame where they are clearly separated.
   If you can't tell who is who with confidence, show me `identify.jpg` and ask before continuing.
3. Write the config from my parameters (defaults for anything blank), with the anchors filled in, and
   run `pipeline.py run`.
4. Check before sending: look at about 4 frames spread across the annotated video. Confirm each name
   stays on the right person and the skeletons sit on the bodies. If identities swap, try a different
   anchor frame and rerun.
5. Send me every output file, then a short summary covering: what each file is for, tracking coverage
   per person, any stretch where tracking drops out or swaps, and what the metrics show (peak values,
   left vs right differences, anything notable for technique). Be clear that these are 2D estimates.

Rules: never modify the original file, never use frame interpolation for slow motion, and keep HDR
sources in HDR for the main output. If one of my extra requests isn't supported by the pipeline,
tell me whether you can do it as a one-off for this video, and how, before doing it.

PARAMETERS
- People (one line each — name | what they look like | colour: cyan, magenta, yellow, lime, orange, red, blue, white):
  - Justin | brown top, black trousers, white sneakers | cyan
  - Angi | black top, blue jeans, silver heels | magenta
- Trim in seconds (start / end, blank = whole video):
- Joint labels (any of: knee, hip, elbow, trunk_lean): knee
- Graph metric (knee, hip, elbow, trunk_lean, none): knee
- Graph position (top, bottom, none): top
- Trails (any of: ankles, wrists, hips, or none): ankles
- Trail length in seconds: 0.4
- Slow-motion speeds: 0.5, 0.25
- Outputs (any of: annotated, slowmo, sdr, side_by_side, frame_step, csv, or all): all
- Quality (CRF, 14 = best / biggest, 18 = smaller): 16
- What I want to learn from this video:
- Extra requests:

---
