# Reusable prompt

Paste everything between the lines into a new Claude Code session that has the `justinwelcome/test`
repository attached, attach your video, and fill in the PARAMETERS block. Blank lines use the defaults.
Remove deliverables you don't need from the list.

---

I'm a dance teacher. I'm attaching a video. Produce the deliverables listed below by following my
`biomech/` toolkit in the repo `justinwelcome/test` (if `biomech/` isn't on the checked-out branch, get
it from branch `ccr-38e59a3d-5bikxl`). Read `biomech/README.md` and `biomech/DELIVERABLES.md` first:
DELIVERABLES.md defines how each deliverable is made and how it must read, so follow it exactly.

Workflow:
1. Set up a Python venv in your scratchpad and `pip install -r biomech/requirements.txt` (CPU is fine).
2. Run `pipeline.py identify`, look at `identify.jpg`, and match each person to my descriptions. If
   you can't tell who is who with confidence, show me `identify.jpg` and ask. Write the config from
   my parameters and run `pipeline.py run`. Check 4 frames of the annotated video before going on.
3. Run `assess.py`. Read the timeline and the assessment, look at each phase's frame, and give every
   phase a dance name. Use these measurements and phase names for everything that follows.
4. Make the remaining deliverables in the order of DELIVERABLES.md. Look at every image you make at
   full size and fix it before sending (arrows on the right body part, nothing covering faces or the
   event watermark).
5. Send me the files, put written deliverables in chat (the growth report as a Claude Doc), and end
   with a short list of anything you could not measure or had to estimate.

Rules: never modify the original video; never use frame interpolation; keep HDR sources in HDR for the
main output; every number must come from this video's analysis; public posts show strengths only.
If one of my extra requests isn't covered by the toolkit, tell me how you'd do it before doing it.

PARAMETERS
- Dance style: Urban Kizomba
- People (one line each — name | role: lead / follow / solo | what they look like | colour: cyan, magenta, yellow, lime, orange, red, blue, white):
  - Justin | lead | brown top, black trousers, white sneakers | cyan
  - Angi | follow | black top, blue jeans, silver heels | magenta
- Event / filmed at (credited in public posts):
- My name or handle for posts:
- Language for written outputs: English
- Trim in seconds (start / end, blank = whole video):
- What I want the dancers to learn from this video:

Videos
- Joint labels (knee, hip, elbow, trunk_lean): knee
- Graph metric (knee, hip, elbow, trunk_lean, none): knee · position (top, bottom, none): top
- Trails (ankles, wrists, hips, none): ankles · length in seconds: 0.4
- Slow-motion speeds: 0.5, 0.25
- Video outputs (annotated, slowmo, sdr, side_by_side, frame_step, csv, or all): all
- Quality (CRF, 14 = best / biggest, 18 = smaller): 16

DELIVERABLES (delete the ones you don't want)
1. Analysis videos + pose data CSV
2. Scorecard: what can / can't be measured, scores per criterion and overall, evidence
3. Growth report for the dancers (Claude Doc, warm and artistic, not scientific)
4. Social captions: Instagram + Facebook (strengths only)
5. "Learn it" caption explaining the moves so viewers can practise them
6. Carousel: cover + one slide per phase with arrows and captions + practice slide, with post caption and alt text
7. Balanced paragraph (scientific / artistic / pedagogical) + short version

Extra requests:

---
