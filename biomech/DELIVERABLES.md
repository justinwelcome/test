# Deliverables playbook

How each output is made and how it should read. `PROMPT.md` points here; follow it step by step.
Order matters: each step reuses what the earlier steps measured.

General rules for every written output:
- Every number comes from `pipeline.py` or `assess.py` output for THIS video. Never invent or round up a
  finding; if a measure was not significant (`*_p >= 0.05`) do not present it as a strength.
- Translate numbers into dancer language: "about 2 cm of up-and-down", "about a tenth of a second",
  "almost the whole time". Keep exact figures for the scorecard and the balanced paragraph.
- Names, roles (lead/follow) and pronouns come from the parameters. Without roles given, say "lead"
  only if `assess.py` finds one person consistently moving first, and say it is inferred.
- Public posts (captions, carousel, paragraph) show strengths and general principles only. Personal
  growth points stay in the private report.
- Credit any event watermark visible in the video (e.g. "Filmed at <event>").

---

## 1. Analysis videos — `pipeline.py`
`identify` → look at `identify.jpg` → set anchors → `run`. Check 4 frames spread across the annotated
video before moving on (names on the right person, skeletons on the bodies).

## 2. Scorecard — `assess.py`
Run it, open `<base>_timeline.png` and `<base>_assessment.json`. Give the user, in chat:
- A short "can measure / can't measure, and why" table for: connection, fluidity, musicality,
  technicality. Limits to state: 2D single camera, camera movement, no partner pressure / intention /
  weight transfer / style, too few steps for beat alignment on short clips, left/right swaps in close
  embrace, clothing hiding joints.
- The score table: rows = each dancer + couple; columns = Connection (30%), Fluidity (25%),
  Musicality (20%), Technicality (25%), Overall. Round to 0.5. Mark low-confidence cells
  (`musicality_confidence`, `_connection_confidence`).
- 2–4 bullets per criterion with the evidence, then "how to make the scores meaningful" (longer
  clip, tripod, side view, same setup each time, a reference couple).
Scores are fixed linear anchors (see `assess.py`): say they compare clips filmed the same way, not
dancers against a norm.

Name the phases: `assessment.json → phases` gives still / walking / busy segments with a suggested
frame each. Look at those frames (and the annotated video) and give each phase a dance name
(walk, shared pause, travelling, turn, ...). These names are reused by every later step.

## 3. Growth report — a Claude Doc (private, for the dancers)
Tone: a teacher writing to their students. Warm, artistic, encouraging, concrete. Not medical, not a
lab report: no "subject", "data shows", p-values or jargon. Numbers only when they make a point
vivid. Each pillar = what's already beautiful → your next step → one named exercise ("Try this").

Structure (one section each):
1. **At a glance** — the story of the dance in 2–3 lines; a table: Pillar | Where you are today |
   In one line. Levels: Seed → Growing → Solid → Shining (score < 5 / 5–6.5 / 6.5–8 / ≥ 8).
   Pillars: Connection & embrace, Flow & grounding, Musicality, Body technique, Presence (observed
   from the video, never scored). One line under the table: levels describe this clip, not the dancer.
2. **Connection & embrace** 3. **Flow & grounding** 4. **Musicality** 5. **Body technique** —
   each: one-paragraph lead, "What's already beautiful" (3 bullets), "Your next step" (2 bullets
   with timestamps from the phases), "Try this: <name>" (one drill, 2–3 sentences).
6. **For <dancer>** — one section per dancer: one-line lead, Keep / Grow bullets, a
   "Question to dance with".
7. **Moments to rewatch** — a strip of annotated stills, one per phase (see below), uploaded to the
   doc, then a numbered list: phase name (start–end s): what to notice.
8. **Two-week practice plan** — table: Week | Together | <dancer 1> | <dancer 2>, built from the
   drills above; then tasks: film a "before" clip, film an "after" clip, send both for a new report.
9. **How to read this report** — a mirror, not a verdict; what video can and can't see; film
   longer, side-on, on a tripod.

Moments strip: render the overlay for each phase's frame with `render.py --png` (labels: knee,
graph: none, trails: ankles, the dancers' names), composite on frames extracted with ffmpeg
(`scale=in_color_matrix=bt709:in_range=tv:out_range=pc` — OpenCV's decode shifts colours), add a
title bar "Phase name / time", tile side by side, upload as an asset to the doc.

## 4. Social captions — Instagram + Facebook (public, strengths only)
- **Instagram**: hook line ("5 seconds. One shared body."), one line on the motion tracking, 3
  emoji-led strengths (flow, connection, musicality) each with one dancer-friendly number, one line on
  the teaching philosophy, "swipe" prompt, 8–10 hashtags for the style.
- **Facebook**: question hook, 1–2 sentences on the teacher's biomechanics approach, the same 3
  strengths in fuller sentences, one line that each dancer also gets a private growth report, a call
  to action, 5 hashtags.
After them: remind the user to get the dancers' consent, suggest the side-by-side video or the
carousel as visuals, and credit the event.

## 5. "Learn it" caption — for the annotated / slow-motion video
A mini-lesson the viewer can practise. Sections: how to read the video (skeleton colours per
dancer, knee numbers, foot trails) → one numbered block per phase: "<Phase> (start–end s): <principle>",
one sentence on what to watch, one "Try it" drill → Timing (tempo, which beat the steps land on,
start slow) → Bonus: film yourself from the side → hashtags incl. a tutorial tag. Note that
timestamps double on the 0.5x video.

## 6. Carousel — `carousel.py` (public)
One slide per phase, plus a cover and a practice slide. Write the spec (copy
`carousel.example.json`): per moment the phase's frame, a 3–6 word headline (the principle), 2–3
callouts whose arrows point at the body part the principle is about, and a one-sentence "Try it".
Prefer joint targets (`Name.r_knee`, `Name.chest`, `between:A.chest|B.chest`, `Name.feet`,
`Name.shoulders`, `Name.head_top`); for a turn, an `axis` line between the two hip centres. Set
`keep_bottom` when an event watermark sits at the bottom of the frame.
Then LOOK at every slide at full size: each arrow tip must land on the named part, no label may
cover a face, and nothing should cover the watermark. Nudge with `dx`/`dy` or pixel targets and
regenerate until all are right.
Give the user: the slides in order, a short post caption (numbered cues, save/tag call, event
credit, hashtags), and one line of alt text per slide.

## 7. Balanced paragraph (+ short version)
One paragraph, about 170–200 words, balanced three ways:
- **Scientific**: 3–5 measured facts with units (vertical travel in cm, trunk sway in degrees,
  partner response delay, step interval against the tempo, knee softness share), described plainly.
- **Artistic**: how the dance feels — phrasing through the phases, the quality of the connection.
- **Pedagogical**: close with the key principles in very few words (e.g. "soft knees for a level
  head, a lead that starts in the chest, stillness shared rather than imposed, one common axis").
Then a 2–3 sentence version for a caption, and one line noting the figures are 2D video estimates if
it will be published formally.
