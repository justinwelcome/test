# Biomechanics overlay pipeline

Turns a dance or sport video into analysis versions: pose skeletons per person, joint-angle labels,
motion trails and a live graph. The original quality is kept: HDR stays HDR, slow motion uses only
real frames, and the source file is never touched.

## Quick start

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -r biomech/requirements.txt          # needs ffmpeg on PATH; CPU is fine

cp biomech/config.example.yaml my.yaml           # set names, colours, metrics, outputs

# 1. Track everyone and write work/identify.jpg with each candidate person numbered
python biomech/pipeline.py identify VIDEO.mov --config my.yaml --work work

# 2. In my.yaml, give each person an anchor from identify.jpg, e.g. anchor: {frame: 105, index: 0}
#    (choose a frame where the people are clearly separated)

# 3. Render everything
python biomech/pipeline.py run VIDEO.mov --config my.yaml --work work --out out
```

Re-running `run` with a changed config reuses the tracking in `work/`, so changing names, colours,
labels or outputs is quick. Delete `work/` to track from scratch (for example after changing `trim`).

## Outputs

| Output key | File | What it is |
|---|---|---|
| `annotated` | `*_annotated_HDR.mov` (or `*_annotated.mp4` for SDR sources) | Full resolution and frame rate, same colour format as the source |
| `slowmo` | `*_slowmo_0.5x.*`, `*_slowmo_0.25x.*` | The annotated stream retimed with no re-encode, so no quality loss and no invented frames |
| `sdr` | `*_annotated_SDR.mp4` | Full resolution, tone-mapped to SDR so it plays on every device and app |
| `side_by_side` | `*_side_by_side_SDR.mp4` | Clean and annotated next to each other |
| `frame_step` | `*_frame_step_SDR.mp4` | 1080p, every frame a keyframe, for exact frame stepping (Kinovea, players) |
| `csv` | `*_pose_data.csv` | Per frame: every joint's x/y in pixels and every metric |

## Metrics

All metrics are 2D estimates measured in the image plane from one camera. They are good for comparing
steps, sides and attempts, but they are not lab motion capture. They are most accurate when the camera
is side-on to the movement.

| Key | Meaning |
|---|---|
| `knee` | Knee bend, left and right. 0° = straight leg |
| `hip` | Hip bend (trunk vs thigh). 0° = straight line shoulder-hip-knee |
| `elbow` | Elbow bend. 0° = straight arm |
| `trunk_lean` | Trunk angle from vertical; the arrow shows direction on screen. Camera tilt adds error |

Weight shift is deliberately left out: from a side view both feet sit at the same screen position, so
"pelvis between the feet" cannot be measured reliably in 2D.

## How it works

1. **Proxy**: HDR (HLG/PQ) is tone-mapped to an SDR copy for detection and for the SDR outputs.
2. **Tracking**: YOLO11x-pose plus ByteTrack on every frame. `detect_size` stays at 640: at larger
   input sizes the model cannot regress boxes tall enough, and people who fill the frame get cut off at
   the knees.
3. **Identity**: each person's anchor detection gives a colour model (torso and leg HSV histograms).
   Every frame is matched by colour similarity plus box overlap with the previous frame, sweeping
   forward and backward from the anchor frame.
4. **Cleanup**: left/right swap correction for legs and arms, short gaps filled, longer gaps left empty
   (the skeleton disappears instead of being guessed), then light Savitzky-Golay smoothing.
5. **Render**: the overlay is drawn as transparent frames and composited onto the decoded source. HDR
   overlays are scaled to reference white so the graphics don't glare.

## Known limits

- Close embrace and crossing legs can briefly put a limb on the wrong person.
- A person hidden behind another for long stretches is dropped for those frames.
- Motion trails are in screen space, so a moving camera adds to them.
