"""Render the biomechanics overlay as transparent RGBA frames on stdout (raw video).

usage: render.py pose.json [--config my.yaml] [--hdr hlg|pq] [--png 0,120,240]

--hdr scales overlay colours to HDR reference white so graphics don't glare on HDR screens.
--png writes preview PNGs of the given frames instead of streaming video.
"""
import argparse, json, os
import numpy as np
import cv2
import yaml
from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser()
ap.add_argument('pose')
ap.add_argument('--config')
ap.add_argument('--hdr', choices=['hlg', 'pq'])
ap.add_argument('--png')
A = ap.parse_args()

cfg = yaml.safe_load(open(os.path.join(HERE, 'config.example.yaml')))
if A.config:
    cfg.update(yaml.safe_load(open(A.config)) or {})
d = json.load(open(A.pose))
W, H, FPS, NAMES = d['w'], d['h'], d['fps'], d['names']
if A.config and len(cfg.get('people', [])) == len(NAMES):   # names from the config win, so renaming needs no re-tracking
    NAMES = [p['name'] for p in cfg['people']]
P = np.array(d['pose'], dtype=float)                       # [person, frame, joint, x/y/conf]
M = [{m: np.array(v, dtype=float) for m, v in mk.items()} for mk in d['metrics']]
K, N = P.shape[:2]
U = min(W, H) / 1080                                       # layout unit: 1 at 1080p

PALETTE = dict(cyan=(0, 215, 255), magenta=(255, 70, 200), yellow=(255, 220, 0), lime=(120, 255, 60),
               orange=(255, 140, 0), red=(255, 60, 60), blue=(60, 140, 255), white=(240, 240, 240))
ORDER = ['cyan', 'magenta', 'yellow', 'lime']


def colour(k):
    c = cfg['people'][k].get('colour') if k < len(cfg.get('people', [])) else None
    if isinstance(c, (list, tuple)):
        return tuple(int(v) for v in c)
    return PALETTE.get(str(c).lower(), PALETTE[ORDER[k % 4]])


COL = [colour(k) for k in range(K)]


def tint(c, t=0.55):
    return tuple(int(v + (255 - v) * t) for v in c)


LIMBS = [(5, 7, 'L'), (7, 9, 'L'), (6, 8, 'R'), (8, 10, 'R'), (5, 6, 'C'), (5, 11, 'L'), (6, 12, 'R'), (11, 12, 'C'),
         (11, 13, 'L'), (13, 15, 'L'), (12, 14, 'R'), (14, 16, 'R')]
FONT = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
FONTB = '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'
f_lab = ImageFont.truetype(FONTB, int(18 * U))
f_small = ImageFont.truetype(FONTB, int(16 * U))
f_axis = ImageFont.truetype(FONT, int(13 * U))
TRAIL = max(1, int(round(cfg['trail_seconds'] * FPS)))
TRAIL_JOINTS = {'ankles': [(15, 'L'), (16, 'R')], 'wrists': [(9, 'L'), (10, 'R')], 'hips': [('hipmid', 'C')]}

# metric -> (label on screen, unit, [(series, side, joint the label sits next to)], graph range)
METRICS = {
    'knee': ('knee bend', '°', [('knee_l', 'L', 13), ('knee_r', 'R', 14)], (0, 60)),
    'hip': ('hip bend', '°', [('hip_l', 'L', 11), ('hip_r', 'R', 12)], (0, 60)),
    'elbow': ('elbow bend', '°', [('elbow_l', 'L', 7), ('elbow_r', 'R', 8)], (0, 150)),
    'trunk_lean': ('trunk lean', '°', [('trunk_lean', 'C', 'trunk')], (-30, 30)),
}


def pt(k, i, j):
    if j == 'hipmid':
        a, b = pt(k, i, 11), pt(k, i, 12)
        return ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2) if a and b else None
    x, y = P[k, i, j, 0], P[k, i, j, 1]
    return None if np.isnan(x) else (x, y)


def side_col(c, s):
    return c if s == 'L' else tint(c) if s == 'R' else tint(c, 0.25)


def fmt(metric, v):
    if np.isnan(v):
        return '–'
    if metric == 'trunk_lean':
        return f'lean {abs(v):.0f}° {"→" if v > 0 else "←"}'
    return f'{v:.0f}°'


PLACED = []


def pill(dr, xy, text, font, fg, bg=(0, 0, 0, 170), anchor='mm'):
    l, t, r, b = dr.textbbox(xy, text, font=font, anchor=anchor)
    p = int(6 * U)
    for _ in range(6):                          # nudge down until it clears earlier labels
        hit = [q for q in PLACED if l - p < q[2] and r + p > q[0] and t - p < q[3] and b + p > q[1]]
        if not hit:
            break
        xy = (xy[0], xy[1] + max(q[3] for q in hit) - (t - p) + 2 * U)
        l, t, r, b = dr.textbbox(xy, text, font=font, anchor=anchor)
    PLACED.append((l - p, t - p, r + p, b + p))
    dr.rounded_rectangle((l - p, t - p, r + p, b + p), radius=int(8 * U), fill=bg)
    dr.text(xy, text, font=font, fill=fg, anchor=anchor)


def graph(dr, i, k, box, metric):
    title, unit, series, (lo, hi) = METRICS[metric]
    x0, y0, x1, y1 = box
    dr.rounded_rectangle(box, radius=int(10 * U), fill=(0, 0, 0, 150))
    c = COL[k]
    gx0, gx1, gy0, gy1 = x0 + 40 * U, x1 - 12 * U, y0 + 36 * U, y1 - 24 * U
    yof = lambda v: gy1 - (gy1 - gy0) * (min(max(v, lo), hi) - lo) / (hi - lo)
    dr.text((x0 + 10 * U, y0 + 8 * U), f'{NAMES[k]} · {title}', font=f_small, fill=c + (255,))
    for v in (lo, (lo + hi) / 2, hi):
        dr.line((gx0, yof(v), gx1, yof(v)), fill=(255, 255, 255, 50), width=max(1, int(U)))
        lab = f'{v:g}{unit}'
        dr.text((gx0 - 5 * U, yof(v)), lab, font=f_axis, fill=(255, 255, 255, 170), anchor='rm')
    right = x1 - 10 * U
    for name, s, _ in series:
        v, colr, pts = M[k][name], side_col(c, s), []
        for t in range(N):
            if np.isnan(v[t]):
                if len(pts) > 1:
                    dr.line(pts, fill=colr + (255,), width=int(2.5 * U), joint='curve')
                pts = []
                continue
            pts.append((gx0 + (gx1 - gx0) * t / max(1, N - 1), yof(v[t])))
        if len(pts) > 1:
            dr.line(pts, fill=colr + (255,), width=int(2.5 * U), joint='curve')
        txt = (f'{s} ' if s != 'C' else '') + fmt(metric, v[i])
        dr.text((right, y0 + 8 * U), txt, font=f_small, fill=colr + (255,), anchor='ra')
        right -= dr.textlength(txt, font=f_small) + 16 * U
    cx = gx0 + (gx1 - gx0) * i / max(1, N - 1)
    dr.line((cx, gy0, cx, gy1), fill=(255, 255, 255, 220), width=int(1.5 * U))
    dr.text((gx0, gy1 + 4 * U), '0 s', font=f_axis, fill=(255, 255, 255, 170), anchor='lt')
    dr.text((gx1, gy1 + 4 * U), f'{(N - 1) / FPS:.1f} s', font=f_axis, fill=(255, 255, 255, 170), anchor='rt')


def label_anchor(k, i, where):
    if where == 'trunk':
        a, b = pt(k, i, 'hipmid'), pt(k, i, 5) and pt(k, i, 6)
        if not (a and b):
            return None
        sh = ((P[k, i, 5, 0] + P[k, i, 6, 0]) / 2, (P[k, i, 5, 1] + P[k, i, 6, 1]) / 2)
        return ((a[0] + sh[0]) / 2, (a[1] + sh[1]) / 2)
    p = pt(k, i, where)
    if not p:
        return None
    hm = pt(k, i, 'hipmid')
    out = -1 if hm and p[0] < hm[0] else 1         # push the label away from the body's midline
    return (p[0] + out * 48 * U, p[1])


def frame(i):
    PLACED.clear()
    im = Image.new('RGBA', (W, H), (0, 0, 0, 0))
    ov = np.zeros((H, W, 4), np.uint8)
    for k in range(K):
        c = COL[k]
        for tr in cfg['trails']:
            for j, s in TRAIL_JOINTS[tr]:
                for t in range(max(1, i - TRAIL), i + 1):
                    a, b = pt(k, t - 1, j), pt(k, t, j)
                    if a and b:
                        al = int(230 * (1 - (i - t) / TRAIL))
                        cv2.line(ov, tuple(map(int, a)), tuple(map(int, b)), side_col(c, s) + (al,), int(5 * U),
                                 cv2.LINE_AA)
    im.alpha_composite(Image.fromarray(ov, 'RGBA'))
    # Skeleton on its own layer so `skeleton_opacity` can fade it as a whole (outline and colour together).
    sk = np.zeros((H, W, 4), np.uint8)
    for k in range(K):
        c = COL[k]
        for a, b, s in LIMBS:
            pa, pb = pt(k, i, a), pt(k, i, b)
            if pa and pb:
                cv2.line(sk, tuple(map(int, pa)), tuple(map(int, pb)), (0, 0, 0, 140), int(9 * U), cv2.LINE_AA)
                cv2.line(sk, tuple(map(int, pa)), tuple(map(int, pb)), side_col(c, s) + (255,), int(5 * U),
                         cv2.LINE_AA)
        for j in range(5, 17):
            p = pt(k, i, j)
            if p:
                cv2.circle(sk, tuple(map(int, p)), int(7 * U), (0, 0, 0, 160), -1, cv2.LINE_AA)
                cv2.circle(sk, tuple(map(int, p)), int(5 * U), side_col(c, 'L' if j % 2 else 'R') + (255,), -1,
                           cv2.LINE_AA)
    op = float(cfg.get('skeleton_opacity', 1.0))
    if op < 1:
        sk[..., 3] = (sk[..., 3] * op).astype(np.uint8)
    im.alpha_composite(Image.fromarray(sk, 'RGBA'))
    dr = ImageDraw.Draw(im)
    for k in range(K):
        c = COL[k]
        for metric in cfg['labels']:
            for name, s, where in METRICS[metric][2]:
                v, xy = M[k][name][i], label_anchor(k, i, where)
                if xy and not np.isnan(v):
                    pill(dr, xy, fmt(metric, v), f_lab, side_col(c, s) + (255,))
        head = [pt(k, i, j) for j in range(5) if pt(k, i, j)]
        if head:
            pill(dr, (np.mean([h[0] for h in head]), min(h[1] for h in head) - 70 * U), NAMES[k], f_small, c + (255,))
    if cfg['graph'] and cfg['graph'] != 'none' and cfg['graph_position'] != 'none':
        cols = min(K, 2) if W < H else min(K, 4)
        rows = -(-K // cols)
        gw, gh, m = (W - (cols + 1) * 16 * U) / cols, 170 * U, 16 * U
        top = m if cfg['graph_position'] == 'top' else H - m - rows * (gh + m) - 26 * U
        for k in range(K):
            r, cc = divmod(k, cols)
            x0, y0 = m + cc * (gw + m), top + r * (gh + m)
            graph(dr, i, k, (x0, y0, x0 + gw, y0 + gh), cfg['graph'])
        note = cfg.get('note') or 'L = solid  ·  R = pale  ·  0° = straight  ·  2D estimate from video'
        dr.text((W - m, top + rows * (gh + m) - 6 * U), note, font=f_axis, fill=(255, 255, 255, 200), anchor='ra')
    a = np.array(im)
    if A.hdr:
        # HLG reference white sits at ~75% signal, PQ 203 nits at ~58%.
        a[..., :3] = (a[..., :3].astype(np.float32) * (0.78 if A.hdr == 'hlg' else 0.58)).astype(np.uint8)
    return a


if A.png:
    for i in [int(x) for x in A.png.split(',')]:
        Image.fromarray(frame(i), 'RGBA').save(f'overlay_{i}.png')
else:
    import sys
    out = sys.stdout.buffer
    for i in range(N):
        out.write(frame(i).tobytes())
