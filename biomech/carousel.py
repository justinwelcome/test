"""Instagram/Facebook carousel slides (1080x1350) with arrow callouts, from a pipeline run.

usage: python biomech/carousel.py --work WORKDIR --spec slides.json --out OUTDIR

Writes 00_cover.jpg, 01_<moment>.jpg ... and a final practice slide, plus contact_sheet.jpg.
See carousel.example.json for the spec. Arrow targets can be:
  [x, y]                                  source-video pixels
  "Justin.r_knee"                         a tracked joint (names from pose.json / the config)
  "Justin.chest", "Angi.head_top", "Justin.feet", "Angi.hip_mid"   derived points
  "between:Justin.chest|Angi.chest"       midpoint of two points
  {"at": "Justin.r_knee", "dx": 40, "dy": -20}   any of the above, nudged (source pixels)
Top-level "keep_bottom": a source y the crop always reaches (keeps an event watermark in view).
Always look at every slide afterwards: tracked joints can be off by a few cm, so nudge where needed.
"""
import argparse, json, math, os, subprocess
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

W, H, BAND = 1080, 1350, 190
FD = '/usr/share/fonts/opentype/inter/'
WHITE, INK = (255, 255, 255), (20, 20, 24)
JOINTS = ['nose', 'l_eye', 'r_eye', 'l_ear', 'r_ear', 'l_shoulder', 'r_shoulder', 'l_elbow', 'r_elbow',
          'l_wrist', 'r_wrist', 'l_hip', 'r_hip', 'l_knee', 'r_knee', 'l_ankle', 'r_ankle']


def font(name, size):
    for path in (FD + name, '/usr/share/fonts/truetype/dejavu/' + ('DejaVuSans-Bold.ttf' if 'Bold' in name or 'Black' in name else 'DejaVuSans.ttf')):
        if os.path.exists(path):
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


F_TAG, F_HEAD, F_BODY, F_PILL = font('InterDisplay-Bold.otf', 30), font('InterDisplay-ExtraBold.otf', 62), font('Inter-Medium.otf', 36), font('Inter-SemiBold.otf', 31)
F_TRY, F_TIP, F_COVER = font('InterDisplay-Bold.otf', 28), font('Inter-Medium.otf', 31), font('InterDisplay-ExtraBold.otf', 84)


class Ctx:
    def __init__(self, work, accent):
        self.work = work
        self.accent = tuple(accent)
        pose = json.load(open(os.path.join(work, 'pose.json')))
        self.P = np.array(pose['pose'], dtype=float)
        self.names = pose['names']
        self.frames = {}

    def frame(self, n):
        """Decode one frame with the correct BT.709 conversion (OpenCV's default decode shifts colours)."""
        if n not in self.frames:
            path = os.path.join(self.work, f'still_{n}.png')
            if not os.path.exists(path):
                subprocess.run(['ffmpeg', '-v', 'error', '-y', '-i', os.path.join(self.work, 'proxy_sdr.mp4'), '-vf',
                                f'select=eq(n\\,{n}),scale=in_color_matrix=bt709:in_range=tv:out_range=pc,format=rgb24',
                                '-frames:v', '1', path], check=True)
            self.frames[n] = Image.open(path).convert('RGB')
        return self.frames[n]

    def point(self, spec, fr):
        if isinstance(spec, dict):
            x, y = self.point(spec['at'], fr)
            return (x + spec.get('dx', 0), y + spec.get('dy', 0))
        if isinstance(spec, (list, tuple)):
            return (float(spec[0]), float(spec[1]))
        if spec.startswith('between:'):
            a, b = [self.point(s, fr) for s in spec[8:].split('|')]
            return ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)
        who, part = spec.rsplit('.', 1)
        k = self.names.index(who)
        J = lambda j: self.P[k, fr, JOINTS.index(j), :2]
        mid = lambda a, b: (J(a) + J(b)) / 2
        derived = {
            'chest': lambda: mid('l_shoulder', 'r_shoulder') * 0.72 + mid('l_hip', 'r_hip') * 0.28,
            'hip_mid': lambda: mid('l_hip', 'r_hip'),
            'feet': lambda: mid('l_ankle', 'r_ankle'),
            'shoulders': lambda: mid('l_shoulder', 'r_shoulder'),
            'head_top': lambda: J('nose') - np.array([0, 0.09 * self.body_h(k, fr)]),
        }
        p = derived[part]() if part in derived else J(part)
        if np.isnan(p).any():
            raise SystemExit(f'{spec} is not tracked in frame {fr}; give pixel coordinates instead')
        return (float(p[0]), float(p[1]))

    def body_h(self, k, fr):
        J = lambda j: self.P[k, fr, JOINTS.index(j), :2]
        return np.linalg.norm((J('l_shoulder') + J('r_shoulder')) / 2 - (J('l_ankle') + J('r_ankle')) / 2) / 0.82

    def span(self, fr):
        """x centre, top of heads and bottom of feet of everyone tracked in this frame."""
        xs, tops, bots = [], [], []
        for k in range(len(self.names)):
            pts = self.P[k, fr, :, :2]
            ok = ~np.isnan(pts[:, 0])
            if not ok.any():
                continue
            bh = self.body_h(k, fr)
            bh = bh if not np.isnan(bh) else np.ptp(pts[ok, 1]) * 1.2
            xs += list(pts[ok, 0])
            tops.append(np.nanmin(pts[:5, 1]) - 0.13 * bh if ok[:5].any() else np.nanmin(pts[ok, 1]) - 0.2 * bh)
            bots.append(np.nanmax(pts[15:17, 1]) + 0.06 * bh if ok[15:17].any() else np.nanmax(pts[ok, 1]) + 0.1 * bh)
        return ((min(xs) + max(xs)) / 2, min(tops), max(bots))


def background(src, cx, cy, h):
    w, s = h * W / H, H / h
    k = max(W * 1.3 / src.width, H * 1.3 / src.height)
    blur = src.resize((int(src.width * k), int(src.height * k))).filter(ImageFilter.GaussianBlur(40))
    blur = Image.blend(blur, Image.new('RGB', blur.size, (12, 12, 16)), 0.55)
    canvas = Image.new('RGB', (W, H))
    canvas.paste(blur, ((W - blur.width) // 2, (H - blur.height) // 2))
    x0, y0 = cx - w / 2, cy - h / 2
    canvas.paste(src.resize((int(src.width * s), int(src.height * s)), Image.LANCZOS), (int(-x0 * s), int(-y0 * s)))
    return canvas, (lambda p: ((p[0] - x0) * s, (p[1] - y0) * s))


def shade(img, top):
    g = Image.new('L', (1, H))
    for y in range(H):
        g.putpixel((0, y), int(210 * max(0, 1 - y / top) ** 1.3) if y < top else 0)
    img.paste(Image.new('RGB', (W, H), (8, 8, 12)), (0, 0), g.resize((W, H)))


def text_block(d, xy, text, f, fill, width, spacing=10):
    lines, cur = [], ''
    for wd in text.split():
        t = (cur + ' ' + wd).strip()
        if d.textlength(t, font=f) <= width:
            cur = t
        else:
            lines.append(cur)
            cur = wd
    lines.append(cur)
    x, y = xy
    for ln in lines:
        d.text((x, y), ln, font=f, fill=fill)
        y += f.size + spacing
    return y


def arrow(d, a, b, width=6, head=26):
    ang = math.atan2(b[1] - a[1], b[0] - a[0])
    end = (b[0] - math.cos(ang) * head * 0.8, b[1] - math.sin(ang) * head * 0.8)
    tip = [b, (b[0] - head * math.cos(ang - 0.45), b[1] - head * math.sin(ang - 0.45)),
           (b[0] - head * math.cos(ang + 0.45), b[1] - head * math.sin(ang + 0.45))]
    c = (sum(p[0] for p in tip) / 3, sum(p[1] for p in tip) / 3)
    d.line([a, end], fill=(0, 0, 0, 120), width=width + 6)
    d.polygon([(c[0] + (p[0] - c[0]) * 1.25, c[1] + (p[1] - c[1]) * 1.25) for p in tip], fill=(0, 0, 0, 120))
    d.line([a, end], fill=WHITE, width=width)
    d.polygon(tip, fill=WHITE)
    d.ellipse((a[0] - 7, a[1] - 7, a[0] + 7, a[1] + 7), fill=WHITE)


def pill_box(d, xy, text, anchor):
    l, t, r, b = d.textbbox(xy, text, font=F_PILL, anchor=anchor)
    return (l - 20, t - 13, r + 20, b + 13)


def pill(d, xy, text, anchor):
    l, t, r, b = pill_box(d, xy, text, anchor)
    d.rounded_rectangle((l + 3, t + 4, r + 3, b + 4), radius=26, fill=(0, 0, 0, 90))
    d.rounded_rectangle((l, t, r, b), radius=26, fill=WHITE)
    d.text(xy, text, font=F_PILL, fill=INK, anchor=anchor)


def edge(box, target):
    l, t, r, b = box
    if l + 30 < target[0] < r - 30:
        return (target[0], b if target[1] > (t + b) / 2 else t)
    return (r if target[0] > r else l, (t + b) / 2)


def dots(d, i, n):
    x = W / 2 - (n - 1) * 11
    for k in range(n):
        r = 6 if k == i else 4
        d.ellipse((x + k * 22 - r, H - 26 - r, x + k * 22 + r, H - 26 + r), fill=WHITE if k == i else (255, 255, 255, 110))


def fitted(ctx, fr, header_bottom, span=None):
    cx, y_top, y_bot = span or ctx.span(fr)
    y_bot = max(y_bot, ctx.keep_bottom or 0)
    sc = (H - BAND - 10 - (header_bottom + 30)) / (y_bot - y_top)
    h = H / sc
    img, P = background(ctx.frame(fr), cx, y_top - (header_bottom + 30) / sc + h / 2, h)
    shade(img, header_bottom + 120)
    d = ImageDraw.Draw(img, 'RGBA')
    d.rectangle((0, H - BAND, W, H), fill=(10, 10, 14, 255))
    return img, P, d


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--work', required=True)
    ap.add_argument('--spec', required=True)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    spec = json.load(open(a.spec))
    ctx = Ctx(a.work, spec.get('accent', [92, 225, 230]))
    if spec.get('names'):
        ctx.names = spec['names']
    ctx.keep_bottom = spec.get('keep_bottom')
    os.makedirs(a.out, exist_ok=True)
    moments = spec['moments']
    total = len(moments) + 2
    probe = ImageDraw.Draw(Image.new('RGB', (W, H)))
    made = []

    c = spec['cover']
    tb = text_block(probe, (64, 128), c['title'], F_COVER, WHITE, W - 128, spacing=4)
    img, P, d = fitted(ctx, c['frame'], tb, c.get('span'))
    d.text((64, 80), c.get('kicker', ''), font=F_TAG, fill=ctx.accent)
    text_block(d, (64, 128), c['title'], F_COVER, WHITE, W - 128, spacing=4)
    d.text((64, H - BAND + 30), c.get('subtitle', ''), font=F_BODY, fill=WHITE)
    d.text((64, H - BAND + 80), c.get('credit', ''), font=font('Inter-Regular.otf', 31), fill=(220, 220, 220))
    d.text((W - 64, H - BAND + 52), 'Swipe  →', font=F_TRY, fill=ctx.accent, anchor='rm')
    dots(d, 0, total)
    made.append(os.path.join(a.out, '00_cover.jpg'))
    img.save(made[-1], quality=93)

    for i, m in enumerate(moments, 1):
        fr = m['frame']
        tag = f"{i}/{len(moments)}  ·  {m['tag'].upper()}"
        hb = text_block(probe, (64, 116), m['headline'], F_HEAD, WHITE, W - 128, spacing=6)
        img, P, d = fitted(ctx, fr, hb, m.get('span'))
        if m.get('axis'):                                   # dashed vertical line, e.g. a shared axis
            x = ctx.point(m['axis']['x'], fr)[0]
            top, bot = P((x, m['axis'].get('top', ctx.span(fr)[1] + 150))), P((x, m['axis'].get('bottom', ctx.span(fr)[2] - 150)))
            y = top[1]
            while y < bot[1]:
                d.line([(top[0], y), (top[0], min(y + 26, bot[1]))], fill=ctx.accent + (255,), width=6)
                y += 44
        later = []
        for co in m['callouts']:
            pos = (48, co['y']) if co.get('side', 'L') == 'L' else (W - 48, co['y'])
            anc = 'lm' if co.get('side', 'L') == 'L' else 'rm'
            box = pill_box(d, pos, co['label'], anc)
            for tgt in co['targets']:
                p = P(ctx.point(tgt, fr))
                arrow(d, edge(box, p), p)
            later.append((pos, co['label'], anc))
        for pos, label, anc in later:                       # labels on top of the arrows
            pill(d, pos, label, anc)
        d.text((64, 70), tag, font=F_TAG, fill=ctx.accent)
        text_block(d, (64, 116), m['headline'], F_HEAD, WHITE, W - 128, spacing=6)
        d.text((64, H - BAND + 22), 'TRY IT', font=F_TRY, fill=ctx.accent)
        text_block(d, (200, H - BAND + 18), m['try'], F_TIP, WHITE, W - 264, spacing=8)
        dots(d, i, total)
        made.append(os.path.join(a.out, f"{i:02d}_{m['tag'].lower().replace(' ', '_')}.jpg"))
        img.save(made[-1], quality=93)

    e = spec['closing']
    img, _ = background(ctx.frame(moments[len(moments) // 2]['frame']), *ctx.span(moments[0]['frame'])[:1], 1900, 3300)
    img = img.filter(ImageFilter.GaussianBlur(18))
    img.paste(Image.new('RGB', (W, H), (8, 8, 12)), (0, 0), Image.new('L', (W, H), 175))
    d = ImageDraw.Draw(img, 'RGBA')
    d.text((64, 110), e.get('kicker', 'TIMING'), font=F_TAG, fill=ctx.accent)
    y = text_block(d, (64, 156), e['title'], F_HEAD, WHITE, W - 128)
    y = text_block(d, (64, y + 20), e['body'], F_BODY, (235, 235, 235), W - 128, spacing=12)
    d.text((64, y + 56), f'THE {len(moments)} CUES', font=F_TAG, fill=ctx.accent)
    y += 104
    for n, cue in enumerate(e.get('cues') or [m['headline'] for m in moments], 1):
        d.ellipse((64, y, 112, y + 48), outline=ctx.accent, width=3)
        d.text((88, y + 24), str(n), font=F_TRY, fill=ctx.accent, anchor='mm')
        d.text((136, y + 24), cue, font=font('InterDisplay-Bold.otf', 38), fill=WHITE, anchor='lm')
        y += 70
    if e.get('bonus'):
        d.text((64, y + 40), 'BONUS', font=F_TAG, fill=ctx.accent)
        text_block(d, (64, y + 86), e['bonus'], F_BODY, (235, 235, 235), W - 128, spacing=12)
    d.rounded_rectangle((64, H - 250, W - 64, H - 120), radius=30, outline=ctx.accent, width=4)
    d.text((W / 2, H - 185), e.get('button', 'Save this for your next practice'), font=font('InterDisplay-Bold.otf', 40), fill=WHITE, anchor='mm')
    dots(d, total - 1, total)
    made.append(os.path.join(a.out, f'{total - 1:02d}_practice.jpg'))
    img.save(made[-1], quality=93)

    thumbs = [Image.open(p).resize((360, 450)) for p in made]
    sheet = Image.new('RGB', (min(3, len(thumbs)) * 370, ((len(thumbs) + 2) // 3) * 460), 'white')
    for i, t in enumerate(thumbs):
        sheet.paste(t, ((i % 3) * 370, (i // 3) * 460))
    sheet.save(os.path.join(a.out, 'contact_sheet.jpg'), quality=85)
    print('\n'.join(made))


if __name__ == '__main__':
    main()
