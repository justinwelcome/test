"""Biomechanics overlay pipeline for dance / sport videos.

Two steps:

  python biomech/pipeline.py identify VIDEO --config my.yaml --work WORKDIR
      Converts the video for analysis, runs pose tracking, and writes
      WORKDIR/identify.jpg: sample frames with every candidate person numbered.
      Use it to fill in each person's `anchor: {frame, index}` in the config.

  python biomech/pipeline.py run VIDEO --config my.yaml --work WORKDIR --out OUTDIR
      Assigns identities, cleans keypoints, computes metrics, renders the overlay
      and encodes every output listed in the config.

The source video is never modified. HDR sources (HLG or PQ) keep their HDR format
in the `annotated` and `slowmo` outputs; SDR copies are tone-mapped separately.
"""
import argparse, json, os, shutil, subprocess, sys
import numpy as np
import cv2
import yaml
from scipy.optimize import linear_sum_assignment
from scipy.signal import savgol_filter

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULTS = yaml.safe_load(open(os.path.join(HERE, 'config.example.yaml')))
JOINTS = ['nose', 'l_eye', 'r_eye', 'l_ear', 'r_ear', 'l_shoulder', 'r_shoulder', 'l_elbow', 'r_elbow',
          'l_wrist', 'r_wrist', 'l_hip', 'r_hip', 'l_knee', 'r_knee', 'l_ankle', 'r_ankle']
# Tone mapping chosen by eye on iPhone HLG footage: keeps warm light without over-saturating.
TONEMAP = 'zscale=t=linear:npl=100,format=gbrpf32le,zscale=p=bt709,tonemap=hable:desat=2,zscale=t=bt709:m=bt709:r=full,format=gbrp'
SDR_TAGS = ['-color_primaries', 'bt709', '-color_trc', 'bt709', '-colorspace', 'bt709', '-color_range', 'tv']


def sh(cmd, **kw):
    print('+', ' '.join(str(c) for c in cmd)[:300], flush=True)
    return subprocess.run(cmd, check=True, **kw)


def load_config(path):
    cfg = dict(DEFAULTS)
    if path:
        cfg.update(yaml.safe_load(open(path)) or {})
    cfg['trim'] = {**DEFAULTS['trim'], **(cfg.get('trim') or {})}
    return cfg


def probe(video):
    out = subprocess.run(['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries',
                          'stream=color_transfer,avg_frame_rate', '-of', 'json', video],
                         capture_output=True, text=True, check=True).stdout
    s = json.loads(out)['streams'][0]
    num, den = s['avg_frame_rate'].split('/')
    has_audio = bool(subprocess.run(['ffprobe', '-v', 'error', '-select_streams', 'a', '-show_entries',
                                     'stream=index', '-of', 'csv=p=0', video], capture_output=True, text=True).stdout.strip())
    transfer = s.get('color_transfer', '')
    return dict(fps=float(num) / float(den), hdr={'arib-std-b67': 'hlg', 'smpte2084': 'pq'}.get(transfer),
                has_audio=has_audio)


def trim_args(cfg):
    a = []
    if cfg['trim']['start'] is not None:
        a += ['-ss', str(cfg['trim']['start'])]
    if cfg['trim']['end'] is not None:
        a += ['-t', str(cfg['trim']['end'] - (cfg['trim']['start'] or 0))]
    return a


# ---------------------------------------------------------------- step 1: proxy + detection

def make_proxy(video, cfg, info, work):
    """Full-resolution SDR copy used for detection and for the SDR outputs' clean side."""
    proxy = os.path.join(work, 'proxy_sdr.mp4')
    if os.path.exists(proxy):
        return proxy
    vf = (TONEMAP + ',scale=out_color_matrix=bt709:out_range=tv,format=yuv420p') if info['hdr'] else 'format=yuv420p'
    sh(['ffmpeg', '-v', 'error', '-y', *trim_args(cfg), '-i', video, '-vf', vf, '-c:v', 'libx264', '-crf', '10',
        '-preset', 'fast', '-an', *SDR_TAGS, proxy])
    return proxy


def detect(proxy, cfg, work):
    out = os.path.join(work, 'raw_pose.json')
    if os.path.exists(out):
        return json.load(open(out))
    from ultralytics import YOLO
    model = YOLO('yolo11x-pose.pt')
    frames = []
    # imgsz stays small on purpose: YOLO can't regress boxes much taller than ~1000 px at the
    # model's input scale, so at larger sizes people who fill the frame get cut off at the knees.
    for r in model.track(proxy, stream=True, imgsz=cfg['detect_size'], conf=0.25, tracker='bytetrack.yaml',
                         verbose=False, persist=True):
        people = []
        if r.boxes is not None and len(r.boxes):
            for b, c, kp, kc in zip(r.boxes.xyxy.tolist(), r.boxes.conf.tolist(), r.keypoints.xy.tolist(),
                                    r.keypoints.conf.tolist()):
                people.append(dict(box=b, conf=c, kp=kp, kc=kc))
        frames.append(people)
        if len(frames) % 60 == 0:
            print(f'  tracked {len(frames)} frames', flush=True)
    d = dict(w=r.orig_shape[1], h=r.orig_shape[0], frames=frames)
    json.dump(d, open(out, 'w'))
    return d


def candidates(frame, H, cfg):
    c = [p for p in frame if (p['box'][3] - p['box'][1]) / H >= cfg['min_person_height']]
    return sorted(c, key=lambda p: -(p['box'][3] - p['box'][1]))


def cmd_identify(args):
    cfg = load_config(args.config)
    os.makedirs(args.work, exist_ok=True)
    info = probe(args.video)
    proxy = make_proxy(args.video, cfg, info, args.work)
    d = detect(proxy, cfg, args.work)
    N, H = len(d['frames']), d['h']
    picks = sorted(set(int(x) for x in np.linspace(0, N - 1, 4)))
    cap = cv2.VideoCapture(proxy)
    tiles, listing = [], {}
    for f in picks:
        cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        ok, im = cap.read()
        cands = candidates(d['frames'][f], H, cfg)
        listing[f] = [[round(v) for v in p['box']] for p in cands]
        s = im.shape[1] / 1080
        for i, p in enumerate(cands):
            x1, y1, x2, y2 = map(int, p['box'])
            cv2.rectangle(im, (x1, y1), (x2, y2), (0, 255, 0), int(4 * s))
            cv2.putText(im, f'#{i}', (x1 + int(8 * s), y1 + int(60 * s)), cv2.FONT_HERSHEY_SIMPLEX, 2 * s,
                        (0, 0, 0), int(10 * s), cv2.LINE_AA)
            cv2.putText(im, f'#{i}', (x1 + int(8 * s), y1 + int(60 * s)), cv2.FONT_HERSHEY_SIMPLEX, 2 * s,
                        (0, 255, 255), int(4 * s), cv2.LINE_AA)
        cv2.putText(im, f'frame {f}', (int(20 * s), int(80 * s)), cv2.FONT_HERSHEY_SIMPLEX, 2.4 * s, (0, 0, 0),
                    int(12 * s), cv2.LINE_AA)
        cv2.putText(im, f'frame {f}', (int(20 * s), int(80 * s)), cv2.FONT_HERSHEY_SIMPLEX, 2.4 * s,
                    (255, 255, 255), int(4 * s), cv2.LINE_AA)
        h = 960
        tiles.append(cv2.resize(im, (int(im.shape[1] * h / im.shape[0]), h)))
    cv2.imwrite(os.path.join(args.work, 'identify.jpg'), cv2.hconcat(tiles))
    json.dump(listing, open(os.path.join(args.work, 'identify.json'), 'w'))
    print(f"\nWrote {os.path.join(args.work, 'identify.jpg')}")
    print(f"Source: {'HDR ' + info['hdr'].upper() if info['hdr'] else 'SDR'}, {info['fps']:.3f} fps, {N} frames, {d['w']}x{H}")
    print('Set each person\'s anchor in the config to {frame: F, index: I} using the #numbers in identify.jpg.')


# ---------------------------------------------------------------- step 2: identities + metrics

def appearance(img, p, cfg):
    """HSV colour histograms of the torso and the legs: what tells people apart frame to frame."""
    kp, kc = np.array(p['kp']), np.array(p['kc'])
    ok = kc > cfg['keypoint_confidence']
    h = p['box'][3] - p['box'][1]
    hists = []
    for part in ((5, 6, 12, 11), ((11, 13), (13, 15), (12, 14), (14, 16))):
        mask = np.zeros(img.shape[:2], np.uint8)
        if isinstance(part[0], int):
            if ok[list(part)].all():
                cv2.fillConvexPoly(mask, kp[list(part)].astype(np.int32), 255)
        else:
            for a, b in part:
                if ok[a] and ok[b]:
                    cv2.line(mask, tuple(map(int, kp[a])), tuple(map(int, kp[b])), 255, max(4, int(0.05 * h)))
        if mask.any():
            hist = cv2.calcHist([cv2.cvtColor(img, cv2.COLOR_BGR2HSV)], [0, 1, 2], mask, [12, 6, 4],
                                [0, 180, 0, 256, 0, 256])
            hists.append(cv2.normalize(hist, None).flatten())
        else:
            hists.append(None)
    return hists


def hist_dist(a, b):
    ds = [cv2.compareHist(x, y, cv2.HISTCMP_BHATTACHARYYA) for x, y in zip(a, b) if x is not None and y is not None]
    return float(np.mean(ds)) if ds else 1.0


def iou(a, b):
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    i = max(0, x2 - x1) * max(0, y2 - y1)
    return i / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i + 1e-9)


def assign_people(d, proxy, cfg):
    H, frames, people = d['h'], d['frames'], cfg['people']
    N, K = len(frames), len(people)
    for p in people:
        if 'anchor' not in p:
            sys.exit(f"Person '{p['name']}' has no anchor. Run `identify` and set anchor: {{frame, index}}.")
    cands = [candidates(f, H, cfg) for f in frames]
    cap = cv2.VideoCapture(proxy)
    feats = []
    for i in range(N):
        ok, img = cap.read()
        feats.append([appearance(img, c, cfg) for c in cands[i]] if ok else [])
    model = []
    for p in people:
        f, idx = p['anchor']['frame'], p['anchor']['index']
        model.append(feats[f][idx])
    sel = [[None] * K for _ in range(N)]

    def sweep(order, start_prev):
        prev = list(start_prev)
        for i in order:
            C = cands[i]
            if not C:
                continue
            cost = np.full((K, len(C)), 9.0)
            for k in range(K):
                for j, c in enumerate(C):
                    cost[k, j] = hist_dist(model[k], feats[i][j])
                    if prev[k] is not None:
                        cost[k, j] += 0.6 * (1 - iou(c['box'], prev[k]['box']))
            rows, cols = linear_sum_assignment(cost)
            for k, j in zip(rows, cols):
                limit = 0.95 if prev[k] is not None else 0.6
                if cost[k, j] < limit:
                    sel[i][k] = C[j]
                    prev[k] = C[j]

    for k, p in enumerate(people):          # anchors are fixed by hand
        sel[p['anchor']['frame']][k] = cands[p['anchor']['frame']][p['anchor']['index']]
    a0 = people[0]['anchor']['frame']
    start = [sel[a0][k] for k in range(K)]
    sweep(range(a0 + 1, N), start)
    sweep(range(a0 - 1, -1, -1), start)
    return sel


def clean(sel, d, cfg):
    N, K, W = len(sel), len(sel[0]), d['w']
    P = np.full((K, N, 17, 3), np.nan)
    for i in range(N):
        for k in range(K):
            p = sel[i][k]
            if p is None:
                continue
            kp, kc = np.array(p['kp']), np.array(p['kc'])
            P[k, i, :, :2] = np.where(kc[:, None] > cfg['keypoint_confidence'], kp, np.nan)
            P[k, i, :, 2] = kc
    # Left/right swap correction for legs and arms against the previous frame.
    for k in range(K):
        for chain in (([13, 15], [14, 16]), ([7, 9], [8, 10])):
            L, R = chain
            for i in range(1, N):
                a = P[k, i - 1, L + R, :2]
                b = P[k, i, L + R, :2]
                if np.isnan(a).any() or np.isnan(b).any():
                    continue
                keep = np.linalg.norm(a - b, axis=1).sum()
                swap = np.linalg.norm(a - b[[2, 3, 0, 1]], axis=1).sum()
                if swap < 0.6 * keep:
                    P[k, i, L + R] = P[k, i, R + L]
    # Fill short gaps, leave long ones empty, then smooth each continuous run.
    S = P.copy()
    win, gap = cfg['smoothing_frames'], cfg['max_gap_frames']
    idx = np.arange(N)
    for k in range(K):
        for j in range(17):
            for c in range(2):
                s = S[k, :, j, c]
                good = ~np.isnan(s)
                if good.sum() < 3:
                    S[k, :, j, c] = np.nan
                    continue
                filled = np.interp(idx, idx[good], s[good])
                t = 0
                while t < N:
                    if not good[t]:
                        e = t
                        while e < N and not good[e]:
                            e += 1
                        if e - t > gap or t == 0 or e == N:
                            filled[t:e] = np.nan
                        t = e
                    else:
                        t += 1
                if win and win >= 5:
                    seg = ~np.isnan(filled)
                    t = 0
                    while t < N:
                        if seg[t]:
                            e = t
                            while e < N and seg[e]:
                                e += 1
                            if e - t >= win:
                                filled[t:e] = savgol_filter(filled[t:e], win | 1, 2)
                            t = e
                        else:
                            t += 1
                S[k, :, j, c] = filled
    return S


def angle(a, b, c):
    v1, v2 = a - b, c - b
    cos = (v1 * v2).sum(-1) / (np.linalg.norm(v1, axis=-1) * np.linalg.norm(v2, axis=-1) + 1e-9)
    return np.degrees(np.arccos(np.clip(cos, -1, 1)))


def metrics(S, W):
    out = []
    for k in range(S.shape[0]):
        J = S[k, :, :, :2]
        mid_sh, mid_hip = (J[:, 5] + J[:, 6]) / 2, (J[:, 11] + J[:, 12]) / 2
        trunk = mid_sh - mid_hip
        out.append(dict(
            # flexion: 0 = straight limb
            knee_l=180 - angle(J[:, 11], J[:, 13], J[:, 15]), knee_r=180 - angle(J[:, 12], J[:, 14], J[:, 16]),
            hip_l=180 - angle(J[:, 5], J[:, 11], J[:, 13]), hip_r=180 - angle(J[:, 6], J[:, 12], J[:, 14]),
            elbow_l=180 - angle(J[:, 5], J[:, 7], J[:, 9]), elbow_r=180 - angle(J[:, 6], J[:, 8], J[:, 10]),
            # lean of the trunk from image vertical; + = toward screen right
            trunk_lean=np.degrees(np.arctan2(trunk[:, 0], -trunk[:, 1])),
        ))
    return out


def write_csv(path, S, M, names, fps):
    import csv
    keys = list(M[0].keys())
    with open(path, 'w', newline='') as fh:
        w = csv.writer(fh)
        hdr = ['frame', 'time_s']
        for n in names:
            tag = n.lower().replace(' ', '_')
            hdr += [f'{tag}_{m}' for m in keys] + [f'{tag}_{j}_{c}' for j in JOINTS for c in ('x_px', 'y_px')]
        w.writerow(hdr)
        for i in range(S.shape[1]):
            row = [i, round(i / fps, 4)]
            for k in range(len(names)):
                row += ['' if np.isnan(M[k][m][i]) else round(float(M[k][m][i]), 2) for m in keys]
                row += ['' if np.isnan(S[k, i, j, c]) else round(float(S[k, i, j, c]), 1) for j in range(17) for c in range(2)]
            w.writerow(row)


# ---------------------------------------------------------------- step 3: render + encode

def overlay_cmd(pose_json, cfg_path, hdr):
    cmd = [sys.executable, os.path.join(HERE, 'render.py'), pose_json]
    if cfg_path:
        cmd += ['--config', cfg_path]
    if hdr:
        cmd += ['--hdr', hdr]
    return cmd


def encode(video, cfg, cfg_path, info, work, out, W, H):
    pose_json = os.path.join(work, 'pose.json')
    fps, outs, crf = info['fps'], cfg['outputs'], str(cfg['quality_crf'])
    raw = ['-f', 'rawvideo', '-pix_fmt', 'rgba', '-s', f'{W}x{H}', '-r', f'{fps}', '-i', '-']
    audio = ['-map', '0:a?', '-c:a', 'copy'] if info['has_audio'] else []
    base = os.path.join(out, cfg.get('basename', 'video'))
    made = []

    def piped(hdr, ffargs):
        r = subprocess.Popen(overlay_cmd(pose_json, cfg_path, hdr), stdout=subprocess.PIPE)
        sh(['ffmpeg', '-v', 'error', '-y', *trim_args(cfg), '-i', video, *raw, *ffargs], stdin=r.stdout)
        r.stdout.close()
        if r.wait():
            sys.exit('overlay renderer failed')

    # Native-format annotated master (HDR stays HDR); slow motion is a retime of its bitstream.
    if 'annotated' in outs or 'slowmo' in outs:
        if info['hdr']:
            trc = 'arib-std-b67' if info['hdr'] == 'hlg' else 'smpte2084'
            master = f'{base}_annotated_HDR.mov'
            piped(info['hdr'], [
                '-filter_complex', '[0:v]scale=in_color_matrix=bt2020:in_range=tv,format=gbrp10le[bg];'
                '[bg][1:v]overlay=format=gbrp:eof_action=pass,scale=out_color_matrix=bt2020:out_range=tv,format=yuv420p10le[v]',
                '-map', '[v]', *audio, '-c:v', 'libx265', '-preset', 'medium', '-crf', crf,
                '-x265-params', f'log-level=error:colorprim=bt2020:transfer={trc}:colormatrix=bt2020nc:range=limited',
                '-color_primaries', 'bt2020', '-color_trc', trc, '-colorspace', 'bt2020nc', '-color_range', 'tv',
                '-tag:v', 'hvc1', '-movflags', '+faststart', master])
            bsf, ext, tag = 'hevc_mp4toannexb', 'hevc', ['-tag:v', 'hvc1', '-color_primaries', 'bt2020',
                                                         '-color_trc', trc, '-colorspace', 'bt2020nc']
        else:
            master = f'{base}_annotated.mp4'
            piped(None, ['-filter_complex', '[0:v][1:v]overlay=eof_action=pass,format=yuv420p[v]', '-map', '[v]',
                         *audio, '-c:v', 'libx264', '-preset', 'medium', '-crf', crf, *SDR_TAGS,
                         '-movflags', '+faststart', master])
            bsf, ext, tag = 'h264_mp4toannexb', 'h264', SDR_TAGS
        made.append(master)
        if 'slowmo' in outs:
            es = os.path.join(work, f'master.{ext}')
            sh(['ffmpeg', '-v', 'error', '-y', '-i', master, '-map', '0:v', '-c:v', 'copy', '-bsf:v', bsf, '-f', ext, es])
            for sp in cfg['slowmo']:
                dst = f"{base}_annotated{'_HDR' if info['hdr'] else ''}_slowmo_{sp}x{os.path.splitext(master)[1]}"
                sh(['ffmpeg', '-v', 'error', '-y', '-r', f'{fps * sp}', '-i', es, '-c:v', 'copy', *tag,
                    '-movflags', '+faststart', dst])
                made.append(dst)
        if 'annotated' not in outs:
            os.remove(master)
            made.remove(master)

    # SDR family, from one render pass.
    sdr_outs = [o for o in ('sdr', 'side_by_side', 'frame_step') if o in outs]
    if sdr_outs:
        to_rgb = TONEMAP if info['hdr'] else 'scale=in_color_matrix=bt709:in_range=tv,format=gbrp'
        hw, hh = (W // 2 // 2 * 2, H // 2 // 2 * 2) if min(W, H) > 1100 else (W, H)
        sc = 1080 / min(W, H) if min(W, H) > 1080 else 1
        fw, fh = int(W * sc) // 2 * 2, int(H * sc) // 2 * 2
        yuv = 'scale=out_color_matrix=bt709:out_range=tv,format=yuv420p'
        n = len(sdr_outs)
        fc = [f'[0:v]{to_rgb},split[c1][c2]', f'[c1][1:v]overlay=format=gbrp:eof_action=pass,split={n}' +
              ''.join(f'[a{i}]' for i in range(n))]
        maps = []
        for i, o in enumerate(sdr_outs):
            if o == 'sdr':
                fc.append(f'[a{i}]{yuv}[o{i}]')
                maps += ['-map', f'[o{i}]', *audio, '-c:v', 'libx264', '-preset', 'medium', '-crf', crf, *SDR_TAGS,
                         '-movflags', '+faststart', f'{base}_annotated_SDR.mp4']
                made.append(f'{base}_annotated_SDR.mp4')
            elif o == 'side_by_side':
                fc.append(f'[c2]scale={hw}:{hh}:flags=lanczos[cs]')
                fc.append(f'[a{i}]scale={hw}:{hh}:flags=lanczos[as]')
                fc.append(f'[cs][as]hstack,{yuv}[o{i}]')
                maps += ['-map', f'[o{i}]', *audio, '-c:v', 'libx264', '-preset', 'medium', '-crf', crf, *SDR_TAGS,
                         '-movflags', '+faststart', f'{base}_side_by_side_SDR.mp4']
                made.append(f'{base}_side_by_side_SDR.mp4')
            else:
                fc.append(f'[a{i}]scale={fw}:{fh}:flags=lanczos,{yuv}[o{i}]')
                maps += ['-map', f'[o{i}]', '-c:v', 'libx264', '-preset', 'medium', '-crf', crf, '-g', '1',
                         '-keyint_min', '1', *SDR_TAGS, '-an', '-movflags', '+faststart', f'{base}_frame_step_SDR.mp4']
                made.append(f'{base}_frame_step_SDR.mp4')
        if 'side_by_side' not in sdr_outs:
            fc[0] = f'[0:v]{to_rgb}[c1]'
        piped(None, ['-filter_complex', ';'.join(fc), *maps])

    if 'csv' in outs:
        shutil.copy(os.path.join(work, 'pose_data.csv'), f'{base}_pose_data.csv')
        made.append(f'{base}_pose_data.csv')
    return made


def cmd_run(args):
    cfg = load_config(args.config)
    os.makedirs(args.work, exist_ok=True)
    os.makedirs(args.out, exist_ok=True)
    info = probe(args.video)
    proxy = make_proxy(args.video, cfg, info, args.work)
    d = detect(proxy, cfg, args.work)
    sel = assign_people(d, proxy, cfg)
    S = clean(sel, d, cfg)
    M = metrics(S, d['w'])
    names = [p['name'] for p in cfg['people']]
    N = S.shape[1]
    nan2none = lambda a: np.where(np.isnan(a), None, a).tolist()
    json.dump(dict(w=d['w'], h=d['h'], fps=info['fps'], names=names, joints=JOINTS, pose=nan2none(S),
                   metrics=[{m: nan2none(v) for m, v in mk.items()} for mk in M]),
              open(os.path.join(args.work, 'pose.json'), 'w'))
    write_csv(os.path.join(args.work, 'pose_data.csv'), S, M, names, info['fps'])
    print('\nTracking coverage (legs fully tracked):')
    for k, n in enumerate(names):
        cov = (~np.isnan(S[k, :, 11:17, 0])).all(1).mean()
        kl, kr = M[k]['knee_l'], M[k]['knee_r']
        rng = f'knee bend L {np.nanmin(kl):.0f}-{np.nanmax(kl):.0f}°, R {np.nanmin(kr):.0f}-{np.nanmax(kr):.0f}°' \
            if not np.isnan(kl).all() else 'no knee data'
        print(f'  {n}: {cov:.0%} of {N} frames, {rng}')
    cfg.setdefault('basename', os.path.splitext(os.path.basename(args.video))[0][:40].replace(' ', '_'))
    made = encode(args.video, cfg, args.config, info, args.work, args.out, d['w'], d['h'])
    print('\nOutputs:')
    for m in made:
        print(f'  {m}  ({os.path.getsize(m) / 1e6:.1f} MB)')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    for name in ('identify', 'run'):
        p = sub.add_parser(name)
        p.add_argument('video')
        p.add_argument('--config')
        p.add_argument('--work', required=True, help='folder for intermediate files (reused between steps)')
        if name == 'run':
            p.add_argument('--out', required=True, help='folder for the finished files')
    a = ap.parse_args()
    (cmd_identify if a.cmd == 'identify' else cmd_run)(a)
