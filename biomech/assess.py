"""Movement assessment from a finished pipeline run: connection, fluidity, musicality, technique.

usage: python biomech/assess.py VIDEO --work WORKDIR --out OUTDIR [--config my.yaml]

Needs WORKDIR from `pipeline.py run` (pose.json, raw_pose.json, proxy_sdr.mp4). Writes:
  <base>_assessment.json  every measure, score, test result and the suggested phases
  <base>_timeline.png     foot activity, pelvis height, trunk lean and partner distance against the beat

All measures are 2D, camera-robust where possible (relative to the body or the partner, scaled to
body height). Scores are 0-10 on fixed linear anchors written below; they compare clips filmed the
same way, they are not norms.
"""
import argparse, json, os, subprocess, sys
import numpy as np
import cv2
import yaml
from scipy.signal import butter, filtfilt, find_peaks

HERE = os.path.dirname(os.path.abspath(__file__))
ASSUMED_HEIGHT_CM = 170          # only used to turn body-height fractions into approximate cm


def load_cfg(path):
    cfg = yaml.safe_load(open(os.path.join(HERE, 'config.example.yaml')))
    if path:
        cfg.update(yaml.safe_load(open(path)) or {})
    return cfg


def lin(v, a0, a10):
    """Linear score: value a0 -> 0, a10 -> 10, clipped."""
    if v is None or np.isnan(v):
        return None
    return float(np.clip((v - a0) / (a10 - a0) * 10, 0, 10))


def camera_roll(work, n):
    """Camera roll per frame from background features (people masked out). Orbiting cameras make
    translation unreliable, but roll is still usable to correct trunk lean."""
    d = json.load(open(os.path.join(work, 'raw_pose.json')))
    cap = cv2.VideoCapture(os.path.join(work, 'proxy_sdr.mp4'))
    s, prev, T = 0.25, None, [np.eye(3)]
    for i in range(n):
        ok, im = cap.read()
        if not ok:
            break
        g = cv2.cvtColor(cv2.resize(im, None, fx=s, fy=s), cv2.COLOR_BGR2GRAY)
        mask = np.full(g.shape, 255, np.uint8)
        for p in d['frames'][i]:
            x1, y1, x2, y2 = [int(v * s) for v in p['box']]
            cv2.rectangle(mask, (x1 - 10, y1 - 10), (x2 + 10, y2 + 10), 0, -1)
        mask[int(g.shape[0] * 0.80):] = 0     # event logos / watermarks sit here and never move
        mask[:int(g.shape[0] * 0.04)] = 0
        if prev is not None:
            pts = cv2.goodFeaturesToTrack(prev[0], 400, 0.01, 8, mask=prev[1])
            M = None
            if pts is not None and len(pts) > 10:
                nxt, st, _ = cv2.calcOpticalFlowPyrLK(prev[0], g, pts, None)
                a, b = pts[st == 1], nxt[st == 1]
                if len(a) > 10:
                    M, _ = cv2.estimateAffinePartial2D(a, b, method=cv2.RANSAC, ransacReprojThreshold=2)
            M = np.eye(3) if M is None else np.vstack([M, [0, 0, 1]])
            T.append(M @ T[-1])
        prev = (g, mask)
    while len(T) < n:
        T.append(T[-1])
    T = np.array(T)
    return np.degrees(np.arctan2(T[:, 1, 0], T[:, 0, 0]))


def band(x, lo, hi, fps):
    y = np.full_like(x, np.nan)
    m = ~np.isnan(x)
    if m.sum() < 20:
        return y
    b, a = butter(2, [lo / (fps / 2), hi / (fps / 2)], 'band')
    y[m] = filtfilt(b, a, x[m])
    return y


def hf_fraction(x, fps, cut=3.0, lo=0.5, hi=8):
    """Share of movement energy above `cut` Hz: higher = more small, quick corrections."""
    x = x[~np.isnan(x)]
    if len(x) < 30:
        return np.nan
    t = np.arange(len(x))
    x = x - np.polyval(np.polyfit(t, x, 2), t)
    f = np.fft.rfftfreq(len(x), 1 / fps)
    p = np.abs(np.fft.rfft(x * np.hanning(len(x)))) ** 2
    return float(p[(f >= cut) & (f < hi)].sum() / max(p[(f >= lo) & (f < hi)].sum(), 1e-12))


def angle(a, b, c):
    v1, v2 = a - b, c - b
    cos = (v1 * v2).sum(-1) / (np.linalg.norm(v1, axis=-1) * np.linalg.norm(v2, axis=-1) + 1e-9)
    return np.degrees(np.arccos(np.clip(cos, -1, 1)))


def audio_beats(video, cfg, work):
    import librosa
    wav = os.path.join(work, 'audio.wav')
    tr = []
    if cfg['trim']['start'] is not None:
        tr += ['-ss', str(cfg['trim']['start'])]
    if cfg['trim']['end'] is not None:
        tr += ['-t', str(cfg['trim']['end'] - (cfg['trim']['start'] or 0))]
    r = subprocess.run(['ffmpeg', '-v', 'error', '-y', *tr, '-i', video, '-vn', '-ac', '1', '-ar', '22050', wav])
    if r.returncode or not os.path.exists(wav):
        return None
    y, sr = librosa.load(wav, sr=None)
    if len(y) < sr or np.sqrt((y ** 2).mean()) < 1e-3:
        return None
    oenv = librosa.onset.onset_strength(y=y, sr=sr)
    tempo, beats = librosa.beat.beat_track(onset_envelope=oenv, sr=sr, units='time')
    if len(beats) < 4:
        return None
    return dict(beats=np.asarray(beats), period=float(np.median(np.diff(beats))),
                onset=oenv / max(oenv.max(), 1e-9), onset_t=np.arange(len(oenv)) * 512 / sr)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('video')
    ap.add_argument('--work', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--config')
    a = ap.parse_args()
    cfg = load_cfg(a.config)
    os.makedirs(a.out, exist_ok=True)
    pose = json.load(open(os.path.join(a.work, 'pose.json')))
    P = np.array(pose['pose'], dtype=float)[..., :2]
    fps, names = pose['fps'], pose['names']
    if a.config and len(cfg.get('people', [])) == len(names):
        names = [p['name'] for p in cfg['people']]
    K, N = P.shape[:2]
    t = np.arange(N) / fps
    roll = camera_roll(a.work, N)
    J = lambda k, j: P[k, :, j]
    mid = lambda k, i, j: (J(k, i) + J(k, j)) / 2
    BH = [np.nanmedian(np.linalg.norm(mid(k, 5, 6) - mid(k, 15, 16), axis=1)) / 0.82 for k in range(K)]

    # ---- per-person signals
    sig = []
    for k in range(K):
        hip = mid(k, 11, 12)
        fl = np.linalg.norm(np.gradient((J(k, 15) - hip) / BH[k], axis=0) * fps, axis=1)
        fr = np.linalg.norm(np.gradient((J(k, 16) - hip) / BH[k], axis=0) * fps, axis=1)
        with np.errstate(invalid='ignore'):
            pel_h = (np.nanmean([J(k, 15)[:, 1], J(k, 16)[:, 1]], axis=0) - hip[:, 1]) / BH[k]
        tr = mid(k, 5, 6) - hip
        lean = np.degrees(np.arctan2(tr[:, 0], -tr[:, 1])) - roll
        kl, kr = 180 - angle(J(k, 11), J(k, 13), J(k, 15)), 180 - angle(J(k, 12), J(k, 14), J(k, 16))
        sig.append(dict(fl=fl, fr=fr, pel_h=pel_h, lean=lean, kl=kl, kr=kr))

    # ---- steps: a foot's swing peak, landing = speed back under 35% of the peak; merge within 50 ms
    def steps(k):
        ev = []
        for f in ('fl', 'fr'):
            s = np.nan_to_num(sig[k][f])
            for p in find_peaks(s, height=0.6, prominence=0.35, distance=int(0.15 * fps))[0]:
                e = p
                while e < N - 1 and s[e] > 0.35 * s[p]:
                    e += 1
                ev.append(e / fps)
        out = []
        for x in sorted(ev):
            if not out or x - out[-1] > 0.05:
                out.append(x)
        return np.array(out)

    music = audio_beats(a.video, cfg, a.work)
    rng = np.random.default_rng(0)
    dur = N / fps
    R = dict(meta=dict(names=names, frames=N, fps=fps, duration_s=round(dur, 2),
                       tempo_bpm=round(60 / music['period'], 1) if music else None,
                       beats_s=np.round(music['beats'], 3).tolist() if music else None),
             people={}, couple=None, scores={}, phases=[], notes=[])

    for k in range(K):
        s = sig[k]
        legs_ok = (~np.isnan(P[k, :, 11:17, 0])).all(1)
        st = steps(k)
        bob = band(s['pel_h'], 0.7, 4, fps)
        lean = band(s['lean'], 0.3, 4, fps)
        with np.errstate(invalid='ignore'):
            stance = np.nanmin(np.vstack([s['kl'], s['kr']]), axis=0)
        person = dict(
            tracked_share=float(legs_ok.mean()),
            body_height_px=float(BH[k]),
            level_bob_pct_height=float(np.nanstd(bob) * 100),
            level_bob_cm_approx=float(np.nanstd(bob) * ASSUMED_HEIGHT_CM),
            jerkiness_hf_share=float(np.nanmean([hf_fraction(s['pel_h'], fps), hf_fraction(s['kl'], fps), hf_fraction(s['kr'], fps)])),
            trunk_sway_deg=float(np.nanstd(lean)),
            knee_bend_mean_deg=float(np.nanmean(np.vstack([s['kl'], s['kr']]))),
            standing_knee_soft_share=float(np.nanmean(stance[~np.isnan(stance)] > 5)) if (~np.isnan(stance)).any() else None,
            steps_s=np.round(st, 3).tolist(),
            step_interval_s=float(np.median(np.diff(st))) if len(st) > 2 else None)
        if music and len(st) >= 6:
            hb = music['period'] / 2
            def onbeat(ts):
                off = np.abs(((ts - music['beats'][0]) / hb + 0.5) % 1 - 0.5) * hb
                return float(np.mean(off < 0.06)), float(np.abs(np.exp(2j * np.pi * (ts - music['beats'][0]) / hb).mean())), float(np.median(off) * 1000)
            w, plv, med = onbeat(st)
            null_w = [onbeat((st + sh) % dur)[0] for sh in rng.uniform(0, music['period'], 2000)]
            person.update(cadence_vs_half_beat=float(person['step_interval_s'] / hb),
                          landings_within_60ms_of_half_beat=w, chance_within_60ms=float(0.12 / hb),
                          beat_alignment_p=float(np.mean(np.array(null_w) >= w)), median_offset_ms=med)
        R['people'][names[k]] = person

    # ---- connection (two people)
    if K >= 2:
        # filter only the stretches where each person is tracked; zero-filling gaps would add false signal
        A = [np.nan_to_num(band(sig[k]['fl'] + sig[k]['fr'], 0.3, 4, fps)) for k in (0, 1)]
        both = (~np.isnan(sig[0]['fl'])) & (~np.isnan(sig[1]['fl']))
        A0, A1 = A[0][both] - A[0][both].mean(), A[1][both] - A[1][both].mean()
        L = int(0.3 * fps)

        def best(a0, a1):
            res = []
            for l in range(-L, L + 1):
                x, y = a0[max(0, -l):len(a0) - max(0, l)], a1[max(0, l):len(a1) - max(0, -l)]
                res.append((np.corrcoef(x, y)[0, 1], l))
            return max(res)
        r, lag = best(A0, A1)
        null = [best(A0, np.roll(A1, sh))[0] for sh in rng.integers(int(0.6 * fps), len(A1) - int(0.6 * fps), 300)]
        dist = np.linalg.norm(mid(0, 5, 6) - mid(1, 5, 6), axis=1) / np.mean(BH[:2])
        dist = dist[~np.isnan(dist)]
        q = [np.nan_to_num(sig[k]['fl'] + sig[k]['fr']) / 2 < 0.35 for k in (0, 1)]
        lag_ms = lag / fps * 1000
        R['couple'] = dict(
            movement_coupling_r=float(r), coupling_chance_r=float(np.mean(null)), coupling_p=float(np.mean(np.array(null) >= r)),
            response_delay_ms=float(abs(lag_ms)),
            initiator=names[0] if lag_ms > 0 else names[1] if lag_ms < 0 else 'simultaneous',
            chest_distance_pct_height=float(np.mean(dist) * 100), chest_distance_variation=float(np.std(dist) / np.mean(dist)),
            shared_stillness=float((q[0] & q[1])[both].mean()), stillness_if_independent=float(q[0][both].mean() * q[1][both].mean()))

    # ---- phases: still / walking / busy, from everyone's mean foot activity
    act = np.nanmean([np.nan_to_num(sig[k]['fl'] + sig[k]['fr']) / 2 for k in range(K)], axis=0)
    w = max(1, int(0.5 * fps))
    act = np.convolve(act, np.ones(w) / w, mode='same')
    lab = np.where(act < 0.22, 'still', np.where(act > 0.42, 'busy', 'walking'))
    segs, s0 = [], 0
    for i in range(1, N + 1):
        if i == N or lab[i] != lab[s0]:
            segs.append([s0, i - 1, str(lab[s0])])
            s0 = i
    merged = []                          # fold fragments shorter than 0.5 s into their neighbour
    for sgm in segs:
        if merged and ((sgm[1] - sgm[0] + 1) / fps < 0.5 or merged[-1][2] == sgm[2]):
            merged[-1][1] = sgm[1]
        else:
            merged.append(sgm)
    if len(merged) > 1 and (merged[0][1] - merged[0][0] + 1) / fps < 0.5:
        merged[1][0] = merged[0][0]
        merged.pop(0)
    for a0, a1, kind in merged:
        ok = [i for i in range(a0, a1 + 1) if all(not np.isnan(P[k, i, 15, 0]) for k in range(K))]
        rep = ok[len(ok) // 2] if ok else (a0 + a1) // 2
        R['phases'].append(dict(start_s=round(a0 / fps, 2), end_s=round((a1 + 1) / fps, 2), kind=kind, suggested_frame=int(rep)))

    # ---- scores (anchors are fixed so clips filmed the same way stay comparable)
    S = {}
    for n, p in R['people'].items():
        flu = [lin(p['level_bob_pct_height'], 4, 0.5), lin(p['jerkiness_hf_share'], 0.35, 0.05)]
        tec = [lin(p['standing_knee_soft_share'] if p['standing_knee_soft_share'] is not None else np.nan, 0.30, 0.95),
               lin(p['trunk_sway_deg'], 6, 1)]
        mus, mus_conf = None, 'not measurable (no usable music or too few steps)'
        if 'cadence_vs_half_beat' in p:
            tempo = lin(abs(p['cadence_vs_half_beat'] - 1), 0.30, 0)
            align = lin(p['landings_within_60ms_of_half_beat'], p['chance_within_60ms'], 1.0)
            mus = 0.6 * tempo + 0.4 * align
            mus_conf = 'tempo match measured; beat alignment above chance' if p['beat_alignment_p'] < 0.05 else \
                'tempo match measured; beat alignment not distinguishable from chance'
        S[n] = dict(fluidity=float(np.mean([x for x in flu if x is not None])),
                    technique=float(np.mean([x for x in tec if x is not None])),
                    musicality=mus, musicality_confidence=mus_conf)
    if R['couple']:
        c = R['couple']
        conn = np.mean([lin(c['movement_coupling_r'], c['coupling_chance_r'], 0.70),
                        lin(c['chest_distance_variation'], 0.40, 0.05),
                        8.0 if 50 <= c['response_delay_ms'] <= 200 else 5.0])
        S['_connection'] = float(conn)
        S['_connection_confidence'] = 'above chance' if c['coupling_p'] < 0.05 else 'not distinguishable from chance'
    weights = dict(connection=0.30, fluidity=0.25, musicality=0.20, technique=0.25)
    for n in R['people']:
        parts = dict(connection=S.get('_connection'), fluidity=S[n]['fluidity'], musicality=S[n]['musicality'],
                     technique=S[n]['technique'])
        used = {k: v for k, v in parts.items() if v is not None}
        S[n]['overall'] = float(sum(weights[k] * v for k, v in used.items()) / sum(weights[k] for k in used))
    S['weights'] = weights
    R['scores'] = S
    R['notes'] = [
        'Everything is measured from one 2D camera view; most accurate when filmed side-on on a tripod.',
        'Weight transfer, partner pressure, intention and style are not measurable from video.',
        'Left/right labels can swap in close embrace; side-specific numbers are low confidence.',
        f'cm values assume a {ASSUMED_HEIGHT_CM} cm dancer and are approximate.',
    ]

    base = cfg.get('basename') or os.path.splitext(os.path.basename(a.video))[0][:40].replace(' ', '_')
    jpath = os.path.join(a.out, f'{base}_assessment.json')
    json.dump(R, open(jpath, 'w'), indent=1, default=float)

    # ---- timeline figure: what Claude (and a teacher) reads to name the phases
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    cols = ['#00b8d4', '#e040a0', '#c8a000', '#4caf50']
    fig, ax = plt.subplots(4, 1, figsize=(12, 9), sharex=True)
    for k in range(K):
        ax[0].plot(t, (np.nan_to_num(sig[k]['fl']) + np.nan_to_num(sig[k]['fr'])) / 2, cols[k], lw=1, label=names[k])
        ax[0].plot(R['people'][names[k]]['steps_s'], [1.6 + 0.15 * k] * len(R['people'][names[k]]['steps_s']), '|', color=cols[k], ms=12)
        ax[1].plot(t, sig[k]['pel_h'], cols[k], label=names[k])
        ax[2].plot(t, sig[k]['lean'], cols[k], label=names[k])
    if K >= 2:
        ax[3].plot(t, np.linalg.norm(mid(0, 5, 6) - mid(1, 5, 6), axis=1) / np.mean(BH[:2]), 'k', label='chest distance (body heights)')
    if music:
        ax[3].plot(music['onset_t'], music['onset'], '0.6', lw=.8, label='audio onsets')
    for ph in R['phases']:
        ax[0].axvspan(ph['start_s'], ph['end_s'], color={'still': '#ddd', 'busy': '#fde2c8', 'walking': 'white'}[ph['kind']], zorder=0)
        ax[0].text((ph['start_s'] + ph['end_s']) / 2, 1.95, ph['kind'], ha='center', fontsize=8)
    for axx in ax:
        if music:
            for b in music['beats']:
                axx.axvline(b, color='orange', lw=.6)
        axx.legend(fontsize=7, loc='upper right')
    ax[0].set_ylim(0, 2.1)
    ax[0].set_ylabel('foot activity\n(body h / s)')
    ax[1].set_ylabel('pelvis height\n(body h)')
    ax[2].set_ylabel('trunk lean (°)')
    ax[3].set_xlabel('time (s)  ·  orange = beats  ·  ticks = detected step landings')
    plt.tight_layout()
    plt.savefig(os.path.join(a.out, f'{base}_timeline.png'), dpi=80)

    # ---- console summary
    print(f"\nTempo: {R['meta']['tempo_bpm']} BPM" if music else '\nNo usable music track found.')
    if R['couple']:
        c = R['couple']
        print(f"Connection: coupling r={c['movement_coupling_r']:.2f} (chance {c['coupling_chance_r']:.2f}, p={c['coupling_p']:.3f}); "
              f"{c['initiator']} moves first by ~{c['response_delay_ms']:.0f} ms; chest distance varies {c['chest_distance_variation']:.0%}")
    for n, p in R['people'].items():
        s = S[n]
        print(f"{n}: overall {s['overall']:.1f} | fluidity {s['fluidity']:.1f} | technique {s['technique']:.1f} | "
              f"musicality {s['musicality'] if s['musicality'] is None else round(s['musicality'], 1)} ({s['musicality_confidence']})")
    if '_connection' in S:
        print(f"Connection score {S['_connection']:.1f} ({S['_connection_confidence']})")
    print('Phases:', ', '.join(f"{p['kind']} {p['start_s']}-{p['end_s']}s (frame {p['suggested_frame']})" for p in R['phases']))
    print(f'\nWrote {jpath}')


if __name__ == '__main__':
    main()
