#!/usr/bin/env python3
"""Index rover blackbox recordings: per-segment metadata, standby-card flag, person-event count.

Usage:
  python scripts/blackbox_index.py \
      --roots ~/Development/ubrobotics/datasets/blackbox \
              ~/Development/ubrobotics/retained-jetson0-blackbox-20260901/blackbox \
      --events ~/Development/ubrobotics/datasets/logs/events.jsonl \
      --out blackbox_index.csv [--sample 150] [--workers 8]

Needs ffmpeg/ffprobe on PATH.

Two things worth knowing before you trust the output:

* Standby detection is a luminance heuristic (mean of a 32x18 grey thumbnail at t=15s < --standby-lum).
  It separates the dark "LIVE SIGNAL OFFLINE" splash from lit indoor footage. It will ALSO flag genuine
  night footage, which is a target condition for this project - re-check the threshold, or switch to OSD
  template matching, before indexing any outdoor night capture.
* Person events are joined only to the archive the events file belongs to (see --events-root). Counting
  them against another archive would invent labels, and would silently pull provenance-restricted
  segments into a training set.
"""
import argparse, csv, glob, json, os, random, statistics, subprocess, sys, bisect
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

GRID_W, GRID_H = 32, 18


def probe(path):
    try:
        out = subprocess.run(['ffprobe', '-v', 'error',
                              '-show_entries', 'format=duration:stream=codec_type,width,height,r_frame_rate',
                              '-of', 'json', path],
                             capture_output=True, text=True, timeout=30).stdout
        j = json.loads(out)
        v = next((s for s in j.get('streams', []) if s.get('codec_type') == 'video'), {})
        a = any(s.get('codec_type') == 'audio' for s in j.get('streams', []))
        return float(j.get('format', {}).get('duration', 0) or 0), v.get('width'), v.get('height'), v.get('r_frame_rate'), a, True
    except Exception:
        return 0, None, None, None, False, False


def standby(path, thresh):
    try:
        raw = subprocess.run(['ffmpeg', '-v', 'error', '-ss', '15', '-i', path, '-frames:v', '1',
                              '-vf', f'scale={GRID_W}:{GRID_H},format=gray', '-f', 'rawvideo', '-'],
                             capture_output=True, timeout=30).stdout
        if len(raw) < GRID_W * GRID_H:
            return None, None
        m = statistics.mean(raw)
        return (m < thresh), round(m, 1)
    except Exception:
        return None, None


def seg_ts(name):
    """Epoch seconds from seg-<digits>[-av|-v].mkv, or None if the clock was wrong."""
    parts = name.lstrip('.').split('-')
    if len(parts) < 2:
        return None
    digits = ''.join(ch for ch in parts[1] if ch.isdigit())
    if not digits:
        return None
    v = int(digits)
    ts = v / 1000.0 if len(digits) == 13 else float(v) if len(digits) == 10 else None
    if ts is None or not (1.5e9 < ts < 2.2e9):   # reject bad-clock names like seg-350795-v.mkv
        return None
    return ts


def load_person_events(path):
    ev = []
    with open(path, 'rb') as fh:
        for line in fh:
            try:
                d = json.loads(line)
                if (d.get('alert') or {}).get('label') == 'person' and 'ts' in d:
                    ev.append(d['ts'])
            except Exception:
                pass
    ev.sort()
    return ev


def resolve_events_root(roots, events, explicit):
    """Which root the events file describes. Events live at <archive>/logs/events.jsonl,
    segments at <archive>/blackbox - so the owning root shares a parent with the events dir."""
    if explicit:
        for r in roots:
            if os.path.basename(os.path.abspath(r)) == explicit or os.path.abspath(r).rstrip('/').endswith(explicit):
                return r
        sys.exit(f'--events-root {explicit!r} does not match any --roots')
    if not events:
        return None
    ev_archive = os.path.dirname(os.path.dirname(os.path.abspath(events)))
    for r in roots:
        if os.path.dirname(os.path.abspath(r).rstrip('/')) == ev_archive:
            return r
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--roots', nargs='+', required=True)
    ap.add_argument('--events')
    ap.add_argument('--events-root', help='root the events file belongs to (default: auto-detected by shared parent)')
    ap.add_argument('--out', default='blackbox_index.csv')
    ap.add_argument('--sample', type=int, default=0, help='index a RANDOM sample of N per root (0 = all)')
    ap.add_argument('--seed', type=int, default=0, help='sampling seed, for reproducible estimates')
    ap.add_argument('--workers', type=int, default=8, help='parallel ffprobe/ffmpeg workers')
    ap.add_argument('--standby-lum', type=float, default=40.0, help='mean-luminance threshold for the standby card')
    a = ap.parse_args()

    ev = load_person_events(a.events) if a.events and os.path.exists(a.events) else []
    ev_root = resolve_events_root(a.roots, a.events, a.events_root)
    if ev and not ev_root:
        print('[warn] could not tell which archive the events file belongs to; '
              'person_events left blank everywhere. Pass --events-root.', file=sys.stderr)
    elif ev:
        print(f'[info] {len(ev)} person events joined to {ev_root} only; '
              f'other archives get a blank person_events column', file=sys.stderr)

    rows = []
    for root in a.roots:
        files = sorted(glob.glob(os.path.join(root, 'seg-*.mkv')) + glob.glob(os.path.join(root, '.seg-*.mkv')))
        total = len(files)
        if a.sample and a.sample < total:
            files = sorted(random.Random(a.seed).sample(files, a.sample))
            print(f'[info] {root}: random sample of {a.sample} of {total} (seed {a.seed})', file=sys.stderr)
        joins_events = (root == ev_root) and bool(ev)
        label = os.path.basename(os.path.dirname(os.path.abspath(root))) if os.path.basename(root) == 'blackbox' \
            else os.path.basename(os.path.abspath(root))

        def index_one(f):
            name = os.path.basename(f)
            ts = seg_ts(name)
            dur, w, h, fps, audio, ok = probe(f)
            sb, lum = standby(f, a.standby_lum) if ok and dur > 16 else (None, None)
            n_person = ''
            if joins_events and ts:
                lo = bisect.bisect_left(ev, ts)
                hi = bisect.bisect_right(ev, ts + (dur or 60))
                n_person = hi - lo
            return dict(root=label, file=name, ts=ts,
                        date=datetime.fromtimestamp(ts, timezone.utc).date().isoformat() if ts else '',
                        duration_s=round(dur, 1), width=w, height=h, fps=fps, audio=audio, readable=ok,
                        standby=sb, luminance=lum, person_events=n_person,
                        events_joined=joins_events,
                        size_mb=round(os.path.getsize(f) / 1048576, 2), hidden=name.startswith('.'))

        with ThreadPoolExecutor(max_workers=a.workers) as pool:
            for i, row in enumerate(pool.map(index_one, files), 1):
                rows.append(row)
                if i % 200 == 0:
                    print(f'{label}: {i}/{len(files)}', file=sys.stderr)

    if not rows:
        sys.exit('no segments found - check --roots (segments live in <archive>/blackbox/)')
    with open(a.out, 'w', newline='') as fh:
        wtr = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        wtr.writeheader()
        wtr.writerows(rows)

    for label in dict.fromkeys(r['root'] for r in rows):
        rs = [r for r in rows if r['root'] == label]
        cam = [r for r in rs if r['readable'] and r['standby'] is False]
        sb = sum(1 for r in rs if r['standby'])
        n = len(rs)
        pct = 100.0 * sb / n if n else 0.0
        # binomial standard error, so a --sample estimate is not quoted as if it were exact
        se = 100.0 * ((pct / 100) * (1 - pct / 100) / n) ** 0.5 if n else 0.0
        withp = sum(1 for r in cam if isinstance(r['person_events'], int) and r['person_events'])
        print(f"{label}: {n} segments; readable {sum(r['readable'] for r in rs)}; "
              f"camera {len(cam)} ({sum(r['duration_s'] for r in cam)/3600:.1f} h); "
              f"standby {sb} ({pct:.1f}% +/- {1.96*se:.1f} at 95%); "
              f"camera segments with person events {withp if any(r['events_joined'] for r in rs) else 'n/a'}")


if __name__ == '__main__':
    main()
