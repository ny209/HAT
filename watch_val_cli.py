# term_plot_val.py
import time
import argparse
from pathlib import Path
import plotext as plt
import re
import math
# term_plot_val.py
import time
import argparse
from pathlib import Path
import plotext as plt
import re
import math

# -----------------------------
# 설정
# -----------------------------
BUCKET = 5000  # 검증 주기(5,000 iter) 기준으로 x를 스냅

# -----------------------------
# 로그 파서 (패턴 강화)
# -----------------------------
RE_ITER   = re.compile(r"(?:iter(?:ation)?|current_iter|it|i)\s*[:=]?\s*\[?([\d,]+)\]?", re.I)
# psnr/ssim 라인에 주석/탭/문자 섞여도 숫자만 뽑히도록
RE_PSNR   = re.compile(r"psnr[^:]*[:=]\s*([0-9]+(?:\.[0-9]+)?)", re.I)
RE_SSIM   = re.compile(r"ssim[^:]*[:=]\s*([0-9]+(?:\.[0-9]+)?)", re.I)
RE_ATITER = re.compile(r"@\s*([\d,]+)\s*iter", re.I)
VAL_HINTS = ("validation", "[val", " val", "valid", "val_")

def _to_int(s: str) -> int:
    return int(s.replace(",", ""))

def latest_log(exp_dir: Path) -> Path:
    cands = sorted(list(exp_dir.rglob("*.log")) + list(exp_dir.rglob("*.txt")),
                   key=lambda p: p.stat().st_mtime, reverse=True)
    if not cands:
        raise FileNotFoundError(f"로그 없음: {exp_dir.resolve()}")
    return cands[0]

def parse_val_rows(log_path: Path):
    """로그에서 (iter, psnr, ssim) 튜플들을 수집.
    같은 검증 블록에서 psnr/ssim이 서로 다른 줄로 찍혀도 각각 한 줄로 반환."""
    rows = []
    last_iter = None
    in_val = False
    ttl = 0
    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        for ln in f:
            mi = RE_ITER.search(ln)
            if mi:
                try:
                    last_iter = _to_int(mi.group(1))
                except:
                    pass

            low = ln.lower()
            if any(h in low for h in VAL_HINTS):
                in_val, ttl = True, 10
                continue

            if in_val and ttl > 0:
                mp, ms = RE_PSNR.search(ln), RE_SSIM.search(ln)
                if mp or ms:
                    try:
                        ps = float(mp.group(1)) if mp else float("nan")
                    except:
                        ps = float("nan")
                    try:
                        ss = float(ms.group(1)) if ms else float("nan")
                    except:
                        ss = float("nan")

                    mit = RE_ATITER.search(ln)
                    if mit:
                        try:
                            it = _to_int(mit.group(1))
                        except:
                            it = last_iter if last_iter is not None else -1
                    else:
                        it = last_iter if last_iter is not None else -1

                    rows.append((it, ps, ss))

                ttl -= 1
                if ttl == 0:
                    in_val = False
    return rows

# -----------------------------
# 유틸
# -----------------------------
def is_finite(x):
    return x is not None and not (isinstance(x, float) and (math.isnan(x) or math.isinf(x)))

def merge_by_bucket(rows, bucket=BUCKET):
    """같은 5,000 버킷에서 psnr/ssim을 병합하되 NaN으로 덮지 않음.
    반환: xs(버킷들), psnr_list, ssim_list"""
    merged = {}  # b -> [psnr, ssim]
    for it, ps, ss in rows:
        if it is None or it < 0:
            continue
        b = round(it / bucket) * bucket  # 5,000 단위 스냅(반올림)
        if b == 0:
            b = bucket
        if b not in merged:
            merged[b] = [float("nan"), float("nan")]
        # NaN으로 기존 값을 덮지 않음
        if is_finite(ps) and not is_finite(merged[b][0]):
            merged[b][0] = ps
        if is_finite(ss) and not is_finite(merged[b][1]):
            merged[b][1] = ss

    xs = sorted(merged.keys())
    ps = [merged[x][0] for x in xs]
    ss = [merged[x][1] for x in xs]
    return xs, ps, ss

def set_x_ticks_5000(xs):
    if not xs:
        return
    xmin, xmax = xs[0], xs[-1]
    start = ((xmin - 1) // BUCKET + 1) * BUCKET
    end   = ((xmax - 1) // BUCKET + 1) * BUCKET
    ticks = list(range(start, end + BUCKET, BUCKET))
    if ticks:
        plt.xticks(ticks)

def set_y_limits_from_data(ys, pad_ratio=0.05, hard=None):
    vals = [v for v in ys if is_finite(v)]
    if not vals:
        return
    y_min, y_max = min(vals), max(vals)
    if hard:
        y_min = max(y_min, hard[0]); y_max = min(y_max, hard[1])
    if y_min == y_max:
        delta = max(1e-3, abs(y_min) * 0.01)
        y_min -= delta; y_max += delta
    pad = (y_max - y_min) * pad_ratio
    plt.ylim(y_min - pad, y_max + pad)

# -----------------------------
# 메인
# -----------------------------
def main():
    ap = argparse.ArgumentParser("Terminal live plot of val PSNR/SSIM")
    ap.add_argument("--exp", required=True, help="experiments/<EXP> (YAML name)")
    ap.add_argument("--interval", type=int, default=60, help="갱신 주기(초)")
    ap.add_argument("--points", type=int, default=50, help="최근 N 포인트만 표시")
    args = ap.parse_args()

    exp_dir = Path("./experiments") / args.exp
    log_path = latest_log(exp_dir)
    print(f"[INFO] watching: {log_path}")

    while True:
        rows = parse_val_rows(log_path)          # [(iter, psnr(or NaN), ssim(or NaN)), ...]
        xs_all, ps_all, ss_all = merge_by_bucket(rows, BUCKET)  # 버킷 병합 + NaN 덮기 방지

        # 최근 N 포인트
        xs   = xs_all[-args.points:]
        psnr = ps_all[-args.points:]
        ssim = ss_all[-args.points:]

        plt.clear_terminal()
        plt.subplots(2, 1)

        # ---- PSNR (dot only) ----
        plt.subplot(1)
        plt.title("Validation PSNR")
        if xs:
            plt.scatter(xs, psnr)  # plotext는 NaN을 무시하므로 그대로 전달 OK
            set_y_limits_from_data(psnr)
            set_x_ticks_5000(xs)
        plt.ylabel("PSNR")

        # ---- SSIM (dot only) ----
        plt.subplot(2)
        plt.title("Validation SSIM")
        if xs:
            plt.scatter(xs, ssim)
            set_y_limits_from_data(ssim, hard=(0.0, 1.0))
            set_x_ticks_5000(xs)
        plt.xlabel("iter")
        plt.ylabel("SSIM")

        plt.show()
        time.sleep(args.interval)

if __name__ == "__main__":
    main()