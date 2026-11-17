"""Forecast GTA VI's launch-day Twitch viewers, hour by hour.

Model: learn the release-day curve from the case's own comparables.

A launch's Twitch category does not grow smoothly from hour 1. The pooled
curve over past launches is bimodal, and the split is driven by *when* the
game actually unlocks relative to 00:00 UTC:

- Launches with a rolling midnight-local unlock (New Zealand ~11-12h before
  00:00Z) already have many hours of play behind them at hour 1, so hour 1
  is large and the rest of the day only rises ~1-2.5x to an evening peak.
- Launches with one simultaneous global unlock later in the day have a
  near-dead hour 1 (hundreds of viewers) and explode 100-2000x when the
  unlock lands, usually in the UTC afternoon/evening.

So hour 1's own size is the dominant predictor of the shape. This program
splits the case's comparables into three buckets by their hour-1 viewers
(<5k, 5k-50k, >=50k), takes each bucket's per-hour median log-ratio
log(viewers_h / viewers_1) and spread, and blends in a second signal:
console-only launches (no PC in their platforms) peak higher and tighter
than the average launch. GTA VI is console-only with a rolling midnight
unlock, so it sits in the intersection of the two most favourable groups.

Per hour h, with v1 the target's hour-1 print:

    q_tau = v1 * exp(med_h + z_tau * sigma_h)

where med_h is the mean of the target bucket's median log-ratio and the
console-only median log-ratio, sigma_h is the larger of the two groups'
spreads, and z_tau are the standard normal quantiles at the five scored
levels (0.05, 0.25, 0.5, 0.75, 0.95). The buckets are hard cuts: the data show a step, not a gradient — a launch
is either already unlocked at hour 1 (tens of thousands of viewers) or
still waiting for its unlock (hundreds), with almost nothing in between, so
interpolating across an edge only blurs the signal.

Everything is recomputed from the case file at run time, so the final case
(which adds the four practice launches to the comparables) is handled with
no code change. A fitted fallback table for the big-launch branch is
embedded in case the comparables cannot be parsed; a persistence fallback
(the house shape) is the last resort. The program never crashes on a case:
any failure still writes a valid forecast file.

Standard library only, no network, deterministic: no randomness, no clock,
no hash-order dependence (comparables are sorted by name).
"""

import json
import math
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

# The quantile levels the contest scores, in the order each row lists them.
LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
# The standard normal's quantile at each level.
Z = [-1.645, -0.674, 0.0, 0.674, 1.645]

# Hour-1 viewer buckets. Launches below 5k at hour 1 had not unlocked yet;
# launches at 50k+ were already hours into a rolling midnight unlock.
BUCKET_EDGES = (5000.0, 50000.0)
# A group with fewer launches than this is too thin to trust; use the pool.
MIN_GROUP = 4
# Never claim a tighter spread than this in log units.
SIGMA_FLOOR = 0.10

# Fallback (median log-ratio, spread) per hour for the big-launch branch,
# fitted on the 37 usable comparable launches in the practice kit: the
# >=50k bucket blended with the console-only launches. Used only when the
# case's comparables cannot be parsed. Hour 2 is index 0.
FALLBACK_BIG = [
    (-0.15, 0.19), (-0.30, 0.32), (-0.43, 0.28), (-0.26, 0.29),
    (-0.18, 0.40), (-0.20, 0.47), (-0.25, 0.45), (-0.32, 0.45),
    (-0.39, 0.46), (-0.38, 0.46), (-0.28, 0.45), (-0.11, 0.42),
    (0.08, 0.41), (0.31, 0.44), (0.37, 0.46), (0.48, 0.43),
    (0.53, 0.46), (0.56, 0.47), (0.59, 0.42), (0.59, 0.42),
    (0.51, 0.45), (0.45, 0.42), (0.42, 0.42),
]

# The house baseline's spread, used by the last-resort fallback only.
HOUSE_SIGMA = 0.15


def round2(x):
    """Round to 2 decimals, half up, returning an int when whole.

    Matches the starter's rounding (JavaScript Math.round(x * 100) / 100),
    not Python's banker's rounding.
    """
    scaled = x * 100.0
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 100.0
    return int(value) if value.is_integer() else value


def median(values):
    """Median of a non-empty list of numbers."""
    ordered = sorted(values)
    n = len(ordered)
    mid = n // 2
    if n % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2.0


def pstdev(values):
    """Population standard deviation; 0.0 for a single value."""
    n = len(values)
    if n < 2:
        return 0.0
    mean = sum(values) / n
    return math.sqrt(sum((v - mean) ** 2 for v in values) / n)


def hour_start(day_start, hour):
    """Timestamp string of the start of hour `hour` (1-based) of the day."""
    base = datetime.strptime(day_start, "%Y-%m-%dT%H:%M:%SZ")
    base = base.replace(tzinfo=timezone.utc)
    return (base + timedelta(hours=hour - 1)).strftime("%Y-%m-%dT%H:%M:%SZ")


def release_day_viewers(comp):
    """The comparable's 24 release-day hourly viewer counts, or None.

    Returns None when the launch has no usable release day (the GTA VI
    pre-release entry in the comparables is not a launch: it has no release
    day in the file) or any hour is missing.
    """
    day_start = comp.get("day_start")
    hourly = comp.get("hourly")
    if not day_start or not isinstance(hourly, list):
        return None
    by_from = {}
    for point in hourly:
        try:
            by_from[point["from"]] = point["viewers"]
        except (KeyError, TypeError):
            continue
    viewers = []
    for hour in range(1, 25):
        value = by_from.get(hour_start(day_start, hour))
        if value is None:
            return None
        viewers.append(value)
    if viewers[0] is None or viewers[0] <= 0:
        return None
    return viewers


def is_console_only(comp):
    """True when the launch's platforms list no PC version."""
    platforms = comp.get("platforms") or ""
    return "pc" not in platforms.lower()


def log_ratios(viewers):
    """Per-hour log(viewers_h / viewers_1), hours 2..24, floor of 1 viewer."""
    first = max(1, viewers[0])
    return [math.log(max(1, v) / first) for v in viewers[1:]]


def group_curve(series_list):
    """Per-hour (median, spread) of a list of log-ratio series (23 each).

    Returns None when the group is too thin to trust.
    """
    if len(series_list) < MIN_GROUP:
        return None
    curve = []
    for h in range(23):
        column = [series[h] for series in series_list]
        curve.append((median(column), max(SIGMA_FLOOR, pstdev(column))))
    return curve


def collect_curves(case):
    """Bucket + console-only + pooled per-hour curves from the comparables.

    Returns (buckets, console_curve, pooled_curve) where buckets is a list
    of three curves (small, mid, big), any of which may be None.
    """
    small, mid, big, console, pooled = [], [], [], [], []
    comparables = sorted(case.get("comparables") or [],
                         key=lambda c: str(c.get("name", "")))
    for comp in comparables:
        viewers = release_day_viewers(comp)
        if viewers is None:
            continue
        ratios = log_ratios(viewers)
        pooled.append(ratios)
        if is_console_only(comp):
            console.append(ratios)
        if viewers[0] < BUCKET_EDGES[0]:
            small.append(ratios)
        elif viewers[0] < BUCKET_EDGES[1]:
            mid.append(ratios)
        else:
            big.append(ratios)
    buckets = [group_curve(small), group_curve(mid), group_curve(big)]
    return buckets, group_curve(console), group_curve(pooled)


def bucket_curve_at(buckets, pooled, log10_v1):
    """The bucket curve for this hour-1 size.

    Hard cut at 5k/50k: the data are a step (unlocked vs not-yet-unlocked),
    not a gradient. Leave-one-out over the 37 usable launches: hard buckets
    0.486 overall vs 0.506 with a +-0.2 dex smoothing band. Falls back to
    the pooled curve when the bucket is too thin.
    """
    if log10_v1 < math.log10(BUCKET_EDGES[0]):
        curve = buckets[0]
    elif log10_v1 < math.log10(BUCKET_EDGES[1]):
        curve = buckets[1]
    else:
        curve = buckets[2]
    return curve if curve is not None else pooled


def hour_quantiles(v1, med, sigma, hour):
    """The five quantiles for one hour around the median forecast."""
    return [round2(min(1e9, max(0.01, v1 * math.exp(med + z * sigma))))
            for z in Z]


def forecast_case(case):
    """Forecast hours 2..24 of one case from its hour-1 print + comparables."""
    known = case.get("known") or []
    first = next((k for k in known if k.get("hour") == 1), None)
    if first is None and known:
        first = known[0]
    if first is not None:
        v1 = max(1, first.get("viewers") or 1)
    else:
        history = case.get("history_hourly") or case.get("history") or []
        v1 = max(1, (history[-1].get("viewers") or 1) if history else 1)

    rows = case.get("forecast") or [{"hour": h} for h in range(2, 25)]
    hours = sorted(int(row["hour"]) for row in rows)

    try:
        buckets, console, pooled = collect_curves(case)
        if pooled is None:
            raise ValueError("no usable comparables")
        bucket = bucket_curve_at(buckets, pooled, math.log10(v1))
        if bucket is None:
            raise ValueError("no bucket curve")
        out_hours = []
        for hour in hours:
            med, sigma = bucket[hour - 2]
            if console is not None:
                c_med, c_sigma = console[hour - 2]
                med = 0.5 * (med + c_med)
                sigma = max(sigma, c_sigma)
            out_hours.append({"hour": hour,
                              "q": hour_quantiles(v1, med, sigma, hour)})
        return {"hours": out_hours}
    except Exception:
        # Last resort: the fitted big-launch table, else the house shape.
        out_hours = []
        for hour in hours:
            if 2 <= hour <= 24:
                med, sigma = FALLBACK_BIG[hour - 2]
            else:
                med, sigma = 0.0, HOUSE_SIGMA * math.sqrt(max(1, hour - 1))
            out_hours.append({"hour": hour,
                              "q": hour_quantiles(v1, med, sigma, hour)})
        return {"hours": out_hours}


def persistence_forecast(v1, hours):
    """The house baseline shape: every hour stays at the hour-1 print."""
    return {"hours": [
        {"hour": h,
         "q": [round2(min(1e9, max(0.01,
                     v1 * math.exp(z * HOUSE_SIGMA * math.sqrt(h - 1)))))
               for z in Z]}
        for h in hours]}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # Every case file, in name order, so two runs write the same bytes.
    for path in sorted(INPUT_DIR.glob("*.in")):
        try:
            case = json.loads(path.read_text(encoding="utf-8"))
            forecast = forecast_case(case)
        except Exception:
            # A case we cannot even read still gets a valid-shaped file.
            forecast = persistence_forecast(
                1, [h for h in range(2, 25)])
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n",
            encoding="utf-8")


if __name__ == "__main__":
    main()
