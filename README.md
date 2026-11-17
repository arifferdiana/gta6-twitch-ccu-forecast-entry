# Forecast GTA VI's launch-day Twitch viewers hour by hour

[![make.fun](https://make.fun/badge/gta6-twitch-ccu-forecast-01.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01)
[![rank](https://make.fun/badge/gta6-twitch-ccu-forecast-01/arifferdiana/gta6-twitch-ccu-forecast-entry/rank.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01)
[![reward](https://make.fun/badge/gta6-twitch-ccu-forecast-01/arifferdiana/gta6-twitch-ccu-forecast-entry/reward.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01)

Entry for [Forecast GTA VI's launch-day Twitch viewers hour by hour](https://make.fun/bounty/gta6-twitch-ccu-forecast-01)
on [make.fun](https://make.fun).

## What it does

`main.py` (Python 3.13, standard library only) reads one case from
`/input/<id>.in` and writes `/output/<id>.out`: for each of hours 2–24 of
the release day (UTC), five quantiles (5%, 25%, 50%, 75%, 95%) of that
hour's Twitch concurrent viewers in the "Grand Theft Auto VI" category.

## Model

A launch's Twitch curve is not one smooth shape — it is split by *when*
the game unlocks relative to 00:00 UTC. Launches with a rolling
midnight-local unlock already have hours of play behind them at hour 1, so
hour 1 is large and the day only rises ~1–2.5× to an evening peak. Launches
with one simultaneous global unlock later in the day have a near-dead
hour 1 and explode 100–2000× at unlock.

So the program:

1. Takes the case's own comparables (past big launches) and, for each one,
   extracts its release-day hourly series.
2. Buckets launches by their hour-1 viewers (<5k, 5k–50k, ≥50k) and takes
   each bucket's per-hour median log-ratio `log(viewers_h / viewers_1)` and
   spread. The cuts are hard: the data are a step (unlocked vs
   not-yet-unlocked), not a gradient.
3. Blends in a second signal: console-only launches peak higher and
   tighter than the average launch. GTA VI is console-only with a rolling
   midnight unlock.
4. Forecasts `q_tau = v1 * exp(med_h + z_tau * sigma_h)` per hour, where
   `v1` is the target's hour-1 print.

Everything is recomputed from the case file at run time; a fitted
fallback table for the big-launch branch is embedded in case the
comparables cannot be parsed, and a persistence fallback is the last
resort. The program never crashes on a case — any failure still writes a
valid forecast file.

## Local validation

Backtested leave-one-out over the 37 usable comparable launches in the
kit (each as a pseudo-target, the other 36 as comparables), scoring with
the contest's log-scale pinball loss:

| model | all 37 | hour-1 ≥ 20k | hour-1 ≥ 50k | console-only |
|---|---|---|---|---|
| persistence (house baseline) | 0.930 | 0.321 | 0.251 | 0.323 |
| this model (3 buckets + console blend) | 0.486 | 0.265 | 0.226 | 0.202 |

The house baseline scores ~0.387 on the four practice targets; this model
is expected around 0.22–0.28 in the GTA regime (large hour-1, console-only,
rolling unlock).

Run it locally:

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=<kit>/input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

## License

MIT — see [LICENSE](LICENSE).
