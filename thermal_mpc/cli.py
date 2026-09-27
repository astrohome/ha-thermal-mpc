"""Command-line entry point: ``thermal-mpc export|fit``."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .config import Config
from .ha_history import export
from .model import fit, validate


def _load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, index_col="time")
    df.index = pd.to_datetime(df.index, utc=True)
    return df


def cmd_export(args: argparse.Namespace) -> None:
    """Download history and write the resampled dataset."""
    config = Config.load(args.config)
    end = pd.Timestamp.now(tz="UTC").floor(config.step)
    start = end - pd.Timedelta(days=args.days)
    df = export(config, start, end)
    if args.append and Path(args.out).exists():
        old = _load_csv(args.out)
        df = pd.concat([old, df])
        df = df[~df.index.duplicated(keep="last")].sort_index()
    df.to_csv(args.out)
    complete = df.notna().all(axis=1).mean()
    print(f"wrote {len(df)} rows to {args.out} ({complete:.0%} complete rows)")
    print(df.notna().mean().rename("coverage").to_string(float_format="{:.0%}".format))


def cmd_fit(args: argparse.Namespace) -> None:
    """Fit on the first part of the data and validate on the rest."""
    config = Config.load(args.config)
    df = _load_csv(args.data)
    split = int(len(df) * (1 - args.holdout))
    train, test = df.iloc[:split], df.iloc[split:]
    model = fit(train, config)
    model.to_json(args.out)

    print(f"model written to {args.out}\n")
    for name, p in model.rooms.items():
        print(f"[{name}]  n={p.n_samples}  one-step RMSE={p.rmse_one_step:.3f} K")
        print(f"  tau_out = {p.tau_out_h:.1f} h   offset = {p.offset:+.3f} K/h")
        for other, g in p.g_rooms.items():
            if g > 1e-6:
                print(f"  couples to {other:<16} tau = {1 / g:.1f} h")
        for inp, b in p.gains.items():
            print(f"  gain {inp:<20} {b:+.3f} K/h per unit")
    if len(test) > 0:
        rmse = validate(model, test, horizon=pd.Timedelta(hours=args.horizon))
        print(f"\nhold-out open-loop RMSE (K), last {args.holdout:.0%} of data:")
        picks = rmse.iloc[[i for i in (0, 11, 35, 71) if i < len(rmse)] + [-1]]
        print(picks.drop_duplicates().to_string(float_format="{:.2f}".format))


def main(argv: list[str] | None = None) -> None:
    """Run the CLI."""
    parser = argparse.ArgumentParser(prog="thermal-mpc")
    parser.add_argument("-c", "--config", default="config.yaml")
    sub = parser.add_subparsers(required=True)

    p = sub.add_parser("export", help="download HA history to CSV")
    p.add_argument("--days", type=float, default=10)
    p.add_argument("--out", default="data/history.csv")
    p.add_argument("--append", action="store_true", help="merge into existing CSV")
    p.set_defaults(func=cmd_export)

    p = sub.add_parser("fit", help="fit and validate the thermal model")
    p.add_argument("--data", default="data/history.csv")
    p.add_argument("--out", default="model.json")
    p.add_argument("--holdout", type=float, default=0.2)
    p.add_argument("--horizon", type=float, default=6, help="validation hours")
    p.set_defaults(func=cmd_fit)

    args = parser.parse_args(argv)
    if getattr(args, "out", None):
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    args.func(args)


if __name__ == "__main__":
    main()
