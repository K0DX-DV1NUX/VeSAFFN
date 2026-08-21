import argparse
import re
import shutil
from pathlib import Path

# Matches the setting string runExp.py builds:
# '{d_model}-{d_data}-ft{d_forecast_type}-sl{d_seq_len}-pl{d_pred_len}-seed{d_seed}'
SETTING_RE = re.compile(
    r'^(?P<model>.+?)-(?P<data>.+?)-ft(?P<forecast>[A-Za-z]+)-sl(?P<seq>\d+)-pl(?P<pred>\d+)-seed(?P<seed>\d+)$'
)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Promotes checkpoints from a normal (non-KD) runExp.py training run into the "
            "per-(dataset, pred_len) layout Exp_KD_Offline's fixed teachers read from: "
            "scripts/<Model>/pre_trained/<Dataset>/pl<PredLen>/checkpoint.pth"
        )
    )
    parser.add_argument("--model", required=True, help="Teacher model name, e.g. PatchTST.")
    parser.add_argument("--checkpoints_dir", default="checkpoints",
                         help="Root directory holding checkpoints/<setting>/checkpoint.pth (default: ./checkpoints).")
    parser.add_argument("--seed", type=int, default=2021,
                         help="Only promote checkpoints trained with this seed -- offline KD uses one "
                              "fixed teacher, so only a single seed's checkpoint is ever needed here "
                              "(default: 2021, matching `runExp.py ... --seed 2021`).")
    parser.add_argument("--dry_run", action="store_true", help="Print what would be copied without copying.")
    args = parser.parse_args()

    checkpoints_root = Path(args.checkpoints_dir)
    if not checkpoints_root.is_dir():
        raise FileNotFoundError(f"No such checkpoints directory: '{checkpoints_root}'")

    promoted = 0
    skipped_no_checkpoint = []

    for setting_dir in sorted(checkpoints_root.iterdir()):
        if not setting_dir.is_dir():
            continue

        match = SETTING_RE.match(setting_dir.name)
        if not match or match.group("model") != args.model or int(match.group("seed")) != args.seed:
            continue

        src = setting_dir / "checkpoint.pth"
        if not src.exists():
            skipped_no_checkpoint.append(setting_dir.name)
            continue

        dataset = match.group("data")
        pred_len = match.group("pred")
        dest_dir = Path("scripts") / args.model / "pre_trained" / dataset / f"pl{pred_len}"
        dest = dest_dir / "checkpoint.pth"

        print(f"{src} -> {dest}")
        if not args.dry_run:
            dest_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dest)
        promoted += 1

    for name in skipped_no_checkpoint:
        print(f"skipped '{name}': matched {args.model}/seed{args.seed} but has no checkpoint.pth (training incomplete?)")

    verb = "Would promote" if args.dry_run else "Promoted"
    print(f"{verb} {promoted} checkpoint(s) for {args.model} (seed {args.seed}).")


if __name__ == "__main__":
    main()
