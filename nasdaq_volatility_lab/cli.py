"""Command-line interface; domain modules handle data, fitting and reporting."""

import argparse
import sys
from pathlib import Path

from .data import fetch_data
from .models import TrainingOptions
from .reporting import METRIC_LABELS, plot_results, render_results
from .storage import load_run
from .training import train


def build_parser():
    parser = argparse.ArgumentParser(
        prog="nasdaq-volatility-lab",
        description="Fetch QQQ/SPY data, train volatility models, and compare saved results.",
    )
    parser.add_argument(
        "--project-dir",
        type=Path,
        default=Path.cwd(),
        help="Data/model/result workspace (default: current directory).",
    )
    parser.add_argument("--debug", action="store_true", help="Show tracebacks on failure.")
    commands = parser.add_subparsers(dest="command", required=True)
    fetch = commands.add_parser(
        "fetch-data", help="Refresh adjusted prices and the validated feature table."
    )
    fetch.add_argument(
        "--start-date",
        default="2005-01-01",
        help="First feature date, inclusive (default: 2005-01-01).",
    )
    fetch.add_argument(
        "--end-date",
        default="2025-12-31",
        help="Last feature date, inclusive (default: 2025-12-31).",
    )
    fit = commands.add_parser("train", help="Fit models using an existing feature table.")
    fit.add_argument("--model", choices=["baseline", "ridge", "xgboost", "all"], default="all")
    fit.add_argument(
        "--validation-start",
        default="2017-01-01",
        help="Inclusive validation start (default: 2017-01-01).",
    )
    fit.add_argument(
        "--validation-end",
        default="2018-01-01",
        help="Exclusive validation end (default: 2018-01-01).",
    )
    fit.add_argument(
        "--alpha", type=float, default=1.0, help="Positive Ridge penalty (default: 1)."
    )
    fit.add_argument(
        "--learning-rate", type=float, default=0.05, help="XGBoost shrinkage (default: 0.05)."
    )
    fit.add_argument(
        "--n-estimators",
        type=int,
        default=1000,
        help="Maximum XGBoost rounds per candidate before early stopping (default: 1000).",
    )
    fit.add_argument(
        "--early-stopping-rounds", type=int, default=30, help="Inner MAE patience (default: 30)."
    )
    fit.add_argument("--max-depth", type=int, help="Fix depth instead of searching depths 2 and 3.")
    fit.add_argument(
        "--reg-lambda", type=float, help="Fix the XGBoost L2 penalty instead of searching 1 and 10."
    )
    show = commands.add_parser(
        "show-results", help="Display a completed run; default is the latest run."
    )
    show.add_argument("--run-id", help="Run ID printed by train.")
    show.add_argument(
        "--metric",
        choices=list(METRIC_LABELS),
        help="Show one metric, sorted by validation performance.",
    )
    show.add_argument(
        "--plot", action="store_true", help="Save a validation prediction comparison PNG."
    )
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    root = args.project_dir.expanduser().resolve()
    try:
        if args.command == "fetch-data":
            schema = fetch_data(root, args.start_date, args.end_date)
            print(
                f"Data ready | rows={schema['row_count']} labeled={schema['labeled_row_count']} dates={'..'.join(schema['date_range'])} features={','.join(schema['feature_columns'])}"
            )
        elif args.command == "train":
            options = TrainingOptions(
                **{name: getattr(args, name) for name in TrainingOptions.__dataclass_fields__}
            )
            record = train(
                root,
                args.model,
                validation_start=args.validation_start,
                validation_end=args.validation_end,
                options=options,
            )
            for item in record["models"]:
                params = item["parameters"]
                if item["model"] == "baseline":
                    detail = f"constant={params['constant'] * 100:.4f}%"
                elif item["model"] == "ridge":
                    detail = f"alpha={params['alpha']:.4f}"
                else:
                    detail = f"learning_rate={params['learning_rate']:.4f} n_estimators={params['n_estimators']} max_depth={params['max_depth']} reg_lambda={params['reg_lambda']:.4f}"
                print(
                    f"Trained {item['model']} | run={record['run_id']} train={item['train_rows']} val={item['val_rows']} time={item['training_seconds']:.4f}s {detail}"
                )
        else:
            record = load_run(root, args.run_id)
            split = record["split"]
            print(
                f"Run {record['run_id']} | train={split['train_rows']} val={split['evaluation_rows']} validation={split['evaluation_start']}..{split['evaluation_end']}"
            )
            print(render_results(record, args.metric))
            if args.plot:
                path = plot_results(root, record)
                print(f"Plot saved | {path}")
    except Exception as exc:
        # The CLI is the exception boundary: keep normal failures concise while
        # --debug preserves the original traceback for diagnosis.
        if args.debug:
            raise
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0
