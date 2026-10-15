"""
Phase 7: the full experimental campaign (every policy x every load pattern x repetitions), followed by the
analysis that produces the numbers the report cites.

    python -m inferno.experiments.run_full_campaign --config configs/experiment_config.yaml
    python -m inferno.experiments.report results/<run_dir>          # re-run analysis on existing data
"""
from .run_adaptive import build_parser, run_campaign


def main():
    args = build_parser("campaign").parse_args()
    run_campaign(args)


if __name__ == "__main__":
    main()
