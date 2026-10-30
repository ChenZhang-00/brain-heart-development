"""Preflight and resumable SDG computation using preprocessed EEG and reviewed R peaks."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config.dbhi_config import make_parser, paths_from_args


def main(argv=None):
    parser = make_parser(__doc__, ("participants", "cache", "reports_dir"))
    parser.add_argument(
        "--subject",
        action="append",
        help="Process only this ID; repeat for multiple IDs. Default: every participant-table ID.",
    )
    parser.add_argument("--n-jobs", type=int, default=4)
    parser.add_argument(
        "--check-only", action="store_true", help="Write input QC without computing model arrays."
    )
    args = parser.parse_args(argv)
    paths = paths_from_args(args)

    import copy
    from analysis.inputs import read_participants
    from sdg.pipeline import (
        CONT_DIR,
        RPEAK_DIR,
        DEFAULT_CONFIG,
        preflight,
        run_all,
        write_csv,
    )

    ids = args.subject or read_participants(paths.participants, full_cohort=False).index.tolist()
    if (
        not ids
        or len(ids) != len(set(ids))
        or any(
            not s.strip() or s != s.strip() or "/" in s or "\\" in s or s in (".", "..")
            for s in ids
        )
    ):
        raise ValueError("Provide nonempty, unique participant IDs without path separators.")
    config = copy.deepcopy(DEFAULT_CONFIG)
    if args.check_only:
        rows = preflight(ids, config)
        write_csv(paths.reports / "input_qc.csv", rows)
        flagged = [r for r in rows if r["status"] != "ok"]
        for row in flagged:
            print(row)
        if any(r["status"] == "failed" for r in rows):
            raise RuntimeError(f"Input checks failed; see {paths.reports / 'input_qc.csv'}")
        print(f"Input checks passed for {len(ids)} participant(s); {len(flagged)} flagged clip(s).")
        return
    run_all(ids, paths.cache, paths.reports, config, n_jobs=args.n_jobs)


if __name__ == "__main__":
    main()
