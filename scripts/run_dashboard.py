"""Launch the artifact-backed ThermoTwin Streamlit dashboard."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

_DASHBOARD_PATH = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "thermotwin"
    / "visualization"
    / "dashboard.py"
)
_SOURCE_DIRECTORY = _DASHBOARD_PATH.parents[2]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--predictive-directory",
        required=True,
        help="directory containing comparison_summary.json",
    )
    parser.add_argument(
        "--fault-directory",
        required=True,
        help="fault run directory or parent containing fault-detection/",
    )
    parser.add_argument(
        "--evaluation-directory",
        default=None,
        help=(
            "optional final-evaluation export (or its evaluation/ folder) for the "
            "benchmark views; defaults to results/phase12-final-evaluation if present"
        ),
    )
    return parser


def _validate_directories(
    parser: argparse.ArgumentParser, predictive: Path, fault: Path
) -> tuple[Path, Path]:
    predictive = predictive.expanduser().resolve()
    fault = fault.expanduser().resolve()
    if not (predictive / "comparison_summary.json").is_file():
        parser.error(
            f"predictive directory must contain comparison_summary.json: {predictive}"
        )
    fault_summary = fault / "fault_summary.json"
    if not fault_summary.is_file():
        fault_summary = fault / "fault-detection" / "fault_summary.json"
    if not fault_summary.is_file():
        parser.error(
            "fault directory must contain fault_summary.json directly or under "
            f"fault-detection/: {fault}"
        )
    return predictive, fault


def main(argv: Sequence[str] | None = None) -> int:
    """Validate artifact roots and return Streamlit's child process status."""
    parser = _parser()
    arguments = parser.parse_args(argv)
    predictive, fault = _validate_directories(
        parser,
        Path(arguments.predictive_directory),
        Path(arguments.fault_directory),
    )
    environment = os.environ.copy()
    environment["THERMOTWIN_PREDICTIVE_RESULTS"] = str(predictive)
    environment["THERMOTWIN_FAULT_RESULTS"] = str(fault)
    if arguments.evaluation_directory is not None:
        evaluation = Path(arguments.evaluation_directory).expanduser().resolve()
        if not (
            (evaluation / "final_evaluation.json").is_file()
            or (evaluation / "evaluation" / "final_evaluation.json").is_file()
        ):
            parser.error(
                "evaluation directory must contain final_evaluation.json directly "
                f"or under evaluation/: {evaluation}"
            )
        environment["THERMOTWIN_EVALUATION_RESULTS"] = str(evaluation)
    existing_pythonpath = environment.get("PYTHONPATH")
    pythonpath_parts = [str(_SOURCE_DIRECTORY)]
    if existing_pythonpath:
        pythonpath_parts.append(existing_pythonpath)
    environment["PYTHONPATH"] = os.pathsep.join(pythonpath_parts)
    command = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(_DASHBOARD_PATH),
        "--server.headless=true",
        # Theme flags mirror .streamlit/config.toml so the launcher works from any
        # working directory.
        "--theme.base=dark",
        "--theme.primaryColor=#27C2D1",
        "--theme.backgroundColor=#071018",
        "--theme.secondaryBackgroundColor=#0D1822",
        "--theme.textColor=#F3F7F9",
        "--client.toolbarMode=minimal",
    ]
    result = subprocess.run(command, env=environment, check=False)
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
