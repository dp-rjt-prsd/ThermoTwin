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
    command = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(_DASHBOARD_PATH),
        "--server.headless=true",
    ]
    result = subprocess.run(command, env=environment, check=False)
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
