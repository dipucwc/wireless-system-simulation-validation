"""
Simulation-configuration logging utility.

Each accepted numerical result should be traceable to:

- the executed source code;
- the selected configuration;
- the random seed;
- the stored CSV output.

This module writes every field of the dataclass configuration to a text
file beside the corresponding CSV. For example,

    results/python_results.csv

produces

    results/python_results_config.txt.

The log includes an ISO-8601 generation timestamp and preserves the exact
field values used by the run.
"""

from __future__ import annotations

import dataclasses
import datetime
import os
import platform
import subprocess
import sys
from pathlib import Path

import matplotlib
import numpy as np
import scipy


def _git_commit() -> str:
    """Return the current git commit hash, or the string not-available."""
    try:
        out = subprocess.run(
            ['git', 'rev-parse', 'HEAD'],
            cwd=Path(__file__).resolve().parent,
            capture_output=True, text=True, timeout=5,
        )
        return out.stdout.strip() if out.returncode == 0 else 'not available'
    except Exception:
        return 'not available'


def write_config_log(
    cfg,
    csv_path: str,
) -> str:
    """Write the complete dataclass configuration beside a CSV file.

    Parameters
    ----------
    cfg:
        Dataclass configuration object.

    csv_path:
        Path of the numerical CSV result whose configuration must be
        archived.

    Returns
    -------
    str
        Path of the generated configuration log.

    Raises
    ------
    TypeError
        If ``cfg`` is not a dataclass instance.
    """


    if not dataclasses.is_dataclass(cfg):
        raise TypeError(
            "cfg must be a dataclass instance."
        )


    csv_file = Path(csv_path)


    log_file = csv_file.with_name(
        f"{csv_file.stem}_config.txt"
    )


    log_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )


    generated_timestamp = (
        datetime.datetime.now().astimezone().isoformat()
    )


    with log_file.open(
        "w",
        encoding="utf-8",
    ) as file_handle:

        file_handle.write(
            f"Configuration log for {csv_file.name}\n"
        )


        file_handle.write(
            f"Generated: {generated_timestamp}\n\n"
        )


        # Environment provenance, so every stored result can be tied to
        # the exact software stack and source revision that produced it.
        file_handle.write(f"python_version      {sys.version}\n")
        file_handle.write(f"numpy_version       {np.__version__}\n")
        file_handle.write(f"scipy_version       {scipy.__version__}\n")
        file_handle.write(f"matplotlib_version  {matplotlib.__version__}\n")
        file_handle.write(f"platform            {platform.platform()}\n")
        file_handle.write(f"architecture        {platform.machine()}\n")
        file_handle.write(f"git_commit          {_git_commit()}\n\n")


        for field_definition in dataclasses.fields(cfg):

            field_value = getattr(
                cfg,
                field_definition.name,
            )


            file_handle.write(
                f"{field_definition.name:<24} {field_value}\n"
            )


    log_path = os.fspath(log_file)


    print(
        f"Saved config log:  {log_path}"
    )


    return log_path
