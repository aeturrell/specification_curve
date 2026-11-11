"""Times the estimation engine on a synthetic firm-year panel.

Run from the repository root with ``uv run python benchmarks/benchmark.py``.

Each run appends its timings to ``benchmarks/results.csv``, together with the
date, the package and pyfixest versions, and a description of the machine. The
file is tracked in git, so it doubles as a log of how performance has changed.
The timings table on the features page of the docs is built from the most
recent run in this file.
"""

import os
import platform
import subprocess
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pyfixest

import specification_curve as sc

RESULTS = Path(__file__).parent / "results.csv"
CONTROLS = ["size", "leverage", "rd", "size:leverage", "size:rd"]


def timed(sco: sc.SpecificationCurve, batch: bool = True, n_jobs: int = 1) -> float:
    """Seconds taken to fit a curve."""
    tic = time.perf_counter()
    sco.fit(progress=False, batch=batch, n_jobs=n_jobs)
    return time.perf_counter() - tic


def scenarios(df: pd.DataFrame) -> list[dict]:
    """Runs each timing scenario on one data set.

    Returns one record per scenario, with a stable ``scenario`` id, a
    human-readable ``description``, the number of specifications and the time
    taken in seconds.
    """
    outcomes = ["log(sales)", "sales"]
    fixed_effects = [None, "firm", "firm + year"]
    one_vcov = sc.SpecificationCurve(
        df, y=outcomes, x="treated", controls=CONTROLS, fe=fixed_effects
    )
    three_vcov = sc.SpecificationCurve(
        df,
        y=outcomes,
        x="treated",
        controls=CONTROLS,
        fe=fixed_effects,
        vcov=["iid", "hetero", {"CRV1": "firm"}],
    )
    poisson = sc.SpecificationCurve(
        df,
        y="patents",
        x="treated",
        controls=CONTROLS,
        fe=[None, "firm"],
        model="fepois",
    )
    small = sc.SpecificationCurve(
        df,
        y="log(sales)",
        x="treated",
        controls=CONTROLS[:3],
        fe=["firm", "firm + year"],
    ).fit(progress=False)
    tic = time.perf_counter()
    small.fit_null(n_boot=50, cluster="firm", seed=0, progress=False)
    null_time = time.perf_counter() - tic

    runs = [
        ("feols_single", "feols, one fit each", one_vcov, timed(one_vcov, batch=False)),
        ("feols_batched", "feols, batched", one_vcov, timed(one_vcov, batch=True)),
        (
            "feols_single_jobs4",
            "feols, one fit each, 4 jobs",
            one_vcov,
            timed(one_vcov, batch=False, n_jobs=4),
        ),
        (
            "feols_batched_jobs4",
            "feols, batched, 4 jobs",
            one_vcov,
            timed(one_vcov, batch=True, n_jobs=4),
        ),
        (
            "feols_3vcov_batched",
            "feols, 3 vcov, batched",
            three_vcov,
            timed(three_vcov),
        ),
        ("fepois_single", "fepois, one fit each", poisson, timed(poisson, batch=False)),
        ("fepois_batched", "fepois, batched", poisson, timed(poisson, batch=True)),
        ("fit_null_50", "fit_null, 50 resamples", small, null_time),
    ]
    return [
        {
            "scenario": scenario,
            "description": description,
            "n_specs": sco.n_specs,
            "seconds": round(seconds, 3),
        }
        for scenario, description, sco, seconds in runs
    ]


def cpu_name() -> str:
    """A short description of the processor, without identifying the machine."""
    try:
        if platform.system() == "Darwin":
            return subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip()
        if platform.system() == "Linux":
            for line in Path("/proc/cpuinfo").read_text().splitlines():
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except (OSError, subprocess.SubprocessError):
        pass
    return platform.processor() or platform.machine()


def main() -> None:
    """Runs the benchmark at two data sizes and appends the timings to the log."""
    warnings.simplefilter("ignore")
    run_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    meta = {
        "run_at": run_at,
        "package_version": sc.__version__,
        "pyfixest_version": pyfixest.__version__,
        "python_version": platform.python_version(),
        "os": platform.system(),
        "cpu": cpu_name(),
        "cpu_count": os.cpu_count(),
    }
    records = []
    for n_firms in (500, 20_000):
        df = sc.load_example_panel(n_firms=n_firms, n_years=10)
        for record in scenarios(df):
            records.append({**meta, "n_rows": len(df), **record})
    new = pd.DataFrame(records)

    table = new.pivot_table(
        index=["description", "n_specs"], columns="n_rows", values="seconds", sort=False
    )
    print(table.round(2).to_string())

    if RESULTS.exists():
        new = pd.concat([pd.read_csv(RESULTS), new], ignore_index=True)
    new.to_csv(RESULTS, index=False)
    print(f"\nAppended {len(records)} timings to {RESULTS}")


if __name__ == "__main__":
    main()
