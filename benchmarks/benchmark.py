"""Times the estimation engine on a synthetic firm-year panel.

Run from the repository root with ``uv run python benchmarks/benchmark.py``.
"""

import time
import warnings
from datetime import datetime
from pathlib import Path

import pandas as pd
import specification_curve as sc

CONTROLS = ["size", "leverage", "rd", "size:leverage", "size:rd"]


def timed(sco: sc.SpecificationCurve, batch: bool = True, n_jobs: int = 1) -> float:
    """Seconds taken to fit a curve."""
    tic = time.perf_counter()
    sco.fit(progress=False, batch=batch, n_jobs=n_jobs)
    return time.perf_counter() - tic


def scenarios(df: pd.DataFrame) -> dict[str, float]:
    """Runs each timing scenario on one data set."""
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
    return {
        f"feols, {one_vcov.n_specs} specs, one fit each": timed(one_vcov, batch=False),
        f"feols, {one_vcov.n_specs} specs, batched": timed(one_vcov, batch=True),
        f"feols, {one_vcov.n_specs} specs, one fit each, 4 jobs": timed(
            one_vcov, batch=False, n_jobs=4
        ),
        f"feols, {one_vcov.n_specs} specs, batched, 4 jobs": timed(
            one_vcov, batch=True, n_jobs=4
        ),
        f"feols, {three_vcov.n_specs} specs (3 vcov), batched": timed(three_vcov),
        f"fepois, {poisson.n_specs} specs, one fit each": timed(poisson, batch=False),
        f"fepois, {poisson.n_specs} specs, batched": timed(poisson, batch=True),
        f"fit_null, {small.n_specs} specs, 50 resamples": null_time,
    }


def main() -> None:
    """Runs the benchmark at two data sizes and saves the timings."""
    warnings.simplefilter("ignore")
    columns = {}
    for n_firms in (500, 20_000):
        df = sc.load_example_panel(n_firms=n_firms, n_years=10)
        columns[f"{len(df):,} rows"] = scenarios(df)
    table = pd.DataFrame(columns).round(2)
    print(table.to_string())
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = Path("outputs") / f"benchmark_time_{stamp}.txt"
    out.parent.mkdir(exist_ok=True)
    out.write_text(table.to_string() + "\n")


if __name__ == "__main__":
    main()
