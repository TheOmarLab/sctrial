"""Participant relabeling: permutation tests and the calibration of the tests they check.

One routine serves every participant-level permutation in the manuscript, so that a
figure and a table cannot report different nulls for the same test:

* two-group designs (TNBC arms, melanoma response groups, COVID-19 severity): the group
  labels are reassigned across participants, keeping the group sizes;
* single-arm paired designs (vaccination, AML, CAR-T): the two visits are swapped
  within participants.

Every assignment is enumerated when there are at most ``MAX_EXACT`` of them. The
permutation p-value is then exact and has no seed. Larger designs draw ``N_DRAWS``
assignments from a seeded generator and use ``(b + 1) / (B + 1)``.

For two-group difference-in-differences the same pass also records, for every
assignment, the analytical test and the wild cluster bootstrap that the clinical
analyses report. Under reassignment the labels carry no information, so the share of
assignments in which a test rejects is its false-positive rate on the cohort at hand.
Each fit goes through ``sctrial.did_fit``, the function behind ``did_table``, and the
observed assignment is checked against ``did_table`` itself.
"""

from __future__ import annotations

import hashlib
import itertools
import os
import pickle
import warnings
from math import comb
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from sctrial.stats.comparisons import between_arm_fit_beta, within_arm_fit_beta
from sctrial.stats.did import did_fit, did_table, get_did_aggregated_df

from ._shared import STATS_FINGERPRINT

MAX_EXACT = 10_000
N_DRAWS = 9_999
SEED = 42
N_BOOT = 999

_VERSION = "v1"
_CACHE_DIR = Path(__file__).resolve().parent / "_cache"


def n_jobs() -> int:
    """Workers for the assignment loops: the SLURM allocation, else all but one core."""
    return max(1, int(os.environ.get("SLURM_CPUS_PER_TASK") or (os.cpu_count() or 2) - 1))


def group_assignments(
    n_units: int, n_first: int, *, max_exact: int = MAX_EXACT, n_draws: int = N_DRAWS,
    seed: int = SEED,
) -> tuple[np.ndarray, bool]:
    """Assignments of ``n_first`` of ``n_units`` participants to the first group.

    Returns ``(assignments, exact)``: an integer array with one row per assignment and
    one 0/1 column per participant, and whether the rows are all the assignments.
    """
    if not 0 < n_first < n_units:
        raise ValueError("both groups need at least one participant")
    if comb(n_units, n_first) <= max_exact:
        out = np.zeros((comb(n_units, n_first), n_units), dtype=np.int8)
        for row, members in enumerate(itertools.combinations(range(n_units), n_first)):
            out[row, list(members)] = 1
        return out, True
    rng = np.random.default_rng(seed)
    base = np.r_[np.ones(n_first, dtype=np.int8), np.zeros(n_units - n_first, dtype=np.int8)]
    return np.stack([rng.permutation(base) for _ in range(n_draws)]), False


def swap_assignments(
    n_units: int, *, max_exact: int = MAX_EXACT, n_draws: int = N_DRAWS, seed: int = SEED,
) -> tuple[np.ndarray, bool]:
    """Assignments of a visit swap (1) or no swap (0) to each of ``n_units`` participants."""
    if n_units < 1:
        raise ValueError("at least one participant is required")
    if n_units < 31 and 2**n_units <= max_exact:
        codes = np.arange(2**n_units)
        return ((codes[:, None] >> np.arange(n_units)) & 1).astype(np.int8), True
    rng = np.random.default_rng(seed)
    return rng.integers(0, 2, size=(n_draws, n_units), dtype=np.int8), False


def permutation_summary(null: np.ndarray, observed: float, *, exact: bool) -> dict:
    """Two-sided permutation p-value and the null summary reported beside it.

    ``null`` holds the statistic under every assignment when ``exact`` (the observed
    assignment is one of them), or under the random draws otherwise.
    """
    null = np.asarray(null, dtype=float)
    null = null[np.isfinite(null)]
    if null.size == 0 or not np.isfinite(observed):
        return {"permutation_p": np.nan, "n_permutations": int(null.size), "exact": exact,
                "null_mean": np.nan, "null_sd": np.nan, "null_95_lo": np.nan, "null_95_hi": np.nan}
    as_extreme = int(np.sum(np.abs(null) >= abs(observed) - 1e-9 * max(1.0, abs(observed))))
    p = as_extreme / null.size if exact else (as_extreme + 1) / (null.size + 1)
    return {
        "permutation_p": float(p),
        "n_permutations": int(null.size),
        "exact": bool(exact),
        "null_mean": float(np.mean(null)),
        "null_sd": float(np.std(null)),
        "null_95_lo": float(np.percentile(null, 2.5)),
        "null_95_hi": float(np.percentile(null, 97.5)),
    }


def _chunks(n_rows: int, n_chunks: int) -> list[np.ndarray]:
    return [c for c in np.array_split(np.arange(n_rows), max(1, min(n_chunks, n_rows))) if c.size]


def _did_chunk(df_use, features, unit, time, arm_bin, position, assignments, rows,
               inference, n_boot, seed) -> list[dict]:
    out = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for row in rows:
            frame = df_use.copy()
            frame[arm_bin] = assignments[row][position].astype(int)
            for feat in features:
                fit = did_fit(frame, y=feat, unit=unit, time=time, arm_bin=arm_bin, standardize=True)
                rec = {"assignment": int(row), "feature": feat,
                       "beta_DiD": fit["beta_DiD"], "se_DiD": fit["se_DiD"],
                       "p_analytic": fit["p_DiD"], "cov_type_used": fit.get("cov_type_used")}
                if inference:
                    boot = did_fit(frame, y=feat, unit=unit, time=time, arm_bin=arm_bin,
                                   standardize=True, use_bootstrap=True, n_boot=n_boot, seed=seed)
                    rec.update(p_boot=boot.get("p_DiD_boot", np.nan),
                               ci_lo_boot=boot.get("ci_lo_boot", np.nan),
                               ci_hi_boot=boot.get("ci_hi_boot", np.nan))
                out.append(rec)
    return out


def did_relabeling(
    df_use: pd.DataFrame, features: list[str], unit: str, time: str, arm_bin: str, *,
    inference: bool = True, n_boot: int = N_BOOT, seed: int = SEED,
    max_exact: int = MAX_EXACT, n_draws: int = N_DRAWS,
) -> pd.DataFrame:
    """Refit the two-group DiD under every reassignment of the participants.

    ``df_use`` is the participant-visit frame of ``get_did_aggregated_df``. Returns one
    row per assignment and feature with ``beta_DiD``, the analytical standard error and
    p-value, and, with ``inference``, the wild cluster bootstrap p-value and interval.
    ``observed`` marks the assignment of the data, which is among the rows whenever the
    assignments are enumerated. ``attrs["exact"]`` says whether they are.
    """
    ids = sorted(df_use[unit].astype(str).unique())
    position = df_use[unit].astype(str).map({u: i for i, u in enumerate(ids)}).to_numpy()
    observed = (df_use.assign(_u=df_use[unit].astype(str)).groupby("_u")[arm_bin].first()
                .reindex(ids).to_numpy().astype(np.int8))
    assignments, exact = group_assignments(len(ids), int(observed.sum()), max_exact=max_exact,
                                           n_draws=n_draws, seed=seed)
    jobs = n_jobs()
    parts = Parallel(n_jobs=jobs, backend="loky")(
        delayed(_did_chunk)(df_use, list(features), unit, time, arm_bin, position, assignments,
                            rows, inference, n_boot, seed)
        for rows in _chunks(len(assignments), jobs * 4)
    )
    table = pd.DataFrame([rec for part in parts for rec in part])
    is_observed = (assignments == observed).all(axis=1)
    table["observed"] = is_observed[table["assignment"].to_numpy()]
    table = table.sort_values(["assignment", "feature"], kind="stable").reset_index(drop=True)
    table.attrs.update(exact=exact, n_assignments=int(len(assignments)), n_units=len(ids),
                       n_first=int(observed.sum()))
    return table


def _cache_path(tag: str, *parts) -> Path:
    key = hashlib.md5("|".join([_VERSION, STATS_FINGERPRINT, *map(str, parts)]).encode()).hexdigest()[:12]  # noqa: S324
    return _CACHE_DIR / f"relabeling_{tag}_{key}.pkl"


def did_relabeling_cached(
    adata, features: list[str], design, visits: tuple[str, str], *, layer: str | None,
    tag: str, inference: bool = True, n_boot: int = N_BOOT, seed: int = SEED,
) -> pd.DataFrame:
    """``did_relabeling`` for a scored AnnData, cached on disk.

    The cache key holds the statistics fingerprint, a digest of the participant-visit
    values and the bootstrap settings, so every module that asks for the same cohort
    reads the same table. On a fresh computation the observed assignment is compared
    with ``did_table`` on the AnnData, which is how the clinical results are produced.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        df_use, unit, time, arm_bin = get_did_aggregated_df(
            adata, features, design, visits, layer=layer, aggregate="participant_visit")
    content = df_use[[unit, time, arm_bin, *features]].astype({unit: str})
    digest = hashlib.md5(  # noqa: S324
        pd.util.hash_pandas_object(content, index=False).to_numpy().tobytes()).hexdigest()
    path = _cache_path(tag, digest, list(features), inference, n_boot, seed, MAX_EXACT, N_DRAWS)
    if path.exists():
        # A cache this module wrote for the same participant-visit values; not external input.
        with open(path, "rb") as fh:
            payload = pickle.load(fh)  # noqa: S301
        table = payload["table"]
        table.attrs.update(payload["attrs"])
        print(f"  Relabeling {tag}: {table.attrs['n_assignments']:,} assignments (cached)")
        return table

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        reference = did_table(adata, features=features, design=design, visits=visits, layer=layer,
                              standardize=True, aggregate="participant_visit",
                              use_bootstrap=inference, n_boot=n_boot, seed=seed)
    table = did_relabeling(df_use, list(features), unit, time, arm_bin, inference=inference,
                           n_boot=n_boot, seed=seed)
    print(f"  Relabeling {tag}: {table.attrs['n_assignments']:,} assignments of "
          f"{table.attrs['n_units']} participants ({table.attrs['n_first']} in the first group), "
          f"{'all enumerated' if table.attrs['exact'] else 'random draws'}")
    check_observed_assignment(table, reference, inference=inference)

    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f".{os.getpid()}.tmp")
    with open(tmp, "wb") as fh:
        pickle.dump({"table": table, "attrs": dict(table.attrs)}, fh, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, path)
    return table


def check_observed_assignment(table: pd.DataFrame, reference: pd.DataFrame, *, inference: bool,
                              atol: float = 1e-10) -> None:
    """Require the observed-assignment rows to reproduce ``did_table`` on the AnnData."""
    if not table.attrs.get("exact", False):
        return
    obs = table[table["observed"]].set_index("feature")
    ref = reference.set_index("feature").loc[obs.index]
    pairs = [("beta_DiD", "beta_DiD"), ("se_DiD", "se_DiD")]
    if inference:
        pairs += [("p_boot", "p_DiD_boot"), ("ci_lo_boot", "ci_lo_boot"), ("ci_hi_boot", "ci_hi_boot")]
    else:
        pairs += [("p_analytic", "p_DiD")]
    for mine, theirs in pairs:
        a = obs[mine].to_numpy(float)
        b = ref[theirs].to_numpy(float)
        same = np.isclose(a, b, rtol=0, atol=atol) | (np.isinf(a) & (a == b))
        if not same.all():
            worst = obs.index[~same][0]
            raise AssertionError(
                f"relabeling does not reproduce did_table for {worst}: {mine} "
                f"{obs.loc[worst, mine]!r} versus {theirs} {ref.loc[worst, theirs]!r}")


def did_permutation_nulls(table: pd.DataFrame) -> dict[str, tuple[np.ndarray, float]]:
    """For each feature of a relabeling table, the null of ``beta_DiD`` and its observed value."""
    out = {}
    for feat, grp in table.groupby("feature", sort=False):
        obs = grp.loc[grp["observed"], "beta_DiD"]
        if len(obs) != 1:
            raise ValueError(f"{feat}: expected one observed assignment, found {len(obs)}")
        out[feat] = (grp["beta_DiD"].to_numpy(float), float(obs.iloc[0]))
    return out


def calibration_rates(table: pd.DataFrame, *, alpha: float = 0.05) -> pd.DataFrame:
    """Share of assignments in which each test rejects at ``alpha``, per feature.

    Also the share of assignments whose bootstrap interval contains zero. The interval
    level is the bootstrap's own (95%), whatever ``alpha`` is.
    """
    for col in ("p_analytic", "p_boot", "ci_lo_boot", "ci_hi_boot"):
        if table[col].isna().any():
            raise ValueError(f"relabeling table has missing values in {col}")
    flags = pd.DataFrame({
        "feature": table["feature"],
        "reject_analytic": table["p_analytic"] < alpha,
        "reject_boot": table["p_boot"] < alpha,
        "zero_in_interval": (table["ci_lo_boot"] <= 0) & (table["ci_hi_boot"] >= 0),
    })
    return flags.groupby("feature", sort=False).mean().reset_index()


def _within_chunk(df_use, features, unit, position, assignments, rows) -> list[list[float]]:
    out = []
    visit = df_use["visit_num"].to_numpy()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for row in rows:
            frame = df_use.copy()
            swap = assignments[row][position].astype(bool)
            frame["visit_num"] = np.where(swap, 1 - visit, visit)
            out.append([within_arm_fit_beta(frame, feat, unit, standardize=True) for feat in features])
    return out


def within_arm_nulls(
    df_use: pd.DataFrame, features: list[str], unit: str, *, max_exact: int = MAX_EXACT,
    n_draws: int = N_DRAWS, seed: int = SEED,
) -> tuple[dict[str, np.ndarray], bool]:
    """Null of the within-arm ``beta_time`` under visit swaps within participants.

    ``df_use`` is the participant-visit frame of ``get_within_arm_aggregated_df`` (two
    visits per participant, ``visit_num`` 0 and 1). Returns the null per feature and
    whether every swap assignment was enumerated.
    """
    counts = df_use.groupby(unit, observed=True)["visit_num"].agg(["size", "nunique"])
    if not ((counts["size"] == 2) & (counts["nunique"] == 2)).all():
        raise ValueError("every participant needs exactly one row per visit")
    ids = sorted(df_use[unit].astype(str).unique())
    position = df_use[unit].astype(str).map({u: i for i, u in enumerate(ids)}).to_numpy()
    assignments, exact = swap_assignments(len(ids), max_exact=max_exact, n_draws=n_draws, seed=seed)
    jobs = n_jobs()
    parts = Parallel(n_jobs=jobs, backend="loky")(
        delayed(_within_chunk)(df_use, list(features), unit, position, assignments, rows)
        for rows in _chunks(len(assignments), jobs * 4)
    )
    null = np.array([rec for part in parts for rec in part], dtype=float)
    return {feat: null[:, j] for j, feat in enumerate(features)}, exact


def _between_chunk(df_cs, features, design, assignments, rows) -> list[list[float]]:
    out = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for row in rows:
            frame = df_cs.copy()
            frame["arm_bin"] = assignments[row].astype(int)
            out.append([between_arm_fit_beta(frame, feat, design) for feat in features])
    return out


def between_arm_nulls(
    df_cs: pd.DataFrame, features: list[str], design, *, max_exact: int = MAX_EXACT,
    n_draws: int = N_DRAWS, seed: int = SEED,
) -> tuple[dict[str, np.ndarray], bool]:
    """Null of the between-arm ``beta_arm`` under reassignment of the group labels.

    ``df_cs`` is the one-row-per-participant frame of ``get_between_arm_aggregated_df``.
    """
    if df_cs[design.participant_col].duplicated().any():
        raise ValueError("one row per participant is required")
    observed = df_cs["arm_bin"].to_numpy().astype(np.int8)
    assignments, exact = group_assignments(len(observed), int(observed.sum()), max_exact=max_exact,
                                           n_draws=n_draws, seed=seed)
    jobs = n_jobs()
    parts = Parallel(n_jobs=jobs, backend="loky")(
        delayed(_between_chunk)(df_cs, list(features), design, assignments, rows)
        for rows in _chunks(len(assignments), jobs * 4)
    )
    null = np.array([rec for part in parts for rec in part], dtype=float)
    return {feat: null[:, j] for j, feat in enumerate(features)}, exact
