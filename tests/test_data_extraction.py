"""Sanity tests for the extracted CSVs."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

DATA = Path(__file__).resolve().parent.parent / "data"


def _maybe_skip(*names):
    missing = [n for n in names if not (DATA / n).exists()]
    if missing:
        pytest.skip(f"Missing CSV(s): {missing}. Run `make data` first.")


@pytest.mark.parametrize("group", ["control", "cka", "cos"])
def test_cka_matrix_is_symmetric_with_unit_diagonal(group: str):
    name = f"cka_matrix_{group}.csv"
    _maybe_skip(name)
    M = pd.read_csv(DATA / name, index_col=0).values
    assert M.shape == (30, 30)
    assert np.allclose(np.diag(M), 1.0, atol=1e-5)
    assert np.allclose(M, M.T, atol=1e-4)


def test_cka_penalty_actually_reduces_pairwise_cka():
    """CKA penalty should drop mean off-diagonal pairwise CKA at least 10x vs control."""
    _maybe_skip("cka_matrix_control.csv", "cka_matrix_cka.csv")
    control = pd.read_csv(DATA / "cka_matrix_control.csv", index_col=0).values
    cka = pd.read_csv(DATA / "cka_matrix_cka.csv", index_col=0).values
    ctrl_off = control[np.triu_indices_from(control, k=1)]
    cka_off  = cka[np.triu_indices_from(cka, k=1)]
    assert ctrl_off.mean() > 0.7, f"unexpected control off-diag mean {ctrl_off.mean():.3f}"
    assert cka_off.mean() < 0.20, f"unexpected CKA off-diag mean {cka_off.mean():.3f}"

    assert ctrl_off.mean() / cka_off.mean() > 10


@pytest.mark.parametrize("group", ["control", "cka", "cos"])
def test_scaling_tables_have_expected_columns_and_30_rows(group: str):
    bench = "mmlu"
    name = f"scaling_{bench}_{group}.csv"
    _maybe_skip(name)
    df = pd.read_csv(DATA / name)
    assert len(df) == 30
    expected_cols = {"Models", "BestSingleAcc", "OracleAcc",
                     "HardVoteAcc", "SoftVoteAcc"}
    assert expected_cols.issubset(df.columns)


def test_oracle_monotone_nondecreasing():
    """Oracle accuracy must be a non-decreasing function of ensemble size."""
    for grp in ["control", "cka", "cos"]:
        for b in ["mmlu", "gsm8k", "humaneval", "arc_challenge", "arc_easy"]:
            name = f"scaling_{b}_{grp}.csv"
            if not (DATA / name).exists():
                continue
            df = pd.read_csv(DATA / name)
            o = df["OracleAcc"].values
            assert all(o[i+1] >= o[i] - 1e-9 for i in range(len(o)-1)), \
                f"{name} has decreasing oracle"


def test_cka_oracle_beats_control_at_k30_on_mmlu():
    """The paper's headline scaling claim, locked in as a regression test."""
    _maybe_skip("scaling_mmlu_cka.csv", "scaling_mmlu_control.csv")
    cka = pd.read_csv(DATA / "scaling_mmlu_cka.csv").iloc[-1].OracleAcc
    ctrl = pd.read_csv(DATA / "scaling_mmlu_control.csv").iloc[-1].OracleAcc
    assert cka >= 0.70 and ctrl <= 0.66 and (cka - ctrl) >= 0.06
