"""Shared statistics against hand-worked values (ADR 0004: every flag rests on these)."""

import math

import numpy as np
import pytest

from ledgerlens.core.stats import (
    benjamini_hochberg,
    bootstrap_ci,
    bootstrap_cohens_d,
    cohens_d,
    cohens_h,
    cohens_kappa,
    fisher_p,
    fleiss_kappa,
    seed_for,
    spearman,
    welch_p,
)


def test_bh_matches_the_hand_worked_step_up():
    # sorted .005 .01 .03 .04 -> p*m/rank = .02 .02 .04 .04, already monotone; back in input order.
    assert benjamini_hochberg([0.01, 0.04, 0.03, 0.005]) == pytest.approx([0.02, 0.04, 0.04, 0.02])
    assert benjamini_hochberg([]) == []


def test_bh_enforces_monotonicity_from_the_top():
    # sorted .02 .04 .05 -> p*m/rank = .06 .06 .05: both earlier values are pulled down to the later .05.
    assert benjamini_hochberg([0.02, 0.05, 0.04]) == pytest.approx([0.05, 0.05, 0.05])


def test_cohens_kappa_hand_worked():
    # observed 3/4, expected (2*3 + 2*1)/16 = 1/2 -> (0.75 - 0.5) / 0.5
    assert cohens_kappa([1, 1, 0, 0], [1, 0, 0, 0]) == pytest.approx(0.5)
    assert cohens_kappa([True, False], [True, False]) == pytest.approx(1.0)
    assert cohens_kappa([True, True], [True, True]) is None  # no variation: undefined, not perfect
    with pytest.raises(ValueError, match="same length"):
        cohens_kappa([1], [1, 0])


def test_fleiss_kappa_hand_worked():
    # P_i = 1, 1, 0 -> P_bar 2/3; p_j = .5, .5 -> P_e .5; kappa = (2/3 - 1/2) / (1/2) = 1/3
    assert fleiss_kappa(np.array([[1, 1], [2, 2], [1, 2]])) == pytest.approx(1 / 3)
    assert fleiss_kappa(np.array([[1, 1, 1], [3, 3, 3], [2, 2, 2]])) == pytest.approx(1.0)
    assert fleiss_kappa(np.array([[2, 2], [2, 2]])) == 0.0  # one category only: undefined, reported as 0


def test_cohens_h_and_d():
    assert cohens_h(0.5, 0.5) == 0.0
    assert cohens_h(1.0, 0.0) == pytest.approx(math.pi)
    assert cohens_h(0.3, 0.5) == pytest.approx(-cohens_h(0.5, 0.3))
    assert cohens_d([1, 2, 3], [2, 3, 4]) == pytest.approx(-1.0)  # means 2 vs 3, pooled sd 1
    assert cohens_d([2, 2, 2], [2, 2, 2]) == 0.0


def test_fisher_matches_the_tea_tasting_table():
    assert fisher_p(3, 4, 1, 4) == pytest.approx(0.4857142857)
    assert fisher_p(10, 20, 10, 20) == pytest.approx(1.0)


def test_degenerate_inputs_return_safe_values():
    assert welch_p([1, 1, 1], [1, 1, 1]) == 1.0
    assert spearman([1, 2, 3], [5, 5, 5]) == 0.0
    assert spearman([1, 2, 3], [10, 20, 30]) == pytest.approx(1.0)
    assert welch_p([0, 0.1, 0.2, 0.1], [5, 5.1, 5.2, 5.1]) < 1e-6


def test_seed_for_is_stable_per_key_and_independent_across_keys():
    assert seed_for(7, "a").random() == seed_for(7, "a").random()
    assert seed_for(7, "a").random() != seed_for(7, "b").random()
    assert seed_for(7, "a").random() != seed_for(8, "a").random()


def test_bootstrap_ci_is_reproducible_and_brackets_the_point():
    rng_values = np.random.default_rng(0).normal(10, 2, 200)
    first = bootstrap_ci(np.mean, rng_values, rng=seed_for(1, "mean"), n_resamples=500)
    second = bootstrap_ci(np.mean, rng_values, rng=seed_for(1, "mean"), n_resamples=500)
    assert first == second
    assert first.ci_low < first.value < first.ci_high
    assert first.ci_high - first.ci_low < 1.0  # sd 2, n 200 -> 95% width about 0.55


def test_vectorised_bootstrap_d_agrees_with_the_generic_bootstrap():
    rng = np.random.default_rng(3)
    a, b = rng.normal(0.5, 1, 80), rng.normal(0, 1, 80)
    fast = bootstrap_cohens_d(a, b, rng=seed_for(2, "d"), n_resamples=2000)
    slow = bootstrap_ci(cohens_d, a, b, rng=seed_for(2, "d2"), n_resamples=2000)
    assert fast.value == pytest.approx(cohens_d(a, b))
    assert fast.ci_low == pytest.approx(slow.ci_low, abs=0.08)
    assert fast.ci_high == pytest.approx(slow.ci_high, abs=0.08)
