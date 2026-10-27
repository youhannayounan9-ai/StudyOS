"""Unit tests for deterministic Laplace-smoothed mastery estimation."""

from studyos.learner.mastery import calculate_concept_mastery
from studyos.learner.models import MasteryCategory


def test_zero_observations_returns_unassessed():
    est = calculate_concept_mastery(observation_count=0, correct_count=0, incorrect_count=0)
    assert est.category == MasteryCategory.UNASSESSED
    assert est.score == 0.0
    assert est.has_sufficient_evidence is False
    assert "No graded assessment observations recorded" in est.reason


def test_low_sample_size_returns_needs_evidence():
    # 1 correct out of 1
    est1 = calculate_concept_mastery(observation_count=1, correct_count=1, incorrect_count=0)
    assert est1.category == MasteryCategory.NEEDS_EVIDENCE
    assert est1.has_sufficient_evidence is False
    # (1+1)/(1+2) = 2/3 = 0.667
    assert est1.score == 0.667
    assert "minimum of 3 observations is required" in est1.reason

    # 0 correct out of 1
    est2 = calculate_concept_mastery(observation_count=1, correct_count=0, incorrect_count=1)
    assert est2.category == MasteryCategory.NEEDS_EVIDENCE
    assert est2.has_sufficient_evidence is False
    # (0+1)/(1+2) = 1/3 = 0.333
    assert est2.score == 0.333

    # 2 observations, 1 correct
    est3 = calculate_concept_mastery(observation_count=2, correct_count=1, incorrect_count=1)
    assert est3.category == MasteryCategory.NEEDS_EVIDENCE
    assert est3.has_sufficient_evidence is False
    # (1+1)/(2+2) = 2/4 = 0.500
    assert est3.score == 0.500


def test_sufficient_sample_size_categorization():
    # n = 3, 0 correct -> (0+1)/(3+2) = 0.200 -> WEAK (< 0.60)
    est_weak = calculate_concept_mastery(observation_count=3, correct_count=0, incorrect_count=3)
    assert est_weak.category == MasteryCategory.WEAK
    assert est_weak.score == 0.200
    assert est_weak.has_sufficient_evidence is True
    assert "Weak evidence" in est_weak.reason

    # n = 4, 3 correct -> (3+1)/(4+2) = 4/6 = 0.667 -> MODERATE (0.60 to 0.84)
    est_mod = calculate_concept_mastery(observation_count=4, correct_count=3, incorrect_count=1)
    assert est_mod.category == MasteryCategory.MODERATE
    assert est_mod.score == 0.667
    assert est_mod.has_sufficient_evidence is True
    assert "Moderate evidence" in est_mod.reason

    # n = 5, 5 correct -> (5+1)/(5+2) = 6/7 = 0.857 -> STRONG (>= 0.85)
    est_strong = calculate_concept_mastery(observation_count=5, correct_count=5, incorrect_count=0)
    assert est_strong.category == MasteryCategory.STRONG
    assert est_strong.score == 0.857
    assert est_strong.has_sufficient_evidence is True
    assert "Currently strong evidence" in est_strong.reason


def test_clamping_and_inconsistent_input_handling():
    # correct count exceeds observation count -> clamped safely
    est = calculate_concept_mastery(observation_count=3, correct_count=5, incorrect_count=0)
    assert est.observation_count >= 3
    assert est.score <= 1.0
    assert est.has_sufficient_evidence is True


def test_normalized_single_correct_observation():
    """1. One fully correct observation contributes 1.0 normalized success."""
    est = calculate_concept_mastery(observation_count=1, success_credits=1.0)
    # M = (1.0 + 1) / (1 + 2) = 2/3 = 0.667
    assert est.observation_count == 1
    assert est.correct_count == 1.0
    assert est.score == 0.667
    assert est.category == MasteryCategory.NEEDS_EVIDENCE
    # sigma = sqrt(((2)(1))/((9)(4))) = sqrt(2/36) ~ 0.2357
    assert 0.2350 <= est.uncertainty <= 0.2360


def test_normalized_single_incorrect_observation():
    """2. One fully incorrect graded observation contributes 0.0."""
    est = calculate_concept_mastery(observation_count=1, success_credits=0.0)
    # M = (0.0 + 1) / (1 + 2) = 1/3 = 0.333
    assert est.observation_count == 1
    assert est.correct_count == 0.0
    assert est.score == 0.333
    assert est.category == MasteryCategory.NEEDS_EVIDENCE
    # sigma = sqrt(((1)(2))/((9)(4))) = sqrt(2/36) ~ 0.2357
    assert 0.2350 <= est.uncertainty <= 0.2360


def test_normalized_partial_credit_observation():
    """3. One partial-credit observation, such as 2/3, contributes 2/3, not 2 raw successes."""
    # 2 earned out of 3 possible = 2/3 = 0.666666...
    est = calculate_concept_mastery(observation_count=1, success_credits=2.0 / 3.0)
    # M = (2/3 + 1) / (1 + 2) = (5/3) / 3 = 5/9 = 0.556
    assert est.observation_count == 1
    assert round(est.correct_count, 3) == 0.667
    assert est.score == 0.556
    assert est.category == MasteryCategory.NEEDS_EVIDENCE
    # Verify it is NOT treated as 2 raw successes: (2+1)/(1+2) would be 1.0!
    assert est.score != 1.0
    # sigma = sqrt(((5/3)*(4/3))/((9)(4))) = sqrt(20/324) ~ 0.2485
    assert 0.2480 <= est.uncertainty <= 0.2490


def test_normalized_different_point_maxima():
    """4. Different point maxima are normalized consistently, such as 1/1, 2/3, and 4/5."""
    # Observations:
    # 1. 1/1 = 1.0
    # 2. 2/3 = 0.666666...
    # 3. 4/5 = 0.8
    # k = 1.0 + 2/3 + 0.8 = 2.466666...
    k = 1.0 + (2.0 / 3.0) + (4.0 / 5.0)
    est = calculate_concept_mastery(observation_count=3, success_credits=k)
    # M = (2.466666... + 1) / (3 + 2) = 3.466666... / 5 = 0.693333... -> 0.693
    assert est.observation_count == 3
    assert round(est.correct_count, 3) == 2.467
    assert est.score == 0.693
    assert est.category == MasteryCategory.MODERATE
    assert est.has_sufficient_evidence is True
    # sigma = sqrt(((3.4667)*(1.5333))/((25)(6))) = sqrt(5.3155/150) ~ 0.1882
    assert 0.1870 <= est.uncertainty <= 0.1890


def test_uncertainty_bounds_and_finiteness():
    """9. The estimate and uncertainty remain finite and within valid bounds [0, 1] and [0, 0.5]."""
    import math

    # Unassessed prior uncertainty ~ 0.2887
    est_zero = calculate_concept_mastery(observation_count=0)
    assert est_zero.uncertainty == 0.2887
    assert math.isfinite(est_zero.uncertainty)

    # Across various sample sizes and success ratios
    for n in [1, 2, 3, 5, 10, 20, 50]:
        for ratio in [0.0, 0.25, 0.5, 0.667, 0.8, 1.0]:
            k = n * ratio
            est = calculate_concept_mastery(observation_count=n, success_credits=k)
            assert 0.0 <= est.score <= 1.0
            assert 0.0 < est.uncertainty <= 0.30
            assert math.isfinite(est.score)
            assert math.isfinite(est.uncertainty)


def test_nan_and_inf_inputs_handled_safely():
    """7. An invalid, NaN, or infinite credit input cannot create an invalid estimate."""
    import math

    est_nan = calculate_concept_mastery(observation_count=3, success_credits=float("nan"))
    assert est_nan.score >= 0.0
    assert math.isfinite(est_nan.score)
    assert math.isfinite(est_nan.uncertainty)

    est_inf = calculate_concept_mastery(observation_count=3, success_credits=float("inf"))
    assert est_inf.score <= 1.0
    assert math.isfinite(est_inf.score)
    assert math.isfinite(est_inf.uncertainty)
