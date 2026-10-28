"""Unit tests for the advanced forgetting-aware scheduler (local-fsrs-v1)."""

from datetime import datetime, timedelta, timezone
import pytest
from studyos.learner.models import ReviewRating
from studyos.learner.scheduler import (
    calculate_retrievability,
    compute_fsrs_schedule,
    compute_next_schedule,
    map_grade_to_rating,
)


def test_retrievability_formula():
    """Verify that retrievability satisfies R(0, S) = 1.0 and R(S, S) = 0.90."""
    stability = 10.0
    r_initial = calculate_retrievability(0.0, stability)
    assert r_initial == 1.0

    r_target = calculate_retrievability(10.0, stability)
    assert round(r_target, 3) == 0.900

    # Over time, retrievability drops
    r_overdue = calculate_retrievability(20.0, stability)
    assert r_overdue < 0.900
    assert r_overdue > 0.0


def test_map_grade_to_rating():
    """Test mapping of graded scores to review ratings."""
    assert map_grade_to_rating(0.5, 1.0) == ReviewRating.AGAIN
    assert map_grade_to_rating(0.59, 1.0) == ReviewRating.AGAIN
    assert map_grade_to_rating(0.60, 1.0) == ReviewRating.HARD
    assert map_grade_to_rating(0.70, 1.0) == ReviewRating.HARD
    assert map_grade_to_rating(0.75, 1.0) == ReviewRating.GOOD
    assert map_grade_to_rating(0.85, 1.0) == ReviewRating.GOOD
    assert map_grade_to_rating(0.90, 1.0) == ReviewRating.EASY
    assert map_grade_to_rating(1.0, 1.0) == ReviewRating.EASY
    # Non-positive max score safely returns AGAIN
    assert map_grade_to_rating(1.0, 0.0) == ReviewRating.AGAIN


def test_fsrs_initial_cold_start():
    """Test cold-start (repetition_count = 0) scheduling across ratings."""
    ref = datetime(2026, 10, 27, 12, 0, tzinfo=timezone.utc)

    # Initial GOOD
    s_good = compute_fsrs_schedule(
        current_stability=1.0,
        current_difficulty=5.0,
        repetition_count=0,
        rating_or_outcome=ReviewRating.GOOD,
        now=ref,
    )
    assert s_good.stability == 3.0
    assert s_good.difficulty == 5.0
    assert s_good.repetition_count == 1
    assert s_good.interval_days == 3.0
    assert s_good.scheduler_policy == "local-fsrs-v1"

    # Initial EASY
    s_easy = compute_fsrs_schedule(
        current_stability=1.0,
        current_difficulty=5.0,
        repetition_count=0,
        rating_or_outcome=ReviewRating.EASY,
        now=ref,
    )
    assert s_easy.stability == 7.0
    assert s_easy.difficulty == 3.0
    assert s_easy.interval_days == 7.0

    # Initial AGAIN
    s_again = compute_fsrs_schedule(
        current_stability=1.0,
        current_difficulty=5.0,
        repetition_count=0,
        rating_or_outcome=ReviewRating.AGAIN,
        now=ref,
    )
    assert s_again.stability == 0.4
    assert s_again.difficulty == 7.0
    assert s_again.interval_days == 1.0  # Clamped to min 1.0 day interval


def test_fsrs_subsequent_progression_and_lapse():
    """Test multi-review progression and lapse recovery."""
    t0 = datetime(2026, 10, 27, 12, 0, tzinfo=timezone.utc)

    # Review 1: Good
    s1 = compute_fsrs_schedule(
        current_stability=1.0,
        current_difficulty=5.0,
        repetition_count=0,
        rating_or_outcome=ReviewRating.GOOD,
        now=t0,
    )
    assert s1.stability == 3.0

    # Review 2: 3 days later, Good again
    t1 = t0 + timedelta(days=3)
    s2 = compute_fsrs_schedule(
        current_stability=s1.stability,
        current_difficulty=s1.difficulty,
        repetition_count=s1.repetition_count,
        rating_or_outcome=ReviewRating.GOOD,
        last_reviewed_at=t0,
        now=t1,
    )
    assert s2.stability > s1.stability
    assert s2.repetition_count == 2

    # Review 3: Lapse (AGAIN)
    t2 = t1 + timedelta(days=int(s2.interval_days))
    s3 = compute_fsrs_schedule(
        current_stability=s2.stability,
        current_difficulty=s2.difficulty,
        repetition_count=s2.repetition_count,
        rating_or_outcome=ReviewRating.AGAIN,
        last_reviewed_at=t1,
        now=t2,
    )
    assert s3.repetition_count == 0  # Reps reset on lapse
    assert s3.stability < s2.stability  # Stability dropped
    assert s3.difficulty > s2.difficulty  # Difficulty increased
    assert s3.interval_days == 1.0  # Interval clamped to 1.0


def test_compute_next_schedule_delegates_to_fsrs():
    """Test compute_next_schedule with policy='local-fsrs-v1'."""
    ref = datetime(2026, 10, 27, 12, 0, tzinfo=timezone.utc)
    s = compute_next_schedule(
        current_interval=3.0,
        current_ease=2.5,
        repetition_count=1,
        rating_or_correctness=ReviewRating.GOOD,
        now=ref,
        policy="local-fsrs-v1",
        stability=3.0,
        difficulty=5.0,
        last_reviewed_at=ref - timedelta(days=3),
    )
    assert s.scheduler_policy == "local-fsrs-v1"
    assert s.stability > 3.0


def test_fsrs_bounds_and_clamping():
    """Verify stability, difficulty, and interval bounds are strictly enforced."""
    ref = datetime(2026, 10, 27, 12, 0, tzinfo=timezone.utc)

    # 1. Repeated EASY: difficulty never drops below 1.0, interval capped at max_interval_days
    s = compute_fsrs_schedule(
        current_stability=300.0,
        current_difficulty=1.2,
        repetition_count=10,
        rating_or_outcome=ReviewRating.EASY,
        now=ref,
        max_interval_days=365.0,
    )
    assert s.difficulty >= 1.0
    assert s.interval_days <= 365.0

    # 2. Repeated AGAIN: difficulty never exceeds 10.0, interval clamped to 1.0
    s_again = compute_fsrs_schedule(
        current_stability=0.5,
        current_difficulty=9.8,
        repetition_count=5,
        rating_or_outcome=ReviewRating.AGAIN,
        now=ref,
    )
    assert s_again.difficulty <= 10.0
    assert s_again.interval_days == 1.0
    assert s_again.stability >= 0.4


def test_partial_credit_mapping_and_bounds():
    """Verify partial-credit outcomes map to correct ratings deterministically."""
    # 2/3 = 0.6667 -> HARD (strenuous recall / partial credit)
    assert map_grade_to_rating(2.0, 3.0) == ReviewRating.HARD

    # 4/5 = 0.80 -> GOOD (solid recall)
    assert map_grade_to_rating(4.0, 5.0) == ReviewRating.GOOD

    # 1/3 = 0.3333 -> AGAIN (recall failed)
    assert map_grade_to_rating(1.0, 3.0) == ReviewRating.AGAIN

    # 95/100 = 0.95 -> EASY
    assert map_grade_to_rating(95.0, 100.0) == ReviewRating.EASY

    # Invalid inputs
    assert map_grade_to_rating(float("nan"), 10.0) == ReviewRating.AGAIN
    assert map_grade_to_rating(5.0, -1.0) == ReviewRating.AGAIN

