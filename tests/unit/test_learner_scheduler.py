"""Unit tests for the bounded spaced repetition review scheduler."""

from datetime import datetime, timedelta, timezone

from studyos.learner.models import ReviewRating
from studyos.learner.scheduler import (
    compute_next_schedule,
    is_review_due,
    is_review_overdue,
)


def test_initial_schedule_progression():
    ref = datetime(2026, 10, 27, 12, 0, tzinfo=timezone.utc)

    # First correct review (reps = 0) -> interval 1.0 day
    s1 = compute_next_schedule(
        current_interval=1.0,
        current_ease=2.5,
        repetition_count=0,
        rating_or_correctness=ReviewRating.GOOD,
        now=ref,
    )
    assert s1.interval_days == 1.0
    assert s1.repetition_count == 1
    assert s1.ease_factor == 2.55
    assert s1.next_due_date == ref + timedelta(days=1)

    # Second correct review (reps = 1) -> interval 3.0 days
    s2 = compute_next_schedule(
        current_interval=s1.interval_days,
        current_ease=s1.ease_factor,
        repetition_count=s1.repetition_count,
        rating_or_correctness=ReviewRating.GOOD,
        now=ref,
    )
    assert s2.interval_days == 3.0
    assert s2.repetition_count == 2
    assert s2.ease_factor == 2.60

    # Third correct review (reps = 2) -> interval = 3.0 * 2.6 = 7.8 days
    s3 = compute_next_schedule(
        current_interval=s2.interval_days,
        current_ease=s2.ease_factor,
        repetition_count=s2.repetition_count,
        rating_or_correctness=ReviewRating.GOOD,
        now=ref,
    )
    assert s3.interval_days == 7.8
    assert s3.repetition_count == 3


def test_again_rating_resets_interval_and_drops_ease():
    ref = datetime(2026, 10, 27, 12, 0, tzinfo=timezone.utc)
    s = compute_next_schedule(
        current_interval=14.0,
        current_ease=2.4,
        repetition_count=4,
        rating_or_correctness=ReviewRating.AGAIN,
        now=ref,
    )
    assert s.interval_days == 1.0
    assert s.repetition_count == 0
    assert s.ease_factor == 2.2
    assert "Review interval reset to 1 day" in s.reason


def test_boolean_correctness_mapping():
    ref = datetime(2026, 10, 27, 12, 0, tzinfo=timezone.utc)
    s_corr = compute_next_schedule(
        current_interval=1.0,
        current_ease=2.5,
        repetition_count=0,
        rating_or_correctness=True,
        now=ref,
    )
    assert s_corr.interval_days == 1.0
    assert s_corr.repetition_count == 1

    s_incorr = compute_next_schedule(
        current_interval=5.0,
        current_ease=2.5,
        repetition_count=2,
        rating_or_correctness=False,
        now=ref,
    )
    assert s_incorr.interval_days == 1.0
    assert s_incorr.repetition_count == 0


def test_bounds_clamping():
    ref = datetime(2026, 10, 27, 12, 0, tzinfo=timezone.utc)
    # Ease cannot drop below 1.3
    s_min_ease = compute_next_schedule(
        current_interval=1.0,
        current_ease=1.35,
        repetition_count=0,
        rating_or_correctness=ReviewRating.AGAIN,
        now=ref,
    )
    assert s_min_ease.ease_factor == 1.3

    # Ease cannot exceed 3.0
    s_max_ease = compute_next_schedule(
        current_interval=1.0,
        current_ease=2.95,
        repetition_count=2,
        rating_or_correctness=ReviewRating.EASY,
        now=ref,
    )
    assert s_max_ease.ease_factor == 3.0

    # Interval clamped at 180 days
    s_max_int = compute_next_schedule(
        current_interval=150.0,
        current_ease=2.5,
        repetition_count=5,
        rating_or_correctness=ReviewRating.EASY,
        now=ref,
    )
    assert s_max_int.interval_days == 180.0


def test_is_review_due_and_overdue():
    now = datetime(2026, 10, 27, 12, 0, tzinfo=timezone.utc)

    # Future date: not due, not overdue
    future = now + timedelta(days=2)
    assert is_review_due(future, now) is False
    assert is_review_overdue(future, now) is False

    # Current moment: due, not overdue
    assert is_review_due(now, now) is True
    assert is_review_overdue(now, now) is False

    # 12 hours ago: due, but not yet overdue (> 24 hours required)
    twelve_hrs_ago = now - timedelta(hours=12)
    assert is_review_due(twelve_hrs_ago, now) is True
    assert is_review_overdue(twelve_hrs_ago, now) is False

    # 36 hours ago: both due and overdue
    thirty_six_hrs_ago = now - timedelta(hours=36)
    assert is_review_due(thirty_six_hrs_ago, now) is True
    assert is_review_overdue(thirty_six_hrs_ago, now) is True
