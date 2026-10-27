"""Deterministic, bounded spaced repetition review scheduler for StudyOS.

Algorithm Overview & Policy:
---------------------------
StudyOS implements a bounded interval-scheduling policy inspired by Leitner and SM-2 principles.
It operates 100% offline, requires no cloud services or external models, and produces predictable,
explainable review intervals.

Policy Parameters:
- Initial Review: 1.0 day (first recall test).
- Interval Bounds: Clamped strictly between 1.0 day and 180.0 days (maximum 6 months).
- Ease Factor Bounds: Clamped strictly between 1.3 (high difficulty) and 3.0 (high retention). Default: 2.5.
- Progression Logic:
  - Rating 'AGAIN' or Incorrect Assessment:
    Recall failed. Reset interval to 1.0 day, reset repetition count to 0, reduce ease factor by 0.2.
  - Rating 'HARD' or Partial Credit:
    Recall was strenuous. Progress interval sub-linearly (factor 1.2), decrease ease factor by 0.15.
  - Rating 'GOOD' or Full Credit:
    Successful recall. Repetition 1 -> 1 day, Repetition 2 -> 3 days, Repetition 3+ -> (interval * ease_factor).
    Increase ease factor by 0.05.
  - Rating 'EASY':
    Effortless recall. Repetition 1 -> 2 days, Repetition 2 -> 5 days, Repetition 3+ -> (interval * ease_factor * 1.3).
    Increase ease factor by 0.15.

Due vs. Overdue Status:
- Due: next_due_date <= current_time (active review window).
- Overdue: next_due_date < current_time - 24 hours (past the due window by more than 1 calendar day).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from studyos.learner.models import ReviewRating


@dataclass(frozen=True)
class ScheduleUpdate:
    """Calculated schedule state for a concept review task."""

    interval_days: float
    ease_factor: float
    repetition_count: int
    next_due_date: datetime
    is_due: bool
    is_overdue: bool
    reason: str


def compute_next_schedule(
    current_interval: float,
    current_ease: float,
    repetition_count: int,
    rating_or_correctness: ReviewRating | bool,
    now: datetime | None = None,
) -> ScheduleUpdate:
    """Calculate the next review interval and due timestamp deterministically.

    Args:
        current_interval: Current review interval in days (>= 1.0).
        current_ease: Current ease factor (1.3 to 3.0).
        repetition_count: Number of successful consecutive reviews completed.
        rating_or_correctness: Explicit ReviewRating enum or bool from an assessment outcome.
        now: Optional reference timestamp (defaults to datetime.now(timezone.utc)).

    Returns:
        ScheduleUpdate with new interval, ease factor, repetition count, and due date.
    """
    ts = now or datetime.now(timezone.utc)
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)

    # Normalize input to ReviewRating
    if isinstance(rating_or_correctness, bool):
        rating = ReviewRating.GOOD if rating_or_correctness else ReviewRating.AGAIN
    else:
        rating = rating_or_correctness

    ease = max(1.3, min(3.0, current_ease))
    interval = max(1.0, current_interval)
    reps = max(0, repetition_count)

    if rating == ReviewRating.AGAIN:
        new_interval = 1.0
        new_reps = 0
        new_ease = max(1.3, ease - 0.2)
        reason = "Recall failed. Review interval reset to 1 day for prompt reinforcement."
    elif rating == ReviewRating.HARD:
        new_interval = max(1.0, round(interval * 1.2, 1))
        new_reps = reps + 1
        new_ease = max(1.3, ease - 0.15)
        reason = f"Recall was difficult. Moderate interval increase to {new_interval:.1f} day(s)."
    elif rating == ReviewRating.GOOD:
        if reps == 0:
            new_interval = 1.0
        elif reps == 1:
            new_interval = 3.0
        else:
            new_interval = round(interval * ease, 1)
        new_reps = reps + 1
        new_ease = min(3.0, ease + 0.05)
        reason = f"Successful recall. Standard progression to {new_interval:.1f} day(s)."
    elif rating == ReviewRating.EASY:
        if reps == 0:
            new_interval = 2.0
        elif reps == 1:
            new_interval = 5.0
        else:
            new_interval = round(interval * ease * 1.3, 1)
        new_reps = reps + 1
        new_ease = min(3.0, ease + 0.15)
        reason = f"Effortless recall. Accelerated interval increase to {new_interval:.1f} day(s)."
    else:
        new_interval = 1.0
        new_reps = reps
        new_ease = ease
        reason = "Unrecognized rating. Interval maintained at 1 day."

    # Clamp bounds
    new_interval = max(1.0, min(180.0, new_interval))
    new_ease = max(1.3, min(3.0, round(new_ease, 2)))

    next_due = ts + timedelta(days=new_interval)
    is_due = next_due <= ts
    is_overdue = next_due < (ts - timedelta(days=1))

    return ScheduleUpdate(
        interval_days=new_interval,
        ease_factor=new_ease,
        repetition_count=new_reps,
        next_due_date=next_due,
        is_due=is_due,
        is_overdue=is_overdue,
        reason=reason,
    )


def is_review_due(due_date: datetime, now: datetime | None = None) -> bool:
    """Return True if due_date is at or before current time."""
    ts = now or datetime.now(timezone.utc)
    if due_date.tzinfo is None:
        due_date = due_date.replace(tzinfo=timezone.utc)
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    return due_date <= ts


def is_review_overdue(due_date: datetime, now: datetime | None = None) -> bool:
    """Return True if due_date is more than 24 hours in the past."""
    ts = now or datetime.now(timezone.utc)
    if due_date.tzinfo is None:
        due_date = due_date.replace(tzinfo=timezone.utc)
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    return due_date < (ts - timedelta(days=1))
