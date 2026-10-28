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
import math
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
    stability: float = 1.0
    difficulty: float = 5.0
    estimated_retention: float = 1.0
    scheduler_policy: str = "local-fsrs-v1"


def calculate_retrievability(elapsed_days: float, stability: float) -> float:
    """Calculate memory retrievability R(t, S) using power-law decay.

    Formula:
        R(t, S) = (1 + (19/81) * (t / S))^(-0.5)
    Guarantees:
        R(0, S) = 1.0 (100% initial recall)
        R(S, S) = (1 + 19/81)^(-0.5) = (100/81)^(-0.5) = 9/10 = 0.90 (90% target retention at interval S)
    """
    if elapsed_days <= 0.0:
        return 1.0
    s = max(0.1, stability)
    decay = 1.0 + (19.0 / 81.0) * (elapsed_days / s)
    r = math.pow(decay, -0.5)
    return max(0.01, min(1.0, r))


def map_grade_to_rating(score_earned: float, score_possible: float) -> ReviewRating:
    """Deterministically map graded assessment scores into review ratings.

    Policy:
        p < 0.60         -> AGAIN (Recall failed)
        0.60 <= p < 0.75 -> HARD  (Strenuous recall / partial credit)
        0.75 <= p < 0.90 -> GOOD  (Solid recall)
        p >= 0.90        -> EASY  (Effortless recall)
    """
    if score_possible <= 0.0 or not math.isfinite(score_earned) or not math.isfinite(score_possible):
        return ReviewRating.AGAIN
    p = max(0.0, min(1.0, score_earned / score_possible))
    if p < 0.60:
        return ReviewRating.AGAIN
    elif p < 0.75:
        return ReviewRating.HARD
    elif p < 0.90:
        return ReviewRating.GOOD
    else:
        return ReviewRating.EASY


def compute_fsrs_schedule(
    current_stability: float,
    current_difficulty: float,
    repetition_count: int,
    rating_or_outcome: ReviewRating | bool | float,
    last_reviewed_at: datetime | None = None,
    now: datetime | None = None,
    max_interval_days: float = 365.0,
) -> ScheduleUpdate:
    """Calculate the next review schedule using the StudyOS Local FSRS (DSR) policy.

    Mathematical Model:
    - Stability (S): Days until retrievability drops to 90%.
    - Difficulty (D): Inherent concept difficulty [1.0 (easiest) to 10.0 (hardest)].
    - Retrievability (R): Power-law retention at elapsed time t.
    """
    ts = now or datetime.now(timezone.utc)
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)

    # Normalize input to ReviewRating
    if isinstance(rating_or_outcome, bool):
        rating = ReviewRating.GOOD if rating_or_outcome else ReviewRating.AGAIN
    elif isinstance(rating_or_outcome, (int, float)):
        rating = map_grade_to_rating(float(rating_or_outcome), 1.0)
    else:
        rating = rating_or_outcome

    s = max(0.1, current_stability)
    d = max(1.0, min(10.0, current_difficulty))
    reps = max(0, repetition_count)

    # Calculate elapsed time in days
    if last_reviewed_at:
        last_ts = last_reviewed_at
        if last_ts.tzinfo is None:
            last_ts = last_ts.replace(tzinfo=timezone.utc)
        elapsed = max(0.0, (ts - last_ts).total_seconds() / 86400.0)
    else:
        elapsed = 0.0

    retention_before = calculate_retrievability(elapsed, s)

    # First review (cold start)
    if reps == 0:
        if rating == ReviewRating.AGAIN:
            new_s = 0.4
            new_d = 7.0
            new_reps = 0
            reason = "Initial recall failed. Stability set to 0.4d for immediate reinforcement."
        elif rating == ReviewRating.HARD:
            new_s = 1.2
            new_d = 6.0
            new_reps = 1
            reason = "Initial recall was difficult. Stability initialized to 1.2d."
        elif rating == ReviewRating.GOOD:
            new_s = 3.0
            new_d = 5.0
            new_reps = 1
            reason = "Initial recall was solid. Stability initialized to 3.0d."
        else:  # EASY
            new_s = 7.0
            new_d = 3.0
            new_reps = 1
            reason = "Initial recall was effortless. Stability initialized to 7.0d."
    else:
        # Subsequent reviews: update S and D based on previous S, D, and retention R
        r_term = math.pow(retention_before, -0.5)

        if rating == ReviewRating.AGAIN:
            # Lapse penalty
            new_s = max(0.4, min(s * 0.25, 2.0))
            new_d = min(10.0, d + 0.8)
            new_reps = 0
            reason = (
                f"Recall failed (estimated retention was {retention_before*100:.1f}%). "
                f"Lapse detected: stability reset to {new_s:.1f}d, difficulty increased to {new_d:.1f}."
            )
        elif rating == ReviewRating.HARD:
            new_s = s * (1.0 + (10.0 - d) * 0.15 * r_term)
            new_d = min(10.0, d + 0.3)
            new_reps = reps + 1
            reason = (
                f"Recall was difficult (retention was {retention_before*100:.1f}%). "
                f"Stability increased to {new_s:.1f}d, difficulty increased to {new_d:.1f}."
            )
        elif rating == ReviewRating.GOOD:
            new_s = s * (1.0 + (10.0 - d) * 0.35 * r_term)
            # Gentle mean-reversion toward 5.0
            new_d = max(1.0, min(10.0, d + 0.1 * (5.0 - d)))
            new_reps = reps + 1
            reason = (
                f"Successful recall (retention was {retention_before*100:.1f}%). "
                f"Stability increased to {new_s:.1f}d."
            )
        else:  # EASY
            new_s = s * (1.0 + (10.0 - d) * 0.60 * r_term) * 1.3
            new_d = max(1.0, d - 0.5)
            new_reps = reps + 1
            reason = (
                f"Effortless recall (retention was {retention_before*100:.1f}%). "
                f"Stability accelerated to {new_s:.1f}d, difficulty reduced to {new_d:.1f}."
            )

    # Next interval targets 90% retention (I = S), but lapses immediately reset interval to 1.0 day
    if rating == ReviewRating.AGAIN:
        interval_raw = 1.0
    else:
        interval_raw = max(1.0, min(max_interval_days, new_s))
    new_interval = round(interval_raw, 1)
    new_s = round(new_s, 2)
    new_d = round(new_d, 2)

    # Calculate compatible ease factor for legacy consumers
    ease_compat = round(3.0 - (new_d - 1.0) * (1.7 / 9.0), 2)
    ease_compat = max(1.3, min(3.0, ease_compat))

    next_due = ts + timedelta(days=new_interval)
    is_due = next_due <= ts
    is_overdue = next_due < (ts - timedelta(days=1))

    return ScheduleUpdate(
        interval_days=new_interval,
        ease_factor=ease_compat,
        repetition_count=new_reps,
        next_due_date=next_due,
        is_due=is_due,
        is_overdue=is_overdue,
        reason=reason,
        stability=new_s,
        difficulty=new_d,
        estimated_retention=1.0,
        scheduler_policy="local-fsrs-v1",
    )


def compute_next_schedule(
    current_interval: float,
    current_ease: float,
    repetition_count: int,
    rating_or_correctness: ReviewRating | bool,
    now: datetime | None = None,
    policy: str = "sm2_legacy",
    stability: float | None = None,
    difficulty: float | None = None,
    last_reviewed_at: datetime | None = None,
) -> ScheduleUpdate:
    """Calculate the next review interval and due timestamp deterministically.

    Supports both legacy SM-2 policy ('sm2_legacy') and advanced forgetting-aware policy ('local-fsrs-v1').
    """
    if policy == "local-fsrs-v1":
        init_s = stability if stability is not None else current_interval
        init_d = difficulty if difficulty is not None else (3.0 - (current_ease - 1.3) * (9.0 / 1.7))
        return compute_fsrs_schedule(
            current_stability=init_s,
            current_difficulty=init_d,
            repetition_count=repetition_count,
            rating_or_outcome=rating_or_correctness,
            last_reviewed_at=last_reviewed_at,
            now=now,
        )

    # Legacy SM-2 implementation
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
        stability=new_interval,
        difficulty=5.0,
        estimated_retention=1.0,
        scheduler_policy="sm2_legacy",
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
