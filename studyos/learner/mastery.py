"""Deterministic, explainable concept mastery estimation for StudyOS.

Mathematical Foundations & Interpretation:
------------------------------------------
StudyOS uses Laplace-smoothed empirical accuracy under a uniform Beta(1,1) prior
to track concept-level learner performance from validated assessment attempts:

    For n eligible observations with score_earned_i and score_possible_i > 0:
        p_i = score_earned_i / score_possible_i   (p_i in [0.0, 1.0])
        k = sum(p_i)                               (0.0 <= k <= n)
        mastery_score = (k + 1.0) / (n + 2.0)

Approximate posterior uncertainty (standard deviation):
    uncertainty = sqrt( ((k + 1.0) * (n - k + 1.0)) / ((n + 2.0)^2 * (n + 3.0)) )

Key Properties:
1. Normalized Evidence Units:
   Each eligible graded observation contributes exactly 1 unit of evidence to n.
   Partial credit contributes fractional success (p_i in [0.0, 1.0]) to k.
   k represents normalized earned-credit equivalents (fractional successes)—not raw unnormalized points.
2. Conservative Prior:
   Before large sample sizes are gathered, the estimate is regularized toward 0.5 (Beta(1,1) prior),
   preventing an early mistake from plunging mastery to 0.0 or a single lucky guess from scoring 1.0.
3. Distinction Between Absence of Evidence and Low Mastery:
   - Zero graded attempts (n = 0) produces category UNASSESSED with mastery_score = 0.0 and uncertainty ~ 0.2887.
     It is explicitly marked as "No graded assessment observations recorded" rather than failure.
   - Low sample size (1 <= n < 3) produces category NEEDS_EVIDENCE. The score is computed as a
     provisional indicator, but has_sufficient_evidence is False.
   - Sufficient sample size (n >= 3) produces categories WEAK (< 0.60), MODERATE (0.60 to 0.84),
     or STRONG (>= 0.85).
4. Treatment of Pending & Skipped Questions:
   - Questions in 'pending_manual_review' status (such as unreviewed short-answer questions)
     are strictly EXCLUDED from n and k until officially finalized.
   - Skipped objective questions (scored 0.0 in Stage 4) count as valid observations with p_i = 0.0.
5. Explainability & Transparent Uncertainty:
   The estimate is an empirical performance heuristic, not a psychological proof of latent mastery.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from studyos.learner.models import MasteryCategory


@dataclass(frozen=True)
class MasteryEstimate:
    """Calculated mastery estimate with transparent reasoning and uncertainty state."""

    score: float
    category: MasteryCategory
    has_sufficient_evidence: bool
    reason: str
    observation_count: int
    correct_count: float
    incorrect_count: float
    uncertainty: float = 0.2887


def calculate_concept_mastery(
    observation_count: int,
    correct_count: float | None = None,
    incorrect_count: float | None = None,
    *,
    success_credits: float | None = None,
) -> MasteryEstimate:
    """Calculate deterministic concept mastery estimate from valid graded observations.

    Args:
        observation_count: Total valid graded question attempts for this concept (n).
        correct_count: Normalized success credits (k) or integer correct count.
        incorrect_count: Optional missed credits (n - k).
        success_credits: Explicit normalized success credits k = sum(p_i) in [0, n].

    Returns:
        MasteryEstimate containing the smoothed score, category, sufficiency flag, reason, and uncertainty.
    """
    if observation_count <= 0:
        return MasteryEstimate(
            score=0.0,
            category=MasteryCategory.UNASSESSED,
            has_sufficient_evidence=False,
            reason="No graded assessment observations recorded for this concept.",
            observation_count=0,
            correct_count=0.0,
            incorrect_count=0.0,
            uncertainty=round(math.sqrt(1.0 / 12.0), 4),  # Prior std dev for Beta(1,1) ~ 0.2887
        )

    n = int(observation_count)

    # Determine k (normalized success credits)
    raw_k: float = 0.0
    if success_credits is not None:
        raw_k = float(success_credits)
    elif correct_count is not None:
        raw_k = float(correct_count)

    # Validate and clamp: k must be finite and within [0.0, n]
    if not math.isfinite(raw_k):
        raw_k = 0.0
    k = max(0.0, min(raw_k, float(n)))
    valid_incorrect = round(float(n) - k, 3)

    # Laplace smoothed accuracy: M = (k + 1.0) / (n + 2.0)
    smoothed_score = round((k + 1.0) / (float(n) + 2.0), 3)

    # Approximate posterior uncertainty: std dev under Beta(1,1) with fractional successes
    numerator = (k + 1.0) * (float(n) - k + 1.0)
    denominator = ((float(n) + 2.0) ** 2) * (float(n) + 3.0)
    uncertainty = round(math.sqrt(numerator / denominator), 4)

    if n < 3:
        return MasteryEstimate(
            score=smoothed_score,
            category=MasteryCategory.NEEDS_EVIDENCE,
            has_sufficient_evidence=False,
            reason=(
                f"Needs more evidence: only {n} graded attempt(s) recorded "
                f"({k:.2f} normalized credits, {valid_incorrect:.2f} missed credits). "
                f"A minimum of 3 observations is required for confident mastery estimation."
            ),
            observation_count=n,
            correct_count=round(k, 3),
            incorrect_count=valid_incorrect,
            uncertainty=uncertainty,
        )

    # For n >= 3, categorize by smoothed score threshold
    if smoothed_score < 0.60:
        category = MasteryCategory.WEAK
        reason = (
            f"Weak evidence: low recent accuracy ({k:.2f}/{n} normalized credits, "
            f"smoothed score {smoothed_score:.2f}). Concept requires targeted review."
        )
    elif smoothed_score < 0.85:
        category = MasteryCategory.MODERATE
        reason = (
            f"Moderate evidence: developing understanding ({k:.2f}/{n} normalized credits, "
            f"smoothed score {smoothed_score:.2f}). Progressing well, periodic review suggested."
        )
    else:
        category = MasteryCategory.STRONG
        reason = (
            f"Currently strong evidence: consistent accuracy ({k:.2f}/{n} normalized credits, "
            f"smoothed score {smoothed_score:.2f}). Concept demonstrates solid retention."
        )

    return MasteryEstimate(
        score=smoothed_score,
        category=category,
        has_sufficient_evidence=True,
        reason=reason,
        observation_count=n,
        correct_count=round(k, 3),
        incorrect_count=valid_incorrect,
        uncertainty=uncertainty,
    )
