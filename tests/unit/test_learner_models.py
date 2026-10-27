"""Unit tests for StudyOS learner domain models and enumerations."""

import pytest
from pydantic import ValidationError

from studyos.learner.models import (
    Concept,
    ConceptMastery,
    ConceptProgressSummary,
    CourseProgressSummary,
    LearnerEvent,
    LearnerProfile,
    LearnerProgressSummary,
    MasteryCategory,
    ReviewRating,
    ReviewStatus,
    StudyRecommendation,
)


def test_mastery_category_enum():
    assert MasteryCategory.UNASSESSED.value == "unassessed"
    assert MasteryCategory.NEEDS_EVIDENCE.value == "needs_evidence"
    assert MasteryCategory.WEAK.value == "weak"
    assert MasteryCategory.MODERATE.value == "moderate"
    assert MasteryCategory.STRONG.value == "strong"


def test_review_rating_enum():
    assert ReviewRating.AGAIN.value == "again"
    assert ReviewRating.HARD.value == "hard"
    assert ReviewRating.GOOD.value == "good"
    assert ReviewRating.EASY.value == "easy"


def test_learner_profile_validation():
    p = LearnerProfile(id="p1", name="Alice")
    assert p.id == "p1"
    assert p.name == "Alice"

    # Empty or whitespace name rejected
    with pytest.raises(ValidationError):
        LearnerProfile(id="p2", name="   ")


def test_concept_validation():
    c = Concept(id="c1", course_id="crs1", name="Dynamic Programming", normalized_name="dynamic programming")
    assert c.id == "c1"
    assert c.name == "Dynamic Programming"

    with pytest.raises(ValidationError):
        Concept(id="c2", course_id="crs1", name="", normalized_name="")


def test_concept_mastery_defaults_and_bounds():
    m = ConceptMastery(
        id="m1",
        learner_id="l1",
        concept_id="c1",
        course_id="crs1",
    )
    assert m.mastery_score == 0.0
    assert m.mastery_category == MasteryCategory.UNASSESSED
    assert m.observation_count == 0
    assert m.interval_days == 1.0
    assert m.ease_factor == 2.5
    assert m.has_sufficient_evidence is False

    # Out of bounds mastery score rejected
    with pytest.raises(ValidationError):
        ConceptMastery(
            id="m2",
            learner_id="l1",
            concept_id="c1",
            course_id="crs1",
            mastery_score=1.5,
        )


def test_study_recommendation_model():
    rec = StudyRecommendation(
        concept_id="c1",
        concept_name="Graph Traversal",
        course_id="crs1",
        action="Targeted Practice",
        reason="Low recent accuracy.",
        priority=2,
        mastery_category=MasteryCategory.WEAK,
        observation_count=3,
        supporting_evidence={"correct": 1, "total": 3},
    )
    assert rec.priority == 2
    assert rec.mastery_category == MasteryCategory.WEAK
    assert rec.supporting_evidence["correct"] == 1
