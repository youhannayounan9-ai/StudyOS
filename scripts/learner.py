"""StudyOS Learner Model, Mastery Tracking & Spaced Revision CLI."""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import json
import sys

from studyos.core.config import get_default_config
from studyos.learner.models import ReviewRating, ReviewStatus
from studyos.learner.service import LearnerService
from studyos.storage.db import Database


import os


def _setup_service() -> LearnerService:
    config = get_default_config()
    db_path = os.getenv("STUDYOS_DB_PATH") or config.db_path
    db = Database(db_path)
    db.migrate()
    return LearnerService(db=db, config=config)


# ---------------------------------------------------------------------------
# Command Handlers
# ---------------------------------------------------------------------------

def handle_progress(args: argparse.Namespace) -> None:
    service = _setup_service()
    since = None
    if args.days:
        since_dt = datetime.now(timezone.utc) - timedelta(days=args.days)
        since = since_dt.isoformat()

    summary = service.get_progress_summary(course_id=args.course, since=since)

    if args.json:
        print(summary.model_dump_json(indent=2))
        return

    print("\n" + "=" * 70)
    print("StudyOS - Learner Progress & Mastery Overview")
    print("=" * 70)
    if args.course:
        print(f"Course Filter:           {args.course}")
    if args.days:
        print(f"Time Window:             Last {args.days} day(s)")
    print(f"Total Concepts:          {summary.total_concepts}")
    print(f"  - Unassessed:          {summary.unassessed_count}")
    print(f"  - Needs Evidence:      {summary.needs_evidence_count} (< 3 observations)")
    print(f"  - Weak Evidence:       {summary.weak_count}")
    print(f"  - Moderate Progress:   {summary.moderate_count}")
    print(f"  - Currently Strong:    {summary.strong_count}")
    print("-" * 70)
    print(f"Total Graded Attempts:   {summary.total_graded_attempts}")
    print(f"Total Correct:           {summary.total_correct}")
    acc = f"{summary.accuracy_rate:.1f}%" if summary.accuracy_rate is not None else "N/A (no observations)"
    print(f"Overall Accuracy Rate:   {acc}")
    print(f"Due Reviews:             {summary.pending_reviews_count}")
    print(f"Overdue Reviews:         {summary.overdue_reviews_count}")
    print("=" * 70)

    if summary.concept_summaries:
        print("\nConcept Breakdown:")
        print(f"{'Concept':<30} {'Category':<18} {'Score':<8} {'Obs':<6} {'Due Date':<20}")
        print("-" * 84)
        for cs in summary.concept_summaries:
            due_display = cs.next_review_due[:16] if cs.next_review_due else "None"
            if cs.is_review_overdue:
                due_display += " (OVERDUE)"
            elif cs.is_review_due:
                due_display += " (DUE)"
            corr_str = f"{int(cs.correct_count)}" if cs.correct_count == int(cs.correct_count) else f"{cs.correct_count:.2f}"
            obs_str = f"{corr_str}/{cs.observation_count}"
            score_str = f"{cs.mastery_score:.2f}" if cs.observation_count > 0 else "0.00"
            print(f"{cs.concept_name[:28]:<30} {cs.status_display:<18} {score_str:<8} {obs_str:<6} {due_display:<20}")
    print()


def handle_concepts(args: argparse.Namespace) -> None:
    service = _setup_service()
    concepts = service.list_concepts(course_id=args.course)
    summary = service.get_progress_summary(course_id=args.course)
    m_map = {cs.concept_id: cs for cs in summary.concept_summaries}

    if args.json:
        data = [
            {
                "id": c.id,
                "course_id": c.course_id,
                "name": c.name,
                "summary": m_map.get(c.id).model_dump() if m_map.get(c.id) else None,
            }
            for c in concepts
        ]
        print(json.dumps(data, indent=2))
        return

    print("\n" + "=" * 80)
    print(f"StudyOS Concepts Catalog ({len(concepts)} registered concepts)")
    print("=" * 80)
    if not concepts:
        print("No concepts registered. Ingest documents or generate/take practice exams to populate concepts.")
        print()
        return

    print(f"{'Concept ID':<18} {'Course':<10} {'Name':<26} {'Category':<15} {'Score':<6} {'Evidence'}")
    print("-" * 88)
    for c in concepts:
        cs = m_map.get(c.id)
        cat = cs.status_display if cs else "Unassessed"
        score = f"{cs.mastery_score:.2f}" if cs and cs.observation_count > 0 else "-"
        if cs and cs.observation_count > 0:
            corr_val = f"{int(cs.correct_count)}" if cs.correct_count == int(cs.correct_count) else f"{cs.correct_count:.2f}"
            ev = f"{corr_val}/{cs.observation_count} credits"
        else:
            ev = "No attempts"
        print(f"{c.id:<18} {c.course_id:<10} {c.name[:24]:<26} {cat:<15} {score:<6} {ev}")
    print()


def handle_history(args: argparse.Namespace) -> None:
    service = _setup_service()
    events = service.repository.list_events(
        learner_id="learner_default",
        course_id=args.course,
        limit=args.limit,
    )

    if args.json:
        print(json.dumps([e.model_dump() for e in events], indent=2))
        return

    print("\n" + "=" * 80)
    print(f"StudyOS Activity History (Showing latest {len(events)} events)")
    print("=" * 80)
    if not events:
        print("No activity recorded yet.")
        print()
        return

    print(f"{'Timestamp':<20} {'Event Type':<24} {'Course':<10} {'Outcome':<10} {'Score'}")
    print("-" * 80)
    for e in events:
        ts = e.created_at[:19] if e.created_at else "Unknown"
        outcome = (
            "Correct"
            if e.is_correct is True
            else ("Incorrect" if e.is_correct is False else "Pending/N/A")
        )
        score_str = f"{e.score_achieved:.1f}/{e.score_max:.1f}" if e.score_achieved is not None and e.score_max is not None else "-"
        print(f"{ts:<20} {e.event_type:<24} {e.course_id:<10} {outcome:<10} {score_str}")
    print()


def handle_reviews(args: argparse.Namespace) -> None:
    service = _setup_service()
    status_filter = ReviewStatus.SCHEDULED if args.status != "all" else None
    only_due = args.status == "due"

    tasks = service.list_pending_reviews(
        course_id=args.course,
        only_due=only_due,
    )
    if args.status == "overdue":
        tasks = [t for t in tasks if t.is_overdue]

    if args.json:
        print(json.dumps([t.model_dump() for t in tasks], indent=2))
        return

    print("\n" + "=" * 80)
    print(f"StudyOS Spaced Repetition Review Queue ({len(tasks)} tasks)")
    print("=" * 80)
    if not tasks:
        print("No review tasks matching the specified criteria.")
        print()
        return

    print(f"{'Task ID':<18} {'Concept':<26} {'Course':<10} {'Due Date':<20} {'Status'}")
    print("-" * 82)
    for t in tasks:
        status_tag = "OVERDUE" if t.is_overdue else ("DUE TODAY" if t.is_due else "Scheduled")
        cname = t.concept_name or t.concept_id
        print(f"{t.id:<18} {cname[:24]:<26} {t.course_id:<10} {t.due_date[:16]:<20} {status_tag}")
    print()


def handle_record_review(args: argparse.Namespace) -> None:
    service = _setup_service()

    try:
        rating = ReviewRating(args.rating.lower())
    except ValueError:
        print(f"Error: Invalid rating '{args.rating}'. Must be one of: again, hard, good, easy.", file=sys.stderr)
        sys.exit(1)

    task_id = args.task_id
    if not task_id:
        if not args.concept_id:
            print("Error: Must provide either --task-id or --concept-id.", file=sys.stderr)
            sys.exit(1)
        # Find active task for concept
        tasks = service.repository.list_review_tasks(
            learner_id="learner_default",
            status=ReviewStatus.SCHEDULED,
        )
        match = next((t for t in tasks if t.concept_id == args.concept_id), None)
        if not match:
            print(f"Error: No active scheduled review task found for concept '{args.concept_id}'.", file=sys.stderr)
            sys.exit(1)
        task_id = match.id

    completed = service.record_review_outcome(task_id=task_id, rating=rating)
    mastery = service.repository.get_concept_mastery("learner_default", completed.concept_id)

    print("\n" + "=" * 60)
    print("Review Successfully Recorded")
    print("=" * 60)
    print(f"Task ID:         {completed.id}")
    print(f"Concept:         {completed.concept_name or completed.concept_id}")
    print(f"Rating Recorded: {rating.value.upper()}")
    if mastery:
        print(f"New Interval:    {mastery.interval_days:.1f} day(s)")
        print(f"Ease Factor:     {mastery.ease_factor:.2f}")
        print(f"Next Due Date:   {mastery.next_review_due}")
    print("=" * 60 + "\n")


def handle_recommendations(args: argparse.Namespace) -> None:
    service = _setup_service()
    recs = service.get_recommendations(course_id=args.course)

    if args.json:
        print(json.dumps([r.model_dump() for r in recs], indent=2))
        return

    print("\n" + "=" * 80)
    print(f"StudyOS Personalized Study Plan ({len(recs)} recommended actions)")
    print("=" * 80)
    if not recs:
        print("No active recommendations. Great job keeping up with your studies!")
        print()
        return

    for idx, r in enumerate(recs, 1):
        print(f"{idx}. [{r.action.upper()}] - Priority {r.priority}")
        print(f"   Concept:  {r.concept_name} ({r.concept_id}) | Course: {r.course_id}")
        print(f"   Status:   Category '{r.mastery_category.value}' ({r.observation_count} graded attempt(s))")
        print(f"   Reason:   {r.reason}")
        print("-" * 80)
    print()


def handle_why(args: argparse.Namespace) -> None:
    service = _setup_service()
    rec = service.explain_recommendation(concept_id=args.concept_id)
    if not rec:
        print(f"Concept '{args.concept_id}' not found.", file=sys.stderr)
        sys.exit(1)

    mastery = service.repository.get_concept_mastery("learner_default", args.concept_id)

    print("\n" + "=" * 70)
    print(f"Diagnostic Explanation for Concept: {rec.concept_name}")
    print("=" * 70)
    print(f"Concept ID:          {rec.concept_id}")
    print(f"Course:              {rec.course_id}")
    print(f"Recommended Action:  {rec.action}")
    print(f"Mastery Category:    {rec.mastery_category.value.upper()}")
    print(f"Observation Count:   {rec.observation_count}")
    if mastery:
        print(f"Mastery Score (M):   {mastery.mastery_score:.3f}")
        print(f"Success Credits (k): {mastery.correct_count:.2f}")
        print(f"Uncertainty (σ):     {mastery.uncertainty:.4f}")
    print(f"Diagnostic Reason:   {rec.reason}")
    print("=" * 70 + "\n")


# ---------------------------------------------------------------------------
# CLI Argument Parser
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="StudyOS Learner Model, Mastery Tracking & Spaced Revision CLI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", help="Learner management commands")

    # progress
    p_prog = subparsers.add_parser("progress", help="Show overall learner progress summary and mastery metrics")
    p_prog.add_argument("--course", help="Optional course ID filter")
    p_prog.add_argument("--days", type=int, help="Optional time window in days")
    p_prog.add_argument("--json", action="store_true", help="Output summary as JSON")

    # concepts
    p_cpt = subparsers.add_parser("concepts", help="List registered concepts and current mastery estimates")
    p_cpt.add_argument("--course", help="Optional course ID filter")
    p_cpt.add_argument("--json", action="store_true", help="Output concepts as JSON")

    # history
    p_hist = subparsers.add_parser("history", help="Show chronological study activity events")
    p_hist.add_argument("--course", help="Optional course ID filter")
    p_hist.add_argument("--limit", type=int, default=30, help="Maximum events to return")
    p_hist.add_argument("--json", action="store_true", help="Output history as JSON")

    # reviews
    p_rev = subparsers.add_parser("reviews", help="Show scheduled, due, or overdue spaced repetition reviews")
    p_rev.add_argument("--course", help="Optional course ID filter")
    p_rev.add_argument("--status", choices=["due", "overdue", "all"], default="all", help="Review status filter")
    p_rev.add_argument("--json", action="store_true", help="Output review queue as JSON")

    # record-review
    p_rec = subparsers.add_parser("record-review", help="Record a completed spaced review outcome")
    p_rec.add_argument("--task-id", help="Review task ID")
    p_rec.add_argument("--concept-id", help="Concept ID (resolves active scheduled task)")
    p_rec.add_argument("--rating", required=True, choices=["again", "hard", "good", "easy"], help="Review rating")

    # recommendations
    p_recs = subparsers.add_parser("recommendations", help="View recommended study plan and next actions")
    p_recs.add_argument("--course", help="Optional course ID filter")
    p_recs.add_argument("--json", action="store_true", help="Output recommendations as JSON")

    # why
    p_why = subparsers.add_parser("why", help="Explain why a specific concept is recommended for study")
    p_why.add_argument("--concept-id", required=True, help="Concept ID to inspect")

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(1)

    if args.command == "progress":
        handle_progress(args)
    elif args.command == "concepts":
        handle_concepts(args)
    elif args.command == "history":
        handle_history(args)
    elif args.command == "reviews":
        handle_reviews(args)
    elif args.command == "record-review":
        handle_record_review(args)
    elif args.command == "recommendations":
        handle_recommendations(args)
    elif args.command == "why":
        handle_why(args)


if __name__ == "__main__":
    main()
