"""StudyOS Learner Model, Knowledge Graph, Adaptive Planning & Spaced Revision CLI."""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import json
import os
import sys

from studyos.core.config import get_default_config
from studyos.core.health import check_ollama_health
from studyos.inference.ollama import OllamaChatProvider
from studyos.learner.graph import CycleDetectedError
from studyos.learner.models import (
    ConceptRelationship,
    ConceptReviewStatus,
    RelationshipProvenance,
    RelationshipStatus,
    RelationshipType,
    ReviewRating,
    ReviewStatus,
)
from studyos.learner.service import LearnerService
from studyos.research import (
    ExternalResearchService,
    FakeSearchProvider,
    ResearchRepository,
    WikipediaSearchProvider,
)
from studyos.storage.db import Database


def _setup_service(require_chat: bool = False) -> LearnerService:
    config = get_default_config()
    db_path = os.getenv("STUDYOS_DB_PATH") or config.db_path
    db = Database(db_path)
    db.migrate()

    chat_provider = None
    health = check_ollama_health(config, model_name=config.tutor_model)
    if health.is_ready:
        chat_provider = OllamaChatProvider.from_config(config)

    return LearnerService(db=db, chat_provider=chat_provider, config=config)


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
    concepts = service.repository.list_concepts(course_id=args.course, status=args.status)
    summary = service.get_progress_summary(course_id=args.course)
    m_map = {cs.concept_id: cs for cs in summary.concept_summaries}

    if args.json:
        data = [
            {
                "id": c.id,
                "course_id": c.course_id,
                "name": c.name,
                "review_status": c.review_status.value,
                "provenance": c.provenance.value,
                "description": c.description,
                "summary": m_map.get(c.id).model_dump() if m_map.get(c.id) else None,
            }
            for c in concepts
        ]
        print(json.dumps(data, indent=2))
        return

    print("\n" + "=" * 80)
    print(f"StudyOS Concepts Catalog ({len(concepts)} concepts matching filter)")
    print("=" * 80)
    if not concepts:
        print("No concepts found matching your query.")
        print()
        return

    print(f"{'Concept ID':<18} {'Status':<10} {'Name':<24} {'Category':<15} {'Score':<6} {'Provenance'}")
    print("-" * 88)
    for c in concepts:
        cs = m_map.get(c.id)
        cat = cs.status_display if cs else "Unassessed"
        score = f"{cs.mastery_score:.2f}" if cs and cs.observation_count > 0 else "-"
        print(f"{c.id:<18} {c.review_status.value:<10} {c.name[:22]:<24} {cat:<15} {score:<6} {c.provenance.value}")
    print()


def handle_discover_concepts(args: argparse.Namespace) -> None:
    service = _setup_service(require_chat=True)
    res = service.discover_concepts(
        course_id=args.course,
        document_id=args.document,
        max_chunks=args.max_chunks,
        auto_accept=args.auto_accept,
    )

    if args.json:
        print(res.model_dump_json(indent=2))
        return

    print("\n" + "=" * 80)
    print("StudyOS Direct Concept Discovery Results")
    print("=" * 80)
    print(f"Course ID:        {res.course_id}")
    print(f"Status:           {res.status.upper()}")
    print(f"Chunks Inspected: {res.chunks_scanned}")
    print(f"Concepts Found:   {res.total_discovered}")
    if res.error_message:
        print(f"Notice:           {res.error_message}")
    print("-" * 80)

    if not res.concepts_discovered:
        print("No concepts could be discovered from the selected document/course chunks.")
        print()
        return

    print(f"{'Concept ID':<18} {'Status':<10} {'Name':<24} {'Definition'}")
    print("-" * 80)
    for dc in res.concepts_discovered:
        desc = (dc.description or "No description")[:35]
        print(f"{dc.concept_id:<18} {dc.review_status.value:<10} {dc.name[:22]:<24} {desc}...")
    print("=" * 80 + "\n")


def handle_relationships(args: argparse.Namespace) -> None:
    service = _setup_service()
    rels = service.list_concept_relationships(
        course_id=args.course,
        status=args.status,
        rel_type=args.type,
    )

    if args.json:
        print(json.dumps([r.model_dump() for r in rels], indent=2))
        return

    print("\n" + "=" * 80)
    print(f"StudyOS Concept Relationships for Course '{args.course}' ({len(rels)} edges)")
    print("=" * 80)
    if not rels:
        print("No relationships recorded yet for this course.")
        print()
        return

    print(f"{'Rel ID':<16} {'Source Concept':<22} {'Type':<14} {'Target Concept':<22} {'Status'}")
    print("-" * 84)
    for r in rels:
        s_name = r.source_concept_name or r.source_concept_id
        t_name = r.target_concept_name or r.target_concept_id
        print(f"{r.id:<16} {s_name[:20]:<22} {r.relationship_type.value:<14} {t_name[:20]:<22} {r.review_status.value}")
    print()


def handle_add_relationship(args: argparse.Namespace) -> None:
    service = _setup_service()
    try:
        rel = service.add_concept_relationship(
            course_id=args.course,
            source_concept_id=args.source,
            target_concept_id=args.target,
            relationship_type=args.type,
            review_status=args.status,
            source_evidence=args.evidence,
            confidence=args.confidence,
        )
    except CycleDetectedError as e:
        print(f"Relationship rejected: {e}", file=sys.stderr)
        sys.exit(1)
    except ValueError as e:
        print(f"Validation error: {e}", file=sys.stderr)
        sys.exit(1)

    print("\n" + "=" * 60)
    print("Concept Relationship Saved Successfully")
    print("=" * 60)
    print(f"Relationship ID: {rel.id}")
    print(f"Course:          {rel.course_id}")
    print(f"Source Concept:  {rel.source_concept_name or rel.source_concept_id}")
    print(f"Relationship:    {rel.relationship_type.value.upper()}")
    print(f"Target Concept:  {rel.target_concept_name or rel.target_concept_id}")
    print(f"Review Status:   {rel.review_status.value}")
    if rel.source_evidence:
        print(f"Evidence:        {rel.source_evidence}")
    print("=" * 60 + "\n")


def handle_update_relationship(args: argparse.Namespace) -> None:
    service = _setup_service()
    try:
        rel = service.update_relationship_status(args.id, args.status)
    except CycleDetectedError as e:
        print(f"Status update rejected: {e}", file=sys.stderr)
        sys.exit(1)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"Relationship '{rel.id}' status updated to '{rel.review_status.value}'.")


def handle_remove_relationship(args: argparse.Namespace) -> None:
    service = _setup_service()
    service.delete_concept_relationship(args.id)
    print(f"Relationship '{args.id}' successfully removed.")


def handle_prerequisites(args: argparse.Namespace) -> None:
    service = _setup_service()
    prereq_ids = service.get_prerequisites(args.course, args.concept_id)

    cpt = service.get_concept(args.concept_id)
    target_name = cpt.name if cpt else args.concept_id

    if args.json:
        print(json.dumps({"target_concept_id": args.concept_id, "prerequisites": prereq_ids}, indent=2))
        return

    print("\n" + "=" * 70)
    print(f"Prerequisite Analysis for Concept: {target_name} ({args.concept_id})")
    print("=" * 70)
    if not prereq_ids:
        print("No prerequisite requirements identified. This concept has no required antecedents.")
        print()
        return

    print(f"Total Prerequisites: {len(prereq_ids)} (in topological study order):\n")
    for idx, pid in enumerate(prereq_ids, 1):
        p_cpt = service.get_concept(pid)
        p_name = p_cpt.name if p_cpt else pid
        m = service.repository.get_concept_mastery("learner_default", pid)
        cat = m.mastery_category.value if m else "unassessed"
        score = f"{m.mastery_score:.2f}" if m else "0.00"
        print(f"  {idx}. {p_name:<28} (ID: {pid:<18}) Status: {cat:<15} Score: {score}")
    print("=" * 70 + "\n")


def handle_plan(args: argparse.Namespace) -> None:
    service = _setup_service()
    plan = service.generate_adaptive_study_plan(
        course_id=args.course,
        target_concept_id=args.target,
        time_budget_minutes=args.time_budget,
        max_items=args.max_items,
    )

    if args.json:
        print(plan.model_dump_json(indent=2))
        return

    print("\n" + "=" * 80)
    print("StudyOS Transparent Adaptive Study Plan")
    print("=" * 80)
    if plan.course_id:
        print(f"Course Scope:        {plan.course_id}")
    if plan.target_concept_name:
        print(f"Target Concept:      {plan.target_concept_name} ({plan.target_concept_id})")
    if plan.time_budget_minutes:
        print(f"Time Budget:         {plan.time_budget_minutes} min (Allocated: {plan.allocated_minutes} min)")
    else:
        print(f"Total Time Estimate: {plan.allocated_minutes} min")
    print(f"Total Study Actions: {plan.total_actions}")
    print("=" * 80)

    if not plan.items:
        print("No pending study items. You are fully caught up!")
        print()
        return

    for idx, item in enumerate(plan.items, 1):
        print(f"\n{idx}. [{item.action_title.upper()}] — {item.estimated_minutes} min (Priority {item.priority})")
        print(f"   Concept:  {item.concept_name} ({item.concept_id}) | Course: {item.course_id}")
        print(f"   Status:   Category '{item.mastery_category.value}' ({item.observation_count} observation(s), score: {item.mastery_score:.2f})")
        print(f"   Reason:   {item.reason}")
        if item.unmet_prerequisites:
            print(f"   Prereqs:  Unmet prerequisite concept IDs: {', '.join(item.unmet_prerequisites)}")
        print("-" * 80)
    print()


def handle_research(args: argparse.Namespace) -> None:
    if not args.external_research:
        print(
            "Error: External research requires explicit user opt-in (--external-research).\n"
            "StudyOS maintains a strict local-first privacy boundary. Run again with --external-research.",
            file=sys.stderr,
        )
        sys.exit(1)

    config = get_default_config()
    db = Database(os.getenv("STUDYOS_DB_PATH") or config.db_path)
    db.migrate()

    provider = WikipediaSearchProvider() if not args.fake else FakeSearchProvider()
    chat_provider = None
    health = check_ollama_health(config, model_name=config.tutor_model)
    if health.is_ready:
        chat_provider = OllamaChatProvider.from_config(config)

    research_svc = ExternalResearchService(
        db=db,
        provider=provider,
        chat_client=chat_provider,
    )

    try:
        summary = research_svc.research(
            query=args.query,
            course_id=args.course,
            max_results=args.max_results,
            fetch_pages=args.fetch_pages,
            explicit_opt_in=True,
        )
    except Exception as e:
        print(f"External research failed: {e}", file=sys.stderr)
        sys.exit(1)

    if args.json:
        print(summary.model_dump_json(indent=2))
        return

    print("\n" + "=" * 80)
    print("StudyOS External Research Report")
    print("=" * 80)
    print(f"Query:               {summary.query}")
    print(f"Status:              {summary.status.value.upper()}")
    print(f"External Sources:    {len(summary.sources)}")
    print("-" * 80)
    print(f"{summary.disclaimer}\n")
    print("-" * 80)
    print("SYNTHESIZED RESEARCH SUMMARY:")
    print(summary.summary_text)
    print("-" * 80)

    if summary.sources:
        print("\nVERIFIED EXTERNAL SOURCES:")
        print(f"{'Citation':<10} {'Publisher':<20} {'Title':<26} {'URL'}")
        print("-" * 84)
        for s in summary.sources:
            pub = s.publisher or "Web Source"
            print(f"{s.citation_marker:<10} {pub[:18]:<20} {s.title[:24]:<26} {s.url}")

    if summary.invalid_citations_rejected:
        print(f"\nRejected Hallucinated Citations: {', '.join(summary.invalid_citations_rejected)}")
    print("=" * 80 + "\n")


def handle_research_history(args: argparse.Namespace) -> None:
    config = get_default_config()
    db = Database(os.getenv("STUDYOS_DB_PATH") or config.db_path)
    db.migrate()
    repo = ResearchRepository(db)

    records = repo.list_research_records(course_id=args.course, limit=args.limit)

    if args.json:
        print(json.dumps([r.model_dump() for r in records], indent=2))
        return

    print("\n" + "=" * 80)
    print(f"StudyOS External Research History ({len(records)} queries)")
    print("=" * 80)
    if not records:
        print("No external research queries recorded.")
        print()
        return

    print(f"{'ID':<16} {'Timestamp':<18} {'Status':<10} {'Query'}")
    print("-" * 80)
    for r in records:
        ts = r.created_at[:16] if r.created_at else "Unknown"
        print(f"{r.id:<16} {ts:<18} {r.status.value:<10} {r.query[:32]}")
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
        print(f"Stability (S):   {mastery.stability:.2f}")
        print(f"Difficulty (D):  {mastery.difficulty:.2f}")
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
        print(f"Stability (S):       {mastery.stability:.2f}")
        print(f"Difficulty (D):      {mastery.difficulty:.2f}")
        print(f"Policy:              {mastery.scheduler_policy}")
    print(f"Diagnostic Reason:   {rec.reason}")
    print("=" * 70 + "\n")


# ---------------------------------------------------------------------------
# CLI Argument Parser
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="StudyOS Learner Model, Knowledge Graph, Adaptive Planning & Spaced Revision CLI",
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
    p_cpt.add_argument("--status", choices=["proposed", "accepted", "rejected"], help="Filter by review status")
    p_cpt.add_argument("--json", action="store_true", help="Output concepts as JSON")

    # discover-concepts
    p_disc = subparsers.add_parser("discover-concepts", help="Direct concept discovery from ingested documents")
    p_disc.add_argument("--course", required=True, help="Course ID")
    p_disc.add_argument("--document", help="Optional document ID")
    p_disc.add_argument("--max-chunks", type=int, default=15, help="Maximum chunks to inspect")
    p_disc.add_argument("--auto-accept", action="store_true", help="Automatically accept discovered concepts")
    p_disc.add_argument("--json", action="store_true", help="Output discovered concepts as JSON")

    # relationships
    p_rels = subparsers.add_parser("relationships", help="List concept relationships for a course")
    p_rels.add_argument("--course", required=True, help="Course ID")
    p_rels.add_argument("--status", choices=["proposed", "accepted", "rejected"], help="Status filter")
    p_rels.add_argument("--type", choices=["prerequisite", "related"], help="Relationship type filter")
    p_rels.add_argument("--json", action="store_true", help="Output as JSON")

    # add-relationship
    p_add_rel = subparsers.add_parser("add-relationship", help="Create a directed concept relationship")
    p_add_rel.add_argument("--course", required=True, help="Course ID")
    p_add_rel.add_argument("--source", required=True, help="Source concept ID")
    p_add_rel.add_argument("--target", required=True, help="Target concept ID")
    p_add_rel.add_argument("--type", default="prerequisite", choices=["prerequisite", "related"], help="Relationship type")
    p_add_rel.add_argument("--status", default="accepted", choices=["accepted", "proposed", "rejected"], help="Status")
    p_add_rel.add_argument("--evidence", help="Optional supporting source evidence text")
    p_add_rel.add_argument("--confidence", type=float, default=1.0, help="Confidence score [0.0, 1.0]")

    # update-relationship
    p_upd_rel = subparsers.add_parser("update-relationship", help="Accept or reject a relationship")
    p_upd_rel.add_argument("--id", required=True, help="Relationship ID")
    p_upd_rel.add_argument("--status", required=True, choices=["accepted", "rejected"], help="New status")

    # remove-relationship
    p_del_rel = subparsers.add_parser("remove-relationship", help="Delete a concept relationship")
    p_del_rel.add_argument("--id", required=True, help="Relationship ID")

    # prerequisites
    p_pre = subparsers.add_parser("prerequisites", help="List topological prerequisites for a concept")
    p_pre.add_argument("--course", required=True, help="Course ID")
    p_pre.add_argument("--concept-id", required=True, help="Target concept ID")
    p_pre.add_argument("--json", action="store_true", help="Output prerequisites as JSON")

    # plan
    p_plan = subparsers.add_parser("plan", help="Generate transparent adaptive study plan for today")
    p_plan.add_argument("--course", help="Optional course ID")
    p_plan.add_argument("--target", help="Optional target concept ID")
    p_plan.add_argument("--time-budget", type=int, help="Optional study time budget in minutes (e.g. 30, 60)")
    p_plan.add_argument("--max-items", type=int, default=10, help="Maximum study items to schedule")
    p_plan.add_argument("--json", action="store_true", help="Output plan as JSON")

    # research
    p_res = subparsers.add_parser("research", help="Run explicit external research with citations")
    p_res.add_argument("--query", required=True, help="External search query text")
    p_res.add_argument("--course", help="Optional course ID to attach findings to")
    p_res.add_argument("--max-results", type=int, default=5, help="Maximum external sources to query")
    p_res.add_argument("--fetch-pages", action="store_true", help="Safely fetch and inspect external web pages")
    p_res.add_argument("--external-research", action="store_true", help="Explicit user opt-in for external network queries")
    p_res.add_argument("--fake", action="store_true", help="Use fake search provider (for offline testing)")
    p_res.add_argument("--json", action="store_true", help="Output research report as JSON")

    # research-history
    p_rhist = subparsers.add_parser("research-history", help="List past external research operations")
    p_rhist.add_argument("--course", help="Optional course ID filter")
    p_rhist.add_argument("--limit", type=int, default=20, help="Maximum records to display")
    p_rhist.add_argument("--json", action="store_true", help="Output as JSON")

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
    elif args.command == "discover-concepts":
        handle_discover_concepts(args)
    elif args.command == "relationships":
        handle_relationships(args)
    elif args.command == "add-relationship":
        handle_add_relationship(args)
    elif args.command == "update-relationship":
        handle_update_relationship(args)
    elif args.command == "remove-relationship":
        handle_remove_relationship(args)
    elif args.command == "prerequisites":
        handle_prerequisites(args)
    elif args.command == "plan":
        handle_plan(args)
    elif args.command == "research":
        handle_research(args)
    elif args.command == "research-history":
        handle_research_history(args)
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
