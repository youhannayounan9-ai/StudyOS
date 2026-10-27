"""CLI tool for generating, taking, and grading practice exams in StudyOS."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from studyos.assessment.models import (
    AttemptStatus,
    Difficulty,
    ExamRequest,
    GradingStatus,
    QuestionType,
    SubmitAnswersRequest,
)
from studyos.assessment.service import AssessmentService
from studyos.core.config import get_default_config
from studyos.core.health import check_ollama_health
from studyos.inference.fake import FakeChatProvider
from studyos.inference.ollama import OllamaChatProvider
from studyos.retrieval.embedding import FakeEmbeddingProvider, OllamaEmbeddingProvider
from studyos.retrieval.hybrid import HybridRetrievalService
from studyos.retrieval.vector_store import ChromaVectorStore
from studyos.storage.db import Database


def _setup_service(
    require_chat: bool = True,
    fake_chat: bool = False,
    fake_embeddings: bool = False,
) -> AssessmentService:
    config = get_default_config()
    db = Database(config.db_path)
    db.migrate()

    # Embedding provider
    if fake_embeddings:
        embed_provider = FakeEmbeddingProvider(dimension=config.embedding_dimension)
    else:
        embed_health = check_ollama_health(config, model_name=config.embedding_model)
        if embed_health.is_ready:
            embed_provider = OllamaEmbeddingProvider.from_config(config)
        else:
            embed_provider = None

    vector_store = ChromaVectorStore.from_config(config, provider=embed_provider)
    retrieval_service = HybridRetrievalService(
        db=db,
        vector_store=vector_store,
        rrf_k=config.rrf_k,
    )

    # Chat provider
    chat_provider = None
    if fake_chat:
        chat_provider = FakeChatProvider(
            model_name="fake-tutor",
            default_response=json.dumps(
                {
                    "questions": [
                        {
                            "type": "mcq",
                            "question": "What is the primary characteristic of Dijkstra's algorithm?",
                            "difficulty": "medium",
                            "options": [
                                {"id": "A", "text": "It handles negative edge weights"},
                                {"id": "B", "text": "It finds single-source shortest paths on non-negative weights"},
                                {"id": "C", "text": "It computes all-pairs shortest paths"},
                                {"id": "D", "text": "It relies on Bellman-Ford relaxations"}
                            ],
                            "correct_answer": "B",
                            "explanation": "Dijkstra solves single-source shortest paths on non-negative graphs [E1].",
                            "evidence_markers": ["E1"]
                        },
                        {
                            "type": "true_false",
                            "question": "Bellman-Ford algorithm can detect negative weight cycles in a graph.",
                            "difficulty": "easy",
                            "correct_answer": "True",
                            "explanation": "If a relaxation succeeds after |V|-1 passes, a negative cycle exists [E1].",
                            "evidence_markers": ["E1"]
                        }
                    ]
                }
            ),
        )
    elif require_chat:
        chat_health = check_ollama_health(config, model_name=config.tutor_model)
        if not chat_health.is_ready:
            print("\n" + "=" * 65, file=sys.stderr)
            print("ERROR: Local AI Chat Service Unavailable", file=sys.stderr)
            print("=" * 65, file=sys.stderr)
            print(f"Status:   {chat_health.status.value.upper()}", file=sys.stderr)
            print(f"Base URL: {chat_health.base_url}", file=sys.stderr)
            print(f"Model:    {chat_health.model_name}", file=sys.stderr)
            print(f"Message:  {chat_health.message}", file=sys.stderr)
            if chat_health.actionable_remediation:
                print("-" * 65, file=sys.stderr)
                print("Action Required:", file=sys.stderr)
                print(chat_health.actionable_remediation, file=sys.stderr)
            print("=" * 65, file=sys.stderr)
            print("Tip: Use '--fake-chat' to generate exams without running Ollama.", file=sys.stderr)
            sys.exit(1)

        chat_provider = OllamaChatProvider.from_config(config)

    return AssessmentService(
        db=db,
        retrieval_service=retrieval_service,
        chat_provider=chat_provider,
        config=config,
    )


def handle_generate(args: argparse.Namespace) -> None:
    service = _setup_service(fake_chat=args.fake_chat, fake_embeddings=args.fake_embeddings)

    # Resolve course code if necessary
    course_id = args.course
    course_obj = service.db.get_course_by_code(course_id)
    if course_obj:
        course_id = course_obj.id

    q_types = [QuestionType(args.type)] if args.type else None

    req = ExamRequest(
        course_id=course_id,
        question_count=args.count,
        question_types=q_types,
        difficulty=Difficulty(args.difficulty),
        topic=args.topic,
        title=args.title,
    )

    try:
        exam = service.generate_exam(req)
        print("\n" + "=" * 70)
        print("StudyOS — Exam Generation Successful")
        print("=" * 70)
        print(f"Exam ID:        {exam.id}")
        print(f"Title:          {exam.title}")
        print(f"Course ID:      {exam.course_id}")
        print(f"Difficulty:     {exam.difficulty.value.upper()}")
        print(f"Question Count: {exam.question_count}")
        print("=" * 70)
        print(f"\nTo inspect and take this exam, run:\n  python scripts/exams.py take --exam-id {exam.id}\n")
    except Exception as e:
        print(f"\nError generating exam: {e}", file=sys.stderr)
        sys.exit(1)


def handle_list(args: argparse.Namespace) -> None:
    service = _setup_service(require_chat=False)
    exams = service.list_exams(course_id=args.course)

    print("\n" + "=" * 70)
    print("StudyOS — Stored Practice Exams")
    print("=" * 70)
    if not exams:
        print("No exams found.")
        print("=" * 70 + "\n")
        return

    for ex in exams:
        print(f"• ID: {ex.id} | {ex.title}")
        print(f"  Course: {ex.course_id} | Questions: {ex.question_count} | Diff: {ex.difficulty.value.upper()}")
        print(f"  Created: {ex.created_at or 'N/A'}")
        print("-" * 70)
    print("=" * 70 + "\n")


def handle_take(args: argparse.Namespace) -> None:
    service = _setup_service(require_chat=False)
    try:
        public_exam = service.get_exam_for_taking(args.exam_id)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    print("\n" + "=" * 70)
    print(f"EXAM: {public_exam.title}")
    print(f"Exam ID: {public_exam.id} | Questions: {public_exam.question_count} | Difficulty: {public_exam.difficulty.value.upper()}")
    print("=" * 70)

    for q in public_exam.questions:
        print(f"\nQuestion {q.question_index} [{q.question_type.value.upper()}]:")
        print(f"  {q.question_text}")
        if q.options:
            for opt in q.options:
                print(f"    ({opt.id}) {opt.text}")

    print("\n" + "=" * 70)
    print("SUBMISSION INSTRUCTIONS:")
    print("Submit your answers using a JSON file or string:")
    print('  python scripts/exams.py submit --exam-id ' + public_exam.id + ' --answers \'{"' + public_exam.questions[0].id + '": "B"}\'')
    print("=" * 70 + "\n")


def handle_submit(args: argparse.Namespace) -> None:
    service = _setup_service(require_chat=False, fake_chat=args.fake_chat)

    raw_answers_str = args.answers
    if not raw_answers_str and args.answers_file:
        raw_answers_str = Path(args.answers_file).read_text(encoding="utf-8")

    if not raw_answers_str:
        print("Error: Must provide --answers JSON string or --answers-file path.", file=sys.stderr)
        sys.exit(1)

    try:
        answers_dict = json.loads(raw_answers_str)
        if not isinstance(answers_dict, dict):
            raise ValueError("Answers payload must be a JSON dictionary mapping question_id -> answer.")
    except Exception as e:
        print(f"Error parsing answers JSON: {e}", file=sys.stderr)
        sys.exit(1)

    # Start attempt
    try:
        attempt = service.start_attempt(args.exam_id)
        result = service.submit_attempt(attempt.id, SubmitAnswersRequest(answers=answers_dict))
    except Exception as e:
        print(f"Submission error: {e}", file=sys.stderr)
        sys.exit(1)

    _display_results(result)


def handle_results(args: argparse.Namespace) -> None:
    service = _setup_service(require_chat=False)
    result = service.get_attempt_result(args.attempt_id)
    if not result:
        print(f"Error: No completed attempt found for ID '{args.attempt_id}'.", file=sys.stderr)
        sys.exit(1)

    _display_results(result)


def handle_review(args: argparse.Namespace) -> None:
    service = _setup_service(require_chat=False)
    try:
        updated = service.grade_short_answer_manually(
            attempt_id=args.attempt_id,
            question_id=args.question_id,
            score=args.score,
            feedback=args.feedback,
        )
        print("\n" + "=" * 70)
        print("StudyOS — Instructor Manual Review Recorded")
        print("=" * 70)
        _display_results(updated)
    except Exception as e:
        print(f"Error recording review: {e}", file=sys.stderr)
        sys.exit(1)


def _display_results(result) -> None:
    print("\n" + "=" * 70)
    print("StudyOS — Graded Exam Report")
    print("=" * 70)
    print(f"Exam:        {result.title} ({result.exam_id})")
    print(f"Attempt ID:  {result.attempt_id}")
    if result.status == AttemptStatus.PENDING_REVIEW or result.needs_manual_review:
        print(f"Status:      PENDING MANUAL REVIEW")
        print(f"Total Score: PENDING (Provisional Subtotal: {result.provisional_score:.1f} / {result.max_score:.1f})")
        prov_pct_str = f"{result.provisional_percentage:.1f}%" if result.provisional_percentage is not None else "0.0%"
        print(f"Percentage:  PENDING (Provisional: {prov_pct_str})")
        print("Notice:      One or more short-answer responses require instructor review.")
    else:
        print(f"Status:      COMPLETED")
        score_str = f"{result.score:.1f}" if result.score is not None else "N/A"
        pct_str = f"{result.percentage:.1f}%" if result.percentage is not None else "N/A"
        print(f"Total Score: {score_str} / {result.max_score:.1f} ({pct_str})")
    print(f"Completed:   {result.completed_at or 'In Review'}")
    print("=" * 70)

    for qr in result.questions:
        print(f"\nQuestion {qr.question_index}: {qr.question_text}")
        print(f"  Your Answer:    {qr.submitted_answer or '<Skipped>'}")
        print(f"  Correct Answer: {qr.correct_answer}")

        if qr.status == GradingStatus.PENDING_MANUAL_REVIEW:
            status_label = "PENDING MANUAL REVIEW"
            score_label = f"[PENDING REVIEW] (Max: {qr.max_score:.1f} pts)"
        elif qr.is_correct is True:
            status_label = "CORRECT"
            score_label = f"{qr.score:.1f}/{qr.max_score:.1f} pts"
        else:
            status_label = "INCORRECT"
            score_label = f"{qr.score:.1f}/{qr.max_score:.1f} pts"

        print(f"  Result:         {status_label} ({score_label})")
        print(f"  Feedback:       {qr.feedback}")
        print(f"  Explanation:    {qr.explanation}")

        if qr.citations:
            print("  Supporting Sources:")
            for c in qr.citations:
                loc = f"{c.location_type} {c.location_index}" if c.location_type and c.location_index else ""
                stale_marker = " [STALE/UNAVAILABLE: historical source from generation time]" if c.is_stale else ""
                print(f"    • [{c.evidence_marker}] {c.filename or 'Document'} {loc}{stale_marker}")
                if c.excerpt:
                    print(f"      Excerpt: \"{c.excerpt}\"")
        print("-" * 70)

    print("=" * 70 + "\n")


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description="StudyOS Practice Exam & Assessment Engine CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # 1. Generate
    gen_parser = subparsers.add_parser("generate", help="Generate a new practice exam from course materials")
    gen_parser.add_argument("--course", "-c", type=str, required=True, help="Course ID or code")
    gen_parser.add_argument("--topic", "-t", type=str, default=None, help="Topic or chapter focus")
    gen_parser.add_argument("--count", "-n", type=int, default=5, help="Question count (default: 5)")
    gen_parser.add_argument("--type", type=str, choices=["mcq", "true_false", "short_answer"], default=None, help="Question type")
    gen_parser.add_argument("--difficulty", "-d", type=str, choices=["easy", "medium", "hard", "mixed"], default="medium")
    gen_parser.add_argument("--title", type=str, default=None, help="Custom exam title")
    gen_parser.add_argument("--fake-chat", action="store_true", help="Use offline deterministic fake chat provider")
    gen_parser.add_argument("--fake-embeddings", action="store_true", help="Use offline fake embeddings")

    # 2. List
    list_parser = subparsers.add_parser("list", help="List stored exams")
    list_parser.add_argument("--course", "-c", type=str, default=None, help="Filter by course ID")

    # 3. Take
    take_parser = subparsers.add_parser("take", help="Inspect exam questions in taking mode (NO answer keys)")
    take_parser.add_argument("--exam-id", "-e", type=str, required=True, help="Exam ID to take")

    # 4. Submit
    sub_parser = subparsers.add_parser("submit", help="Submit answers for an exam and grade the attempt")
    sub_parser.add_argument("--exam-id", "-e", type=str, required=True, help="Exam ID being taken")
    sub_parser.add_argument("--answers", type=str, default=None, help="JSON string mapping question_id -> answer")
    sub_parser.add_argument("--answers-file", type=str, default=None, help="Path to JSON file containing answers")
    sub_parser.add_argument("--fake-chat", action="store_true", help="Use fake chat for short-answer evaluation")

    # 5. Results
    res_parser = subparsers.add_parser("results", help="View results of a completed attempt")
    res_parser.add_argument("--attempt-id", "-a", type=str, required=True, help="Attempt ID")

    # 6. Review (Instructor manual grading for short answers)
    rev_parser = subparsers.add_parser("review", help="Record instructor review for a pending short-answer question")
    rev_parser.add_argument("--attempt-id", "-a", type=str, required=True, help="Attempt ID to review")
    rev_parser.add_argument("--question-id", "-q", type=str, required=True, help="Question ID to grade")
    rev_parser.add_argument("--score", "-s", type=float, required=True, help="Score to award (0.0 to max_score)")
    rev_parser.add_argument("--feedback", "-f", type=str, default=None, help="Optional instructor feedback")

    args = parser.parse_args()

    if args.command == "generate":
        handle_generate(args)
    elif args.command == "list":
        handle_list(args)
    elif args.command == "take":
        handle_take(args)
    elif args.command == "submit":
        handle_submit(args)
    elif args.command == "results":
        handle_results(args)
    elif args.command == "review":
        handle_review(args)


if __name__ == "__main__":
    main()
