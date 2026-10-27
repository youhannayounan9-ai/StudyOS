"""Diagnostic CLI utility to verify single-document ingestion in StudyOS."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

from studyos.core.config import AppConfig
from studyos.ingestion.service import IngestionService
from studyos.storage.db import Database


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Verify document ingestion into StudyOS SQLite store."
    )
    parser.add_argument(
        "--file",
        "-f",
        required=True,
        type=Path,
        help="Path to document file (PDF, DOCX, PPTX, TXT, MD).",
    )
    parser.add_argument(
        "--course",
        "-c",
        required=True,
        type=str,
        help="Course code or name (e.g. CS101, Algorithms).",
    )
    parser.add_argument(
        "--db",
        "-d",
        type=Path,
        default=None,
        help="Custom path to SQLite database (defaults to AppConfig location).",
    )
    parser.add_argument(
        "--force-reindex",
        action="store_true",
        help="Force re-indexing if document already exists with matching hash.",
    )

    args = parser.parse_args()
    file_path: Path = args.file.resolve()

    if not file_path.is_file():
        print(f"Error: File not found: {file_path}", file=sys.stderr)
        return 1

    config = AppConfig()
    db_path = args.db.resolve() if args.db else config.db_path
    print(f"Connecting to database: {db_path}")

    try:
        db = Database(db_path=db_path)
        db.migrate()

        # Find or create target course
        course = db.get_course_by_code(args.course)
        if not course:
            # Check by name or create course
            all_courses = db.list_courses()
            matching = [c for c in all_courses if c.name.lower() == args.course.lower()]
            if matching:
                course = matching[0]
            else:
                course = db.create_course(
                    name=args.course,
                    code=args.course.upper(),
                    description=f"Course {args.course} created by verification CLI.",
                )
                print(f"Created new course: {course.name} ({course.id})")
        else:
            print(f"Found existing course: {course.name} ({course.id})")

        service = IngestionService(db=db, config=config)
        print(f"Ingesting file: {file_path.name} ({file_path.stat().st_size} bytes)...")

        result = service.ingest_file(
            file_path=file_path,
            course_id=course.id,
            force_reindex=args.force_reindex,
        )

        print("\n--- Ingestion Summary ---")
        print(f"Status:         {result.status.upper()}")
        print(f"Document ID:    {result.document_id}")
        print(f"SHA-256 Hash:   {result.sha256_hash}")
        print(f"Elements:       {result.element_count}")
        print(f"Chunks:         {result.chunk_count}")
        print(f"FTS5 Rows:      {db.count_fts_rows()}")

        if result.warnings:
            print("\nWarnings:")
            for w in result.warnings:
                print(f"  [!] {w}")

        if result.status == "duplicate":
            print("\n[Notice] File was recognized as an existing duplicate; duplicate processing was skipped.")
            print("Use --force-reindex to replace and re-chunk.")

        return 0

    except Exception as e:
        print(f"\n[Ingestion Failed] {type(e).__name__}: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
