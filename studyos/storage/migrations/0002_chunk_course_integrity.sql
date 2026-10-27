-- StudyOS Migration 0002: Enforce chunk-document course_id consistency
-- Ensures a chunk's course_id strictly matches the course_id of its parent document.

-- Step 1: Repair any pre-existing inconsistent chunks where chunk.course_id != parent document.course_id.
-- Updating course_id fires the existing chunks_au trigger, keeping chunks_fts synchronized.
UPDATE chunks
SET course_id = (
    SELECT documents.course_id
    FROM documents
    WHERE documents.id = chunks.document_id
)
WHERE EXISTS (
    SELECT 1
    FROM documents
    WHERE documents.id = chunks.document_id
      AND documents.course_id != chunks.course_id
);

-- Step 2: Install validation triggers preventing future invariant violations.
CREATE TRIGGER IF NOT EXISTS chunks_validate_course_ai BEFORE INSERT ON chunks
BEGIN
    SELECT RAISE(ABORT, 'Chunk course_id does not match parent document course_id')
    FROM documents
    WHERE id = new.document_id AND course_id != new.course_id;
END;

CREATE TRIGGER IF NOT EXISTS chunks_validate_course_au BEFORE UPDATE OF document_id, course_id ON chunks
BEGIN
    SELECT RAISE(ABORT, 'Chunk course_id does not match parent document course_id')
    FROM documents
    WHERE id = new.document_id AND course_id != new.course_id;
END;

CREATE TRIGGER IF NOT EXISTS documents_validate_course_au BEFORE UPDATE OF course_id ON documents
BEGIN
    SELECT RAISE(ABORT, 'Cannot update document course_id: child chunks exist with mismatched course_id')
    FROM chunks
    WHERE document_id = old.id AND course_id != new.course_id;
END;
