-- StudyOS Migration 0007: Concept Graph, Discovery Runs, Advanced Scheduler & External Research
-- Adds normalized tables for course-scoped concept relationships (prerequisites & related),
-- concept discovery tracking, advanced scheduler state (stability/difficulty/retrievability),
-- and external research queries and validated sources.

-- 1. Concept Relationships (Course-Scoped Concept Graph)
CREATE TABLE IF NOT EXISTS concept_relationships (
    id TEXT PRIMARY KEY,
    course_id TEXT NOT NULL REFERENCES courses(id) ON DELETE CASCADE,
    source_concept_id TEXT NOT NULL REFERENCES concepts(id) ON DELETE CASCADE,
    target_concept_id TEXT NOT NULL REFERENCES concepts(id) ON DELETE CASCADE,
    relationship_type TEXT NOT NULL CHECK (relationship_type IN ('prerequisite', 'related')),
    provenance TEXT NOT NULL DEFAULT 'manual' CHECK (provenance IN ('manual', 'metadata_derived', 'model_proposed', 'discovered')),
    source_evidence TEXT,
    confidence REAL DEFAULT 1.0 CHECK (confidence IS NULL OR (confidence >= 0.0 AND confidence <= 1.0)),
    review_status TEXT NOT NULL DEFAULT 'accepted' CHECK (review_status IN ('proposed', 'accepted', 'rejected')),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    CHECK (source_concept_id != target_concept_id),
    UNIQUE(course_id, source_concept_id, target_concept_id, relationship_type)
);

CREATE INDEX IF NOT EXISTS idx_concept_rel_course_id ON concept_relationships(course_id);
CREATE INDEX IF NOT EXISTS idx_concept_rel_source ON concept_relationships(source_concept_id);
CREATE INDEX IF NOT EXISTS idx_concept_rel_target ON concept_relationships(target_concept_id);
CREATE INDEX IF NOT EXISTS idx_concept_rel_type ON concept_relationships(relationship_type);
CREATE INDEX IF NOT EXISTS idx_concept_rel_status ON concept_relationships(review_status);

-- Database-level cross-course integrity triggers
CREATE TRIGGER IF NOT EXISTS trg_concept_rel_insert_course_integrity
BEFORE INSERT ON concept_relationships
FOR EACH ROW
BEGIN
    SELECT RAISE(ABORT, 'Cross-course concept relationship rejected: source_concept_id does not belong to course_id')
    WHERE NOT EXISTS (SELECT 1 FROM concepts WHERE id = NEW.source_concept_id AND course_id = NEW.course_id);

    SELECT RAISE(ABORT, 'Cross-course concept relationship rejected: target_concept_id does not belong to course_id')
    WHERE NOT EXISTS (SELECT 1 FROM concepts WHERE id = NEW.target_concept_id AND course_id = NEW.course_id);
END;

CREATE TRIGGER IF NOT EXISTS trg_concept_rel_update_course_integrity
BEFORE UPDATE ON concept_relationships
FOR EACH ROW
BEGIN
    SELECT RAISE(ABORT, 'Cross-course concept relationship rejected: source_concept_id does not belong to course_id')
    WHERE NOT EXISTS (SELECT 1 FROM concepts WHERE id = NEW.source_concept_id AND course_id = NEW.course_id);

    SELECT RAISE(ABORT, 'Cross-course concept relationship rejected: target_concept_id does not belong to course_id')
    WHERE NOT EXISTS (SELECT 1 FROM concepts WHERE id = NEW.target_concept_id AND course_id = NEW.course_id);
END;

-- Database-level cycle prevention triggers for accepted prerequisites
CREATE TRIGGER IF NOT EXISTS trg_concept_rel_insert_cycle_check
BEFORE INSERT ON concept_relationships
FOR EACH ROW
WHEN NEW.relationship_type = 'prerequisite' AND NEW.review_status = 'accepted'
BEGIN
    SELECT RAISE(ABORT, 'Cycle detected in prerequisite graph')
    WHERE NEW.source_concept_id = NEW.target_concept_id
       OR EXISTS (
           WITH RECURSIVE reach(node) AS (
               SELECT NEW.target_concept_id
               UNION
               SELECT cr.target_concept_id
               FROM concept_relationships cr
               JOIN reach r ON cr.source_concept_id = r.node
               WHERE cr.course_id = NEW.course_id
                 AND cr.relationship_type = 'prerequisite'
                 AND cr.review_status = 'accepted'
           )
           SELECT 1 FROM reach WHERE node = NEW.source_concept_id
       );
END;

CREATE TRIGGER IF NOT EXISTS trg_concept_rel_update_cycle_check
BEFORE UPDATE ON concept_relationships
FOR EACH ROW
WHEN NEW.relationship_type = 'prerequisite' AND NEW.review_status = 'accepted'
BEGIN
    SELECT RAISE(ABORT, 'Cycle detected in prerequisite graph')
    WHERE NEW.source_concept_id = NEW.target_concept_id
       OR EXISTS (
           WITH RECURSIVE reach(node) AS (
               SELECT NEW.target_concept_id
               UNION
               SELECT cr.target_concept_id
               FROM concept_relationships cr
               JOIN reach r ON cr.source_concept_id = r.node
               WHERE cr.id != OLD.id
                 AND cr.course_id = NEW.course_id
                 AND cr.relationship_type = 'prerequisite'
                 AND cr.review_status = 'accepted'
           )
           SELECT 1 FROM reach WHERE node = NEW.source_concept_id
       );
END;

-- 2. Concept Discovery Runs
CREATE TABLE IF NOT EXISTS concept_discovery_runs (
    id TEXT PRIMARY KEY,
    course_id TEXT NOT NULL REFERENCES courses(id) ON DELETE CASCADE,
    document_id TEXT REFERENCES documents(id) ON DELETE SET NULL,
    status TEXT NOT NULL DEFAULT 'completed', -- 'started', 'completed', 'failed'
    concepts_discovered INTEGER NOT NULL DEFAULT 0,
    chunks_processed INTEGER NOT NULL DEFAULT 0,
    error_message TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_discovery_runs_course_id ON concept_discovery_runs(course_id);
CREATE INDEX IF NOT EXISTS idx_discovery_runs_doc_id ON concept_discovery_runs(document_id);

-- 3. Extend concepts table with provenance and review status
ALTER TABLE concepts ADD COLUMN review_status TEXT NOT NULL DEFAULT 'accepted';
ALTER TABLE concepts ADD COLUMN provenance TEXT NOT NULL DEFAULT 'assessment_derived';
ALTER TABLE concepts ADD COLUMN learning_objectives TEXT;

CREATE TRIGGER IF NOT EXISTS trg_concepts_insert_status
BEFORE INSERT ON concepts
FOR EACH ROW
BEGIN
    SELECT RAISE(ABORT, 'Invalid concept review_status')
    WHERE NEW.review_status NOT IN ('proposed', 'accepted', 'rejected');

    SELECT RAISE(ABORT, 'Invalid concept provenance')
    WHERE NEW.provenance NOT IN ('manual', 'assessment_derived', 'discovered', 'model_proposed');
END;

CREATE TRIGGER IF NOT EXISTS trg_concepts_update_status
BEFORE UPDATE ON concepts
FOR EACH ROW
BEGIN
    SELECT RAISE(ABORT, 'Invalid concept review_status')
    WHERE NEW.review_status NOT IN ('proposed', 'accepted', 'rejected');

    SELECT RAISE(ABORT, 'Invalid concept provenance')
    WHERE NEW.provenance NOT IN ('manual', 'assessment_derived', 'discovered', 'model_proposed');
END;

-- Enforce concept course scope immutability when dependent records exist
CREATE TRIGGER IF NOT EXISTS concepts_validate_course_au
BEFORE UPDATE OF course_id ON concepts
FOR EACH ROW
WHEN NEW.course_id != OLD.course_id
BEGIN
    SELECT RAISE(ABORT, 'Cannot update concept course_id: concept participates in relationships')
    WHERE EXISTS (
        SELECT 1 FROM concept_relationships
        WHERE source_concept_id = OLD.id OR target_concept_id = OLD.id
    );

    SELECT RAISE(ABORT, 'Cannot update concept course_id: concept has chunk provenance')
    WHERE EXISTS (
        SELECT 1 FROM concept_chunks
        WHERE concept_id = OLD.id
    );

    SELECT RAISE(ABORT, 'Cannot update concept course_id: concept has mastery records')
    WHERE EXISTS (
        SELECT 1 FROM concept_mastery
        WHERE concept_id = OLD.id
    );

    SELECT RAISE(ABORT, 'Cannot update concept course_id: concept has question associations')
    WHERE EXISTS (
        SELECT 1 FROM question_concepts
        WHERE concept_id = OLD.id
    );
END;

-- Enforce chunk provenance course integrity at the database layer
CREATE TRIGGER IF NOT EXISTS trg_concept_chunks_insert_course_integrity
BEFORE INSERT ON concept_chunks
FOR EACH ROW
WHEN NEW.chunk_id IS NOT NULL
BEGIN
    SELECT RAISE(ABORT, 'Cross-course chunk provenance rejected: chunk course_id != concept course_id')
    WHERE (SELECT course_id FROM concepts WHERE id = NEW.concept_id) !=
          (SELECT course_id FROM chunks WHERE id = NEW.chunk_id);
END;

CREATE TRIGGER IF NOT EXISTS trg_concept_chunks_update_course_integrity
BEFORE UPDATE ON concept_chunks
FOR EACH ROW
WHEN NEW.chunk_id IS NOT NULL
BEGIN
    SELECT RAISE(ABORT, 'Cross-course chunk provenance rejected: chunk course_id != concept course_id')
    WHERE (SELECT course_id FROM concepts WHERE id = NEW.concept_id) !=
          (SELECT course_id FROM chunks WHERE id = NEW.chunk_id);
END;

-- 4. Extend concept_mastery table with advanced forgetting-aware scheduler fields (FSRS DSR model)
ALTER TABLE concept_mastery ADD COLUMN stability REAL NOT NULL DEFAULT 1.0;
ALTER TABLE concept_mastery ADD COLUMN difficulty REAL NOT NULL DEFAULT 5.0;
ALTER TABLE concept_mastery ADD COLUMN retrievability REAL NOT NULL DEFAULT 1.0;
ALTER TABLE concept_mastery ADD COLUMN scheduler_policy TEXT NOT NULL DEFAULT 'local-fsrs-v1';

-- 5. Extend review_tasks table with scheduler policy and metrics
ALTER TABLE review_tasks ADD COLUMN scheduler_policy TEXT NOT NULL DEFAULT 'local-fsrs-v1';
ALTER TABLE review_tasks ADD COLUMN scheduled_interval REAL DEFAULT 1.0;
ALTER TABLE review_tasks ADD COLUMN estimated_retention REAL DEFAULT 1.0;

-- 6. External Research Records
CREATE TABLE IF NOT EXISTS external_research_records (
    id TEXT PRIMARY KEY,
    query TEXT NOT NULL,
    course_id TEXT REFERENCES courses(id) ON DELETE SET NULL,
    summary TEXT,
    status TEXT NOT NULL DEFAULT 'success', -- 'success', 'empty', 'failed'
    error_message TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_ext_research_course ON external_research_records(course_id);
CREATE INDEX IF NOT EXISTS idx_ext_research_created ON external_research_records(created_at);

-- 7. External Research Sources (Strict Source Provenance & Citations)
CREATE TABLE IF NOT EXISTS external_research_sources (
    id TEXT PRIMARY KEY,
    research_id TEXT NOT NULL REFERENCES external_research_records(id) ON DELETE CASCADE,
    citation_index INTEGER NOT NULL, -- e.g. 1 for [EXT-1]
    url TEXT NOT NULL,
    title TEXT NOT NULL,
    publisher TEXT,
    published_date TEXT,
    retrieved_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    snippet TEXT,
    source_type TEXT NOT NULL DEFAULT 'web_page',
    is_valid INTEGER NOT NULL DEFAULT 1,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(research_id, citation_index)
);

CREATE INDEX IF NOT EXISTS idx_ext_sources_research_id ON external_research_sources(research_id);
CREATE INDEX IF NOT EXISTS idx_ext_sources_url ON external_research_sources(url);
