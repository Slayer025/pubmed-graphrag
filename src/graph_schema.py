"""Graph schema definitions for the PubMed GraphRAG Neo4j knowledge graph."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, TypedDict

# Node labels
ARTICLE_LABEL = "Article"
CHUNK_LABEL = "Chunk"
ENTITY_LABEL = "Entity"

# Relationship types
REL_HAS_CHUNK = "HAS_CHUNK"
REL_MENTIONS = "MENTIONS"

RelationshipType = Literal["HAS_CHUNK", "MENTIONS"]


class ArticleNode(TypedDict):
    """An Article node in the graph."""

    article_id: str
    abstract: str


class ChunkNode(TypedDict):
    """A Chunk node in the graph."""

    chunk_id: str
    article_id: str
    text: str
    strategy: str
    embedding: str  # semicolon-separated float values for CSV transport


class EntityNode(TypedDict):
    """An Entity node in the graph."""

    entity_id: str
    name: str
    label: str


class MentionRel(TypedDict):
    """A MENTIONS relationship from a Chunk to an Entity."""

    chunk_id: str
    entity_id: str


class HasChunkRel(TypedDict):
    """A HAS_CHUNK relationship from an Article to a Chunk."""

    article_id: str
    chunk_id: str


@dataclass(frozen=True)
class GraphSchema:
    """Collects node/relationship files and Cypher statements for one export."""

    output_dir: str
    articles_csv: str = "articles.csv"
    chunks_csv: str = "chunks.csv"
    entities_csv: str = "entities.csv"
    has_chunk_csv: str = "has_chunk.csv"
    mentions_csv: str = "mentions.csv"
    cypher_file: str = "schema.cypher"

    @property
    def cypher(self) -> str:
        """Return the Cypher script to load the exported CSV files into Neo4j."""
        return _CYPHER_TEMPLATE.format(
            articles_csv=self.articles_csv,
            chunks_csv=self.chunks_csv,
            entities_csv=self.entities_csv,
            has_chunk_csv=self.has_chunk_csv,
            mentions_csv=self.mentions_csv,
        )


_CYPHER_TEMPLATE = """// Neo4j graph schema and import script for PubMed GraphRAG.
// Run this in the Neo4j Browser or via `cypher-shell` after copying the CSV
// files next to this script (or adjusting the file:/// paths).

// Constraints and indexes
CREATE CONSTRAINT article_id IF NOT EXISTS
    FOR (a:Article) REQUIRE a.article_id IS UNIQUE;

CREATE CONSTRAINT chunk_id IF NOT EXISTS
    FOR (c:Chunk) REQUIRE c.chunk_id IS UNIQUE;

CREATE CONSTRAINT entity_id IF NOT EXISTS
    FOR (e:Entity) REQUIRE e.entity_id IS UNIQUE;

CREATE INDEX article_abstract_search IF NOT EXISTS
    FOR (a:Article) ON (a.abstract);

CREATE INDEX chunk_text_search IF NOT EXISTS
    FOR (c:Chunk) ON (c.text);

// Article nodes
LOAD CSV WITH HEADERS FROM 'file:///{articles_csv}' AS row
MERGE (a:Article {{article_id: row.article_id}})
SET a.abstract = row.abstract;

// Chunk nodes
LOAD CSV WITH HEADERS FROM 'file:///{chunks_csv}' AS row
MERGE (c:Chunk {{chunk_id: row.chunk_id}})
SET c.article_id = row.article_id,
    c.text = row.text,
    c.strategy = row.strategy,
    c.embedding = CASE
        WHEN row.embedding IS NULL OR row.embedding = '' THEN []
        ELSE [x IN split(row.embedding, ';') | toFloat(trim(x))]
    END;

// Entity nodes
LOAD CSV WITH HEADERS FROM 'file:///{entities_csv}' AS row
MERGE (e:Entity {{entity_id: row.entity_id}})
SET e.name = row.name,
    e.label = row.label;

// Article -> Chunk relationships
LOAD CSV WITH HEADERS FROM 'file:///{has_chunk_csv}' AS row
MATCH (a:Article {{article_id: row.article_id}})
MATCH (c:Chunk {{chunk_id: row.chunk_id}})
MERGE (a)-[:HAS_CHUNK]->(c);

// Chunk -> Entity relationships
LOAD CSV WITH HEADERS FROM 'file:///{mentions_csv}' AS row
MATCH (c:Chunk {{chunk_id: row.chunk_id}})
MATCH (e:Entity {{entity_id: row.entity_id}})
MERGE (c)-[:MENTIONS]->(e);
"""
