import os
from neo4j import GraphDatabase
from hypergraphrag.utils import set_logger, logger


COUNTS_QUERIES = {
    "entities": "MATCH (n) WHERE n.role = 'entity' RETURN count(*) AS count",
    "hyperedges": "MATCH (n) WHERE n.role = 'hyperedge' RETURN count(*) AS count",
    "edges": "MATCH ()-[r:DIRECTED]->() RETURN count(r) AS count",
}

SAMPLES_QUERIES = {
    "entities": """
        MATCH (n) WHERE n.role = 'entity'
        RETURN coalesce(labels(n)[0], '') AS entity,
               coalesce(n.entity_type, 'UNKNOWN') AS type,
               coalesce(n.description, 'UNKNOWN') AS description
        LIMIT $limit
    """,
    "hyperedges": """
        MATCH (h) WHERE h.role = 'hyperedge'
        RETURN coalesce(labels(h)[0], '') AS hyperedge,
               coalesce(h.weight, 0)      AS weight,
               coalesce(h.source_id, '')  AS sources
        LIMIT $limit
    """,
    "relationships": """
        MATCH (h)-[r:DIRECTED]->(e)
        WHERE h.role='hyperedge' AND e.role='entity'
        RETURN coalesce(labels(h)[0], '') AS hyperedge,
               coalesce(labels(e)[0], '') AS entity,
               coalesce(r.weight, 0)      AS weight,
               coalesce(r.source_id, '')  AS sources
        LIMIT $limit
    """,
}


def _print_kv(title: str, value):
    logger.info(f"{title}: {value}")


def main():
    set_logger("hypergraphrag_v2.log")

    uri = os.environ.get("NEO4J_URI", "bolt://localhost:7688")
    user = os.environ.get("NEO4J_USERNAME", "neo4j")
    password = os.environ.get("NEO4J_PASSWORD", "myneo4jgraph")
    sample_limit = int(os.environ.get("NEO4J_SAMPLE_LIMIT", "10"))

    logger.info(f"Connecting Neo4j at {uri} as {user}")
    driver = GraphDatabase.driver(uri, auth=(user, password))

    try:
        with driver.session() as session:
            # Counts
            logger.info("=== Counts ===")
            for name, cypher in COUNTS_QUERIES.items():
                result = session.run(cypher)
                count = result.single().get("count", 0)
                _print_kv(name, count)

            # Samples
            logger.info("=== Sample Entities ===")
            for record in session.run(SAMPLES_QUERIES["entities"], limit=sample_limit):
                logger.info(
                    f"- entity={record['entity']}, type={record['type']}, desc={record['description'][:120]}"
                )

            logger.info("=== Sample Hyperedges ===")
            for record in session.run(SAMPLES_QUERIES["hyperedges"], limit=sample_limit):
                logger.info(
                    f"- hyperedge={record['hyperedge']}, weight={record['weight']}, sources={record['sources'][:120]}"
                )

            logger.info("=== Sample Relationships (hyperedge -> entity) ===")
            for record in session.run(SAMPLES_QUERIES["relationships"], limit=sample_limit):
                logger.info(
                    f"- {record['hyperedge']} -> {record['entity']}, weight={record['weight']}, sources={record['sources'][:120]}"
                )

    finally:
        driver.close()


if __name__ == "__main__":
    main()

