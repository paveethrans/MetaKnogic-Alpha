import asyncio
import os
from dataclasses import dataclass
from typing import Any, Union, Tuple, List, Dict
import inspect
from hypergraphrag.utils import logger, compute_mdhash_id
from ..base import BaseGraphStorage
from neo4j import (
    AsyncGraphDatabase,
    exceptions as neo4jExceptions,
    AsyncDriver,
    AsyncManagedTransaction,
)


from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)


@dataclass
class Neo4JStorage(BaseGraphStorage):
    @staticmethod
    def load_nx_graph(file_name):
        print("no preloading of graph with neo4j in production")

    def __init__(self, namespace, global_config, embedding_func):
        super().__init__(
            namespace=namespace,
            global_config=global_config,
            embedding_func=embedding_func,
        )
        self._driver = None
        self._driver_lock = asyncio.Lock()
        URI = os.environ["NEO4J_URI"]
        USERNAME = os.environ["NEO4J_USERNAME"]
        PASSWORD = os.environ["NEO4J_PASSWORD"]
        # Connection pool and timeout tuning via env with sensible defaults
        pool_size = 64
        conn_acq_timeout = 120  # seconds
        conn_timeout = 30  # seconds
        max_tx_retry_time = 60  # seconds
        max_conn_lifetime = 3600  # seconds
        try:
            self._driver: AsyncDriver = AsyncGraphDatabase.driver(
                URI,
                auth=(USERNAME, PASSWORD),
                max_connection_pool_size=pool_size,
                connection_acquisition_timeout=conn_acq_timeout,
                connection_timeout=conn_timeout,
                max_transaction_retry_time=max_tx_retry_time,
                max_connection_lifetime=max_conn_lifetime,
            )
        except TypeError as e:
            # Fallback for older driver versions without these kwargs
            logger.warning(f"Neo4j driver advanced pool params not supported, fallback to defaults: {e}")
            self._driver: AsyncDriver = AsyncGraphDatabase.driver(
                URI, auth=(USERNAME, PASSWORD)
            )
        # Static-id mode uses :Entity/:Hyperedge with {id} instead of dynamic labels
        self._static_mode = str(os.environ.get("HGRAG_NEO4J_STATIC", "0")).lower() in ("1", "true", "yes")
        return None

    def __post_init__(self):
        self._node_embed_algorithms = {
            "node2vec": self._node2vec_embed,
        }

    async def close(self):
        if self._driver:
            await self._driver.close()
            self._driver = None

    async def __aexit__(self, exc_type, exc, tb):
        if self._driver:
            await self._driver.close()

    async def index_done_callback(self):
        print("KG successfully indexed.")

    async def has_node(self, node_id: str) -> bool:
        if getattr(self, "_static_mode", False):
            label, static_id = self._name_to_label_and_id(node_id)
            async with self._driver.session() as session:
                query = f"MATCH (n:{label} {{id:$id}}) RETURN count(n) > 0 AS node_exists"
                result = await session.run(query, id=static_id)
                rec = await result.single()
                return bool(rec and rec["node_exists"])
        entity_name_label = node_id.strip('"')

        async with self._driver.session() as session:
            query = (
                f"MATCH (n:`{entity_name_label}`) RETURN count(n) > 0 AS node_exists"
            )
            result = await session.run(query)
            single_result = await result.single()
            logger.debug(
                f'{inspect.currentframe().f_code.co_name}:query:{query}:result:{single_result["node_exists"]}'
            )
            return single_result["node_exists"]

    async def has_edge(self, source_node_id: str, target_node_id: str) -> bool:
        if getattr(self, "_static_mode", False):
            src_label, src_id = self._name_to_label_and_id(source_node_id)
            tgt_label, tgt_id = self._name_to_label_and_id(target_node_id)
            # Prefer directed hyperedge -> entity
            async with self._driver.session() as session:
                if src_label == "Hyperedge" and tgt_label == "Entity":
                    query = """
                    MATCH (a:Hyperedge {id:$src})-[r:DIRECTED]->(b:Entity {id:$tgt})
                    RETURN COUNT(r) > 0 AS edgeExists
                    """
                    result = await session.run(query, src=src_id, tgt=tgt_id)
                else:
                    query = """
                    MATCH (a {id:$src})-[r]-(b {id:$tgt})
                    RETURN COUNT(r) > 0 AS edgeExists
                    """
                    result = await session.run(query, src=src_id, tgt=tgt_id)
                rec = await result.single()
                return bool(rec and rec["edgeExists"])
        entity_name_label_source = source_node_id.strip('"')
        entity_name_label_target = target_node_id.strip('"')

        async with self._driver.session() as session:
            query = (
                f"MATCH (a:`{entity_name_label_source}`)-[r]-(b:`{entity_name_label_target}`) "
                "RETURN COUNT(r) > 0 AS edgeExists"
            )
            result = await session.run(query)
            single_result = await result.single()
            logger.debug(
                f'{inspect.currentframe().f_code.co_name}:query:{query}:result:{single_result["edgeExists"]}'
            )
            return single_result["edgeExists"]

    async def get_node(self, node_id: str) -> Union[dict, None]:
        if getattr(self, "_static_mode", False):
            label, static_id = self._name_to_label_and_id(node_id)
            async with self._driver.session() as session:
                query = f"MATCH (n:{label} {{id:$id}}) RETURN n LIMIT 1"
                result = await session.run(query, id=static_id)
                record = await result.single()
                if record:
                    return dict(record["n"])
                return None
        async with self._driver.session() as session:
            entity_name_label = node_id.strip('"')
            query = f"MATCH (n:`{entity_name_label}`) RETURN n"
            result = await session.run(query)
            record = await result.single()
            if record:
                node = record["n"]
                node_dict = dict(node)
                logger.debug(
                    f"{inspect.currentframe().f_code.co_name}: query: {query}, result: {node_dict}"
                )
                return node_dict
            return None

    async def node_degree(self, node_id: str) -> int:
        if getattr(self, "_static_mode", False):
            label, static_id = self._name_to_label_and_id(node_id)
            async with self._driver.session() as session:
                query = f"MATCH (n:{label} {{id:$id}}) RETURN COUNT{{ (n)--() }} AS totalEdgeCount"
                result = await session.run(query, id=static_id)
                record = await result.single()
                return int(record["totalEdgeCount"]) if record else 0
        entity_name_label = node_id.strip('"')

        async with self._driver.session() as session:
            query = f"""
                MATCH (n:`{entity_name_label}`)
                RETURN COUNT{{ (n)--() }} AS totalEdgeCount
            """
            result = await session.run(query)
            record = await result.single()
            if record:
                edge_count = record["totalEdgeCount"]
                logger.debug(
                    f"{inspect.currentframe().f_code.co_name}:query:{query}:result:{edge_count}"
                )
                return edge_count
            else:
                return None

    async def edge_degree(self, src_id: str, tgt_id: str) -> int:
        if getattr(self, "_static_mode", False):
            src_degree = await self.node_degree(src_id)
            trg_degree = await self.node_degree(tgt_id)
        else:
            entity_name_label_source = src_id.strip('"')
            entity_name_label_target = tgt_id.strip('"')
            src_degree = await self.node_degree(entity_name_label_source)
            trg_degree = await self.node_degree(entity_name_label_target)

        # Convert None to 0 for addition
        src_degree = 0 if src_degree is None else src_degree
        trg_degree = 0 if trg_degree is None else trg_degree

        degrees = int(src_degree) + int(trg_degree)
        logger.debug(
            f"{inspect.currentframe().f_code.co_name}:query:src_Degree+trg_degree:result:{degrees}"
        )
        return degrees

    async def get_edge(
        self, source_node_id: str, target_node_id: str
    ) -> Union[dict, None]:
        if getattr(self, "_static_mode", False):
            src_label, src_id = self._name_to_label_and_id(source_node_id)
            tgt_label, tgt_id = self._name_to_label_and_id(target_node_id)
            async with self._driver.session() as session:
                query = """
                MATCH (start:Hyperedge {id:$src})-[r:DIRECTED]->(end:Entity {id:$tgt})
                RETURN properties(r) as edge_properties
                LIMIT 1
                """
                result = await session.run(query, src=src_id, tgt=tgt_id)
                record = await result.single()
                if record:
                    return dict(record["edge_properties"])
                return None
        entity_name_label_source = source_node_id.strip('"')
        entity_name_label_target = target_node_id.strip('"')
        """
        Find all edges between nodes of two given labels

        Args:
            source_node_label (str): Label of the source nodes
            target_node_label (str): Label of the target nodes

        Returns:
            list: List of all relationships/edges found
        """
        async with self._driver.session() as session:
            query = f"""
            MATCH (start:`{entity_name_label_source}`)-[r]->(end:`{entity_name_label_target}`)
            RETURN properties(r) as edge_properties
            LIMIT 1
            """.format(
                entity_name_label_source=entity_name_label_source,
                entity_name_label_target=entity_name_label_target,
            )

            result = await session.run(query)
            record = await result.single()
            if record:
                result = dict(record["edge_properties"])
                logger.debug(
                    f"{inspect.currentframe().f_code.co_name}:query:{query}:result:{result}"
                )
                return result
            else:
                return None

    async def get_node_edges(self, source_node_id: str) -> List[Tuple[str, str]]:
        if getattr(self, "_static_mode", False):
            # Hyperedge neighbors or entity neighbors
            s = (source_node_id or "").strip()
            is_hyper = s.startswith("<") or s.startswith("rel-")
            async with self._driver.session() as session:
                if is_hyper:
                    hid = s if s.startswith("rel-") else compute_mdhash_id(s, prefix="rel-")
                    query = """
                    MATCH (h:Hyperedge {id:$id})-[r:DIRECTED]->(e:Entity)
                    RETURN COALESCE(h.hyperedge_name, h.id) AS src, COALESCE(e.entity_name, e.id) AS tgt
                    """
                    results = await session.run(query, id=hid)
                    edges = []
                    async for record in results:
                        edges.append((record["src"], record["tgt"]))
                    return edges
                else:
                    base = s.strip('"').upper()
                    eid = s if s.startswith("ent-") else compute_mdhash_id(base, prefix="ent-")
                    query = """
                    MATCH (h:Hyperedge)-[r:DIRECTED]->(e:Entity {id:$id})
                    RETURN COALESCE(h.hyperedge_name, h.id) AS src, COALESCE(e.entity_name, e.id) AS tgt
                    """
                    results = await session.run(query, id=eid)
                    edges = []
                    async for record in results:
                        edges.append((record["src"], record["tgt"]))
                    return edges
        node_label = source_node_id.strip('"')

        """
        Retrieves all edges (relationships) for a particular node identified by its label.
        :return: List of dictionaries containing edge information
        """
        query = f"""MATCH (n:`{node_label}`)
                OPTIONAL MATCH (n)-[r]-(connected)
                RETURN n, r, connected"""
        async with self._driver.session() as session:
            results = await session.run(query)
            edges = []
            async for record in results:
                source_node = record["n"]
                connected_node = record["connected"]

                source_label = (
                    list(source_node.labels)[0] if source_node.labels else None
                )
                target_label = (
                    list(connected_node.labels)[0]
                    if connected_node and connected_node.labels
                    else None
                )

                if source_label and target_label:
                    edges.append((source_label, target_label))

            return edges

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type(
            (
                neo4jExceptions.ServiceUnavailable,
                neo4jExceptions.TransientError,
                neo4jExceptions.WriteServiceUnavailable,
                neo4jExceptions.ClientError,
            )
        ),
    )
    async def upsert_node(self, node_id: str, node_data: Dict[str, Any]):
        """
        Upsert a node in the Neo4j database.

        Args:
            node_id: The unique identifier for the node (used as label)
            node_data: Dictionary of node properties
        """
        label = node_id.strip('"')
        properties = node_data

        async def _do_upsert(tx: AsyncManagedTransaction):
            query = f"""
            MERGE (n:`{label}`)
            SET n += $properties
            """
            await tx.run(query, properties=properties)
            logger.debug(
                f"Upserted node with label '{label}' and properties: {properties}"
            )

        try:
            async with self._driver.session() as session:
                await session.execute_write(_do_upsert)
        except Exception as e:
            logger.error(f"Error during upsert: {str(e)}")
            raise

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type(
            (
                neo4jExceptions.ServiceUnavailable,
                neo4jExceptions.TransientError,
                neo4jExceptions.WriteServiceUnavailable,
            )
        ),
    )
    async def upsert_edge(
        self, source_node_id: str, target_node_id: str, edge_data: Dict[str, Any]
    ):
        """
        Upsert an edge and its properties between two nodes identified by their labels.

        Args:
            source_node_id (str): Label of the source node (used as identifier)
            target_node_id (str): Label of the target node (used as identifier)
            edge_data (dict): Dictionary of properties to set on the edge
        """
        source_node_label = source_node_id.strip('"')
        target_node_label = target_node_id.strip('"')
        edge_properties = edge_data

        async def _do_upsert_edge(tx: AsyncManagedTransaction):
            query = f"""
            MATCH (source:`{source_node_label}`)
            WITH source
            MATCH (target:`{target_node_label}`)
            MERGE (source)-[r:DIRECTED]->(target)
            SET r += $properties
            RETURN r
            """
            await tx.run(query, properties=edge_properties)
            logger.debug(
                f"Upserted edge from '{source_node_label}' to '{target_node_label}' with properties: {edge_properties}"
            )

        try:
            async with self._driver.session() as session:
                await session.execute_write(_do_upsert_edge)
        except Exception as e:
            logger.error(f"Error during edge upsert: {str(e)}")
            raise

    async def _node2vec_embed(self):
        print("Implemented but never called.")

    # -------- helper functions for static-id mode --------
    def _name_to_label_and_id(self, node_id: str) -> Tuple[str, str]:
        """
        Map a dynamic name-based identifier used by legacy paths (e.g. '"FAD2"', '<hyperedge>"...something..."')
        to a static schema label and id (md5-based) used by :Entity/:Hyperedge nodes.
        """
        s = (node_id or "").strip()
        # Accept direct static ids from Milvus: ent-..., rel-...
        if s.startswith("ent-"):
            return "Entity", s
        if s.startswith("rel-"):
            return "Hyperedge", s
        # Hyperedges can carry a '<hyperedge>' prefix
        if s.startswith("<"):
            return "Hyperedge", compute_mdhash_id(s, prefix="rel-")
        # Otherwise treat as an entity name (possibly quoted); uppercased for stable hashing
        base = s[1:-1] if (len(s) >= 2 and s[0] == '"' and s[-1] == '"') else s
        return "Entity", compute_mdhash_id(str(base).upper(), prefix="ent-")
