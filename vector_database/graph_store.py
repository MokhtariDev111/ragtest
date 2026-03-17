"""
vector_database/graph_store.py
================================
A local, in-memory Knowledge Graph database built on NetworkX.
Provides functionality for storing Entities and Relationships
and querying graph-based neighborhood contexts.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Optional

import networkx as nx
from loguru import logger

class LocalGraphStore:
    """
    Manages an in-memory Knowledge Graph using NetworkX.
    Includes persistence to JSON for disk storage.
    """
    
    def __init__(self, persist_dir: Optional[str] = None):
        self.persist_dir = persist_dir
        self.graph = nx.DiGraph()
        
        if persist_dir:
            os.makedirs(persist_dir, exist_ok=True)
            self._load_from_disk()

    def _get_path(self) -> Path:
        return Path(self.persist_dir) / "nx_graph.json" if self.persist_dir else None

    def _load_from_disk(self):
        """Loads graph state from JSON if it exists."""
        path = self._get_path()
        if path and path.exists():
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.graph = nx.node_link_graph(data)
                logger.debug(f"Graph loaded: {self.get_node_count()} nodes, {self.get_edge_count()} edges.")
            except Exception as e:
                logger.error(f"Failed to load graph from disk: {e}")

    def save_to_disk(self):
        """Persists the NetworkX graph to JSON."""
        path = self._get_path()
        if path:
            try:
                data = nx.node_link_data(self.graph)
                with open(path, "w", encoding="utf-8") as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
            except Exception as e:
                logger.error(f"Failed to save graph to disk: {e}")

    def add_node(self, node_id: str, **attributes):
        """Adds a node to the graph if it doesn't exist."""
        node_id = str(node_id).strip().lower()
        if not self.graph.has_node(node_id):
            self.graph.add_node(node_id, **attributes)
        else:
            # Update attributes if needed
            self.graph.nodes[node_id].update(attributes)

    def add_edge(self, source: str, target: str, relation: str, context: str = ""):
        """Adds a directed edge representing a relationship between two entities."""
        source = str(source).strip().lower()
        target = str(target).strip().lower()
        relation = str(relation).strip().lower()
        
        self.add_node(source)
        self.add_node(target)
        self.graph.add_edge(source, target, relation=relation, context=context)

    def query_neighborhood(self, entity_ids: list[str], depth: int = 2) -> list[str]:
        """
        Given a list of entities (e.g. from a user question),
        traverses `depth` steps outwards and returns the relationships 
        as text strings to feed into the RAG context.
        """
        collected_context = []
        visited_edges = set()
        
        for entity in set([str(e).strip().lower() for e in entity_ids]):
            if not self.graph.has_node(entity):
                continue
                
            # Perform BFS up to `depth`
            queue = [(entity, 0)]
            visited_nodes = {entity}
            
            while queue:
                current_node, current_depth = queue.pop(0)
                if current_depth >= depth:
                    continue
                    
                # Outgoing edges
                for neighbor in self.graph.successors(current_node):
                    edge_str = f"({current_node}) -[{self.graph.edges[current_node, neighbor].get('relation', 'related_to')}]-> ({neighbor})"
                    if edge_str not in visited_edges:
                        visited_edges.add(edge_str)
                        collected_context.append(edge_str)
                    
                    if neighbor not in visited_nodes:
                        visited_nodes.add(neighbor)
                        queue.append((neighbor, current_depth + 1))
                        
                # Incoming edges (Knowledge graphs are bi-directional in RAG context)
                for neighbor in self.graph.predecessors(current_node):
                    edge_str = f"({neighbor}) -[{self.graph.edges[neighbor, current_node].get('relation', 'related_to')}]-> ({current_node})"
                    if edge_str not in visited_edges:
                        visited_edges.add(edge_str)
                        collected_context.append(edge_str)
                        
                    if neighbor not in visited_nodes:
                        visited_nodes.add(neighbor)
                        queue.append((neighbor, current_depth + 1))
                        
        return collected_context

    def get_node_count(self) -> int:
        return self.graph.number_of_nodes()
        
    def get_edge_count(self) -> int:
        return self.graph.number_of_edges()

    def clear(self):
        self.graph.clear()
        self.save_to_disk()
