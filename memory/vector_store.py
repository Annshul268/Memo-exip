"""
Vector storage layer using ChromaDB for semantic retrieval.
Synchronizes with SQLite as the authoritative source of truth,
and provides reconciliation capabilities.
"""

from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
import chromadb
from chromadb.config import Settings

from memory.schema import MemoryRecord


class ChromaVectorStore:
    """Semantic vector index using ChromaDB."""

    def __init__(self, persist_dir: str = "data/chroma_db", collection_name: str = "agent_temporal_memories"):
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.collection_name = collection_name
        
        # Initialize persistent client
        self.client = chromadb.PersistentClient(
            path=str(self.persist_dir),
            settings=Settings(anonymized_telemetry=False)
        )
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"description": "Semantic index for temporal memory records"}
        )

    def index_memory(self, record: MemoryRecord) -> None:
        """Add or update an individual memory record in the vector index."""
        doc_text = record.to_natural_language()
        metadata = {
            "record_id": record.id,
            "subject": record.subject.lower(),
            "attribute": record.attribute.lower(),
            "status": record.status.value,
            "valid_from": record.valid_from.isoformat(),
            "valid_until": record.valid_until.isoformat() if record.valid_until else "",
            "importance": float(record.importance),
            "confidence": float(record.confidence)
        }
        
        self.collection.upsert(
            ids=[record.id],
            documents=[doc_text],
            metadatas=[metadata]
        )

    def search(self, query: str, top_k: int = 5, where: Optional[Dict[str, Any]] = None) -> List[Tuple[str, float]]:
        """
        Semantic search returning (record_id, similarity_score) tuples.
        Similarity score is normalized to [0.0, 1.0].
        """
        if self.collection.count() == 0:
            return []
            
        k = min(top_k, self.collection.count())
        query_kwargs: Dict[str, Any] = {
            "query_texts": [query],
            "n_results": k,
        }
        if where:
            query_kwargs["where"] = where
            
        results = self.collection.query(**query_kwargs)
        
        ranked_results: List[Tuple[str, float]] = []
        if results and results.get("ids") and results["ids"][0]:
            ids = results["ids"][0]
            # ChromaDB returns distances (squared L2 or cosine distance depending on setup)
            # By default ChromaDB uses L2 distance or cosine distance:
            # cosine similarity ~ 1.0 - distance / 2 or 1.0 / (1.0 + distance)
            distances = results.get("distances", [[]])[0] if results.get("distances") else []
            
            for idx, rid in enumerate(ids):
                dist = distances[idx] if idx < len(distances) else 0.5
                # Normalize distance to [0, 1] similarity
                similarity = max(0.0, min(1.0, 1.0 / (1.0 + float(dist))))
                ranked_results.append((rid, similarity))
                
        return ranked_results

    def delete_memory(self, record_id: str) -> None:
        """Remove a record from the vector index."""
        try:
            self.collection.delete(ids=[record_id])
        except Exception:
            pass

    def reconcile_with_sqlite(self, authoritative_records: List[MemoryRecord]) -> Dict[str, int]:
        """
        Reconcile ChromaDB against SQLite source-of-truth:
        1. Identifies missing records and indexes them.
        2. Identifies stale or deleted records and removes them.
        Returns statistics: {'reindexed': N, 'purged': M}.
        """
        existing_count = self.collection.count()
        auth_dict = {r.id: r for r in authoritative_records}
        
        # Get all IDs in collection
        indexed_ids = set()
        if existing_count > 0:
            existing_data = self.collection.get()
            indexed_ids = set(existing_data.get("ids", []))
            
        # 1. Purge IDs not in authoritative SQLite
        to_purge = indexed_ids - set(auth_dict.keys())
        if to_purge:
            self.collection.delete(ids=list(to_purge))
            
        # 2. Re-index/upsert all authoritative records
        if authoritative_records:
            ids = [r.id for r in authoritative_records]
            docs = [r.to_natural_language() for r in authoritative_records]
            metadatas = [{
                "record_id": r.id,
                "subject": r.subject.lower(),
                "attribute": r.attribute.lower(),
                "status": r.status.value,
                "valid_from": r.valid_from.isoformat(),
                "valid_until": r.valid_until.isoformat() if r.valid_until else "",
                "importance": float(r.importance),
                "confidence": float(r.confidence)
            } for r in authoritative_records]
            
            self.collection.upsert(ids=ids, documents=docs, metadatas=metadatas)
            
        return {
            "authoritative_total": len(authoritative_records),
            "reindexed": len(authoritative_records),
            "purged": len(to_purge)
        }

    def clear(self) -> None:
        """Purge all vector entries."""
        try:
            self.client.delete_collection(self.collection_name)
        except Exception:
            pass
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"description": "Semantic index for temporal memory records"}
        )
