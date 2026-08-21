"""Batch embed all Semantic Graph nodes with BGE-M3-Q8_0."""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from hakua_memory.semantic_graph.embedding.llama_cpp import LlamaCppEmbeddingBackend
from hakua_memory.semantic_graph.embedding.serializer import (
    serialize_embedding_node,
    source_text_hash,
)
from hakua_memory.semantic_graph.store import SemanticGraphStore

DB_PATH = Path("C:/Users/downl/.hermes/semantic-graph/semantic_graph.db")
BACKEND = LlamaCppEmbeddingBackend(
    endpoint="http://127.0.0.1:8081",
    model="bge-m3-q8_0",
    revision="q8_0",
    dimensions=1024,
    serializer_version=1,
    timeout_seconds=60.0,
)
BATCH_SIZE = 50


def main():
    store = SemanticGraphStore(DB_PATH)
    namespace = BACKEND.identity.namespace

    # Get nodes without embeddings in our namespace
    with store._connect() as conn:
        rows = conn.execute(
            "SELECT n.node_id, n.node_type, n.subtype, n.label, n.summary "
            "FROM nodes n "
            "LEFT JOIN node_embeddings ne ON n.node_id = ne.node_id AND ne.namespace = ? "
            "WHERE ne.node_id IS NULL",
            (namespace,),
        ).fetchall()

    total = len(rows)
    print(f"Nodes to embed: {total}")
    if not rows:
        print("Nothing to do.")
        return

    # Process in batches
    embedded = 0
    errors = 0
    start = time.time()

    for i in range(0, total, BATCH_SIZE):
        batch = rows[i:i + BATCH_SIZE]
        texts = [serialize_embedding_node(dict(r)) for r in batch]

        try:
            vectors = BACKEND.embed_documents(texts)
            for row, vec in zip(batch, vectors):
                store.upsert_node_embedding(
                    node_id=row["node_id"],
                    identity=BACKEND.identity,
                    vector=vec,
                    source_text_hash=source_text_hash(serialize_embedding_node(dict(row))),
                )
                embedded += 1
        except Exception as e:
            errors += len(batch)
            print(f"  Batch {i // BATCH_SIZE + 1} failed: {e}")
            continue

        if (i // BATCH_SIZE + 1) % 20 == 0 or i + BATCH_SIZE >= total:
            elapsed = time.time() - start
            rate = embedded / elapsed if elapsed > 0 else 0
            print(f"  [{i + len(batch)}/{total}] {embedded} ok, {errors} err, {rate:.1f}/s")

    elapsed = time.time() - start
    print(f"\nDone: {embedded} embedded, {errors} errors, {elapsed:.1f}s total")


if __name__ == "__main__":
    main()
