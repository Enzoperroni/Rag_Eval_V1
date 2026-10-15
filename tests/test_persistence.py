import json
import subprocess
import sys
from xunit_rag.storage import IndexStore


def test_native_hnsw_reopens_in_another_process(config):
    store = IndexStore(config)
    collection = store.client.create_collection(
        "persistent_native_path_test", embedding_function=None, configuration={"hnsw": {"space": "cosine"}}
    )
    # Cross the actual HNSW persistence threshold, which a tiny in-process example cannot verify.
    collection.upsert(
        ids=[f"test_{i}" for i in range(1100)],
        embeddings=[[1.0, 0.0, 0.0]] * 1100,
        documents=["Temporary test data"] * 1100,
    )
    code = """
import json, sys
from pathlib import Path
from xunit_rag.config import Config
from xunit_rag.storage import IndexStore
config = Config(root=Path(sys.argv[1]))
config.initialize()
collection = IndexStore(config).client.get_collection('persistent_native_path_test', embedding_function=None)
assert collection.count() == 1100
result = collection.query(query_embeddings=[[1.0, 0.0, 0.0]], n_results=3)
print(json.dumps(result['ids'][0]))
"""
    completed = subprocess.run(
        [sys.executable, "-c", code, str(config.root)], capture_output=True, text=True, check=True, timeout=60
    )
    assert len(json.loads(completed.stdout)) == 3
