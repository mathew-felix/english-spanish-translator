"""Build a persistent ChromaDB translation memory over Europarl sentence pairs.
The index stores up to 50K English rows from the training CSV and their Spanish metadata.
"""

import csv
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

os.environ.setdefault("ANONYMIZED_TELEMETRY", "FALSE")

import chromadb
from chromadb.config import Settings
from chromadb.errors import InvalidCollectionException
from sentence_transformers import SentenceTransformer

from rag.retriever import COLLECTION_NAME, EMBEDDING_MODEL_NAME


MAX_ROWS = 50000
EMBED_BATCH_SIZE = 256
CHROMA_ADD_BATCH_SIZE = 5000


def _repo_root() -> str:
    """Return the repository root so build paths stay stable.
    The index builder should work regardless of the caller's working directory.
    """
    return REPO_ROOT


def _train_csv_path() -> str:
    """Return the training CSV used to source Europarl memory rows.
    The builder uses the project `data/train.csv` file already created by preprocessing.
    """
    return os.path.join(_repo_root(), "data", "train.csv")


def _db_path() -> str:
    """Return the persistent Chroma directory for the translation memory.
    This path is gitignored because it is a local generated artifact.
    """
    return os.path.join(_repo_root(), "rag", "chroma_db")


def _load_europarl_rows(limit: int = MAX_ROWS) -> list[dict]:
    """Load up to `limit` Europarl rows from the training CSV.
    Rows missing either side of the bilingual pair are skipped.
    """
    csv_path = _train_csv_path()
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"Training CSV not found at '{csv_path}'.")

    selected_rows = []
    with open(csv_path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for source_index, row in enumerate(reader):
            if row.get("Corpus") != "Europarl":
                continue

            english = (row.get("English") or "").strip()
            spanish = (row.get("Spanish") or "").strip()
            if not english or not spanish:
                continue

            selected_rows.append(
                {
                    "id": f"europarl_{source_index}",
                    "english": english,
                    "spanish": spanish,
                    "corpus": "Europarl",
                    "source_index": source_index,
                }
            )
            if len(selected_rows) >= limit:
                break
    return selected_rows


def _reset_collection(client) -> None:
    """Delete the existing translation-memory collection if present.
    Rebuilding should replace the prior index rather than append duplicate rows.
    """
    try:
        client.delete_collection(name=COLLECTION_NAME)
    except (InvalidCollectionException, ValueError):
        return


def build_translation_memory(limit: int = MAX_ROWS) -> int:
    """Build the persistent Chroma translation-memory collection.
    Returns the number of indexed bilingual pairs written to disk.
    """
    rows = _load_europarl_rows(limit=limit)
    if not rows:
        raise RuntimeError("No Europarl rows were found for the translation memory.")

    os.makedirs(_db_path(), exist_ok=True)
    client = chromadb.PersistentClient(
        path=_db_path(),
        settings=Settings(anonymized_telemetry=False),
    )
    _reset_collection(client)

    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    documents = [row["english"] for row in rows]
    embeddings = model.encode(
        documents,
        batch_size=EMBED_BATCH_SIZE,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).tolist()

    metadatas = [
        {
            "spanish": row["spanish"],
            "corpus": row["corpus"],
            "source_index": row["source_index"],
        }
        for row in rows
    ]

    for start in range(0, len(rows), CHROMA_ADD_BATCH_SIZE):
        end = start + CHROMA_ADD_BATCH_SIZE
        collection.add(
            ids=[row["id"] for row in rows[start:end]],
            documents=documents[start:end],
            metadatas=metadatas[start:end],
            embeddings=embeddings[start:end],
        )
    return len(rows)


def main() -> None:
    """Build the Europarl translation memory from the local training CSV.
    The default build size is 50K rows to keep the index focused and practical.
    """
    count = build_translation_memory()
    print(f"Built translation memory with {count} Europarl sentence pairs.")


if __name__ == "__main__":
    main()
