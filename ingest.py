import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from loguru import logger

from config_loader import get_config
from embed import EmbeddingGenerator
from search_utils import BM25Store, ChromaVectorStore

load_dotenv()


class VectorStoreIngester:
    def __init__(
        self,
        chroma_path: str,
        bm25_index_path: str,
        bm25_metadata_path: str,
    ):
        self.chroma_path = chroma_path
        self.bm25_index_path = bm25_index_path
        self.bm25_metadata_path = bm25_metadata_path

    def ingest_to_chroma(self, chunks_filepath: str, vectors_filepath: str):
        chunks_df = pd.read_json(chunks_filepath, lines=True)
        embedding_vectors = joblib.load(vectors_filepath)

        if len(chunks_df) != len(embedding_vectors):
            logger.error(f"Vectors: {len(embedding_vectors)}, chunks: {len(chunks_df)}")
            raise ValueError("Vectors and chunks must have the same length")

        embeddings_array = np.array(embedding_vectors, dtype=np.float32)
        logger.info(f"Vector shape: {embeddings_array.shape}")

        chroma_store = ChromaVectorStore(chroma_path=self.chroma_path)
        chroma_store.create_from_embeddings(embeddings_array, chunks_df)

        logger.info(f"Successfully created Chroma index at {self.chroma_path}")

    def ingest_to_bm25(self, chunks_filepath: str):
        chunks_df = pd.read_json(chunks_filepath, lines=True)
        text_corpus = chunks_df["text"].tolist()

        bm25_store = BM25Store(
            metadata_path=self.bm25_metadata_path,
            bm25_index_path=self.bm25_index_path
        )
        bm25_store.create_from_texts(text_corpus, chunks_df)

        logger.info(f"Successfully created BM25 index at {self.bm25_index_path}")

    def ingest_all(
        self,
        chunks_filepath: str,
        embedding_model: str,
        vectors_output_path: str,
        batch_size: int,
        checkpoint_dir: Path,
    ):
        chunks_df = pd.read_json(chunks_filepath, lines=True)
        text_corpus = chunks_df["text"].tolist()

        logger.info("Creating embeddings...")
        generator = EmbeddingGenerator(model_name=embedding_model)
        embedding_vectors = generator.embed_batch(text_corpus, batch_size=batch_size, checkpoint_dir=checkpoint_dir)

        vectors_path = Path(vectors_output_path)
        vectors_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(embedding_vectors, vectors_path)
        logger.info(f"Saved vectors to {vectors_path}")

        logger.info("Creating Chroma index...")
        self.ingest_to_chroma(chunks_filepath, vectors_output_path)

        logger.info("Creating BM25 index...")
        self.ingest_to_bm25(chunks_filepath)

        logger.info("Ingestion complete!")


if __name__ == "__main__":
    config = get_config()

    logger.info("Starting ingestion process...")

    ingester = VectorStoreIngester(
        chroma_path=config.get("paths.chroma_path"),
        bm25_index_path=config.get("paths.bm25_index"),
        bm25_metadata_path=config.get("paths.bm25_metadata")
    )

    ingester.ingest_all(
        chunks_filepath=config.get("paths.annotated_chunks"),
        embedding_model=config.get("models.embedding.name"),
        vectors_output_path=config.get("paths.vectors_file"),
        batch_size=config.get("embedding.batch_size"),
        checkpoint_dir=config.get_path("embedding.checkpoint_dir"),
    )

    logger.info("Ingestion complete!")