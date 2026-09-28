from functools import lru_cache

from sentence_transformers import SentenceTransformer


@lru_cache(maxsize=4)
def _get_model(
    model_name: str,
) -> SentenceTransformer:
    return SentenceTransformer(model_name)


class TextEmbedder:
    """Generate dense vector representations for text."""

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
    ) -> None:
        self.model = _get_model(model_name)

    def encode(
        self,
        texts: list[str],
    ) -> list[list[float]]:
        embeddings = self.model.encode(
            texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )

        return embeddings.tolist()

    def encode_query(
        self,
        query: str,
    ) -> list[float]:
        embedding = self.model.encode(
            query,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )

        return embedding.tolist()