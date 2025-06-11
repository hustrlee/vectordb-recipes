from functools import cached_property
from typing import List

import numpy as np
from lancedb.embeddings import TextEmbeddingFunction
from lancedb.embeddings.registry import register
from lancedb.util import attempt_import_or_raise


@register("tei")
class TEI(TextEmbeddingFunction):
    name: str = "tei"
    model: str = "http://localhost/embed"
    dim: int = None

    def generate_embeddings(self, texts: List[str] | np.ndarray) -> List[np.ndarray]:
        valid_embeddings = self._tei_client.feature_extraction(texts)
        return list(valid_embeddings[: len(texts)])

    def ndims(self) -> int:
        if self.dim is None:
            self.dim = len(self.generate_embeddings(["foo"])[0])
        return self.dim

    @cached_property
    def _tei_client(self):
        hf_hub = attempt_import_or_raise("huggingface_hub")
        return hf_hub.InferenceClient(model=self.model)
