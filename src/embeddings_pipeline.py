from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression


class AGNewsEmbeddingClassifier:
    """
    SentenceTransformer embeddings + Logistic Regression
    for 4-class AG News classification.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.embedder = SentenceTransformer(model_name)
        self.clf = LogisticRegression(
            max_iter=3000,
            multi_class="auto"
        )
        self.is_fitted = False

    def encode(self, texts: List[str]) -> np.ndarray:
        return self.embedder.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=True
        )

    def fit(self, texts: List[str], labels: List[int]) -> None:
        X = self.encode(texts)
        y = np.array(labels)
        self.clf.fit(X, y)
        self.is_fitted = True

    def predict(self, texts: List[str]) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before predict().")
        X = self.encode(texts)
        return self.clf.predict(X)
