"""Semantic Caching — Buổi 8 (Production Optimization).

Cache question-answer pairs không dựa vào token mà dựa vào NGỮ NGHĨA.
Nếu câu hỏi mới gần giống (theo embedding similarity) với câu cũ, trả
lại câu trả lời cũ ngay — không gọi LLM lần nữa.

Production dùng vector store (Redis, Chroma) thay cho in-memory list.
Đây là phiên bản simplistic để học khái niệm.
"""

from __future__ import annotations

import numpy as np


class SemanticCache:
    """Bộ nhớ cache dựa trên similarity embedding.

    threshold: ngưỡng cosine similarity để coi là "gần giống" (0-1).
              Quá thấp -> cache nhầm; quá cao -> ít hit.
              Recommend: 0.90-0.95 cho most cases.
    """

    def __init__(self, embedder, threshold: float = 0.92):
        """embedder: function(text: str) -> np.ndarray (normalized)."""
        self.embedder = embedder
        self.threshold = threshold
        self.questions: list[str] = []
        self.embeddings: list[np.ndarray] = []
        self.answers: list[str] = []

    def get(self, question: str) -> str | None:
        """Tìm câu trả lời cached cho question nếu similarity đủ cao.
        Trả về None nếu cache miss."""
        if not self.embeddings:
            return None

        q_emb = self.embedder(question)
        # Tính cosine similarity vs all cached embeddings
        sims = np.dot(np.array(self.embeddings), q_emb)
        best_idx = int(np.argmax(sims))
        print(f"SemanticCache: best similarity={sims[best_idx]:.4f} (threshold={self.threshold})")
        if sims[best_idx] >= self.threshold:
            return self.answers[best_idx]  # Cache HIT
        return None

    def set(self, question: str, answer: str) -> None:
        """Lưu (question, answer) vào cache."""
        self.questions.append(question)
        self.embeddings.append(self.embedder(question))
        self.answers.append(answer)

    def clear(self) -> None:
        """Xóa toàn bộ cache."""
        self.questions.clear()
        self.embeddings.clear()
        self.answers.clear()

    def size(self) -> int:
        """Số item trong cache."""
        return len(self.answers)
