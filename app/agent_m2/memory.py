"""Long-term memory — Module II, Bài 3 (Memory & Context Engineering), Section 5.

Vector Memory: lưu "sự thật đáng nhớ" về user vào vector store, chỉ RETRIEVE phần
liên quan tới câu hỏi hiện tại (giống RAG nhưng cho memory thay vì tài liệu).

Khác handbook (dùng LangChain Chroma + OpenAIEmbeddings): ở đây TÁI DÙNG embeddings
native đã xây ở Module I (app.retrieval.embeddings, OpenAI text-embedding-3-small,
Buổi 4) — giữ triết lý "native SDK" của repo, không kéo thêm langchain_community/
chromadb. Dùng Qdrant client + COLLECTION RIÊNG (user_memory), tách hẳn khỏi
collection legal_docs của Module I để không lẫn tài liệu luật với memory user.

Long-term memory sống Ở NGOÀI graph state: đọc vào đầu mỗi lượt (recall_node) và
ghi ra sau mỗi lượt (extract_and_store_node) — xem nodes.py. State chỉ giữ
short-term (messages), đúng như bảng phân loại Section 1.

Phân tách theo `user_id` (metadata filter): mỗi user chỉ recall memory của chính
mình, kể cả khi dùng chung collection.
"""

from __future__ import annotations

import time
import uuid
from functools import lru_cache

from app.config import settings


def _use_qdrant() -> bool:
    """Có key OpenAI để embed không? Nếu không → fallback in-memory (demo/test chạy được).

    Long-term memory cần embedding (OpenAI Cloud). Khi chưa cấu hình key, thay vì
    crash, dùng store in-memory đơn giản (keyword match) để phần còn lại của agent
    vẫn chạy — kém chính xác hơn nhưng đủ minh hoạ luồng recall → store.
    """
    return bool(settings.api_keys)


# ── Qdrant collection riêng cho user memory ───────────────────────────────────

@lru_cache(maxsize=1)
def _client():
    from qdrant_client import QdrantClient

    if settings.qdrant_url == ":memory:":
        return QdrantClient(location=":memory:")
    return QdrantClient(url=settings.qdrant_url)


def _ensure_collection() -> None:
    from qdrant_client.models import Distance, VectorParams

    client = _client()
    name = settings.agent_memory_collection
    if not client.collection_exists(name):
        client.create_collection(
            collection_name=name,
            vectors_config=VectorParams(
                size=settings.embedding_dim, distance=Distance.COSINE
            ),
        )


# ── Fallback store (in-memory, không cần embedding) ───────────────────────────
# Chỉ dùng khi không có OPENAI_API_KEYS. list[(user_id, fact)].
_FALLBACK: list[tuple[str, str]] = []


def _fallback_save(user_id: str, fact: str) -> None:
    _FALLBACK.append((user_id, fact))


def _fallback_recall(user_id: str, query: str, k: int) -> list[str]:
    # "Retrieve" thô: xếp fact theo số từ khoá trùng query (rồi tới fact mới nhất).
    mine = [fact for uid, fact in _FALLBACK if uid == user_id]
    query_words = {w.lower() for w in query.split() if len(w) > 2}
    scored = sorted(
        reversed(mine),  # mới nhất trước khi hoà điểm
        key=lambda f: len(query_words & {w.lower() for w in f.split()}),
        reverse=True,
    )
    return scored[:k]


# ── Public API

def save_to_long_term(user_id: str, fact: str) -> None:
    """Lưu 1 sự thật dài hạn về user vào vector store (hoặc fallback in-memory)."""
    fact = fact.strip()
    if not fact:
        return

    if not _use_qdrant():
        _fallback_save(user_id, fact)
        return

    from qdrant_client.models import PointStruct

    from app.retrieval.embeddings import embed_passages

    _ensure_collection()
    vector = embed_passages([fact])[0]
    _client().upsert(
        collection_name=settings.agent_memory_collection,
        points=[
            PointStruct(
                id=str(uuid.uuid4()),
                vector=vector,
                payload={"text": fact, "user_id": user_id, "type": "semantic", "ts": time.time()},
            )
        ],
    )


def recall_long_term(user_id: str, query: str, k: int = 3) -> list[str]:
    """Truy xuất tối đa k memory liên quan nhất tới query, CHỈ của user này."""
    if not _use_qdrant():
        return _fallback_recall(user_id, query, k)

    client = _client()
    if not client.collection_exists(settings.agent_memory_collection):
        return []

    from qdrant_client.models import FieldCondition, Filter, MatchValue

    from app.retrieval.embeddings import embed_query

    response = client.query_points(
        collection_name=settings.agent_memory_collection,
        query=embed_query(query),
        query_filter=Filter(
            must=[FieldCondition(key="user_id", match=MatchValue(value=user_id))]
        ),
        limit=k,
    )
    return [h.payload.get("text", "") for h in response.points]
