"""State schema — Module II, Bài 2-3 (LangGraph + Memory & Context Engineering).

Khác với app/agent/state.py (CRAG, Module I): agent này KHÔNG phải RAG pipeline
cố định mà là ReAct loop tổng quát (agent tự quyết định gọi tool nào, bao nhiêu
lần) — nên state đơn giản hơn nhiều.

Bài 3 (Memory): `messages` là SHORT-TERM memory (session hiện tại, sống trong
state). LONG-TERM memory KHÔNG sống ở đây — nó ở vector store ngoài (memory.py),
đọc vào đầu lượt (recall_node) và ghi ra sau lượt (extract_and_store_node).
State chỉ mang thêm `user_id` để biết recall/store memory của AI.
"""

from __future__ import annotations

from typing import Annotated, TypedDict

from langgraph.graph.message import add_messages


class AssistantState(TypedDict, total=False):
    """State xuyên suốt ReAct loop: agent ⇄ tools.

    `Annotated[list, add_messages]` — reducer tự APPEND message mới thay vì
    ghi đè, giữ lịch sử hội thoại qua nhiều vòng agent → tools → agent.

    total=False: `user_id` optional để giữ tương thích ngược với Bài 2 (các
    lời gọi cũ chỉ truyền messages vẫn chạy — recall/store bỏ qua nếu thiếu).
    """

    messages: Annotated[list, add_messages]
    user_id: str  # Bài 3: định danh user để recall/store long-term memory
