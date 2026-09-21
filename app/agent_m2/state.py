"""State schema — Module II, Bài 2 (Building Agents với LangGraph).

Khác với app/agent/state.py (CRAG, Module I): agent này KHÔNG phải RAG pipeline
cố định mà là ReAct loop tổng quát (agent tự quyết định gọi tool nào, bao nhiêu
lần) — nên state đơn giản hơn nhiều, chỉ cần lịch sử hội thoại.
"""

from __future__ import annotations

from typing import Annotated, TypedDict

from langgraph.graph.message import add_messages


class AssistantState(TypedDict):
    """State xuyên suốt ReAct loop: agent ⇄ tools.

    `Annotated[list, add_messages]` — reducer tự APPEND message mới thay vì
    ghi đè, giữ lịch sử hội thoại qua nhiều vòng agent → tools → agent.
    """

    messages: Annotated[list, add_messages]
