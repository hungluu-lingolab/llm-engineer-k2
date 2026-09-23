"""Graph — Module II, Bài 2-3 (LangGraph + Memory & Context Engineering).

Luồng Bài 2 (ReAct loop):  START → agent → (tool call?) → tools → agent → ... → END

Luồng Bài 3 (thêm memory + compaction):

    START → recall → [vượt ngưỡng 40%?] ─(có)→ compact ─┐
                                        └─(không)────────┴→ agent → (tool call?)
                                                                       ├─(có)→ tools → agent → ...
                                                                       └─(không)→ store → END

  - recall (đầu mỗi lượt):  đọc long-term memory liên quan (memory.py) → chèn vào context.
  - compact (đầu mỗi lượt, CÓ ĐIỀU KIỆN): nén history cũ, GHI ĐÈ state.messages
    (RemoveMessage + summary) — bản nén PERSIST nên các vòng agent⇄tools sau đó
    trong CÙNG lượt kế thừa, không nén lại. Tách khỏi agent_node có chủ đích, xem
    docstring compact_node/agent_node trong nodes.py (kỷ luật "one node one
    function" + lý do hiệu năng: tránh nén lặp lại mỗi vòng lặp mà không tái dùng).
  - store (cuối mỗi lượt): trích 1 sự thật đáng nhớ → ghi long-term memory.

short-term (messages) vẫn sống trong state; long-term ở vector store ngoài.

Bài 2, Section 5 (Memory kỹ thuật): compile với MemorySaver — state lưu theo
`thread_id`, agent "nhớ" hội thoại qua nhiều lượt invoke() (SHORT-TERM). Khác
với long-term memory Bài 3 (xuyên nhiều thread/session, ở vector store).

Bài 2, Section 6 (HITL): compile với `interrupt_before=["tools"]` — graph luôn
dừng TRƯỚC khi chạy tool, chờ con người phê duyệt qua resume_conversation().
"""

from __future__ import annotations

from functools import lru_cache

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode

from app.agent_m2.nodes import (
    agent_node,
    compact_node,
    extract_and_store_node,
    recall_node,
    should_compact_route,
    should_continue,
)
from app.agent_m2.state import AssistantState
from app.agent_m2.tools import TOOLS

# MemorySaver lưu trong RAM — đủ cho demo/dev. Production dùng SqliteSaver/
# PostgresSaver (bền vững qua restart), xem ghi chú trong bài học Section 5.
_checkpointer = MemorySaver()


@lru_cache(maxsize=1)
def _build_graph():
    graph = StateGraph(AssistantState)
    graph.add_node("recall", recall_node)      # Bài 3: đọc long-term memory
    graph.add_node("compact", compact_node)    # Bài 3: nén history (persist state)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", ToolNode(TOOLS))
    graph.add_node("store", extract_and_store_node)  # Bài 3: ghi long-term memory

    graph.add_edge(START, "recall")
    # Sau recall: history vượt ngưỡng 40% → compact (nén + ghi đè state) rồi mới agent;
    # chưa vượt → thẳng tới agent. Bản nén được persist nên vòng sau không nén lại.
    graph.add_conditional_edges(
        "recall", should_compact_route, {"compact": "compact", "agent": "agent"}
    )
    graph.add_edge("compact", "agent")
    # Xong vòng lặp (không còn tool call) → store thay vì END trực tiếp.
    graph.add_conditional_edges("agent", should_continue, {"tools": "tools", END: "store"})
    graph.add_edge("tools", "agent")
    graph.add_edge("store", END)

    return graph.compile(checkpointer=_checkpointer, interrupt_before=["tools"])


def _pending_tool_call(result: dict) -> dict | None:
    """Trích tool call agent đang chờ duyệt từ state, None nếu graph đã kết thúc bình thường."""
    last_message = result["messages"][-1]
    tool_calls = getattr(last_message, "tool_calls", None)
    return tool_calls[0] if tool_calls else None


def start_conversation(thread_id: str, message: str, user_id: str = "") -> dict:
    """Gửi lượt đầu/tiếp theo của hội thoại. Có thể dừng giữa chừng chờ duyệt tool.

    `user_id` (Bài 3): định danh user để recall/store long-term memory. Bỏ trống
    → agent chạy như Bài 2 (không dùng long-term memory).

    Returns:
        {"status": "done", "answer": str} — agent trả lời xong, không cần duyệt gì.
        {"status": "pending_approval", "tool_call": {...}} — graph dừng trước tool node.
    """
    app = _build_graph()
    config = {"configurable": {"thread_id": thread_id}}

    initial: AssistantState = {"messages": [{"role": "user", "content": message}]}
    if user_id:
        initial["user_id"] = user_id
    result = app.invoke(initial, config=config)
    return _to_response(result)


def resume_conversation(thread_id: str, approve: bool, rejection_note: str = "") -> dict:
    """Con người phê duyệt/từ chối tool call đang chờ, rồi tiếp tục graph (Section 6).

    Từ chối: thay vì để agent gọi tool, chèn 1 tool message báo lỗi/từ chối —
    agent đọc được lý do và tự điều chỉnh (giống error recovery ở Section 3),
    thay vì graph crash hoặc treo.
    """
    app = _build_graph()
    config = {"configurable": {"thread_id": thread_id}}

    if not approve:
        state = app.get_state(config)
        last_message = state.values["messages"][-1]
        call = last_message.tool_calls[0]
        app.update_state(
            config,
            {
                "messages": [
                    {
                        "role": "tool",
                        "tool_call_id": call["id"],
                        "content": f"Người dùng TỪ CHỐI hành động này. Lý do: {rejection_note or 'không nêu rõ'}.",
                    }
                ]
            },
            as_node="tools",
        )

    result = app.invoke(None, config=config)
    return _to_response(result)


def _to_response(result: dict) -> dict:
    pending = _pending_tool_call(result)
    if pending is not None:
        return {"status": "pending_approval", "tool_call": {"name": pending["name"], "args": pending["args"]}}
    return {"status": "done", "answer": result["messages"][-1].content}
