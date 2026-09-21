"""Graph — Module II, Bài 2 (Building Agents với LangGraph).

Luồng (ReAct loop, Section 1-2):
    START → agent → (có tool call?) → tools → agent → ... → END

Section 5 (Memory): compile với MemorySaver — state được lưu theo `thread_id`,
agent "nhớ" hội thoại qua nhiều lượt gọi invoke() khác nhau.

Section 6 (HITL): compile với `interrupt_before=["tools"]` — graph luôn dừng
TRƯỚC khi chạy bất kỳ tool nào, chờ con người phê duyệt qua resume_agent().
Đơn giản hơn "chỉ chặn tool nhạy cảm" (yêu cầu 1 conditional interrupt riêng)
nhưng minh hoạ đúng cơ chế cốt lõi mà bài học dạy.
"""

from __future__ import annotations

from functools import lru_cache

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode

from app.agent_m2.nodes import agent_node, should_continue
from app.agent_m2.state import AssistantState
from app.agent_m2.tools import TOOLS

# MemorySaver lưu trong RAM — đủ cho demo/dev. Production dùng SqliteSaver/
# PostgresSaver (bền vững qua restart), xem ghi chú trong bài học Section 5.
_checkpointer = MemorySaver()


@lru_cache(maxsize=1)
def _build_graph():
    graph = StateGraph(AssistantState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", ToolNode(TOOLS))
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
    graph.add_edge("tools", "agent")

    return graph.compile(checkpointer=_checkpointer, interrupt_before=["tools"])


def _pending_tool_call(result: dict) -> dict | None:
    """Trích tool call agent đang chờ duyệt từ state, None nếu graph đã kết thúc bình thường."""
    last_message = result["messages"][-1]
    tool_calls = getattr(last_message, "tool_calls", None)
    return tool_calls[0] if tool_calls else None


def start_conversation(thread_id: str, message: str) -> dict:
    """Gửi lượt đầu/tiếp theo của hội thoại. Có thể dừng giữa chừng chờ duyệt tool.

    Returns:
        {"status": "done", "answer": str} — agent trả lời xong, không cần duyệt gì.
        {"status": "pending_approval", "tool_call": {...}} — graph dừng trước tool node.
    """
    app = _build_graph()
    config = {"configurable": {"thread_id": thread_id}}

    result = app.invoke({"messages": [{"role": "user", "content": message}]}, config=config)
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
