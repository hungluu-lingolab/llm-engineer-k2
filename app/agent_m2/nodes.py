"""Nodes — Module II, Bài 2 (Building Agents với LangGraph).

Khác với app/agent/nodes.py (CRAG, Module I) vốn gọi thẳng app/llm/completion.py
(native OpenAI SDK): agent này dùng `ChatOpenAI.bind_tools()` (langchain_openai)
vì `ToolNode` (LangGraph prebuilt, Section 2) chỉ tự thực thi tool khi model được
gọi qua interface LangChain — đây là ngoại lệ CÓ CHỦ ĐÍCH so với triết lý "native
SDK" của README, đổi lấy vòng lặp tool-calling khỏi phải viết tay (đã viết tay ở
Bài 1/tools/registry.py, và ở app/agent/nodes.py bằng chat_with_tools).
"""

from __future__ import annotations

from datetime import date
from functools import lru_cache

from langgraph.graph import END

from app.agent_m2.state import AssistantState
from app.agent_m2.tools import TOOLS
from app.config import settings


def _system_prompt() -> str:
    # Chèn ngày hiện tại — không có dòng này, model không biết "hôm nay" là
    # ngày nào nên hay bịa năm/tháng sai khi tính "tuần này"/"ngày mai".
    return (
        "Bạn là trợ lý cá nhân tiếng Việt, thân thiện và ngắn gọn. "
        f"Hôm nay là {date.today().isoformat()}. "
        "Dùng tool khi cần thông tin lịch, đặt lời nhắc, hoặc tìm nhà hàng. "
        "Với câu hỏi về một khoảng thời gian (vd: 'tuần này'), tự tính "
        "start_date/end_date rồi gọi check_calendar MỘT LẦN DUY NHẤT — "
        "không gọi lặp lại cho từng ngày riêng lẻ."
    )


@lru_cache(maxsize=1)
def _llm_with_tools():
    """Lazy-init: tránh gọi ra ngoài (đọc API key) lúc import module cho test.

    Truyền thẳng `api_key` từ settings.api_keys[0] thay vì để ChatOpenAI tự đọc
    biến môi trường OPENAI_API_KEY — repo này dùng OPENAI_API_KEYS (số nhiều,
    hỗ trợ key rotation ở app/llm/client.py), không phải biến số ít mặc định
    của langchain_openai.
    """
    from langchain_openai import ChatOpenAI

    llm = ChatOpenAI(
        model=settings.llm_model,
        temperature=settings.llm_temperature,
        api_key=settings.api_keys[0] if settings.api_keys else None,
    )
    return llm.bind_tools(TOOLS)


# ── Loop termination (Section 3): phát hiện agent lặp lại cùng 1 tool call ────

def _detect_repetition(state: AssistantState, window: int = 4) -> bool:
    """True nếu `window` tool call gần nhất giống hệt nhau — dấu hiệu agent bị kẹt."""
    recent_calls = [
        m.tool_calls[0]
        for m in state["messages"][-window:]
        if getattr(m, "tool_calls", None)
    ]
    if len(recent_calls) < window:
        return False
    return len({str(c) for c in recent_calls}) == 1


def agent_node(state: AssistantState) -> dict:
    """Gọi LLM (kèm tool binding). Chèn cảnh báo nếu phát hiện vòng lặp lặp lại."""
    messages = [{"role": "system", "content": _system_prompt()}] + state["messages"]

    if _detect_repetition(state):
        messages = messages + [
            {
                "role": "system",
                "content": "Bạn đang lặp lại cùng 1 hành động. Hãy thử cách tiếp cận hoàn toàn khác.",
            }
        ]

    response = _llm_with_tools().invoke(messages)
    return {"messages": [response]}


def should_continue(state: AssistantState) -> str:
    """Conditional edge: có tool call → "tools", không thì dừng graph (Section 1, 3)."""
    last_message = state["messages"][-1]
    if not getattr(last_message, "tool_calls", None):
        return END
    return "tools"
