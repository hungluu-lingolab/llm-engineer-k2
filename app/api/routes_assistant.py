"""Assistant routes — Module II, Bài 2 (Building Agents với LangGraph).

Endpoint riêng khỏi /chat (Module I) vì đây là track khác: không phải RAG,
mà là ReAct agent tổng quát với tool calling + memory + HITL (Section 7).
"""

from __future__ import annotations

from fastapi import APIRouter

from app.agent_m2.graph import resume_conversation, start_conversation
from app.api.schemas import AssistantApprovalRequest, AssistantMessageRequest, AssistantResponse

router = APIRouter(prefix="/assistant", tags=["assistant"])


@router.post("/message", response_model=AssistantResponse)
def send_message(req: AssistantMessageRequest) -> AssistantResponse:
    """Gửi 1 lượt tin nhắn. Có thể trả về `pending_approval` nếu agent muốn gọi tool.

    `user_id` (Bài 3): bật long-term memory — agent recall thông tin đã biết về
    user (dị ứng, sở thích...) và tự lưu sự thật mới sau mỗi lượt.
    """
    return AssistantResponse(
        **start_conversation(req.thread_id, req.message, req.user_id)
    )


@router.post("/approve", response_model=AssistantResponse)
def approve_tool_call(req: AssistantApprovalRequest) -> AssistantResponse:
    """Duyệt/từ chối tool call đang chờ (HITL, Section 6), rồi tiếp tục graph."""
    return AssistantResponse(
        **resume_conversation(req.thread_id, req.approve, req.rejection_note)
    )
