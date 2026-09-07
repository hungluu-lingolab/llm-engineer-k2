"""Pipeline orchestrator — nối guardrails + retrieval + prompt + LLM.

Đây là "xương sống" RAG. Ở Buổi 1, bước retrieve trả [] nên thực chất chỉ là
chatbot thuần LLM. Từ Buổi 5, chỉ cần retriever.retrieve() trả chunk thật là
toàn bộ pipeline thành RAG — KHÔNG phải sửa file này.

Luồng:
    guardrails.check_input(question)   — raise GuardrailViolation nếu injection
    → retrieve(query)
    → build_messages(q, ctx)
    → llm.chat / chat_stream / chat_parsed
    guardrails.check_output(answer, ctx) — chỉ ở answer() non-streaming; xem lý
                                            do trong docstring answer_stream().
"""

from __future__ import annotations

from collections.abc import Iterator

from app.guardrails.checks import check_input, check_output
from app.llm import completion
from app.llm.params import GenerationParams
from app.monitoring.tracing import trace_answer, trace_stream
from app.prompts.templates import build_messages
from app.retrieval.retriever import retrieve
from app.schemas.domain import LegalAnswer


def answer(question: str, params: GenerationParams | None = None) -> str:
    """Trả lời dạng text (non-streaming). Có đủ input + output guardrails."""
    check_input(question)
    with trace_answer("answer", question) as t:
        chunks = retrieve(question)  # Buổi 1: [] → trả lời thuần parametric knowledge
        messages = build_messages(question, chunks)
        raw_answer = completion.chat(messages, params)

        result = check_output(raw_answer, [c.text for c in chunks])
        t["output"] = result.answer
    return result.answer


def answer_stream(
    question: str, params: GenerationParams | None = None
) -> Iterator[str]:
    """Trả lời dạng streaming (Section 6).

    Chỉ có input guardrail. Output guardrail không áp dụng được ở đây: token
    đã gửi tới client ngay khi sinh ra, nên không có cách nào "thay bằng
    fallback" sau khi phát hiện vấn đề — muốn kiểm tra output cho luồng
    streaming cần buffer toàn bộ trước (mất lợi ích của streaming) hoặc chấp
    nhận đánh đổi latency-thấp/không-guardrail-output.
    """
    check_input(question)
    chunks = retrieve(question)
    messages = build_messages(question, chunks)
    yield from trace_stream(
        "answer_stream", question, completion.chat_stream(messages, params)
    )


def answer_structured(
    question: str, params: GenerationParams | None = None
) -> LegalAnswer:
    """Trả lời dạng structured output theo schema LegalAnswer (Section 4).

    Chỉ có input guardrail — LegalAnswer đã tự mang confidence/needs_lawyer,
    một hình thức "self-reported groundedness" riêng của schema này.
    """
    check_input(question)
    with trace_answer("answer_structured", question) as t:
        chunks = retrieve(question)
        messages = build_messages(question, chunks)
        result = completion.chat_parsed(messages, LegalAnswer, params)
        t["output"] = result.model_dump_json()
    return result
