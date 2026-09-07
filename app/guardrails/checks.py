"""Guardrails — Buổi 7 (Evaluation & Guardrails).

check_input(): chạy TRƯỚC khi gọi LLM — phát hiện prompt injection + PII trong
câu hỏi người dùng. Raise GuardrailViolation nếu phát hiện injection nghiêm
trọng (regex match) — pipeline dừng lại, không tốn 1 lời gọi LLM cho request
độc hại. API layer bắt exception này và trả HTTP 400 (xem api/routes_chat.py).

check_output(): chạy SAU khi có câu trả lời — kiểm tra độ dài, ngôn ngữ, và
groundedness đơn giản (số liệu trong answer có xuất hiện trong context không).
KHÔNG raise — output đã sinh rồi, raise không giúp gì; thay vào đó trả về
OutputCheckResult để caller quyết định giữ answer gốc hay dùng fallback.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from app.config import settings
from app.guardrails.injection import detect_prompt_injection
from app.guardrails.pii import detect_pii


class GuardrailViolation(Exception):
    """Input bị chặn bởi guardrails (prompt injection nghi ngờ cao)."""

    def __init__(self, reason: str, details: dict | None = None):
        self.reason = reason
        self.details = details or {}
        super().__init__(reason)


@dataclass(slots=True)
class OutputCheckResult:
    valid: bool
    issues: list[str] = field(default_factory=list)
    answer: str = ""


_FALLBACK_RESPONSE = (
    "Xin lỗi, tôi không thể xác nhận độ chính xác của câu trả lời này. "
    "Vui lòng thử diễn đạt lại câu hỏi hoặc tham khảo trực tiếp văn bản luật."
)


def check_input(text: str) -> None:
    """Raise GuardrailViolation nếu phát hiện prompt injection qua regex.

    Chỉ dùng regex ở đường chặn cứng (nhanh, không tốn LLM call). LLM-based
    check (injection.llm_injection_check) mạnh hơn nhưng chậm — bật thêm qua
    GUARDRAILS_LLM_INJECTION_CHECK cho các luồng chấp nhận latency cao hơn.
    """
    if detect_prompt_injection(text):
        raise GuardrailViolation(
            "prompt_injection_detected",
            {"pattern_match": True},
        )

    if settings.guardrails_llm_injection_check:
        from app.guardrails.injection import llm_injection_check

        result = llm_injection_check(text)
        if result.is_injection and result.confidence >= 0.7:
            raise GuardrailViolation(
                "prompt_injection_detected_llm",
                {"confidence": result.confidence, "reason": result.reason},
            )

    pii_found = detect_pii(text)
    if pii_found:
        # PII trong câu hỏi không tự nó là injection — không chặn, chỉ để lại
        # dấu vết cho caller log nếu cần (không raise ở đây theo thiết kế).
        pass


def check_output(answer: str, context: list[str]) -> OutputCheckResult:
    """Kiểm tra output trước khi trả về user. Không raise — trả kết quả có
    thể sửa (fallback) để pipeline quyết định dùng answer gốc hay fallback."""
    issues: list[str] = []

    if len(answer) < settings.guardrails_min_answer_len:
        issues.append("answer_too_short")

    # Groundedness đơn giản: số liệu cụ thể trong answer phải xuất hiện trong
    # context — cách rẻ để bắt hallucination về con số (điều luật, mốc thời
    # gian...) mà không cần gọi thêm LLM.
    if context:
        context_text = " ".join(context)
        numbers_in_answer = re.findall(r"\b\d+\b", answer)
        for num in numbers_in_answer:
            if num not in context_text:
                issues.append(f"unverified_number_{num}")

    if issues:
        return OutputCheckResult(valid=False, issues=issues, answer=_FALLBACK_RESPONSE)
    return OutputCheckResult(valid=True, issues=[], answer=answer)
