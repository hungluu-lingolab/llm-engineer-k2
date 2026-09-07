"""LLM-as-Judge — Buổi 7, Section 1.

Dùng LLM để chấm điểm câu trả lời trên 4 tiêu chí: accuracy, completeness,
clarity, groundedness. Dùng native chat_parsed (Buổi 1/4) — không thêm dep mới.

QUAN TRỌNG — 6 loại bias của LLM-as-Judge (xem bài học) và cách file này giảm
thiểu từng loại:

  1. Position bias    — không áp dụng ở đây (chấm tuyệt đối, không so sánh A/B).
     Nếu bạn mở rộng sang so sánh 2 câu trả lời, PHẢI chấm cả 2 chiều thứ tự
     rồi lấy trung bình (xem judge_pairwise()).
  2. Verbosity bias    — rubric yêu cầu chấm "completeness" tách biệt với độ dài;
     prompt nhắc rõ không thưởng câu trả lời dài dòng.
  3. Self-preference    — nếu judge model trùng họ với model sinh câu trả lời,
     kết quả có thể bị thổi phồng. Khuyến nghị: dùng judge khác họ model chat
     khi có thể, hoặc ít nhất ghi nhận giới hạn này khi đọc kết quả.
  4. Style bias         — rubric chấm "clarity" độc lập với "accuracy"; không để
     format đẹp (markdown, giọng trang trọng) ảnh hưởng điểm nội dung.
  5. Calibration drift  — pin model version cụ thể trong production (không chỉ
     "gpt-4o-mini" mà "gpt-4o-mini-2024-07-18") để điểm số ổn định qua thời gian.
  6. Preference leakage — tránh dùng đúng model đang được đánh giá làm judge cho
     chính nó trên cùng benchmark.

Giảm thiểu chung: chấm theo RUBRIC TUYỆT ĐỐI (1-5 mỗi tiêu chí) thay vì ranking
so sánh — đây là khuyến nghị chính của bài học để giảm verbosity + position bias.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from app.llm import completion
from app.llm.params import GenerationParams


class JudgeScore(BaseModel):
    """Điểm chấm theo rubric tuyệt đối — không so sánh với câu trả lời khác."""

    accuracy: int = Field(ge=1, le=5, description="Thông tin có chính xác không")
    completeness: int = Field(ge=1, le=5, description="Có trả lời đủ câu hỏi không")
    clarity: int = Field(ge=1, le=5, description="Có rõ ràng, dễ hiểu không")
    groundedness: int = Field(
        ge=1, le=5, description="Có bám sát tài liệu không (không hallucinate)"
    )
    reasoning: str = Field(description="Giải thích ngắn gọn cho điểm số")

    @property
    def overall(self) -> float:
        return (self.accuracy + self.completeness + self.clarity + self.groundedness) / 4


_JUDGE_SYSTEM = """Bạn là giám khảo đánh giá chất lượng câu trả lời của trợ lý pháp lý.

Chấm điểm 1-5 cho MỖI tiêu chí một cách ĐỘC LẬP — không để một tiêu chí ảnh
hưởng tiêu chí khác. Không thưởng điểm cho câu trả lời dài hoặc format đẹp nếu
nội dung không tương xứng (verbosity bias / style bias)."""


def judge_answer(
    question: str,
    answer: str,
    reference: str | None = None,
    context: str | None = None,
) -> JudgeScore:
    """Chấm điểm một câu trả lời theo rubric tuyệt đối.

    Args:
        question: câu hỏi gốc.
        answer: câu trả lời cần chấm.
        reference: câu trả lời chuẩn (tùy chọn, giúp chấm accuracy chính xác hơn).
        context: tài liệu đã dùng để trả lời (tùy chọn, dùng cho groundedness).
    """
    parts = [f"Câu hỏi: {question}", f"Câu trả lời cần chấm: {answer}"]
    if reference:
        parts.append(f"Câu trả lời chuẩn (tham khảo): {reference}")
    if context:
        parts.append(f"Tài liệu nguồn (để chấm groundedness): {context}")

    messages = [
        {"role": "system", "content": _JUDGE_SYSTEM},
        {"role": "user", "content": "\n\n".join(parts)},
    ]
    return completion.chat_parsed(messages, JudgeScore, GenerationParams(temperature=0.0))


def judge_batch(
    items: list[dict],
) -> list[JudgeScore]:
    """Chấm một batch. Mỗi item: {question, answer, reference?, context?}."""
    return [
        judge_answer(
            item["question"],
            item["answer"],
            item.get("reference"),
            item.get("context"),
        )
        for item in items
    ]
