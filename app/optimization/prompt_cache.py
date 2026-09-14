"""Prompt Caching — Buổi 8 (Production Optimization).

Prefix dài (system prompt, few-shot, RAG context) được cache ở server LLM.
Request sau với cùng prefix chỉ tính tiền cho phần thay đổi.

Anthropic: dùng cache_control: {"type": "ephemeral"} trên message với type="text".
OpenAI: tự động cache prompt > 1024 tokens (v1.0+ SDK), không cần config.

Chi tiết:
- OpenAI: trả về response.usage.cache_creation_input_tokens + cache_read_input_tokens
- Anthropic: output có cache_creation_input_tokens + cache_read_input_tokens
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from openai.types.chat import ChatCompletion


def format_prompt_for_caching(
    system_instruction: str,
    static_context: str,
    user_query: str,
) -> list[dict]:
    """Format message list để tối ưu Prompt Caching.

    NGUYÊN TẮC VÀNG: tĩnh trước (hệ thống, context), động sau (câu hỏi).

    Args:
        system_instruction: hệ thống role/nhân vật (tĩnh).
        static_context: tài liệu context, few-shot examples (tĩnh, có thể dài).
        user_query: câu hỏi người dùng (động, nhỏ).

    Returns:
        messages list ready for OpenAI/Anthropic API.
    """
    return [
        {"role": "system", "content": system_instruction},
        {"role": "system", "content": static_context},  # Đặt context tĩnh lên đầu
        {"role": "user", "content": user_query},  # Phần động ở cuối
    ]


class PromptCacheStats:
    """Thống kê cache hit/miss từ response."""

    def __init__(self, completion: ChatCompletion):
        """Parse OpenAI response để lấy cache stats."""
        self.cache_creation_input_tokens = getattr(
            completion.usage, "cache_creation_input_tokens", 0
        )
        self.cache_read_input_tokens = getattr(
            completion.usage, "cache_read_input_tokens", 0
        )
        self.input_tokens = completion.usage.input_tokens
        self.output_tokens = completion.usage.output_tokens

    def cache_hit_ratio(self) -> float:
        """Tỷ lệ token được đọc từ cache (0-1)."""
        total_input = self.cache_creation_input_tokens + self.cache_read_input_tokens
        if total_input == 0:
            return 0.0
        return self.cache_read_input_tokens / total_input

    def __str__(self) -> str:
        hit_ratio = self.cache_hit_ratio()
        return (
            f"PromptCache: "
            f"created={self.cache_creation_input_tokens} tokens, "
            f"read={self.cache_read_input_tokens} tokens "
            f"(hit ratio: {hit_ratio:.1%})"
        )
