"""PII Filtering — Buổi 7, Section 3.

Regex-based cho tiếng Việt (theo bài học) — KHÔNG dùng Presidio: thư viện đó
nặng và hỗ trợ tiếng Việt kém, trong khi các pattern SĐT/CCCD/email của VN có
format đủ chuẩn để regex bắt tốt.
"""

from __future__ import annotations

import re

_PATTERNS: dict[str, re.Pattern] = {
    # SĐT di động VN: 0[3-9]xxxxxxxx (10 số) hoặc +84xxxxxxxxx
    "phone": re.compile(r"(0[3-9]\d{8}|\+84\d{9})"),
    # CCCD/CMND mới: 12 chữ số liên tiếp
    "cccd": re.compile(r"\b\d{12}\b"),
    "email": re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
}


def detect_pii(text: str) -> dict[str, list[str]]:
    """Trả về các match tìm được, theo loại PII. Dict rỗng nếu không có gì."""
    found = {}
    for pii_type, pattern in _PATTERNS.items():
        matches = pattern.findall(text)
        if matches:
            found[pii_type] = matches
    return found


def redact_pii(text: str) -> str:
    """Thay mọi PII tìm được bằng placeholder [LOAI_REDACTED]."""
    for pii_type, pattern in _PATTERNS.items():
        text = pattern.sub(f"[{pii_type.upper()}_REDACTED]", text)
    return text
