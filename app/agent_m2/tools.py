"""Tools — Module II, Bài 2, Section 7 (Vietnamese Personal Assistant).

`@tool` (langchain_core) sinh JSON schema từ type hint + docstring, tương
đương TOOLS/DISPATCH viết tay ở app/tools/registry.py (Bài 1) — nhưng
`ToolNode` (Section 2) cần tool ở định dạng này để tự thực thi mà không
phải parse tool_calls thủ công.

Dữ liệu giả lập, giống tinh thần `lookup_article` ở Bài 1: đủ để minh hoạ
control flow, không phải tích hợp thật.
"""

from __future__ import annotations

from langchain_core.tools import tool


@tool
def check_calendar(start_date: str, end_date: str = "") -> str:
    """Kiểm tra lịch làm việc trong một khoảng ngày (YYYY-MM-DD).

    Để trống end_date nếu chỉ hỏi 1 ngày. Với câu hỏi kiểu "tuần này"/"tuần sau",
    hãy tự tính start_date/end_date rồi gọi tool NÀY MỘT LẦN DUY NHẤT cho cả
    khoảng — không gọi lặp lại cho từng ngày riêng lẻ.
    """
    end_date = end_date or start_date
    return (
        f"Từ {start_date} đến {end_date}: Thứ 2 14h họp team, "
        f"Thứ 4 16h30 gọi khách hàng, Thứ 6 10h review sprint."
    )


@tool
def send_reminder(message: str, time: str) -> str:
    """Đặt lời nhắc. Hành động này cần con người phê duyệt trước khi gửi."""
    return f"Đã đặt lời nhắc '{message}' lúc {time}."


@tool
def search_restaurant(location: str, cuisine: str = "") -> str:
    """Tìm nhà hàng theo khu vực và loại món ăn."""
    return f"3 nhà hàng {cuisine} gần {location}: Quán A, Quán B, Quán C."


TOOLS = [check_calendar, send_reminder, search_restaurant]

# Tool có side-effect (gửi lời nhắc thật) — cần HITL approval trước khi chạy.
# Xem graph.py: interrupt_before=["tools"] áp dụng cho MỌI tool call; danh sách
# này chỉ dùng để routes_agent.py hiển thị rõ cho UI "vì sao cần duyệt".
SENSITIVE_TOOLS = {"send_reminder"}
