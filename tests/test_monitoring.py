"""Test Buổi 7 — Monitoring (LangFuse hooks, tối thiểu).

Khi MONITORING_ENABLED=false (mặc định), trace_answer/trace_stream phải là
no-op hoàn toàn — không import package `langfuse` (chưa cài trong dev env).
Khi bật, mock `_get_langfuse` để không cần key/network thật.
"""

from __future__ import annotations

from app.monitoring import tracing


def test_trace_answer_is_noop_when_disabled(monkeypatch):
    monkeypatch.setattr(tracing.settings, "monitoring_enabled", False)

    with tracing.trace_answer("answer", "câu hỏi") as t:
        t["output"] = "trả lời"

    # Không raise, không import langfuse (nếu import sẽ ModuleNotFoundError
    # vì package chưa cài trong dev env).


def test_trace_stream_is_noop_when_disabled(monkeypatch):
    monkeypatch.setattr(tracing.settings, "monitoring_enabled", False)

    tokens = iter(["a", "b", "c"])
    result = list(tracing.trace_stream("answer_stream", "câu hỏi", tokens))
    assert result == ["a", "b", "c"]


def test_trace_answer_calls_langfuse_when_enabled(monkeypatch):
    monkeypatch.setattr(tracing.settings, "monitoring_enabled", True)

    calls = {}

    class FakeSpan:
        def update(self, **kwargs):
            calls["update"] = kwargs

        def end(self):
            calls["ended"] = True

    class FakeLangfuse:
        def start_observation(self, **kwargs):
            calls["start_observation"] = kwargs
            return FakeSpan()

        def flush(self):
            calls["flushed"] = True

    monkeypatch.setattr(tracing, "_get_langfuse", lambda: FakeLangfuse())

    with tracing.trace_answer("answer", "câu hỏi") as t:
        t["output"] = "trả lời"

    assert calls["start_observation"]["input"] == "câu hỏi"
    assert calls["update"]["output"] == "trả lời"
    assert calls["ended"] is True
    assert calls["flushed"] is True


def test_trace_stream_calls_langfuse_when_enabled(monkeypatch):
    monkeypatch.setattr(tracing.settings, "monitoring_enabled", True)

    calls = {}

    class FakeSpan:
        def end(self):
            calls["ended"] = True

    class FakeLangfuse:
        def start_observation(self, **kwargs):
            calls["start_observation"] = kwargs
            return FakeSpan()

        def flush(self):
            calls["flushed"] = True

    monkeypatch.setattr(tracing, "_get_langfuse", lambda: FakeLangfuse())

    tokens = iter(["a", "b"])
    result = list(tracing.trace_stream("answer_stream", "q", tokens))

    assert result == ["a", "b"]
    assert calls["start_observation"]["output"] == "ab"
    assert calls["ended"] is True
    assert calls["flushed"] is True
