from typing import Any

import pytest

from hybrid_rag_search.providers.fake import FakeGenerationProvider
from hybrid_rag_search.providers.generation import (
    GenerationFinishReason,
    GenerationMessage,
    GenerationProvider,
    GenerationRequest,
    GenerationResult,
    GenerationUsage,
    MessageRole,
)


def user(content: str = "What is the vacation policy?") -> GenerationMessage:
    return GenerationMessage(MessageRole.USER, content)


@pytest.mark.parametrize("role", list(MessageRole))
def test_message_accepts_each_role(role: MessageRole) -> None:
    assert GenerationMessage(role, "content").role == role


@pytest.mark.parametrize(
    "role,content", [("tool", "content"), (MessageRole.USER, ""), (MessageRole.USER, " \n")]
)
def test_message_validation(role: Any, content: str) -> None:
    with pytest.raises(ValueError):
        GenerationMessage(role, content)


@pytest.mark.parametrize(
    "messages,max_tokens,message",
    [
        ((), 10, "nonempty tuple"),
        ([user()], 10, "nonempty tuple"),
        (("bad",), 10, "GenerationMessage"),
        ((GenerationMessage(MessageRole.SYSTEM, "instructions"),), 10, "user message"),
        ((user(),), 0, "positive integer"),
        ((user(),), True, "positive integer"),
    ],
)
def test_request_validation(messages: Any, max_tokens: Any, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        GenerationRequest(messages, max_tokens)


@pytest.mark.parametrize(
    "text,finish_reason,model,usage",
    [
        ("", GenerationFinishReason.COMPLETE, "model", GenerationUsage()),
        ("text", GenerationFinishReason.COMPLETE, "", GenerationUsage()),
        ("text", "unknown", "model", GenerationUsage()),
        ("text", GenerationFinishReason.COMPLETE, "model", GenerationUsage(input_tokens=-1)),
        (
            "text",
            GenerationFinishReason.COMPLETE,
            "model",
            GenerationUsage(latency_ms=float("inf")),
        ),
    ],
)
def test_result_validation(
    text: str, finish_reason: Any, model: str, usage: GenerationUsage
) -> None:
    with pytest.raises(ValueError):
        GenerationResult(text, finish_reason, model, "fake", usage)


@pytest.mark.anyio
async def test_fake_is_repeatable_and_contains_no_prompt_text() -> None:
    provider: GenerationProvider = FakeGenerationProvider()
    request = GenerationRequest(
        (
            GenerationMessage(MessageRole.SYSTEM, "Answer from supplied evidence."),
            user("Private vacation policy"),
        ),
        max_output_tokens=100,
    )
    first = await provider.generate(request)
    second = await FakeGenerationProvider().generate(request)
    assert first == second
    assert first.text.startswith("Synthetic response ")
    assert "Private" not in first.text
    assert first.finish_reason == GenerationFinishReason.COMPLETE
    assert first.model == "fake-generation-v1" and first.adapter == "fake"
    assert first.usage.billed_input_tokens == 0
    assert first.usage.billed_output_tokens == 0
    assert first.usage.estimated_cost_usd == 0
    assert first.usage.input_tokens is None and first.usage.output_tokens is None
    assert first.usage.latency_ms is None


@pytest.mark.anyio
async def test_request_changes_fake_response() -> None:
    provider = FakeGenerationProvider()
    baseline = await provider.generate(GenerationRequest((user("first"),), 100))
    changed_text = await provider.generate(GenerationRequest((user("second"),), 100))
    changed_limit = await provider.generate(GenerationRequest((user("first"),), 101))
    assert len({baseline.text, changed_text.text, changed_limit.text}) == 3
