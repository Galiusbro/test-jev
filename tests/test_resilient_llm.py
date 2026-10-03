from __future__ import annotations

import itertools
import json
from collections.abc import Callable, Iterator
from typing import Any

import httpx
import pytest
import respx
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from pydantic import BaseModel, SecretStr

from jev_agent.config import ModelTier, Reasoning, Settings
from jev_agent.llm import CallLog, LLMConfigError, LLMError, ResilientChatModel, chat_model
from jev_agent.llm.reasoning import reasoning_params

BASE = "https://nvidia.test/v1"
URL = f"{BASE}/chat/completions"


def sse(*chunks: dict[str, Any], comments: int = 0) -> bytes:
    lines = [": keep-alive"] * comments
    lines += [f"data: {json.dumps(c)}" for c in chunks]
    lines.append("data: [DONE]")
    return ("\n\n".join(lines) + "\n\n").encode()


def delta(model: str = "m", **d: Any) -> dict[str, Any]:
    return {"model": model, "choices": [{"index": 0, "delta": d, "finish_reason": None}]}


def finish(reason: str, model: str = "m") -> dict[str, Any]:
    return {"model": model, "choices": [{"index": 0, "delta": {}, "finish_reason": reason}]}


def usage(inp: int, out: int) -> dict[str, Any]:
    return {
        "choices": [],
        "usage": {"prompt_tokens": inp, "completion_tokens": out, "total_tokens": inp + out},
    }


def by_model(handlers: dict[str, Callable[[], httpx.Response]]) -> Callable[..., httpx.Response]:
    def handle(request: httpx.Request) -> httpx.Response:
        return handlers[json.loads(request.content)["model"]]()

    return handle


def ok(*chunks: dict[str, Any]) -> Callable[[], httpx.Response]:
    return lambda: httpx.Response(200, content=sse(*chunks))


def make(
    models: list[str],
    reasoning: Reasoning = "default",
    clock: Callable[[], float] | None = None,
) -> ResilientChatModel:
    extra: dict[str, Any] = {"clock": clock} if clock else {}
    return ResilientChatModel(
        models=models,
        api_key=SecretStr("nvapi-x"),
        base_url=BASE,
        reasoning=reasoning,
        first_token_timeout_s=15,
        sleep=lambda _s: None,
        **extra,
    )


def bodies(route: respx.Route) -> list[dict[str, Any]]:
    return [json.loads(c.request.content) for c in route.calls]


@respx.mock
def test_streams_text_reasoning_and_usage() -> None:
    route = respx.post(URL).mock(
        side_effect=by_model(
            {
                "a": ok(
                    delta("a", reasoning_content="17*3 is "),
                    delta("a", reasoning_content="51."),
                    delta("a", content="5"),
                    delta("a", content="1"),
                    finish("stop", "a"),
                    usage(12, 7),
                )
            }
        )
    )
    llm = make(["a"])
    message = llm.invoke("What is 17*3?")

    assert isinstance(message, AIMessage)
    assert message.content == "51"
    assert message.additional_kwargs["reasoning_content"] == "17*3 is 51."
    assert message.response_metadata == {"model_name": "a", "finish_reason": "stop"}
    assert message.usage_metadata == {"input_tokens": 12, "output_tokens": 7, "total_tokens": 19}

    body = bodies(route)[0]
    assert body["stream"] is True and body["stream_options"] == {"include_usage": True}
    assert body["messages"] == [{"role": "user", "content": "What is 17*3?"}]
    (record,) = llm.call_log.records
    assert record.ok and record.model == "a"
    assert (record.input_tokens, record.output_tokens) == (12, 7)


@respx.mock
def test_falls_back_through_errors_timeouts_and_empty_answers() -> None:
    def read_timeout() -> httpx.Response:
        raise httpx.ReadTimeout("stalled")

    respx.post(URL).mock(
        side_effect=by_model(
            {
                "retired": lambda: httpx.Response(404, text="Not found"),
                "broken": lambda: httpx.Response(500, text="Internal server error"),
                "stalled": read_timeout,
                "silent": ok(delta("silent", content="  "), finish("stop", "silent")),
                "good": ok(delta("good", content="pong"), finish("stop", "good")),
            }
        )
    )
    llm = make(["retired", "broken", "stalled", "silent", "good"])
    assert llm.invoke("ping").content == "pong"

    records = llm.call_log.records
    assert [r.model for r in records] == ["retired", "broken", "stalled", "silent", "good"]
    assert [r.ok for r in records] == [False, False, False, False, True]
    assert records[0].error is not None and "404" in records[0].error
    assert records[1].error is not None and "500" in records[1].error
    assert records[2].error is not None and "timeout" in records[2].error
    assert records[3].error is not None and records[3].error.startswith("empty response")


@respx.mock
def test_queue_stall_after_200_detected_by_first_token_timeout() -> None:
    # The server says 200 and sends keep-alives, but no tokens: abandon it.
    respx.post(URL).mock(
        side_effect=by_model(
            {
                "queued": lambda: httpx.Response(
                    200, content=sse(delta("queued", content="late"), comments=1)
                ),
                "good": ok(delta("good", content="pong")),
            }
        )
    )
    ticks: Iterator[float] = itertools.chain([0.0, 20.0], itertools.repeat(0.0))
    llm = make(["queued", "good"], clock=lambda: next(ticks))

    assert llm.invoke("ping").content == "pong"
    assert llm.call_log.records[0].error == "no tokens in 15s"


@respx.mock
def test_all_models_failing_raises_with_every_reason() -> None:
    respx.post(URL).respond(status_code=503, text="busy")
    llm = make(["a", "b"])
    with pytest.raises(LLMError, match=r"a: HTTP 503.*b: HTTP 503"):
        llm.invoke("ping")


@respx.mock
def test_bad_key_stops_immediately() -> None:
    route = respx.post(URL).respond(status_code=401, text="unauthorized")
    with pytest.raises(LLMError, match="rejected the key"):
        make(["a", "b"]).invoke("ping")
    assert route.call_count == 1


@respx.mock
def test_malformed_stream_falls_back() -> None:
    respx.post(URL).mock(
        side_effect=by_model(
            {
                "bad": lambda: httpx.Response(200, content=b"data: {not json\n\n"),
                "good": ok(delta("good", content="ok")),
            }
        )
    )
    llm = make(["bad", "good"])
    assert llm.invoke("x").content == "ok"
    assert llm.call_log.records[0].error == "malformed stream"


class ReadFile(BaseModel):
    """Read a file from the repository."""

    path: str


@respx.mock
def test_tool_calls_assembled_from_fragments() -> None:
    route = respx.post(URL).mock(
        side_effect=by_model(
            {
                "a": ok(
                    delta(
                        "a",
                        tool_calls=[
                            {
                                "index": 0,
                                "id": "c1",
                                "function": {"name": "ReadFile", "arguments": ""},
                            }
                        ],
                    ),
                    delta("a", tool_calls=[{"index": 0, "function": {"arguments": '{"path": '}}]),
                    delta("a", tool_calls=[{"index": 0, "function": {"arguments": '"a.py"}'}}]),
                    delta("a", content="\n"),
                    finish("tool_calls", "a"),
                )
            }
        )
    )
    message = make(["a"]).bind_tools([ReadFile]).invoke("open a.py")

    assert isinstance(message, AIMessage)
    assert message.content == ""
    assert message.tool_calls == [
        {"name": "ReadFile", "args": {"path": "a.py"}, "id": "c1", "type": "tool_call"}
    ]
    body = bodies(route)[0]
    assert body["tools"][0]["function"]["name"] == "ReadFile"
    assert "tool_choice" not in body


@respx.mock
def test_invalid_tool_arguments_reported() -> None:
    respx.post(URL).mock(
        side_effect=by_model(
            {
                "a": ok(
                    delta(
                        "a",
                        tool_calls=[
                            {"index": 0, "function": {"name": "ReadFile", "arguments": "{oops"}}
                        ],
                    )
                )
            }
        )
    )
    message = make(["a"]).bind_tools([ReadFile]).invoke("x")
    assert isinstance(message, AIMessage)
    assert message.tool_calls == []
    assert message.invalid_tool_calls[0]["args"] == "{oops"
    assert message.invalid_tool_calls[0]["id"] == "call_0"


@respx.mock
def test_structured_output_forces_tool_choice() -> None:
    route = respx.post(URL).mock(
        side_effect=by_model(
            {
                "a": ok(
                    delta(
                        "a",
                        tool_calls=[
                            {
                                "index": 0,
                                "id": "c1",
                                "function": {"name": "ReadFile", "arguments": '{"path": "x.py"}'},
                            }
                        ],
                    )
                )
            }
        )
    )
    result = make(["a"]).with_structured_output(ReadFile).invoke("which file?")
    assert result == ReadFile(path="x.py")
    assert bodies(route)[0]["tool_choice"] == "required"


@respx.mock
def test_tool_choice_by_name_and_history_conversion() -> None:
    route = respx.post(URL).mock(side_effect=by_model({"a": ok(delta("a", content="done"))}))
    history = [
        HumanMessage("open a.py"),
        AIMessage("", tool_calls=[{"name": "ReadFile", "args": {"path": "a.py"}, "id": "c1"}]),
        ToolMessage("print('hi')", tool_call_id="c1"),
    ]
    make(["a"]).bind_tools([ReadFile], tool_choice="ReadFile").invoke(history, stop=["END"])

    body = bodies(route)[0]
    assert body["tool_choice"] == {"type": "function", "function": {"name": "ReadFile"}}
    assert body["stop"] == ["END"]
    assert [m["role"] for m in body["messages"]] == ["user", "assistant", "tool"]
    assert body["messages"][2]["tool_call_id"] == "c1"


@respx.mock
def test_reasoning_params_sent_per_model_family() -> None:
    route = respx.post(URL).mock(
        side_effect=by_model(
            {
                "openai/gpt-oss-20b": lambda: httpx.Response(500),
                "nvidia/nemotron-x": ok(delta("n", content="ok")),
            }
        )
    )
    make(["openai/gpt-oss-20b", "nvidia/nemotron-x"], reasoning="off").invoke("x")
    first, second = bodies(route)
    assert first["reasoning_effort"] == "low" and "chat_template_kwargs" not in first
    assert second["chat_template_kwargs"] == {"enable_thinking": False}


@pytest.mark.parametrize(
    ("model", "mode", "expected"),
    [
        ("nvidia/x", "default", {}),
        ("nvidia/x", "on", {"chat_template_kwargs": {"enable_thinking": True}}),
        ("nvidia/x", "off", {"chat_template_kwargs": {"enable_thinking": False}}),
        ("nvidia/x", "low", {"reasoning_effort": "low"}),
        ("openai/gpt-oss-20b", "on", {"reasoning_effort": "high"}),
        ("openai/gpt-oss-20b", "off", {"reasoning_effort": "low"}),
        ("openai/gpt-oss-20b", "default", {}),
    ],
)
def test_reasoning_params(model: str, mode: Reasoning, expected: dict[str, Any]) -> None:
    assert reasoning_params(model, mode) == expected


def test_chat_model_from_settings() -> None:
    log = CallLog()
    settings = Settings(
        _env_file=None,
        nvidia_api_key=SecretStr("nvapi-x"),
        model_coder=" a/one , b/two ,",
        reasoning_coder="low",
        llm_first_token_timeout_s=7,
    )
    llm = chat_model(ModelTier.CODER, settings, call_log=log)
    assert llm.models == ["a/one", "b/two"]
    assert llm.reasoning == "low"
    assert llm.first_token_timeout_s == 7
    assert llm.call_log is log


def test_chat_model_requires_key() -> None:
    with pytest.raises(LLMConfigError, match="NVIDIA_API_KEY"):
        chat_model(ModelTier.FAST, Settings(_env_file=None))


@pytest.mark.parametrize(
    "text",
    [
        '[[{"name": "ReadFile", "parameters": {"path": "a.py"}}]]',
        '{"name": "ReadFile", "arguments": {"path": "a.py"}}',
        '```json\n{"name": "ReadFile", "arguments": "{\\"path\\": \\"a.py\\"}"}\n```',
        '<tool_call>{"name": "ReadFile", "parameters": {"path": "a.py"}}</tool_call>',
    ],
)
@respx.mock
def test_tool_call_written_as_text_is_recovered(text: str) -> None:
    respx.post(URL).mock(side_effect=by_model({"a": ok(delta("a", content=text))}))
    message = make(["a"]).bind_tools([ReadFile]).invoke("open a.py")
    assert isinstance(message, AIMessage)
    assert message.content == ""
    assert message.tool_calls == [
        {"name": "ReadFile", "args": {"path": "a.py"}, "id": "text_call_0", "type": "tool_call"}
    ]


@pytest.mark.parametrize(
    "text",
    [
        'I will call {"name": "ReadFile", "parameters": {"path": "a.py"}} next.',  # prose
        '{"name": "DeleteRepo", "parameters": {}}',  # unknown tool
        '[{"name": "ReadFile", "parameters": {"path": "a"}}, {"name": "Nope"}]',  # mixed
        '{"name": "ReadFile", "parameters": "not json"}',
        '{"name": "ReadFile", "parameters": [1, 2]}',
    ],
)
@respx.mock
def test_text_that_is_not_a_clean_tool_call_stays_text(text: str) -> None:
    respx.post(URL).mock(side_effect=by_model({"a": ok(delta("a", content=text))}))
    message = make(["a"]).bind_tools([ReadFile]).invoke("x")
    assert isinstance(message, AIMessage)
    assert message.tool_calls == [] and message.content == text


@respx.mock
def test_no_recovery_without_tools() -> None:
    text = '{"name": "ReadFile", "parameters": {"path": "a.py"}}'
    respx.post(URL).mock(side_effect=by_model({"a": ok(delta("a", content=text))}))
    message = make(["a"]).invoke("x")
    assert message.content == text


@respx.mock
def test_second_pass_after_whole_chain_failed() -> None:
    responses = iter(
        [
            httpx.Response(503),
            httpx.Response(503),
            httpx.Response(200, content=sse(delta("a", content="back"))),
        ]
    )
    respx.post(URL).mock(side_effect=lambda _r: next(responses))
    pauses: list[float] = []
    llm = make(["a", "b"])
    llm.sleep = pauses.append
    assert llm.invoke("x").content == "back"
    assert pauses == [2.0]
    assert [r.ok for r in llm.call_log.records] == [False, False, True]


@respx.mock
def test_first_token_timeout_grows_with_prompt_size() -> None:
    respx.post(URL).mock(
        side_effect=by_model(
            {
                "a": lambda: httpx.Response(200, content=sse(delta("a", content="x"), comments=1)),
            }
        )
    )
    # 16s without a token: too slow for a short prompt, fine for a ~20k-token one.
    for prompt, expect_ok in (("short", False), ("y" * 80_000, True)):
        ticks = itertools.chain([0.0, 16.0], itertools.repeat(16.0))
        llm = make(["a"], clock=lambda t=ticks: next(t))  # type: ignore[misc]
        llm.chain_passes = 1
        if expect_ok:
            assert llm.invoke(prompt).content == "x"
        else:
            with pytest.raises(LLMError, match="no tokens in 15s"):
                llm.invoke(prompt)


@pytest.mark.parametrize(
    "text",
    [
        '[[{"name": "ReadFile", "parameters": {"path": "a.py"}}]',  # Nemotron: [[ ... ]
        '[[{"name": "readfile", "parameters": {"path": "a.py"}}]]',  # wrong case
        '{"name": "ReadFile", "parameters": {"path": "a]}.py"}',  # brackets inside a string
    ],
)
@respx.mock
def test_unbalanced_or_miscased_text_tool_call_recovered(text: str) -> None:
    respx.post(URL).mock(side_effect=by_model({"a": ok(delta("a", content=text))}))
    message = make(["a"]).bind_tools([ReadFile]).invoke("x")
    assert isinstance(message, AIMessage)
    assert [tc["name"] for tc in message.tool_calls] == ["ReadFile"]
    assert message.tool_calls[0]["args"]["path"] in ("a.py", "a]}.py")


@respx.mock
def test_api_tool_call_name_case_fixed() -> None:
    respx.post(URL).mock(
        side_effect=by_model(
            {
                "a": ok(
                    delta(
                        "a",
                        tool_calls=[
                            {
                                "index": 0,
                                "id": "c1",
                                "function": {"name": "readfile", "arguments": '{"path": "a.py"}'},
                            }
                        ],
                    )
                )
            }
        )
    )
    result = make(["a"]).with_structured_output(ReadFile).invoke("x")
    assert result == ReadFile(path="a.py")
