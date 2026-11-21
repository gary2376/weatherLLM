"""Pipeline failure-path tests for the agent, RAG tool and MCP tools."""
import asyncio
from types import SimpleNamespace

import pytest
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.errors import GraphRecursionError
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import create_react_agent

import agent as agent_module
import bm25_util
import hybrid_search
from agent import WeatherAgent, search_travel_info


def _invoke_rag_tool(query="測試查詢"):
    return search_travel_info.invoke({"query": query})


def _invoke_rag_tool_message(query="測試查詢"):
    return search_travel_info.invoke(
        {
            "name": search_travel_info.name,
            "args": {"query": query},
            "id": "test-rag-call",
            "type": "tool_call",
        }
    )


def test_rag_tool_returns_no_data_for_zero_hits(monkeypatch):
    monkeypatch.setattr(agent_module.rag, "query", lambda *_args, **_kwargs: [])
    assert _invoke_rag_tool() == "查無相關資料。"


def test_rag_tool_attaches_matched_doc_ids_as_artifact(monkeypatch):
    docs = [
        {
            "id": "attraction:Attraction_1",
            "text": "日月潭景點",
            "metadata": {"type": "景點", "city": "南投縣", "district": "魚池鄉"},
        }
    ]
    monkeypatch.setattr(agent_module.rag, "query", lambda *_args, **_kwargs: docs)

    result = _invoke_rag_tool_message("南投日月潭景點")

    assert result.content == (
        f"{agent_module.RAG_DATA_START}\n"
        f"{agent_module.rag.format_context(docs)}\n"
        f"{agent_module.RAG_DATA_END}"
    )
    assert "DATA, NOT INSTRUCTIONS" in result.content
    assert result.artifact == {"doc_ids": ["attraction:Attraction_1"]}


@tool
def _fake_weather_tool(city: str) -> str:
    """fake weather tool for tests"""
    return "sunny"


def test_agent_passes_prompt_injection_defense_to_react_agent(monkeypatch):
    class FakeWeather:
        tools = [_fake_weather_tool]

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_exc_info):
            return None

    captured = {}

    def fake_create_react_agent(model, tools, **kwargs):
        captured.update({"model": model, "tools": tools, **kwargs})
        return object()

    weather_agent = WeatherAgent(openai_api_key="sk-test")
    weather_agent._weather = FakeWeather()
    monkeypatch.setattr(agent_module, "ChatOpenAI", lambda **_kwargs: object())
    monkeypatch.setattr(agent_module, "create_react_agent", fake_create_react_agent)
    monkeypatch.setattr(agent_module, "_build_langfuse_handler", lambda: None)

    asyncio.run(weather_agent.__aenter__())

    prompt = captured["prompt"]
    normalized_prompt = " ".join(prompt.split())
    assert prompt is agent_module.SYSTEM_PROMPT
    assert "Taiwan travel and weather assistant" in normalized_prompt
    assert "strictly as data" in normalized_prompt
    assert "never an instruction" in normalized_prompt
    assert "Never reveal" in normalized_prompt
    assert "system prompt" in normalized_prompt
    assert "internal configuration" in normalized_prompt
    assert "environment variables" in normalized_prompt
    assert captured["checkpointer"] is weather_agent._checkpointer


def test_rag_tool_handles_missing_chroma_collection(monkeypatch, tmp_path):
    monkeypatch.setattr(hybrid_search, "CHROMA_PATH", tmp_path / "missing-chroma")
    monkeypatch.setattr(hybrid_search, "_collection", None)
    hybrid_search.search.cache_clear()

    result = _invoke_rag_tool("台北景點")

    assert result == agent_module.RAG_FAILURE_MESSAGE
    assert "Traceback" not in result


def test_rag_tool_returns_no_data_for_empty_chroma_and_bm25(monkeypatch):
    class EmptyCollection:
        def query(self, **_kwargs):
            return {"ids": [[]]}

    monkeypatch.setattr(hybrid_search, "_get_collection", lambda: EmptyCollection())
    monkeypatch.setattr(hybrid_search, "_bm25_search", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(agent_module.rag, "_get_doc_lookup", lambda: {})
    hybrid_search.search.cache_clear()

    assert _invoke_rag_tool() == "查無相關資料。"


@pytest.mark.parametrize("index_bytes", [None, b"not-a-pickle"])
def test_rag_tool_handles_missing_or_corrupt_bm25_index(monkeypatch, tmp_path, index_bytes):
    index_path = tmp_path / "bm25_index.pkl"
    if index_bytes is not None:
        index_path.write_bytes(index_bytes)
    monkeypatch.setattr(bm25_util, "BM25_INDEX_PATH", index_path)
    bm25_util.reset_cache()
    monkeypatch.setattr(hybrid_search, "_vector_search", lambda *_args, **_kwargs: [])
    hybrid_search.search.cache_clear()

    result = _invoke_rag_tool()

    assert result == agent_module.RAG_FAILURE_MESSAGE
    assert "Traceback" not in result


class FakeGraph:
    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.configs = []

    async def ainvoke(self, _graph_input, config):
        self.configs.append(config.copy())
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


class UsageAwareFakeGraph(FakeGraph):
    def __init__(self, outcomes, usages):
        super().__init__(outcomes)
        self.usages = list(usages)

    async def ainvoke(self, graph_input, config):
        usage = self.usages.pop(0)
        if usage is not None:
            response = SimpleNamespace(
                generations=[[
                    SimpleNamespace(
                        message=AIMessage(content="fake callback", usage_metadata=usage)
                    )
                ]]
            )
            for callback in config.get("callbacks", []):
                if hasattr(callback, "on_llm_end"):
                    callback.on_llm_end(response)
        return await super().ainvoke(graph_input, config)


class ScriptedChatModel(BaseChatModel):
    """Deterministic chat model that still exercises LangChain's callback manager."""

    responses: list[AIMessage]
    index: int = 0
    calls: int = 0

    @property
    def _llm_type(self) -> str:
        return "weatherllm-test"

    def bind_tools(self, _tools, *, tool_choice=None, **_kwargs):
        return self

    def _generate(self, _messages, stop=None, run_manager=None, **_kwargs):
        self.calls += 1
        response = self.responses[self.index]
        if self.index < len(self.responses) - 1:
            self.index += 1
        return ChatResult(generations=[ChatGeneration(message=response)])


@tool
def _echo_for_integration_test(text: str) -> str:
    """Echo text for a deterministic ReAct integration test."""
    return text


def _compiled_graph_after_model(model, *, error=None, loop=False):
    """Compile a real graph whose node invokes a real LangChain chat-model path."""

    async def invoke_model(state, config):
        await model.ainvoke(state["messages"], config=config)
        if error is not None:
            raise error
        return {}

    builder = StateGraph(MessagesState)
    builder.add_node("model", invoke_model)
    builder.add_edge(START, "model")
    builder.add_edge("model", "model" if loop else END)
    return builder.compile(checkpointer=MemorySaver())


def _agent_with_graph(graph, langfuse_handler=None):
    weather_agent = WeatherAgent()
    weather_agent._graph = graph
    weather_agent._langfuse_handler = langfuse_handler
    return weather_agent


def test_agent_entry_failure_closes_started_weather_lifecycle(monkeypatch):
    class FakeWeather:
        tools = []

        def __init__(self):
            self.closed = False

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_exc_info):
            self.closed = True

    weather_agent = WeatherAgent(openai_api_key="sk-test")
    fake_weather = FakeWeather()
    weather_agent._weather = fake_weather
    monkeypatch.setattr(agent_module, "ChatOpenAI", lambda **_kwargs: object())
    monkeypatch.setattr(
        agent_module,
        "create_react_agent",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("forced graph init failure")),
    )

    with pytest.raises(RuntimeError, match="forced graph init failure"):
        asyncio.run(weather_agent.__aenter__())

    assert fake_weather.closed


def test_agent_blocks_hallucinated_answer_after_rag_failure():
    messages = [
        ToolMessage(
            content=agent_module.RAG_FAILURE_MESSAGE,
            tool_call_id="rag-1",
            name=search_travel_info.name,
            artifact={"doc_ids": []},
        ),
        AIMessage(content="這是模型自行編造的景點答案。"),
    ]
    weather_agent = _agent_with_graph(FakeGraph([{"messages": messages}]))

    reply, doc_ids = asyncio.run(weather_agent.ask("rag-failure", "推薦景點"))

    assert reply == agent_module.RAG_FAILURE_MESSAGE
    assert doc_ids == []
    assert "編造" not in reply


def test_agent_returns_ordered_deduplicated_rag_doc_ids():
    messages = [
        HumanMessage(content="上一輪問題"),
        ToolMessage(
            content="上一輪資料",
            tool_call_id="old-rag",
            name=search_travel_info.name,
            artifact={"doc_ids": ["attraction:OLD"]},
        ),
        AIMessage(content="上一輪回答"),
        HumanMessage(content="這一輪問題"),
        ToolMessage(
            content="景點資料",
            tool_call_id="rag-1",
            name=search_travel_info.name,
            artifact={"doc_ids": ["attraction:A1", "lodging:L1"]},
        ),
        ToolMessage(
            content="餐廳資料",
            tool_call_id="rag-2",
            name=search_travel_info.name,
            artifact={"doc_ids": ["lodging:L1", "restaurant:R1"]},
        ),
        AIMessage(content="正常回答"),
    ]
    weather_agent = _agent_with_graph(FakeGraph([{"messages": messages}]))

    reply, doc_ids = asyncio.run(weather_agent.ask("rag-success", "推薦行程"))

    assert reply == "正常回答"
    assert doc_ids == ["attraction:A1", "lodging:L1", "restaurant:R1"]


def test_weather_only_turn_does_not_reuse_previous_rag_doc_ids():
    messages = [
        HumanMessage(content="上一輪景點問題"),
        ToolMessage(
            content="上一輪景點資料",
            tool_call_id="old-rag",
            name=search_travel_info.name,
            artifact={"doc_ids": ["attraction:OLD"]},
        ),
        AIMessage(content="上一輪回答"),
        HumanMessage(content="台北天氣如何"),
        ToolMessage(
            content="台北目前晴朗",
            tool_call_id="weather-1",
            name="get_weather_forecast",
        ),
        AIMessage(content="台北目前晴朗"),
    ]
    weather_agent = _agent_with_graph(FakeGraph([{"messages": messages}]))

    reply, doc_ids = asyncio.run(weather_agent.ask("weather-only", "台北天氣如何"))

    assert reply == "台北目前晴朗"
    assert doc_ids == []


def test_previous_rag_failure_does_not_poison_later_weather_turn():
    messages = [
        HumanMessage(content="上一輪景點問題"),
        ToolMessage(
            content=agent_module.RAG_FAILURE_MESSAGE,
            tool_call_id="old-rag-failure",
            name=search_travel_info.name,
            artifact={"doc_ids": []},
        ),
        AIMessage(content=agent_module.RAG_FAILURE_MESSAGE),
        HumanMessage(content="台北天氣如何"),
        AIMessage(content="台北目前晴朗"),
    ]
    weather_agent = _agent_with_graph(FakeGraph([{"messages": messages}]))

    reply, doc_ids = asyncio.run(weather_agent.ask("old-rag-failure", "台北天氣如何"))

    assert reply == "台北目前晴朗"
    assert doc_ids == []


def test_agent_cost_only_adds_latest_turn_from_accumulated_history():
    previous_usage = {"input_tokens": 100, "output_tokens": 20, "total_tokens": 120}
    current_usage = {"input_tokens": 10, "output_tokens": 2, "total_tokens": 12}
    messages = [
        HumanMessage(content="上一輪問題"),
        AIMessage(content="上一輪回答", usage_metadata=previous_usage),
        HumanMessage(content="這一輪問題"),
        AIMessage(content="這一輪回答", usage_metadata=current_usage),
    ]
    thread_id = "latest-turn-cost"
    previous_cost = agent_module._response_cost_usd(
        [AIMessage(content="上一輪回答", usage_metadata=previous_usage)]
    )
    current_cost = agent_module._response_cost_usd(
        [AIMessage(content="這一輪回答", usage_metadata=current_usage)]
    )
    agent_module._thread_costs[thread_id] = previous_cost
    try:
        weather_agent = _agent_with_graph(FakeGraph([{"messages": messages}]))
        asyncio.run(weather_agent.ask(thread_id, "這一輪問題"))
        assert weather_agent.cost_so_far(thread_id) == pytest.approx(
            previous_cost + current_cost
        )
    finally:
        agent_module._thread_costs.pop(thread_id, None)


def test_weather_or_other_graph_failure_returns_clean_message():
    weather_agent = _agent_with_graph(FakeGraph([TimeoutError("forced weather timeout")]))

    reply, doc_ids = asyncio.run(weather_agent.ask("weather-failure", "台北天氣"))

    assert reply == agent_module.AGENT_FAILURE_MESSAGE
    assert doc_ids == []
    assert "Traceback" not in reply
    assert "forced weather timeout" not in reply


def test_recursion_limit_returns_clean_aborted_message():
    weather_agent = _agent_with_graph(FakeGraph([GraphRecursionError("forced loop")]))

    reply, doc_ids = asyncio.run(weather_agent.ask("recursion-failure", "一直找工具"))

    assert reply == agent_module.RECURSION_LIMIT_MESSAGE
    assert doc_ids == []
    assert "Traceback" not in reply


def test_cost_cap_returns_clean_aborted_message_without_graph_call():
    graph = FakeGraph([])
    weather_agent = _agent_with_graph(graph)
    thread_id = "cost-cap"
    agent_module._thread_costs[thread_id] = agent_module.COST_THRESHOLD_USD
    try:
        reply, doc_ids = asyncio.run(weather_agent.ask(thread_id, "不應送進graph"))
    finally:
        agent_module._thread_costs.pop(thread_id, None)

    assert "安全上限" in reply
    assert doc_ids == []
    assert "Traceback" not in reply
    assert not graph.configs


def test_graph_failure_with_langfuse_handler_retries_without_handler(caplog):
    normal = {"messages": [AIMessage(content="正常回答")]}
    graph = FakeGraph([RuntimeError("forced graph failure"), normal])
    langfuse_handler = SimpleNamespace()
    weather_agent = _agent_with_graph(graph, langfuse_handler=langfuse_handler)

    reply, doc_ids = asyncio.run(weather_agent.ask("trace-failure", "正常問題"))

    assert reply == "正常回答"
    assert doc_ids == []
    assert len(graph.configs) == 2
    assert langfuse_handler in graph.configs[0]["callbacks"]
    assert langfuse_handler not in graph.configs[1]["callbacks"]
    assert weather_agent._langfuse_handler is None
    assert "停用handler後防禦性重試一次" in caplog.text


def test_defensive_retry_counts_usage_from_both_attempts():
    first_usage = {"input_tokens": 100, "output_tokens": 20, "total_tokens": 120}
    retry_usage = {"input_tokens": 50, "output_tokens": 10, "total_tokens": 60}
    normal = {"messages": [AIMessage(content="正常回答", usage_metadata=retry_usage)]}
    graph = UsageAwareFakeGraph(
        [RuntimeError("forced graph failure"), normal],
        [first_usage, retry_usage],
    )
    thread_id = "trace-failure-with-cost"
    weather_agent = _agent_with_graph(graph, langfuse_handler=SimpleNamespace())
    expected_cost = agent_module._response_cost_usd(
        [
            AIMessage(content="第一次", usage_metadata=first_usage),
            AIMessage(content="重試", usage_metadata=retry_usage),
        ]
    )
    try:
        reply, _ = asyncio.run(weather_agent.ask(thread_id, "正常問題"))
        assert reply == "正常回答"
        assert weather_agent.cost_so_far(thread_id) == pytest.approx(expected_cost)
    finally:
        agent_module._thread_costs.pop(thread_id, None)


def test_defensive_retry_without_first_attempt_usage_counts_retry_only():
    retry_usage = {"input_tokens": 50, "output_tokens": 10, "total_tokens": 60}
    normal = {"messages": [AIMessage(content="正常回答", usage_metadata=retry_usage)]}
    graph = UsageAwareFakeGraph(
        [RuntimeError("forced graph failure before LLM"), normal],
        [None, retry_usage],
    )
    thread_id = "trace-failure-without-first-cost"
    weather_agent = _agent_with_graph(graph, langfuse_handler=SimpleNamespace())
    expected_cost = agent_module._response_cost_usd(
        [AIMessage(content="重試", usage_metadata=retry_usage)]
    )
    try:
        reply, _ = asyncio.run(weather_agent.ask(thread_id, "正常問題"))
        assert reply == "正常回答"
        assert weather_agent.cost_so_far(thread_id) == pytest.approx(expected_cost)
    finally:
        agent_module._thread_costs.pop(thread_id, None)


def test_real_react_graph_callback_accumulates_all_model_calls():
    first_usage = {"input_tokens": 100, "output_tokens": 20, "total_tokens": 120}
    second_usage = {"input_tokens": 50, "output_tokens": 10, "total_tokens": 60}
    model = ScriptedChatModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[{
                    "name": _echo_for_integration_test.name,
                    "args": {"text": "hello"},
                    "id": "echo-1",
                    "type": "tool_call",
                }],
                usage_metadata=first_usage,
            ),
            AIMessage(content="正常回答", usage_metadata=second_usage),
        ]
    )
    graph = create_react_agent(
        model,
        [_echo_for_integration_test],
        checkpointer=MemorySaver(),
    )
    handler = agent_module._CostTrackingHandler()

    result = asyncio.run(
        graph.ainvoke(
            {"messages": [{"role": "user", "content": "測試"}]},
            config={
                "configurable": {"thread_id": "real-callback-multiple-calls"},
                "callbacks": [handler],
            },
        )
    )

    expected_cost = agent_module._response_cost_usd(
        [
            AIMessage(content="第一次", usage_metadata=first_usage),
            AIMessage(content="第二次", usage_metadata=second_usage),
        ]
    )
    assert result["messages"][-1].content == "正常回答"
    assert model.calls == 2
    assert handler.has_usage
    assert handler.cost_usd == pytest.approx(expected_cost)


def test_real_graph_recursion_after_model_cost_is_recorded(monkeypatch):
    usage = {"input_tokens": 80, "output_tokens": 8, "total_tokens": 88}
    model = ScriptedChatModel(
        responses=[AIMessage(content="繼續", usage_metadata=usage)]
    )
    graph = _compiled_graph_after_model(model, loop=True)
    thread_id = "real-recursion-with-cost"
    weather_agent = _agent_with_graph(graph)
    monkeypatch.setattr(agent_module, "RECURSION_LIMIT", 2)
    try:
        reply, doc_ids = asyncio.run(weather_agent.ask(thread_id, "一直循環"))
        single_call_cost = agent_module._response_cost_usd(
            [AIMessage(content="一次", usage_metadata=usage)]
        )
        assert reply == agent_module.RECURSION_LIMIT_MESSAGE
        assert doc_ids == []
        assert model.calls >= 1
        assert weather_agent.cost_so_far(thread_id) == pytest.approx(
            single_call_cost * model.calls
        )
    finally:
        agent_module._thread_costs.pop(thread_id, None)


def test_real_graph_retry_failure_records_both_attempt_costs():
    usage = {"input_tokens": 60, "output_tokens": 6, "total_tokens": 66}
    model = ScriptedChatModel(
        responses=[AIMessage(content="呼叫完成", usage_metadata=usage)]
    )
    graph = _compiled_graph_after_model(model, error=RuntimeError("node failed"))
    thread_id = "real-retry-both-fail-with-cost"
    weather_agent = _agent_with_graph(graph, langfuse_handler=BaseCallbackHandler())
    try:
        reply, doc_ids = asyncio.run(weather_agent.ask(thread_id, "觸發兩次失敗"))
        single_call_cost = agent_module._response_cost_usd(
            [AIMessage(content="一次", usage_metadata=usage)]
        )
        assert reply == agent_module.AGENT_FAILURE_MESSAGE
        assert doc_ids == []
        assert model.calls == 2
        assert weather_agent.cost_so_far(thread_id) == pytest.approx(single_call_cost * 2)
    finally:
        agent_module._thread_costs.pop(thread_id, None)


def test_real_graph_retry_can_end_in_recursion_error(monkeypatch):
    usage = {"input_tokens": 40, "output_tokens": 4, "total_tokens": 44}
    model = ScriptedChatModel(
        responses=[AIMessage(content="仍在執行", usage_metadata=usage)]
    )
    invocations = 0

    async def fail_once_then_loop(state, config):
        nonlocal invocations
        await model.ainvoke(state["messages"], config=config)
        invocations += 1
        if invocations == 1:
            raise RuntimeError("first attempt failed")
        return {}

    builder = StateGraph(MessagesState)
    builder.add_node("model", fail_once_then_loop)
    builder.add_edge(START, "model")
    builder.add_edge("model", "model")
    graph = builder.compile(checkpointer=MemorySaver())
    thread_id = "real-retry-recursion"
    weather_agent = _agent_with_graph(graph, langfuse_handler=BaseCallbackHandler())
    monkeypatch.setattr(agent_module, "RECURSION_LIMIT", 2)
    try:
        reply, doc_ids = asyncio.run(weather_agent.ask(thread_id, "先失敗再循環"))
        single_call_cost = agent_module._response_cost_usd(
            [AIMessage(content="一次", usage_metadata=usage)]
        )
        assert reply == agent_module.RECURSION_LIMIT_MESSAGE
        assert doc_ids == []
        assert model.calls >= 2
        assert weather_agent.cost_so_far(thread_id) == pytest.approx(
            single_call_cost * model.calls
        )
    finally:
        agent_module._thread_costs.pop(thread_id, None)


def test_real_graph_malformed_usage_does_not_break_answer(caplog):
    class InvalidUsage(dict):
        def get(self, *_args, **_kwargs):
            raise TypeError("invalid usage value")

    response = AIMessage(content="正常回答")
    object.__setattr__(
        response,
        "usage_metadata",
        InvalidUsage(input_tokens=1, output_tokens=1, total_tokens=2),
    )
    model = ScriptedChatModel(responses=[response])
    graph = create_react_agent(model, [], checkpointer=MemorySaver())
    thread_id = "real-malformed-usage"
    weather_agent = _agent_with_graph(graph)
    try:
        reply, doc_ids = asyncio.run(weather_agent.ask(thread_id, "正常問題"))
        assert reply == "正常回答"
        assert doc_ids == []
        assert weather_agent.cost_so_far(thread_id) == 0.0
        assert "無法解析LLM token usage" in caplog.text
    finally:
        agent_module._thread_costs.pop(thread_id, None)


def test_cost_callback_ignores_malformed_generation_structure():
    handler = agent_module._CostTrackingHandler()

    handler.on_llm_end(SimpleNamespace(generations=None))

    assert handler.cost_usd == 0.0
    assert not handler.has_usage
