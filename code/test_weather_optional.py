"""天氣是選用功能：使用者金鑰優先、沒有金鑰或天氣啟動失敗時退化成只提供旅遊問答。"""
import asyncio

import pytest
from langchain_core.tools import tool

import agent as agent_module
from agent import WEATHER_UNAVAILABLE_NOTE, SYSTEM_PROMPT, WeatherAgent, search_travel_info


@tool
def fake_weather_tool(city: str) -> str:
    """fake weather tool"""
    return "sunny"


class FakeWeather:
    def __init__(self, tools=(), fail=None):
        self.tools = list(tools)
        self.fail = fail
        self.closed = False

    async def __aenter__(self):
        if self.fail:
            raise self.fail
        return self

    async def __aexit__(self, *_):
        self.closed = True


@pytest.fixture
def captured(monkeypatch):
    seen = {}
    monkeypatch.setattr(agent_module, "load_dotenv", lambda *a, **k: None)
    monkeypatch.setattr(agent_module, "ChatOpenAI", lambda **_k: object())
    monkeypatch.setattr(agent_module, "_build_langfuse_handler", lambda: None)

    def fake_create(llm, tools, prompt=None, checkpointer=None):
        seen["tools"] = [t.name for t in tools]
        seen["prompt"] = prompt
        return object()

    monkeypatch.setattr(agent_module, "create_react_agent", fake_create)
    return seen


def _enter(agent):
    return asyncio.run(agent.__aenter__())


def test_user_key_takes_priority_over_operator_key(monkeypatch, captured):
    monkeypatch.setenv("CWA_API_KEY", "CWA-operator")
    agent = WeatherAgent(openai_api_key="sk-test", cwa_api_key="  CWA-user  ")
    assert agent.cwa_key_source == "user"
    assert agent._weather._cwa_api_key == "CWA-user"


def test_operator_key_used_when_user_leaves_it_empty(monkeypatch, captured):
    monkeypatch.setenv("CWA_API_KEY", "CWA-operator")
    for empty in (None, "", "   "):
        agent = WeatherAgent(openai_api_key="sk-test", cwa_api_key=empty)
        assert agent.cwa_key_source == "operator"
        assert agent._weather._cwa_api_key == "CWA-operator"


def test_no_key_at_all_runs_travel_only_with_honest_note(monkeypatch, captured):
    monkeypatch.delenv("CWA_API_KEY", raising=False)
    agent = WeatherAgent(openai_api_key="sk-test")
    assert agent.cwa_key_source == "none" and agent._weather is None
    _enter(agent)
    assert captured["tools"] == [search_travel_info.name]
    assert captured["prompt"] == SYSTEM_PROMPT + WEATHER_UNAVAILABLE_NOTE
    assert agent.weather_enabled is False
    asyncio.run(agent.__aexit__(None, None, None))  # 沒有天氣連線也不能出錯


def test_weather_tools_are_added_when_available(monkeypatch, captured):
    monkeypatch.delenv("CWA_API_KEY", raising=False)
    agent = WeatherAgent(openai_api_key="sk-test", cwa_api_key="CWA-user")
    agent._weather = FakeWeather([fake_weather_tool])
    _enter(agent)
    assert captured["tools"] == [search_travel_info.name, "fake_weather_tool"]
    assert captured["prompt"] == SYSTEM_PROMPT
    assert agent.weather_enabled is True
    asyncio.run(agent.__aexit__(None, None, None))
    assert agent._weather.closed


def test_weather_startup_failure_degrades_instead_of_crashing(monkeypatch, captured):
    monkeypatch.setenv("CWA_API_KEY", "CWA-operator")
    agent = WeatherAgent(openai_api_key="sk-test")
    agent._weather = FakeWeather(fail=RuntimeError("cwa-mcp-server missing"))
    _enter(agent)
    assert agent.weather_enabled is False and agent._weather is None
    assert captured["tools"] == [search_travel_info.name]
    assert captured["prompt"].endswith(WEATHER_UNAVAILABLE_NOTE)


def test_cancellation_during_weather_startup_is_not_swallowed(monkeypatch, captured):
    agent = WeatherAgent(openai_api_key="sk-test", cwa_api_key="CWA-user")
    agent._weather = FakeWeather(fail=asyncio.CancelledError())
    with pytest.raises(asyncio.CancelledError):
        _enter(agent)


def test_resolve_cwa_key_priority_and_whitespace(monkeypatch):
    monkeypatch.setattr(agent_module, "load_dotenv", lambda *a, **k: None)
    monkeypatch.setenv("CWA_API_KEY", "  CWA-operator  ")
    assert agent_module.resolve_cwa_key("  CWA-user ") == ("user", "CWA-user")
    assert agent_module.resolve_cwa_key("   ") == ("operator", "CWA-operator")
    monkeypatch.setenv("CWA_API_KEY", "   ")  # 只有空白視為沒有金鑰，UI與agent一致
    assert agent_module.resolve_cwa_key(None) == ("none", "")
    monkeypatch.delenv("CWA_API_KEY", raising=False)
    assert agent_module.resolve_cwa_key("") == ("none", "")


def test_mcp_server_returning_zero_tools_degrades_with_note(monkeypatch, captured):
    monkeypatch.setenv("CWA_API_KEY", "CWA-operator")
    agent = WeatherAgent(openai_api_key="sk-test")
    agent._weather = FakeWeather(tools=[])
    _enter(agent)
    assert agent.weather_enabled is False
    assert captured["tools"] == [search_travel_info.name]
    assert captured["prompt"].endswith(WEATHER_UNAVAILABLE_NOTE)
    asyncio.run(agent.__aexit__(None, None, None))
    assert agent._weather.closed
