from types import SimpleNamespace

import pytest

from secrag import llm


@pytest.fixture(autouse=True)
def isolated_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(llm.settings, "CACHE_DIR", tmp_path)
    llm._response_cache.cache_clear()
    yield
    llm._response_cache.cache_clear()


def test_models_yaml_is_valid():
    models = llm.load_models()
    assert models, "models.yaml defines no aliases"
    for spec in models.values():
        assert "/" in spec.model and spec.family and spec.roles


def test_every_judge_has_a_cross_family_generator():
    models = llm.load_models().values()
    gen_families = {m.family for m in models if "gen" in m.roles}
    for judge in (m for m in models if "judge" in m.roles):
        assert gen_families - {judge.family}


def test_cache_breakpoint_marks_end_of_system_prefix_only():
    msgs = [
        {"role": "system", "content": "rules"},
        {"role": "user", "content": "question"},
    ]
    out = llm.add_cache_breakpoint(msgs)
    assert out[0]["content"] == [
        {"type": "text", "text": "rules", "cache_control": {"type": "ephemeral"}}
    ]
    assert out[1] == msgs[1]
    assert msgs[0]["content"] == "rules"  # input not mutated


def test_cache_breakpoint_noop_without_system():
    msgs = [{"role": "user", "content": "q"}]
    assert llm.add_cache_breakpoint(msgs) == msgs


def test_cache_key_is_order_independent_and_sample_sensitive():
    m = [{"role": "user", "content": "q"}]
    a = llm.cache_key("x/y", m, {"a": 1, "b": 2}, 0)
    assert a == llm.cache_key("x/y", m, {"b": 2, "a": 1}, 0)
    assert a != llm.cache_key("x/y", m, {"a": 1, "b": 2}, 1)


def test_normalize_usage_handles_openai_and_anthropic_shapes():
    openai = {
        "prompt_tokens": 10,
        "completion_tokens": 2,
        "prompt_tokens_details": {"cached_tokens": 8},
    }
    anthropic = {
        "prompt_tokens": 10,
        "completion_tokens": 2,
        "cache_read_input_tokens": 5,
        "cache_creation_input_tokens": 3,
    }
    assert llm.normalize_usage(openai)["cache_read_input_tokens"] == 8
    assert llm.normalize_usage(anthropic)["cache_creation_input_tokens"] == 3
    assert llm.normalize_usage(None)["input"] == 0


def test_complete_serves_repeat_calls_from_disk_cache(monkeypatch):
    calls = []

    def fake_completion(model, messages, **params):
        calls.append(model)
        return SimpleNamespace(
            model_dump=lambda: {
                "choices": [{"message": {"content": "10-K"}}],
                "usage": {"prompt_tokens": 12, "completion_tokens": 3},
            }
        )

    monkeypatch.setattr(llm.litellm, "completion", fake_completion)
    monkeypatch.setattr(llm.litellm, "completion_cost", lambda completion_response: 0.001)
    msgs = [{"role": "user", "content": "annual report form?"}]

    first = llm.complete("claude-haiku", msgs)
    second = llm.complete("claude-haiku", msgs)

    assert len(calls) == 1
    assert (first.response_cache_hit, second.response_cache_hit) == (False, True)
    assert (first.cost_usd, second.cost_usd) == (0.001, 0.0)
    assert second.text == "10-K"
