"""Tests for the runnable classifier transports (no network)."""

import run_prompt_classifier as rpc
from decider import DeciderState, Question, System, build_spark_request


def _request(model):
    return {"model": model, "state": "s", "questions": {"q1": {}}}


def test_jev_transport_posts_system_one_request(monkeypatch):
    calls = {}

    def fake_post(url, payload, token):
        calls.update(url=url, payload=payload, token=token)
        return {"model": "jev-latest", "answers": {"q1": {"type": "noul", "noul": 0.9}}}

    monkeypatch.setattr(rpc, "_post_json", fake_post)
    monkeypatch.setenv("JEV_API_URL", "https://example.test/v1/systemone")
    monkeypatch.setenv("JEV_API_KEY", "secret")

    request = {"model": "jev-latest", "state": "s", "questions": {"q1": {}}}
    assert rpc.jev_transport(request)["answers"]["q1"]["noul"] == 0.9
    assert calls["url"] == "https://example.test/v1/systemone"
    assert calls["payload"] == request
    assert calls["token"] == "secret"


def test_jev_transport_defaults_to_official_endpoint(monkeypatch):
    calls = {}

    def fake_post(url, payload, token):
        calls["url"] = url
        return {}

    monkeypatch.setattr(rpc, "_post_json", fake_post)
    monkeypatch.delenv("JEV_API_URL", raising=False)
    monkeypatch.setenv("JEV_API_KEY", "secret")
    rpc.jev_transport({"model": "jev-latest", "state": "s", "questions": {}})
    assert calls["url"] == "https://api.typesafe.ai/v1/systemone"


def test_spark_prompt_shows_expected_only_for_q2(monkeypatch):
    prompts = []

    def fake_post(url, payload, token):
        prompts.append(payload["messages"][0]["content"])
        return {"choices": [{"message": {"content": '{"answer": "Y", "confidence": 0.9}'}}]}

    monkeypatch.setattr(rpc, "_post_json", fake_post)
    monkeypatch.setenv("META_API_KEY", "secret")
    system = System(id="s", name="S", spec_ref=None, context="assumed_known")
    state = DeciderState(system=system, prompt_text="p", response="r", expected="e")

    def ask(qid):
        question = Question(id=qid, text="t?", on_Y="q3", on_N="q4")
        rpc.spark_transport(build_spark_request(state, question, "ctx"))

    ask("q1")
    ask("q2")
    assert "Question (answer Y or N): t?" in prompts[0]
    assert "expected_system_output:" not in prompts[0]
    assert "expected_system_output: e" in prompts[1]


def test_clef_transport_routes_by_request_model(monkeypatch):
    calls = []

    def fake_post(url, payload, token):
        calls.append(url)
        return {"result": {"answers": {"q1": {"type": "noul", "noul": 0.9}}}}

    monkeypatch.setattr(rpc, "_post_json", fake_post)
    monkeypatch.setenv("CLOUDFLARE_ACCOUNT_ID", "acct123")
    monkeypatch.setenv("CLOUDFLARE_API_TOKEN", "tok")

    rpc.clef_transport(_request("clef-flash"))
    rpc.clef_transport(_request("clef"))

    assert calls == [
        "https://api.cloudflare.com/client/v4/accounts/acct123"
        "/ai/run/@cf/cloudflare/clef-flash",
        "https://api.cloudflare.com/client/v4/accounts/acct123"
        "/ai/run/@cf/cloudflare/clef",
    ]
