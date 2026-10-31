"""Tests for the runnable classifier transports (no network)."""

import run_prompt_classifier as rpc


def _request(model):
    return {"model": model, "state": "s", "questions": {"q1": {}}}


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
