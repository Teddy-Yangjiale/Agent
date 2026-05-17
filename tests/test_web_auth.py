from __future__ import annotations

from fastapi.testclient import TestClient

from agent_harness.web.server import AgentAPI


class FakeAgent:
    name = "fake"


class FakeTools:
    def __len__(self):
        return 0

    def list_all(self):
        return []


class FakeExecutor:
    agent = FakeAgent()
    tools = FakeTools()


def test_web_api_allows_requests_without_configured_key():
    client = TestClient(AgentAPI(FakeExecutor()).app)

    response = client.get("/api/health")

    assert response.status_code == 200


def test_web_api_requires_key_when_configured():
    client = TestClient(AgentAPI(FakeExecutor(), api_key="secret").app)

    assert client.get("/api/health").status_code == 401
    assert client.get("/api/health", headers={"x-api-key": "secret"}).status_code == 200
