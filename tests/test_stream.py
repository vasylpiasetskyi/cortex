import pytest

pytestmark = pytest.mark.asyncio


async def test_stream_returns_chunks(client):
    response = await client.post(
        "/chat/stream",
        json={"session_id": "stream-test-1", "message": "Count from 1 to 3"},
    )
    assert response.status_code == 200
    assert "text/event-stream" in response.headers["content-type"]
    lines = [line for line in response.text.split("\n") if line.startswith("data:")]
    assert len(lines) > 0


async def test_stream_saves_to_history(client):
    session_id = "stream-memory-1"
    await client.post(
        "/chat/stream",
        json={"session_id": session_id, "message": "My dog is named Rex."},
    )
    response = await client.post(
        "/chat", json={"session_id": session_id, "message": "What is my dog's name?"}
    )
    assert "Rex" in response.json()["answer"]
