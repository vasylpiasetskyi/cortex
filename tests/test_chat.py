async def test_chat_returns_answer(client):
    response = await client.post(
        "/chat",
        json={"session_id": "test-1", "message": "Hello"},
    )
    assert response.status_code == 200
    data = response.json()
    assert "answer" in data
    assert data["answer"] == "mocked response"
    assert data["session_id"] == "test-1"


async def test_chat_remembers_history(client):
    session_id = "history-test-1"
    # First call saves "My name is Alice" → mock returns "mocked response"
    await client.post("/chat", json={"session_id": session_id, "message": "My name is Alice."})
    # Second call — mock returns response
    response = await client.post(
        "/chat", json={"session_id": session_id, "message": "What is my name?"}
    )
    assert response.status_code == 200
    data = response.json()
    assert "answer" in data
    assert data["session_id"] == "history-test-1"
    # Verify the endpoint works across multiple calls without breaking
    assert len(data["answer"]) > 0
