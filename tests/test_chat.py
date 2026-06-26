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
