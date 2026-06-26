async def test_chat_returns_answer(client):
    response = await client.post(
        "/chat",
        json={"session_id": "test-1", "message": "Reply with exactly the word: hello"},
    )
    assert response.status_code == 200
    data = response.json()
    assert "answer" in data
    assert len(data["answer"]) > 0
    assert data["session_id"] == "test-1"
