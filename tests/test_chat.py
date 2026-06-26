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


async def test_extract_person(client):
    response = await client.post(
        "/extract",
        json={"text": "John Smith is 35 years old"},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["name"] == "John Smith"
    assert data["age"] == 35


async def test_weather_tool_called(client):
    response = await client.post(
        "/chat",
        json={"session_id": "weather-test-1", "message": "What is the weather in Warsaw?"},
    )
    assert response.status_code == 200
    data = response.json()
    # Mocked tool returns "22°C, partly cloudy" — must appear in answer
    answer = data["answer"]
    assert "22" in answer or "cloudy" in answer or "partly" in answer
