from python_a2a import A2AClient, Message, TextContent, MessageRole

client = A2AClient("http://localhost:5001/a2a")

response = client.send_message(
    Message(content=TextContent(text="Analyze AAPL"), role=MessageRole.USER)
)
print(response.content.text)
