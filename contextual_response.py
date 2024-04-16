import os
import vertexai
from vertexai.generative_models import GenerativeModel, ChatSession, Content, Part

# Project Information
project_id = os.environ.get("ARBITRATOR_PROJECT_ID")
location = "us-central1"
vertexai.init(project=project_id, location=location)
model = GenerativeModel("gemini-1.0-pro")
chat = model.start_chat()


def get_chat_response(chat: ChatSession, prompt: str) -> str:
    text_response = []
    responses = chat.send_message(prompt, stream=True)
    for chunk in responses:
        text_response.append(chunk.text)
    return "".join(text_response)


# Sample chat history the model should pick up from.
messages = [
    {"role": "user", "parts": {"text": "Tell a knock, knock joke."}},
    {"role": "model", "parts": {"text": "Knock, knock."}},
    {"role": "user", "parts": {"text": "Who's there?"}},
    {"role": "model", "parts": {"text": "Olive."}},
]

# Preload chat history to Gemini model's response.
for message in messages:
    chat.history.append(
        Content(
            role=message["role"],
            parts=[Part.from_text(message["parts"]["text"])],
        )
    )

# Send the chat history and the last message response.
prompt = "Olive who?"
chat_response = get_chat_response(chat, prompt)

# Print entire chat history
for message in chat.history:
    print(f"{message.role}\t{message.parts[0].text}")
