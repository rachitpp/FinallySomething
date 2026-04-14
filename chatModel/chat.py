from dotenv import load_dotenv
from langchain.chat_models import init_chat_model

load_dotenv()

# If you want some creative task to be done, you can use high temperature value like 0.9, and if you want more precise task to be done, you can use low temperature value like 0.1
model = init_chat_model("google_genai:gemini-2.5-flash-lite",temperature=0.9,max_tokens =20)
response = model.invoke("Write a poem on AI")
print(response.content)
