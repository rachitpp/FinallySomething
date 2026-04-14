# Importing dotenv to load the enviorment variables
from dotenv import load_dotenv
# Loading the enviorment variables
load_dotenv()
# Importing the Required langchain modules 
from langchain_mistralai import ChatMistralAI
from langchain_core.messages import AIMessage, SystemMessage, HumanMessage

# Initializing the model
model = ChatMistralAI(model = 'mistral-small-latest',temperature=0.9)

# Initializing the conversation history
message = [SystemMessage(content="You are a funny assistant")]
print("-------------------- welcome type 0 to exit the application------------------")

# Main loop for the chatbot
while True:
    prompt = input("You:")
    # Adding the user message to the conversation history as a HumanMessage
    message.append(HumanMessage(content=prompt))
    if(prompt =='0'):
        break
    response = model.invoke(message)
    # Adding the model response to the conversation history as a AIMessage
    message.append(AIMessage(content=response.content))
    print("Bot:",response.content)

print("----Conversation History----")
print(message)