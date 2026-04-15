from dotenv import load_dotenv

load_dotenv()

from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import TextLoader
model = ChatMistralAI(model = 'mistral-small-latest')

# Loading the data from the text file
data = TextLoader("documentLoaders/notes.txt")
# Loading the data into the document list
documents = data.load()

# creating a prompt template

template = ChatPromptTemplate.from_messages([("system","You are a Ai that summarises the text"),("human","{data}")])


# Formatting the prompt with the data
finalPrompt = template.format_messages(data= documents[0].page_content)

# Invoking the model with the formatted prompt
response = model.invoke(finalPrompt)
print(response.content)