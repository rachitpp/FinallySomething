from dotenv import load_dotenv
load_dotenv()


from pydantic import BaseModel
from typing import List,Optional
from langchain_core.output_parsers import PydanticOutputParser
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate

class Movie(BaseModel):
    title: str
    release_year: Optional[int]
    genre: List[str]
    director: Optional[str]
    cast: List[str]
    imdb_rating: Optional[float]
    summary: str

parser = PydanticOutputParser(pydantic_object = Movie)


model = ChatMistralAI(model = 'mistral-small-latest')

prompt = ChatPromptTemplate.from_messages([
    ("system","""
    Extract movie information from the Paragraph {format_instructions}
    """),

    ("human","{paragraph}")
])

paragraph = input(" Give your Paragraph : ")
final_prompt = prompt.invoke({"paragraph":paragraph,"format_instructions":parser.get_format_instructions()})
response = model.invoke(final_prompt)
print(f"\n\n{response.content}")