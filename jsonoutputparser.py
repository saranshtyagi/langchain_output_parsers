from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

load_dotenv()


llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.2-1B-Instruct",
    task="text-generation",
    max_new_tokens=512,
    temperature=0.7
)

model = ChatHuggingFace(llm = llm)

parser = JsonOutputParser()

template = PromptTemplate(
    template='Give me the name, age and city of a fictional person \n {format_instructions}', 
    input_variables=[], 
    partial_variables={"format_instructions": parser.get_format_instructions()}
)

chain = template | model | parser

result = chain.invoke({})

print(result)