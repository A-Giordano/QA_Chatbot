import chainlit as cl
from chatbot_agent import get_agent
import uuid


@cl.langchain_factory(use_async=False)
def factory():
    # prompt = PromptTemplate(template=template, input_variables=["question"])
    # llm_chain = LLMChain(prompt=prompt, llm=OpenAI(temperature=0), verbose=True)
    index_name = "qa-chatbot"

    return get_agent(index_name)
