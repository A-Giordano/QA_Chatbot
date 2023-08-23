import chainlit as cl
from chatbot_agent import get_agent, get_memory_agent, get_conversation_qa
import uuid


@cl.on_chat_start
def init():
    # prompt = PromptTemplate(template=template, input_variables=["question"])
    # llm_chain = LLMChain(prompt=prompt, llm=OpenAI(temperature=0), verbose=True)
    index_name = "qa-chatbot"
    chain = get_conversation_qa(index_name)
    # chain = get_memory_agent(index_name)
    cl.user_session.set("chain", chain)


@cl.on_message
async def main(message):
    chain = cl.user_session.get("chain")  # type: RetrievalQAWithSourcesChain
    # cb = cl.AsyncLangchainCallbackHandler(
    #     stream_final_answer=True, answer_prefix_tokens=["FINAL", "ANSWER"]
    # )
    # cb.answer_reached = True
    res = chain.run(message)
    # print(res)

    # answer = res["answer"]
    await cl.Message(content=res).send()

