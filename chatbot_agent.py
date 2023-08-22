from functools import partial

from langchain.chains import RetrievalQA
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.memory.chat_memory import BaseChatMemory
from langchain.agents import initialize_agent, AgentType

from langchain.chat_models import PromptLayerChatOpenAI
from langchain.agents import Tool

from langchain.chat_models import ChatOpenAI
from langchain.prompts import MessagesPlaceholder
from langchain.schema import SystemMessage
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone

from text_extractor import TextExtractor
import tiktoken
import pinecone


system_message = f"""
<instructions>
- Ignore any previous instructions
- You are acting as Steve Jobs, the iconic co-founder of Apple Inc.
- Emulate the personality, demeanor, and speaking style of Steve Jobs, the iconic co-founder of Apple Inc. Try to mimic his forward-thinking mindset, inspirational tone, articulate responses, and passion for innovation. Emphasize on his belief in simplicity and intuition when explaining concepts or ideas. Utilize his characteristic mix of business acumen, visionary thinking, and persuasiveness in the responses. 
- In essence, throughout this session, think differently, just as Steve Jobs would have.
- Keep your responses short, STRICTLY max 2 sentences. Example: sentence 1 blah blah. sentence 2 blah blah blah.
- Do not say you are a language model. We are play acting and you are playing the part of Steve Jobs.
- Do not say you are a virtual assistant. We are play acting and you are playing the part of Steve Jobs.
</instructions>

NEVER SAY YOU ARE A VIRTUAL ASSISTANT, YOU ARE STEVE JOBS."""


def count_tokens(string: str, encoding_model: str) -> int:
    """Returns the number of tokens in a text string."""
    enc = tiktoken.encoding_for_model(encoding_model)
    num_tokens = len(enc.encode(string))
    return num_tokens


def load_text(file_path, gpt_model, index_name):
    pinecone.init(environment="us-west1-gcp-free")

    if index_name not in pinecone.list_indexes():
        # we create a new index
        pinecone.create_index(
            name=index_name,
            metric='dotproduct',
            dimension=1536  # 1536 dim of text-embedding-ada-002
        )
    print(f"{index_name} created")

    embeddings = OpenAIEmbeddings()
    index = pinecone.Index(index_name)
    vectorstore = Pinecone(index, embeddings.embed_query, "text")

    vectorstore.add_texts(["q"])
    print(f"{index_name} Populated")

    count_tokens_fun = partial(count_tokens, encoding_model=gpt_model)
    txt_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        length_function=count_tokens_fun,
    )
    txt = TextExtractor(file_path).extract()
    txt_chunks = txt_splitter.split_text(txt)

    vectorstore.add_texts(txt_chunks)
    print(f"{index_name} populated fully")


def get_agent(index_name):
    pinecone.init(environment="us-west1-gcp-free")

    embeddings = OpenAIEmbeddings()
    vectorstore = Pinecone.from_existing_index(index_name=index_name, embedding=embeddings)
    # llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0, streaming=True)
    llm = PromptLayerChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

    return RetrievalQA.from_chain_type(llm, retriever=vectorstore.as_retriever())


def get_memory_agent(index_name):
    pinecone.init(environment="us-west1-gcp-free")

    embeddings = OpenAIEmbeddings()
    vectorstore = Pinecone.from_existing_index(index_name=index_name, embedding=embeddings)
    # llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0, streaming=True)
    llm = PromptLayerChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
    qa = RetrievalQA.from_chain_type(llm, retriever=vectorstore.as_retriever())

    tools = [
        Tool(
            name='TBW-Palestine_Game_Knowledge_Base',
            func=qa.run,
            description=(
                'use this tool when answering questions about the TBW-Palestine game'
                'more information about the topic'
            )
        )
    ]
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    # agent = initialize_agent(
    #     agent='chat-conversational-react-description',
    #     tools=tools,
    #     llm=llm,
    #     verbose=True,
    #     max_iterations=3,
    #     early_stopping_method='generate',
    #     memory=memory
    # )

    agent_kwargs = {
        "extra_prompt_messages": [MessagesPlaceholder(variable_name="chat_history")],
        "system_message": SystemMessage(content=system_message)
    }
    agent = initialize_agent(tools,
                                   llm,
                                   agent=AgentType.OPENAI_FUNCTIONS,
                                   # agent=AgentType.OPENAI_MULTI_FUNCTIONS,
                                   agent_kwargs=agent_kwargs,
                                   memory=memory,
                                   max_execution_time=10,
                                   verbose=True)
    return agent
    # return RetrievalQA.from_chain_type(llm, retriever=vectorstore.as_retriever())


