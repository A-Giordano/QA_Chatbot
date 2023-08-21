from functools import partial

from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone

from text_extractor import TextExtractor
import tiktoken
import pinecone


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
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0, streaming=True)
    return RetrievalQA.from_chain_type(llm, retriever=vectorstore.as_retriever())
