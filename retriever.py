import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from langchain.chains import RetrievalQA

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX = os.getenv("PINECONE_INDEX")

llm = ChatOpenAI(
    openai_api_key=OPENAI_API_KEY,
    model_name="gpt-3.5-turbo",
    temperature=0.01,
    max_tokens=500
)

embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)

vectorstore = PineconeVectorStore(
    pinecone_api_key=PINECONE_API_KEY,
    embedding=embeddings,
    index_name=PINECONE_INDEX,
    namespace="global_warming"
)

qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever()
)

answer = qa.invoke("what is the document about?")

print(answer)
