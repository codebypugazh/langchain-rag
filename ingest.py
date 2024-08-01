import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX = os.getenv("PINECONE_INDEX")

loader = TextLoader("kb.txt")
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
docs = loader.load_and_split(text_splitter=text_splitter)

embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)
vectorstore = PineconeVectorStore(
    pinecone_api_key=PINECONE_API_KEY,
    embedding=embeddings,
    index_name=PINECONE_INDEX,
    namespace="global_warming"
)

vectorstore.add_documents(docs)

print("done")
