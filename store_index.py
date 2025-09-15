# Loading Important Libraries
from dotenv import load_dotenv
import os
from src.helper import load_pdf_files, filter_to_minimal_docs, text_split, download_embeddings
from pinecone import Pinecone
from pinecone import ServerlessSpec
from langchain_pinecone import PineconeVectorStore


# API Keys
load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

os.environ['PINECONE_API_KEY'] = PINECONE_API_KEY
os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY


# Loading Dataset
data_path = 'Dataset/'
extracted_data = load_pdf_files(data_path)

# Filtering Dataset
filter_data = filter_to_minimal_docs(extracted_data)

# Chunking data
text_chunks = text_split(filter_data)

# Downloading embedding model
embeddings = download_embeddings()


pinecine_api_key = PINECONE_API_KEY
pc = Pinecone(api_key=pinecine_api_key)

index_name = "medical-chatbot"

if not pc.has_index(index_name):
    pc.create_index(
        name=index_name,
        dimension = 384,
        metric = 'cosine',
        spec = ServerlessSpec(cloud='aws', region= 'us-east-1')
    )
    
index = pc.Index(index_name)
doc_search = PineconeVectorStore.from_documents(
    documents = text_chunks,
    embedding = embeddings,
    index_name = index_name
)