from langchain_community.document_loaders import YoutubeLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.embeddings import HuggingFaceEmbeddings

# Load environment variables from .env file
load_dotenv()

embedding = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)

def create_vector_db_from_youtube_url(video_url:str)-> FAISS:
    loader =YoutubeLoader.from_youtube_url(video_url)
    transcript=loader.load()

    text_splitter=RecursiveCharacterTextSplitter(chunk_size =1000,chunk_overlap=1000)
    docs=text_splitter.split_documents(transcript)

    db=FAISS.from_doc


