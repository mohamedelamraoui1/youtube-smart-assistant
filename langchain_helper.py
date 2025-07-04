"""
Enhanced LangChain Helper for YouTube Video Analysis
=================================================
This module provides functionality to process YouTube videos and answer questions
about their content using LangChain, FAISS vector database, and Groq LLM.

Dependencies:
- langchain_community: For YouTube video loading
- langchain: For text processing and LLM chains
- langchain_groq: For Groq LLM integration
- faiss: For vector similarity search
- huggingface: For text embeddings
"""

from langchain_community.document_loaders import YoutubeLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.embeddings import HuggingFaceEmbeddings
from dotenv import load_dotenv
import os
import logging

# Configure logging for better debugging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables from .env file
# This should contain your GROQ_API_KEY
load_dotenv()

# Initialize the embedding model from HuggingFace
# all-MiniLM-L6-v2 is a lightweight, fast model good for semantic similarity
embedding = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"},  # Use CPU for compatibility
    # Don't normalize embeddings to preserve original vector magnitudes
    # This keeps the raw embedding values which can be useful for certain similarity calculations
    encode_kwargs={"normalize_embeddings": False},
)


def create_vector_db_from_youtube_url(video_url: str) -> FAISS:
    """
    Creates a FAISS vector database from a YouTube video transcript.

    This function:
    1. Loads the YouTube video transcript
    2. Splits the transcript into manageable chunks
    3. Creates embeddings for each chunk
    4. Stores embeddings in a FAISS vector database for fast similarity search

    Args:
        video_url (str): The YouTube video URL to process

    Returns:
        FAISS: A vector database containing the video transcript chunks

    Raises:
        Exception: If video loading fails or URL is invalid
    """
    try:
        logger.info(f"Loading YouTube video: {video_url}")

        # Load YouTube video transcript using LangChain's YoutubeLoader
        loader = YoutubeLoader.from_youtube_url(video_url)
        transcript = loader.load()

        if not transcript:
            raise ValueError("No transcript found for this video")

        # Split transcript into chunks for better processing
        # chunk_size=1000: Each chunk will be ~1000 characters
        # chunk_overlap=100: 100 characters overlap between chunks to maintain context
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100,  # Fixed: was 1000, should be smaller for better context
            length_function=len,
            separators=["\n\n", "\n", " ", ""],
        )

        # Split the transcript into smaller chunks
        docs = text_splitter.split_documents(transcript)

        # Create FAISS vector database from document chunks
        # This converts text chunks into numerical vectors for similarity search
        db = FAISS.from_documents(docs, embedding)

        logger.info(f"✅ Vector DB created successfully with {len(docs)} chunks")
        return db

    except Exception as e:
        logger.error(f"Error creating vector DB: {str(e)}")
        raise Exception(f"Failed to process YouTube video: {str(e)}")


def get_response_from_query(db: FAISS, query: str, k: int = 4) -> str:
    """
    Generates an answer to a user query using the vector database and Groq LLM.

    This function implements Retrieval-Augmented Generation (RAG):
    1. Searches for relevant transcript chunks using vector similarity
    2. Combines relevant chunks as context
    3. Uses Groq LLM to generate an answer based on the context

    Args:
        db (FAISS): The vector database containing video transcript chunks
        query (str): The user's question about the video
        k (int): Number of most similar chunks to retrieve (default: 4)

    Returns:
        str: Generated answer based on the video transcript

    Raises:
        Exception: If API key is missing or LLM call fails
    """
    try:
        logger.info(f"Processing query: {query}")

        # Step 1: Retrieve top-k most similar document chunks
        # This finds the most relevant parts of the transcript for the question
        docs = db.similarity_search(query, k=k)

        if not docs:
            return "I couldn't find relevant information in the video transcript."

        # Step 2: Combine retrieved chunks into context
        # Join all relevant chunks to provide comprehensive context to the LLM
        docs_page_content = "\n\n".join([d.page_content for d in docs])

        # Step 3: Initialize Groq LLM
        # Groq provides fast inference for large language models
        groq_api_key = os.getenv("GROQ_API_KEY")
        if not groq_api_key:
            raise ValueError("GROQ_API_KEY not found in environment variables")

        llm = ChatGroq(
            groq_api_key=groq_api_key,
            model="llama3-70b-8192",  # High-quality model for better responses
            temperature=0.3,  # Lower temperature for more focused answers
            max_tokens=1000,  # Limit response length
            timeout=30,  # 30 second timeout
        )

        # Step 4: Create structured prompt template
        # This guides the LLM to provide accurate, relevant answers
        prompt = PromptTemplate(
            input_variables=["question", "docs"],
            template="""You are an expert YouTube video assistant that provides accurate answers based on video transcripts.

CONTEXT FROM VIDEO TRANSCRIPT:
{docs}

USER QUESTION: {question}

INSTRUCTIONS:
- Answer the question using ONLY information from the provided transcript
- Be detailed and comprehensive in your response
- If the transcript doesn't contain enough information, clearly state "I don't have enough information in the transcript to answer this question"
- Provide specific details and examples when available
- Structure your answer clearly with proper paragraphs

ANSWER:""",
        )

        # Step 5: Execute the LLM chain
        # This combines the prompt template with the LLM to generate the response
        chain = LLMChain(llm=llm, prompt=prompt)
        response = chain.run(question=query, docs=docs_page_content)

        # Step 6: Clean and format the response
        # Remove excessive whitespace while preserving paragraph structure
        response = response.strip()
        logger.info("✅ Response generated successfully")

        return response

    except Exception as e:
        logger.error(f"Error generating response: {str(e)}")
        return f"Error generating response: {str(e)}"


def validate_youtube_url(url: str) -> bool:
    """
    Validates if the provided URL is a valid YouTube URL.

    Args:
        url (str): URL to validate

    Returns:
        bool: True if valid YouTube URL, False otherwise
    """
    youtube_domains = ["youtube.com", "youtu.be", "www.youtube.com", "m.youtube.com"]
    return any(domain in url.lower() for domain in youtube_domains)


def get_video_info(video_url: str) -> dict:
    """
    Extracts basic information about a YouTube video.

    Args:
        video_url (str): YouTube video URL

    Returns:
        dict: Video information including title, length, etc.
    """
    try:
        loader = YoutubeLoader.from_youtube_url(video_url)
        # This is a simplified version - you might want to extend this
        # to extract more metadata from the video
        return {"status": "valid", "message": "Video accessible"}
    except Exception as e:
        return {"status": "error", "message": str(e)}
