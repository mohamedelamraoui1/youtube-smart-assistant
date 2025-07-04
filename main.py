"""
Enhanced YouTube Assistant - Streamlit Interface
==============================================
A user-friendly interface for analyzing YouTube videos and asking questions
about their content using advanced AI techniques.

Features:
- Clean, intuitive UI with better user experience
- Input validation and error handling
- Progress indicators and status messages
- Responsive design with better formatting
- Session state management for better performance
"""

import streamlit as st
import langchain_helper as lch
import textwrap
import time
from urllib.parse import urlparse

# Page configuration - must be the first Streamlit command
st.set_page_config(
    page_title="YouTube AI Assistant",
    page_icon="🎥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #ff6b6b;
        text-align: center;
        margin-bottom: 2rem;
    }
    .stAlert > div {
        padding: 1rem;
        border-radius: 10px;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .info-box {
        background-color: #e7f3ff;
        border: 1px solid #bee5eb;
        color: #0c5460;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state variables
if 'db' not in st.session_state:
    st.session_state.db = None
if 'current_video_url' not in st.session_state:
    st.session_state.current_video_url = ""
if 'processing_complete' not in st.session_state:
    st.session_state.processing_complete = False

# Main title and description
st.markdown('<h1 class="main-header">🎥 YouTube AI Assistant</h1>', 
            unsafe_allow_html=True)

st.markdown("""
<div class="info-box">
    <strong>How it works:</strong><br>
    1. 📹 Paste a YouTube video URL<br>
    2. ❓ Ask any question about the video content<br>
    3. 🤖 Get AI-powered answers based on the video transcript
</div>
""", unsafe_allow_html=True)

# Create two columns for better layout
col1, col2 = st.columns([1, 2])

with col1:
    st.markdown("### 🎯 Input Section")
    
    # Video URL input with validation
    st.markdown("#### 📹 YouTube Video URL")
    youtube_url = st.text_input(
        label="Enter YouTube URL:",
        placeholder="https://www.youtube.com/watch?v=...",
        help="Paste the full YouTube video URL here",
        label_visibility="collapsed"
    )
    
    # URL validation
    if youtube_url:
        if not lch.validate_youtube_url(youtube_url):
            st.error("❌ Please enter a valid YouTube URL")
        else:
            st.success("✅ Valid YouTube URL detected")
    
    # Question input
    st.markdown("#### ❓ Your Question")
    query = st.text_area(
        label="Ask about the video:",
        placeholder="What is the main topic discussed in this video?",
        height=100,
        help="Ask any question about the video content",
        label_visibility="collapsed"
    )
    
    # Advanced options in an expander
    with st.expander("⚙️ Advanced Options"):
        num_chunks = st.slider(
            "Number of transcript chunks to analyze:",
            min_value=2,
            max_value=10,
            value=4,
            help="More chunks = more comprehensive but slower analysis"
        )
        
        show_sources = st.checkbox(
            "Show source transcript chunks",
            help="Display the transcript portions used to generate the answer"
        )
    
    # Submit button with better styling
    submit_button = st.button(
        "🚀 Analyze Video",
        type="primary",
        use_container_width=True,
        disabled=not (youtube_url and query and lch.validate_youtube_url(youtube_url))
    )

with col2:
    st.markdown("### 🎯 Results Section")
    
    # Process the request when submit is clicked
    if submit_button and youtube_url and query:
        try:
            # Check if we need to process a new video
            if (st.session_state.current_video_url != youtube_url or 
                st.session_state.db is None):
                
                # Show processing status
                with st.spinner("🔄 Processing video transcript..."):
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    # Update progress
                    status_text.text("Loading video transcript...")
                    progress_bar.progress(25)
                    
                    # Create vector database
                    st.session_state.db = lch.create_vector_db_from_youtube_url(youtube_url)
                    st.session_state.current_video_url = youtube_url
                    
                    progress_bar.progress(75)
                    status_text.text("Creating vector database...")
                    
                    progress_bar.progress(100)
                    status_text.text("✅ Video processed successfully!")
                    
                    # Clear progress indicators after a short delay
                    time.sleep(1)
                    progress_bar.empty()
                    status_text.empty()
                
                st.markdown("""
                <div class="success-box">
                    <strong>✅ Video processed successfully!</strong><br>
                    The video transcript has been analyzed and indexed for questioning.
                </div>
                """, unsafe_allow_html=True)
            
            # Generate response
            with st.spinner("🤖 Generating AI response..."):
                response = lch.get_response_from_query(
                    st.session_state.db, 
                    query, 
                    k=num_chunks
                )
            
            # Display the answer
            st.markdown("#### 💡 Answer:")
            
            # Format the response nicely
            if response:
                # Use markdown for better formatting
                st.markdown(f"""
                <div style="background-color: #f8f9fa; padding: 1.5rem; 
                           border-radius: 10px; border-left: 4px solid #007bff;">
                    {response}
                </div>
                """, unsafe_allow_html=True)
                
                # Show source chunks if requested
                if show_sources:
                    with st.expander("📄 Source Transcript Chunks"):
                        docs = st.session_state.db.similarity_search(query, k=num_chunks)
                        for i, doc in enumerate(docs):
                            st.markdown(f"**Chunk {i+1}:**")
                            st.text(textwrap.fill(doc.page_content, width=80))
                            st.markdown("---")
            else:
                st.error("❌ Could not generate a response. Please try again.")
                
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
            st.markdown("""
            **Troubleshooting tips:**
            - Make sure the YouTube URL is valid and accessible
            - Check if the video has captions/transcript available
            - Verify your GROQ_API_KEY is set correctly
            """)
    
    # Show helpful information when no query is made
    elif not submit_button:
        st.markdown("""
        ### 🎯 Ready to analyze!
        
        **Tips for better results:**
        - 📝 Use specific questions about the video content
        - 🎯 Ask about main topics, key points, or specific details
        - 🔍 Try questions like:
          - "What are the main points discussed?"
          - "What examples are given?"
          - "What is the conclusion?"
        
        **Requirements:**
        - ✅ Valid YouTube URL
        - ✅ Clear question about the video
        - ✅ GROQ_API_KEY in your environment
        """)

# Sidebar with additional information
with st.sidebar:
    st.markdown("### 📊 Session Info")
    
    if st.session_state.db is not None:
        st.success("✅ Video database ready")
        if st.session_state.current_video_url:
            st.info(f"📹 Current video: {st.session_state.current_video_url[:50]}...")
    else:
        st.info("⏳ No video processed yet")
    
    # Clear session button
    if st.button("🗑️ Clear Session", use_container_width=True):
        st.session_state.db = None
        st.session_state.current_video_url = ""
        st.session_state.processing_complete = False
        st.rerun()
    
    st.markdown("---")
    st.markdown("### ℹ️ About")
    st.markdown("""
    This app uses:
    - **LangChain** for document processing
    - **FAISS** for vector similarity search
    - **Groq** for fast AI inference
    - **HuggingFace** for text embeddings
    
    Made with ❤️ and Streamlit
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 1rem;">
    <small>💡 Tip: For best results, use videos with clear audio and available transcripts</small>
</div>
""", unsafe_allow_html=True)