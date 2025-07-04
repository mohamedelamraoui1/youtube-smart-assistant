# 🎥 YouTube AI Assistant

A powerful AI-powered application that allows you to analyze YouTube videos and ask intelligent questions about their content. Built with Streamlit, LangChain, and Groq LLM for an enhanced user experience.

## ✨ Features

- **🎯 Smart Video Analysis**: Extract and analyze YouTube video transcripts automatically
- **💬 Interactive Q&A**: Ask questions about video content and get intelligent responses
- **🔍 Semantic Search**: Advanced vector-based search through video content
- **🎨 Modern UI**: Clean, responsive interface built with Streamlit
- **⚡ Fast Processing**: Optimized with FAISS vector database for quick responses
- **🔒 Secure**: Environment-based API key management

## 🛠️ Technology Stack

- **Frontend**: Streamlit
- **LLM**: Groq (Llama models)
- **Vector Database**: FAISS
- **Embeddings**: HuggingFace Sentence Transformers
- **Framework**: LangChain
- **Language**: Python 3.8+

## 📋 Prerequisites

- Python 3.8 or higher
- Groq API key (free tier available)
- Internet connection for YouTube video access

## 🚀 Quick Start

### 1. Clone the Repository
```bash
git clone <your-repo-url>
cd youtube_assistant
```

### 2. Create Virtual Environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Set Up Environment Variables
Create a `.env` file in the project root:
```env
GROQ_API_KEY=your_groq_api_key_here
```

**Get your Groq API key:**
1. Visit [Groq Console](https://console.groq.com/)
2. Sign up for a free account
3. Navigate to API Keys section
4. Create a new API key
5. Copy and paste it into your `.env` file

### 5. Run the Application
```bash
streamlit run main.py
```

The application will open in your default browser at `http://localhost:8501`

## 📱 How to Use

1. **Enter YouTube URL**: Paste any YouTube video URL in the input field
2. **Process Video**: Click "Process Video" to analyze the transcript
3. **Ask Questions**: Once processed, ask any questions about the video content
4. **Get Answers**: Receive AI-generated responses based on the video content

### Supported YouTube URL Formats
- `https://www.youtube.com/watch?v=VIDEO_ID`
- `https://youtu.be/VIDEO_ID`
- `https://youtube.com/watch?v=VIDEO_ID`

## 🎯 Example Use Cases

- **📚 Educational**: Analyze lecture videos and ask study questions
- **📰 News**: Get summaries and insights from news videos
- **🎬 Entertainment**: Understand complex movie explanations or reviews
- **💼 Business**: Extract key points from webinars and presentations
- **🔬 Research**: Analyze documentary content and interviews

## 📁 Project Structure

```
youtube_assistant/
├── main.py                 # Streamlit web application
├── langchain_helper.py     # Core AI processing logic
├── requirements.txt        # Python dependencies
├── .env                   # Environment variables (create this)
├── .gitignore            # Git ignore rules
└── README.md             # Project documentation
```

## ⚙️ Configuration

### Environment Variables
- `GROQ_API_KEY`: Your Groq API key for LLM access

### Customization Options
- **Chunk Size**: Modify text splitting parameters in `langchain_helper.py`
- **Model Selection**: Change the Groq model in the LLM initialization
- **Embedding Model**: Switch HuggingFace embedding models for different languages

## 🔧 Troubleshooting

### Common Issues

**1. "No transcript found for this video"**
- Some videos don't have transcripts available
- Try videos with closed captions enabled

**2. "Invalid API key"**
- Verify your Groq API key in the `.env` file
- Check for extra spaces or invalid characters

**3. "Module not found" errors**
- Ensure all dependencies are installed: `pip install -r requirements.txt`
- Activate your virtual environment

**4. Slow processing**
- Large videos may take longer to process
- Consider using shorter videos for testing

### Performance Tips
- Use videos under 2 hours for optimal performance
- Clear browser cache if the interface becomes unresponsive
- Restart the application if memory usage becomes high

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Groq** for providing fast LLM inference
- **LangChain** for the powerful AI framework
- **Streamlit** for the amazing web framework
- **HuggingFace** for embedding models
- **Meta** for the FAISS vector database

## 📞 Support

If you encounter any issues or have questions:
1. Check the troubleshooting section above
2. Search existing issues in the repository
3. Create a new issue with detailed information
4. Join our community discussions

---

**Happy analyzing! 🎉**

*Built with ❤️ for the AI community*