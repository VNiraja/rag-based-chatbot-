# LLM Hackathon Problem Statement Generator

## Overview
**llm1.py** is an interactive web application built with **Streamlit** that uses **LangChain** and the **Groq API** to generate creative and tailored hackathon problem statements. Users can customize their problem statement requests based on domain, difficulty level, and problem type.

## Features
- **Interactive Domain Selection**: Choose from Healthcare, Education, Finance, Environment, or Entertainment
- **Difficulty Levels**: Select between Beginner, Intermediate, or Advanced
- **Problem Types**: Get Research, Implementation, or Optimization-focused problems
- **AI-Powered Generation**: Uses Groq's Llama 3.3 70B model with temperature tuning for creative output
- **Streamlit UI**: Simple, user-friendly web interface

## Prerequisites
- Python 3.8+
- Virtual Environment (venv)
- Required API Keys:
  - Groq API Key (for LLM access)

## Installation

### 1. Create and Activate Virtual Environment
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Configure Environment Variables
Create a `.env` file in the project root:
```
GROQ_API_KEY=your_groq_api_key_here
```

## Running the Application
```bash
cd LLMs
streamlit run llm1.py
```

The app will open at `http://localhost:8501`

## Project Structure
```
langchain_models/
├── LLMs/
│   ├── llm1.py          # Main Streamlit application
│   ├── llm.py           # Groq LLM implementation
│   ├── llm3.py          # Alternative implementations
│   ├── huggingface.py   # HuggingFace model support
│   └── hug.py           
├── ChatModels/          # Chat model implementations
├── EmbeddedModels/      # Embedding models for semantic search
├── requirements.txt     # Python dependencies
├── test.py              # Testing utilities
├── .env                 # API credentials (not in repo)
└── venv/                # Virtual environment
```

## Dependencies
- **langchain**: LLM framework for building AI applications
- **langchain-groq**: Groq provider integration
- **streamlit**: Web framework for data apps
- **python-dotenv**: Environment variable management
- **openai, anthropic, google-generativeai**: Multi-provider LLM support

See `requirements.txt` for complete list.

## How It Works

1. **User Selects Parameters**:
   - Domain (Healthcare, Education, Finance, Environment, Entertainment)
   - Difficulty Level (Beginner, Intermediate, Advanced)
   - Problem Type (Research, Implementation, Optimization)

2. **Template Formatting**:
   - Uses LangChain's `PromptTemplate` to format the user inputs
   - Template: `"suggest me some good {difficulty} problem statements to make a hackathon project on {idea} for {level} level participants"`

3. **LLM Invocation**:
   - Sends formatted prompt to Groq's Llama 3.3 70B model
   - Temperature: 1.2 (more creative responses)
   - Max tokens: 1000 (for detailed responses)

4. **Display Results**:
   - Displays generated problem statements in the Streamlit UI

## Configuration

### LLM Settings
- **Model**: `llama-3.3-70b-versatile` (via Groq)
- **Temperature**: 1.2 (higher = more creative, lower = more focused)
- **Max Tokens**: 1000 (maximum response length)

Modify these in `llm1.py` lines 9-11 to adjust behavior.

## Troubleshooting

### Error: "File does not exist"
Ensure you're running the command from the correct directory:
```bash
cd LLMs
streamlit run llm1.py
```

### Missing API Key
Verify your `.env` file contains `GROQ_API_KEY` and is in the project root.

### Slow Responses
Check your internet connection and Groq API status. High token limits may increase response time.

## Future Enhancements
- Add more domains and customization options
- Support for multiple LLM providers
- Caching for frequently requested problems
- Export functionality (PDF, JSON)
- Problem difficulty scoring

## License
MIT

## Contact
For questions or issues, refer to the project documentation or LangChain/Groq docs.
