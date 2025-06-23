# AI Blog Generator

An AI-powered blog content generation tool that leverages Google's Gemini AI and various web search strategies to create SEO-optimized blog posts.

## Features

- 🤖 AI-powered blog content generation
- 🔍 Multiple search engine integrations (SearX, DuckDuckGo, Startpage, Bing, Yandex)
- 📱 Web interface using Streamlit
- 🚀 REST API using FastAPI
- 📊 SEO-optimized content generation
- 🌐 No paid API dependencies for web search

## Prerequisites

- Python 3.8+
- Google AI API Key (for Gemini)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/karimzade/blog-agent.git
cd blog-agent
```

2. Set up Python virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate  # On Linux/Mac
# or
.\venv\Scripts\activate  # On Windows
```

3. Upgrade pip and install dependencies:
```bash
python3 -m pip install --upgrade pip
pip install -r requirements.txt
```

4. Create a `.env` file in the project root:
```
GOOGLE_API_KEY=your_google_api_key_here
```

## Usage

### Web Interface

Run the Streamlit app:
```bash
streamlit run app.py
```

Visit `http://localhost:8501` in your browser.

### API

Run the FastAPI server:
```bash
uvicorn api:app --reload
```

The API will be available at `http://localhost:8000`

API endpoints:
- POST `/generate-blog`: Generate a blog post
- GET `/health`: Health check endpoint

## Project Structure

```
blog-agent/
├── app.py         # Main application & Streamlit interface
├── api.py         # FastAPI REST API
├── .env           # Environment variables
└── requirements.txt
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.
