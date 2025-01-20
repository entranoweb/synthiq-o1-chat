# O1 Chat

A powerful chat interface for Azure OpenAI's O1 model with advanced features including:
- Multiple model support (O1 and O1 Mini)
- Chat history management
- Vision analysis
- Structured output
- Custom tools/functions
- Web search integration

## Setup

1. Clone this repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Create a `.env` file with your API keys:
```env
AZURE_OPENAI_O1_ENDPOINT=your_endpoint
AZURE_OPENAI_O1_API_KEY=your_api_key
AZURE_OPENAI_O1_DEPLOYMENT=your_deployment
AZURE_OPENAI_O1_MINI_ENDPOINT=your_mini_endpoint
AZURE_OPENAI_O1_MINI_API_KEY=your_mini_api_key
AZURE_OPENAI_O1_MINI_DEPLOYMENT=your_mini_deployment
PERPLEXITY_API_KEY=your_perplexity_key
```

4. Run the app:
```bash
streamlit run o1-chat.py
```

## Features

- **Multiple Models**: Switch between O1 and O1 Mini models
- **Chat History**: Save and load chat sessions
- **Vision Analysis**: Upload and analyze images
- **Structured Output**: Get responses in JSON format
- **Custom Tools**: Add Python functions for the model to use
- **Web Search**: Integrated Perplexity search
- **Copy Functionality**: Easy copying of messages
- **Modern UI**: Dark theme with clean design

## Deployment

This app can be deployed to Streamlit Community Cloud:

1. Push code to GitHub
2. Connect your GitHub repo to Streamlit Cloud
3. Add your environment variables in Streamlit Cloud
4. Deploy! 