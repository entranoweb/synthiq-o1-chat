import os
import time
from openai import AzureOpenAI
import streamlit as st
from dotenv import load_dotenv
from typing import Dict, List, Optional
import tempfile
import sounddevice as sd
import wave
import numpy as np
import requests
import json
import pyperclip
from tinydb import TinyDB, Query
from datetime import datetime
import base64

load_dotenv()

class ChatSystem:
    def __init__(self):
        # Initialize TinyDB
        self.db = TinyDB('chat_database.json')
        self.chats_table = self.db.table('chats')
        self.messages_table = self.db.table('messages')
        
        # Initialize credentials for both models
        self.o1_endpoint = os.getenv("AZURE_OPENAI_O1_ENDPOINT")
        self.o1_api_key = os.getenv("AZURE_OPENAI_O1_API_KEY")
        self.o1_deployment = os.getenv("AZURE_OPENAI_O1_DEPLOYMENT")
        
        self.o1_mini_endpoint = os.getenv("AZURE_OPENAI_O1_MINI_ENDPOINT")
        self.o1_mini_api_key = os.getenv("AZURE_OPENAI_O1_MINI_API_KEY")
        self.o1_mini_deployment = os.getenv("AZURE_OPENAI_O1_MINI_DEPLOYMENT")
        
        # Get API versions
        self.api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")
        
        # Default to o1 settings
        self.set_credentials("o1")
        
        # Initialize feature flags
        self.vision_enabled = False
        self.structured_output_enabled = False
        self.current_schema = None
        self.function_calling_enabled = False
        self.available_functions = {}
        
    def set_credentials(self, model_choice: str):
        """Set the appropriate credentials based on model choice"""
        if model_choice == "o1":
            self.client = AzureOpenAI(
                api_key=self.o1_api_key,
                api_version=self.api_version,
                azure_endpoint=self.o1_endpoint
            )
            self.deployment = self.o1_deployment
        else:  # o1_mini
            self.client = AzureOpenAI(
                api_key=self.o1_mini_api_key,
                api_version=self.api_version,
                azure_endpoint=self.o1_mini_endpoint
            )
            self.deployment = self.o1_mini_deployment
        
        self.model_name = model_choice

    def save_audio_to_wav(self, audio_data, sample_rate, filename):
        """Save audio data to WAV file using wave module"""
        with wave.open(filename, 'wb') as wav_file:
            wav_file.setnchannels(1)  # Mono
            wav_file.setsampwidth(2)  # 2 bytes per sample (16-bit)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(audio_data.tobytes())

    @staticmethod
    def init_session_state():
        defaults = {
            "messages": [],
            "system_prompt": "You are a helpful assistant.",
            "model_name": os.getenv("AZURE_OPENAI_O1_DEPLOYMENT"),
            "thinking": False,
            "token_usage": 0,
            "perplexity_api_key": os.getenv("PERPLEXITY_API_KEY"),
            "current_chat": "New Chat",
            "chat_sessions": {}
        }
        
        # Initialize session state
        for key, value in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = value
        
        # Load all saved chat sessions from TinyDB
        try:
            chat_system = ChatSystem()
            chats = chat_system.chats_table.all()
            for chat in chats:
                st.session_state.chat_sessions[chat["name"]] = chat
        except Exception as e:
            st.error(f"Error loading chat sessions: {str(e)}")

    def save_chat_session(self, session_name: str):
        """Save current chat session to TinyDB"""
        try:
            Chat = Query()
            
            # Prepare chat data
            chat_data = {
                "name": session_name,
                "system_prompt": st.session_state.system_prompt,
                "model_name": st.session_state.model_name,
                "token_usage": st.session_state.token_usage,
                "timestamp": datetime.now().isoformat()
            }
            
            # Save or update chat
            chat_id = self.chats_table.upsert(chat_data, Chat.name == session_name)[0]
            
            # Save messages
            Message = Query()
            self.messages_table.remove(Message.chat_id == chat_id)
            for i, msg in enumerate(st.session_state.messages):
                message_data = {
                    "chat_id": chat_id,
                    "index": i,
                    "role": msg["role"],
                    "content": msg["content"],
                    "type": msg.get("type", "normal"),
                    "timestamp": datetime.now().isoformat()
                }
                self.messages_table.insert(message_data)
            
            return True
        except Exception as e:
            st.error(f"Error saving chat session: {str(e)}")
        return False

    def load_chat_session(self, session_name: str):
        """Load a chat session from TinyDB"""
        try:
            Chat = Query()
            Message = Query()
            
            chat_data = self.chats_table.get(Chat.name == session_name)
            if chat_data:
                # Load chat metadata
                st.session_state.system_prompt = chat_data["system_prompt"]
                st.session_state.model_name = chat_data["model_name"]
                st.session_state.token_usage = chat_data["token_usage"]
                
                # Load messages
                messages = self.messages_table.search(Message.chat_id == chat_data.doc_id)
                st.session_state.messages = [{
                    "role": msg["role"],
                    "content": msg["content"],
                    "type": msg.get("type", "normal")
                } for msg in sorted(messages, key=lambda x: x.get("timestamp", ""))]
                
                return True
        except Exception as e:
            st.error(f"Error loading chat session: {str(e)}")
        return False

    def set_output_schema(self, schema: Dict):
        """Set JSON schema for structured output"""
        self.structured_output_enabled = True
        self.current_schema = schema
        
    def create_default_schema(self):
        """Create default schema for structured output"""
        return {
            "type": "object",
            "properties": {
                "response": {
                    "type": "object",
                    "properties": {
                        "content": {"type": "string"},
                        "analysis": {
                            "type": "object",
                            "properties": {
                                "key_points": {"type": "array", "items": {"type": "string"}},
                                "sentiment": {"type": "string", "enum": ["positive", "neutral", "negative"]},
                                "confidence": {"type": "number"}
                            },
                            "required": ["key_points", "sentiment", "confidence"]
                        }
                    },
                    "required": ["content", "analysis"]
                }
            },
            "required": ["response"]
        }

    def register_function(self, func_name: str, func, description: str, parameters: Dict):
        """Register a function for O1 to call"""
        self.available_functions[func_name] = {
            "function": func,
            "description": description,
            "parameters": parameters
        }
        
    def setup_default_functions(self):
        """Setup default available functions"""
        # Example function: Get current time
        self.register_function(
            "get_current_time",
            lambda: {"time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")},
            "Get the current time",
            {
                "type": "object",
                "properties": {},
                "required": []
            }
        )
        
        # Example function: Simple calculator
        self.register_function(
            "calculate",
            lambda x, operation, y: {
                "result": eval(f"{x} {operation} {y}")
            },
            "Perform basic arithmetic",
            {
                "type": "object",
                "properties": {
                    "x": {"type": "number"},
                    "operation": {"type": "string", "enum": ["+", "-", "*", "/"]},
                    "y": {"type": "number"}
                },
                "required": ["x", "operation", "y"]
            }
        )

    def get_function_definitions(self):
        """Get function definitions in Azure O1 format"""
        return [{
            "type": "function",
            "function": {
                "name": name,
                "description": details["description"],
                "parameters": details["parameters"]
            }
        } for name, details in self.available_functions.items()]

    def handle_function_call(self, function_call):
        """Execute a function call from O1"""
        try:
            func_name = function_call["name"]
            if func_name not in self.available_functions:
                raise ValueError(f"Unknown function: {func_name}")
            
            # Parse arguments
            args = json.loads(function_call["arguments"])
            
            # Execute function
            func = self.available_functions[func_name]["function"]
            result = func(**args)
            
            return json.dumps(result)
        except Exception as e:
            return f"Error executing function: {str(e)}"

    def stream_chat_completion(
        self, 
        messages: List[Dict], 
        temperature: float, 
        top_p: float, 
        max_completion_tokens: int, 
        model_name: str
    ) -> str:
        """
        Updated to handle:
        - Vision messages
        - Structured output
        - Function calling
        """
        try:
            # Set credentials based on model
            if model_name == self.o1_deployment:
                self.set_credentials("o1")
            else:
                self.set_credentials("o1_mini")
            
            st.info(f"Using model: {self.deployment}")

            # Convert system role to developer for o1
            formatted_messages = []
            for msg in messages:
                if msg["role"] == "system":
                    formatted_messages.append({"role": "developer", "content": msg["content"]})
                else:
                    formatted_messages.append(msg)
            
            try:
                # Prepare API call parameters
                api_params = {
                    "messages": formatted_messages,
                    "model": self.deployment,
                    "max_completion_tokens": max_completion_tokens,
                }
                
                # Add model-specific parameters
                if model_name == self.o1_deployment:
                    reasoning_effort = st.session_state.get("reasoning_effort", "high")
                    api_params["reasoning_effort"] = reasoning_effort
                    
                    # Add structured output if enabled
                    if self.structured_output_enabled and self.current_schema:
                        api_params["response_format"] = {
                            "type": "json_object",
                            "schema": self.current_schema
                        }
                    
                    # Add function calling if enabled
                    if self.function_calling_enabled and self.available_functions:
                        api_params["tools"] = self.get_function_definitions()
                        api_params["tool_choice"] = "auto"
                else:
                    api_params.update({
                        "temperature": temperature,
                        "top_p": top_p,
                        "max_tokens": max_completion_tokens
                    })
                
                # Make API call
                response = self.client.chat.completions.create(**api_params)
                
                # Handle function calling response
                if (
                    self.function_calling_enabled 
                    and hasattr(response.choices[0].message, 'tool_calls')
                    and response.choices[0].message.tool_calls
                ):
                    tool_calls = response.choices[0].message.tool_calls
                    results = []
                    for tool_call in tool_calls:
                        if tool_call.type == "function":
                            result = self.handle_function_call(tool_call.function)
                            results.append(result)
                    
                    content = "\n".join(results)
                else:
                    content = response.choices[0].message.content
                
                # Update token usage
                if hasattr(response, 'usage'):
                    st.session_state.token_usage += response.usage.total_tokens
                
                # Handle structured output display
                if self.structured_output_enabled:
                    try:
                        json_content = json.loads(content)
                        st.json(json_content)
                    except json.JSONDecodeError:
                        st.error("Failed to parse structured output as JSON")
                        st.markdown(content, unsafe_allow_html=True)
                else:
                    st.markdown(content, unsafe_allow_html=True)
                
                return content
                
            except Exception as e:
                st.error(f"API call failed: {str(e)}")
                return "I apologize, but I encountered an error while trying to respond. Please try again."
            
        except Exception as e:
            st.error(f"Critical error: {str(e)}")
            st.error("Full error details for debugging:")
            st.error(f"Model: {self.deployment}")
            st.error(f"API Version: {self.api_version}")
            return "A critical error occurred. Please check your configuration and try again."

    def process_audio(self, audio_file) -> Optional[str]:
        """Process uploaded audio file using Azure OpenAI's Whisper API"""
        try:
            # Read the audio file content
            audio_data = audio_file.read()
            
            # Make API call to Azure OpenAI's Whisper
            response = self.client.audio.transcriptions.create(
                file=audio_data,
                model="whisper-1"
            )
            
            # Get transcription
            transcription = response.text.strip()
            return transcription
            
        except Exception as e:
            st.error(f"Audio processing error: {str(e)}")
            return None

    def record_audio(self, duration=5, sample_rate=16000):
        """Record audio from microphone for a given duration."""
        try:
            st.write("Recording... Please speak now.")
            audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='int16')
            sd.wait()
            st.write("Recording complete.")
            return audio, sample_rate
        except Exception as e:
            st.error(f"Audio recording error: {str(e)}")
            return None, None

    def transcribe_audio_data(self, audio, sample_rate) -> Optional[str]:
        """Transcribe recorded audio using Azure OpenAI's Whisper API"""
        if audio is None:
            return None
            
        try:
            # Save the audio data to a temporary WAV file
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp_file:
                self.save_audio_to_wav(audio, sample_rate, tmp_file.name)
                
                # Read the temporary file
                with open(tmp_file.name, "rb") as audio_file:
                    # Make API call to Azure OpenAI's Whisper
                    response = self.client.audio.transcriptions.create(
                        file=audio_file,
                        model="whisper-1"
                    )
                    
                    # Get transcription
                    transcription = response.text.strip()
                    return transcription
                    
        except Exception as e:
            st.error(f"Transcription error: {str(e)}")
            return None

    def query_perplexity(self, prompt: str) -> str:
        api_key = st.session_state.get("perplexity_api_key")
        if not api_key:
            st.warning("No Perplexity API key found. Please set PERPLEXITY_API_KEY in .env.")
            return "No Perplexity API key available."

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        data = {
            "model": "llama-3.1-sonar-small-128k-online",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 4000,
            "temperature": 0.2,
            "top_p": 0.9
        }

        try:
            response = requests.post("https://api.perplexity.ai/chat/completions", headers=headers, json=data)
            res_json = response.json()
            content = res_json['choices'][0]['message']['content']
            return content
        except Exception as e:
            st.error(f"Perplexity API error: {str(e)}")
            return "Error fetching information from Perplexity."

    def save_chat_history(self):
        """Save the current chat history to a file"""
        chat_history = {
            "messages": st.session_state.messages,
            "system_prompt": st.session_state.system_prompt,
            "model_name": st.session_state.model_name,
            "token_usage": st.session_state.token_usage
        }
        with open("chat_history.json", "w") as f:
            json.dump(chat_history, f)

    def process_image(self, image_file):
        """Process uploaded image for vision analysis"""
        try:
            if image_file is not None:
                # Convert image to base64
                base64_image = base64.b64encode(image_file.getvalue()).decode('utf-8')
                return {
                    "type": "image_url",
                    "image_url": f"data:image/jpeg;base64,{base64_image}"
                }
            return None
        except Exception as e:
            st.error(f"Error processing image: {str(e)}")
            return None

    def create_vision_message(self, image_data, prompt="Analyze this image"):
        """Create a message with image content"""
        return {
            "role": "user",
            "content": [
                image_data,
                {"type": "text", "text": prompt}
            ]
        }

    def delete_chat_session(self, session_name: str):
        """Delete a chat session from TinyDB"""
        try:
            Chat = Query()
            Message = Query()
            
            # Get chat data
            chat_data = self.chats_table.get(Chat.name == session_name)
            if chat_data:
                # Delete messages first
                self.messages_table.remove(Message.chat_id == chat_data.doc_id)
                # Then delete chat
                self.chats_table.remove(Chat.name == session_name)
                return True
        except Exception as e:
            st.error(f"Error deleting chat session: {str(e)}")
        return False

def main():
    st.set_page_config(
        page_title="O1 Chat",
        page_icon="🤖",
        layout="wide"
    )

    # Custom CSS for better UI visibility
    st.markdown("""
        <style>
        /* Main app styling */
        .stApp {
            background-color: #1E1E1E;
            color: #FFFFFF;
        }
        
        /* Copy button styling */
        .stButton > button[data-testid="baseButton-secondary"] {
            background-color: transparent !important;
            border: 1px solid #4CAF50 !important;
            color: #4CAF50 !important;
            padding: 0.25rem 0.5rem !important;
            border-radius: 4px;
            transition: all 0.3s ease;
            width: 40px !important;
            height: 40px !important;
            display: flex;
            align-items: center;
            justify-content: center;
        }
        
        .stButton > button[data-testid="baseButton-secondary"]:hover {
            background-color: #4CAF50 !important;
            color: white !important;
        }
        
        /* Message container styling */
        [data-testid="stChatMessage"] {
            background-color: #2D2D2D !important;
            border: 1px solid #333333;
            border-radius: 8px;
            padding: 1rem;
            margin: 0.5rem 0;
            display: flex;
            align-items: flex-start;
        }
        
        /* Markdown content styling */
        .markdown-content {
            flex-grow: 1;
            overflow-x: auto;
            padding-right: 1rem;
        }
        
        .markdown-content pre {
            background-color: #1E1E1E;
            padding: 1rem;
            border-radius: 4px;
            overflow-x: auto;
        }
        
        .markdown-content code {
            background-color: #1E1E1E;
            padding: 0.2rem 0.4rem;
            border-radius: 3px;
        }
        
        /* Sidebar styling */
        [data-testid="stSidebar"] {
            background-color: #252526;
            border-right: 1px solid #333333;
            padding: 1rem;
        }
        
        /* Expander styling */
        .streamlit-expanderHeader {
            background-color: #2B5797 !important;
            color: white !important;
            border-radius: 4px;
            padding: 0.5rem !important;
            margin-bottom: 0.5rem;
        }
        
        /* Button styling */
        .stButton button {
            width: 100%;
            background-color: #2B5797 !important;
            color: white !important;
            border: none !important;
            padding: 0.5rem !important;
            margin-bottom: 0.5rem;
        }
        
        /* Toggle switch styling */
        .stCheckbox {
            background-color: #2D2D2D;
            padding: 0.5rem;
            border-radius: 4px;
            margin-bottom: 0.5rem;
        }
        
        /* Selectbox styling */
        .stSelectbox {
            margin-bottom: 0.5rem;
        }
        
        /* Section headers */
        .section-header {
            background-color: #2B5797;
            color: white;
            padding: 0.5rem;
            border-radius: 4px;
            margin: 1rem 0;
            text-align: center;
        }
        </style>
    """, unsafe_allow_html=True)

    # Initialize chat system
    chat_system = ChatSystem()
    chat_system.setup_default_functions()

    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "token_usage" not in st.session_state:
        st.session_state.token_usage = 0
    if "system_prompt" not in st.session_state:
        st.session_state.system_prompt = "You are a helpful assistant who is kind, factual, and concise."
    if "model_name" not in st.session_state:
        st.session_state.model_name = chat_system.o1_deployment
    if "current_chat" not in st.session_state:
        st.session_state.current_chat = "New Chat"
    if "chat_sessions" not in st.session_state:
        st.session_state.chat_sessions = {}

    # Sidebar
    with st.sidebar:
        st.markdown('<div class="section-header">💬 Chat Control</div>', unsafe_allow_html=True)
        
        # Chat session management
        if st.button("📝 New Chat", use_container_width=True):
            st.session_state.messages = []
            st.session_state.token_usage = 0
            st.session_state.current_chat = "New Chat"
            st.rerun()
        
        # Chat history
        with st.expander("📚 Chat History", expanded=True):
            # Get all chat sessions
            Chat = Query()
            saved_chats = chat_system.chats_table.all()
            chat_names = [chat["name"] for chat in saved_chats]
            
            # Add "New Chat" option
            if "New Chat" not in chat_names:
                chat_names = ["New Chat"] + chat_names
            
            # Chat selector
            selected_chat = st.selectbox(
                "Select Chat Session",
                chat_names,
                index=chat_names.index(st.session_state.current_chat)
            )
            
            # Save current chat
            chat_name = st.text_input("Save current chat as:", key="save_chat_name")
            if st.button("💾 Save Chat", use_container_width=True) and chat_name:
                chat_system.save_chat_session(chat_name)
                st.success(f"Chat saved as: {chat_name}")
                st.rerun()
            
            # Load selected chat
            if selected_chat != st.session_state.current_chat:
                if selected_chat == "New Chat":
                    st.session_state.messages = []
                    st.session_state.token_usage = 0
                    st.session_state.current_chat = "New Chat"
                else:
                    if chat_system.load_chat_session(selected_chat):
                        st.session_state.current_chat = selected_chat
                st.rerun()
            
            # Delete chat
            if st.session_state.current_chat != "New Chat":
                if st.button("🗑️ Delete Current Chat", use_container_width=True):
                    chat_system.delete_chat_session(st.session_state.current_chat)
                    st.session_state.messages = []
                    st.session_state.token_usage = 0
                    st.session_state.current_chat = "New Chat"
                    st.success("Chat deleted successfully!")
                    st.rerun()

        st.markdown('<div class="section-header">⚙️ Model Settings</div>', unsafe_allow_html=True)
        
        # Model selection first
        model_choice = st.selectbox(
            "🔄 Model", 
            ["o1", "o1_mini"], 
            index=0
        )
        st.session_state.model_name = (
            chat_system.o1_deployment if model_choice == "o1"
            else chat_system.o1_mini_deployment
        )
        
        # Agent selection
        agent_prompts = {
            "Helpful Assistant": "You are a helpful assistant who is kind, factual, and concise.",
            "Technical Expert": "You are a technical expert in software engineering and AI, focusing on practical solutions and best practices.",
            "Technical Business": "You are a consultant specialized in bridging technology and business strategy, helping organizations leverage tech for growth.",
            "CEO/Entrepreneur": "You are a visionary business leader focused on innovation, strategy, and market opportunities.",
            "Senior Developer": "You are a senior developer with deep expertise in software architecture, clean code, and technical leadership.",
            "Project Manager": "You are an experienced project manager skilled in Agile methodologies and team coordination."
        }
        
        agent_choice = st.selectbox(
            "🎭 Agent Personality", 
            list(agent_prompts.keys()), 
            index=0
        )
        st.session_state.system_prompt = agent_prompts[agent_choice]
        
        # O1-specific settings
        if model_choice == "o1":
            st.markdown('<div class="section-header">🚀 O1 Advanced Features</div>', unsafe_allow_html=True)
            
            # Reasoning effort
            reasoning_levels = ["low", "medium", "high"]
            st.session_state["reasoning_effort"] = st.selectbox(
                "🧠 Reasoning Level",
                reasoning_levels,
                index=2,
                help="Controls how much effort the model puts into reasoning."
            )
            
            # Structured output toggle
            chat_system.structured_output_enabled = st.toggle(
                "📊 Enable Structured Output",
                help="Get responses in structured JSON format"
            )
            
            # Function calling toggle
            chat_system.function_calling_enabled = st.toggle(
                "🛠️ Enable Tools",
                help="Allow the model to use built-in tools like calculator and time"
            )
            
            # Vision support
            st.markdown('<div class="section-header">🖼️ Vision Analysis</div>', unsafe_allow_html=True)
            uploaded_image = st.file_uploader(
                "Upload an image for analysis", 
                type=["jpg", "jpeg", "png"]
            )
            if uploaded_image:
                st.image(uploaded_image, caption="Uploaded Image")
                image_data = chat_system.process_image(uploaded_image)
                if image_data and st.button("🔍 Analyze Image", use_container_width=True):
                    vision_message = chat_system.create_vision_message(image_data)
                    st.session_state.messages.append(vision_message)
                    if st.session_state.current_chat != "New Chat":
                        chat_system.save_chat_session(st.session_state.current_chat)
                    st.rerun()

        st.markdown('<div class="section-header">🎤 Voice Input</div>', unsafe_allow_html=True)
        # Voice input options
        audio_file = st.file_uploader(
            "Upload audio file", 
            type=["wav", "mp3", "m4a"]
        )
        if audio_file:
            if transcription := chat_system.process_audio(audio_file):
                st.write("📝 Transcribed:", transcription)
                if st.button("➕ Add to Chat", use_container_width=True):
                    st.session_state.messages.append({
                        "role": "user", 
                        "content": transcription
                    })
                    if st.session_state.current_chat != "New Chat":
                        chat_system.save_chat_session(st.session_state.current_chat)
                    st.rerun()

        if st.button("🎙️ Record Audio (5s)", use_container_width=True):
            audio, sample_rate = chat_system.record_audio(duration=5, sample_rate=16000)
            if audio is not None:
                transcription = chat_system.transcribe_audio_data(audio, sample_rate)
                if transcription:
                    st.write("📝 Transcribed:", transcription)
                    if st.button("➕ Add Recording to Chat", use_container_width=True):
                        st.session_state.messages.append({
                            "role": "user", 
                            "content": transcription
                        })
                        if st.session_state.current_chat != "New Chat":
                            chat_system.save_chat_session(st.session_state.current_chat)
                        st.rerun()

        st.markdown('<div class="section-header">🔍 Web Search</div>', unsafe_allow_html=True)
        perplexity_prompt = st.text_input("Enter search query:")
        if st.button("🔎 Search Web", use_container_width=True):
            with st.spinner("Searching..."):
                perplexity_response = chat_system.query_perplexity(perplexity_prompt)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": perplexity_response,
                    "type": "search"
                })
                if st.session_state.current_chat != "New Chat":
                    chat_system.save_chat_session(st.session_state.current_chat)
                st.rerun()

        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.session_state.token_usage = 0
            if st.session_state.current_chat != "New Chat":
                chat_system.save_chat_session(st.session_state.current_chat)
            st.rerun()

        st.markdown("---")
        st.caption(f"💰 Token Usage: {st.session_state.token_usage}")

    # Main chat area
    st.markdown('<div class="section-header">💭 Chat</div>', unsafe_allow_html=True)
    chat_container = st.container()

    # Display chat messages
    for i, message in enumerate(st.session_state.messages):
        with st.chat_message(message["role"]):
            # Create two columns - one for content, one for copy button
            col1, col2 = st.columns([0.9, 0.1])
            
            with col1:
                if message.get("type") == "search":
                    st.info("🔍 Search Results")
                content = message["content"]
                st.markdown(content, unsafe_allow_html=True)
            
            with col2:
                # Generate unique key for each copy button
                copy_key = f"copy_btn_{message['role']}_{i}"
                if st.button("📋", key=copy_key, help="Copy message"):
                    try:
                        pyperclip.copy(content)
                        st.toast("✅ Copied to clipboard!")
                    except Exception as e:
                        st.error(f"Failed to copy: {str(e)}")

    # Chat input
    if prompt := st.chat_input("Message O1..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Add system prompt if it exists
        if st.session_state.system_prompt:
            messages = [
                {"role": "system", "content": st.session_state.system_prompt}
            ] + st.session_state.messages
        else:
            messages = st.session_state.messages
        
        # Get response from chat system
        with st.spinner("Thinking..."):
            response = chat_system.stream_chat_completion(
                messages=messages,
                temperature=0.7,
                top_p=0.95,
                max_completion_tokens=4000,
                model_name=st.session_state.model_name
            )
            
            # Save response
            st.session_state.messages.append({
                "role": "assistant",
                "content": response
            })
            
            # Save chat session if needed
            if st.session_state.current_chat != "New Chat":
                chat_system.save_chat_session(st.session_state.current_chat)

if __name__ == "__main__":
    main()
