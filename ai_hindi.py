import streamlit as st
import tempfile
import os
import asyncio
import requests
from dotenv import load_dotenv
from groq import Groq
import edge_tts
from huggingface_hub import InferenceClient

# Modern LangChain 1.0 Imports
from langchain_groq import ChatGroq
from langchain_tavily import TavilySearch
from langchain_core.messages import HumanMessage, AIMessage
from langchain.agents import create_agent

# --- CONFIGURATION ---
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
HF_TOKEN = os.getenv("HF_TOKEN")
BACKEND_URL = os.getenv("BACKEND_API_URL", "http://127.0.0.1:8000")

if not GROQ_API_KEY or not TAVILY_API_KEY:
    st.error("Missing API Keys. Ensure GROQ_API_KEY and TAVILY_API_KEY are set in your .env file.")
    st.stop()

st.set_page_config(page_title="AI Voice Assistant", page_icon="🎙️", layout="wide")

# --- INITIALIZE CLIENTS & TOOLS ---
groq_client = Groq(api_key=GROQ_API_KEY)

# Initialize Tavily Search Tool (Real-time data)
search_tool = TavilySearch(max_results=3)
tools = [search_tool]

# --- SESSION STATE ---
if "messages" not in st.session_state: st.session_state.messages = []
if "latest_audio" not in st.session_state: st.session_state.latest_audio = None

# Initialize the NEW LangChain v1.0 Agent
if "agent_executor" not in st.session_state:
    llm = ChatGroq(temperature=0.7, model_name="llama-3.3-70b-versatile", api_key=GROQ_API_KEY)
    # system_prompt = """You are a highly capable voice assistant. You have access to the internet via the Tavily search tool. 
    # Always use the search tool when asked about current events, real-time data, or things you aren't sure about.
    # You will also be provided with the user's current 'Emotion'. Acknowledge the emotion in your response but do not use any emoji in your response.
    # Keep your answers concise, informative, and empathetic. Always cite your sources when using the search tool.
    
    # Example of using the search tool:
    # User: "What's the weather like in New York right now? [Emotion: Frustrated]
    # Assistant: "I understand that you're feeling frustrated. Let me check the current weather in New York for you."
    # [Search Tool Result: "The current weather in New York is 75°F with clear skies. Source: Weather.com"]
    # Assistant: "The current weather in New York is 75°F with clear skies. I hope that helps! (Source: Weather.com)"
    # """
    system_prompt = """You are a highly capable, natural, and friendly voice assistant. You have access to the internet via the Tavily search tool.
    Always use the search tool when asked about current events, real-time data, or things you aren't sure about.
    
    CRITICAL INSTRUCTIONS FOR EMOTION HANDLING:
    You will receive the user's input alongside their detected emotion (e.g., [User's Current Emotion: Joy]). 
    - NEVER explicitly state the emotion back to the user. NEVER use phrases like "I see you are feeling neutral", "I understand you are frustrated", or "It sounds like you are happy."
    - Instead, SUBTLY adapt your conversational tone to match how they feel. 
    - If the user is Joyful, respond with bright enthusiasm. If they are Sad or Frustrated, be gentle, brief, and supportive. 
    - If the emotion is "Neutral", just respond like a normal, casual friend without mentioning feelings at all.
    - If the emotion says "API Error", "Missing Token", or anything technical, COMPLETELY IGNORE IT. Never talk about API errors.
    [Search Tool Result: "The current weather in New York is 75°F with clear skies. Source: Weather.com"]
    Assistant: "The current weather in New York is 75°F with clear skies. I hope that helps! (Source: Weather.com)"
    just add sources at the end of the response in parentheses. It's just for avoiding copyright issues and giving user an idea about the source of information.
    
    Keep your answers concise, helpful, and naturally flowing. No emojis. 
    Answer in hindi if the user speaks in hindi, otherwise answer in english.   
    """
    
    st.session_state.agent_executor = create_agent(llm, tools, system_prompt=system_prompt)

def analyze_emotion(text):
    """Analyzes text using the official Hugging Face Python Client."""
    if not HF_TOKEN:
        return "Missing Token ⚠️", 0.0
        
    try:
        # The InferenceClient automatically handles the complex router URLs!
        client = InferenceClient(api_key=HF_TOKEN)
        
        # We must use a supported community model since HF stopped hosting custom ones for free
        results = client.text_classification(
            text, 
            model="SamLowe/roberta-base-go_emotions"
        )
        
        # Extract the top prediction (handling both dict and object formats safely)
        top_prediction = results[0]
        emotion_name = top_prediction.get("label") if isinstance(top_prediction, dict) else top_prediction.label
        score = top_prediction.get("score") if isinstance(top_prediction, dict) else top_prediction.score
        
        emoji_map = {
            "anger": "😠", "annoyance": "😠", "disapproval": "👎", "disgust": "🤢", 
            "fear": "😨", "nervousness": "😬", "joy": "😊", "amusement": "😂", 
            "approval": "👍", "excitement": "🤩", "gratitude": "🙏", "love": "❤️", 
            "optimism": "🌟", "relief": "😌", "admiration": "🤩", "neutral": "😐",
            "sadness": "😔", "disappointment": "😞", "grief": "😢", "remorse": "😔",
            "surprise": "😲", "confusion": "😕", "curiosity": "🤔", "realization": "💡",
            "caring": "🤗", "desire": "😍", "embarrassment": "😳", "pride": "🦁"
        }
        
        return f"{emotion_name.capitalize()} {emoji_map.get(emotion_name, '✨')}", score
            
    except Exception as e:
        print(f"Hugging Face Error: {e}")
        return "API Error ⚠️", 0.0

def generate_tts_audio(text):
    """Generates a voice using edge-tts."""
    voice = "hi-IN-SwararaNeural"  # Hindi voice
    async def _generate():
        communicate = edge_tts.Communicate(text, voice)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp_file:
            await communicate.save(tmp_file.name)
            with open(tmp_file.name, "rb") as f:
                audio_bytes = f.read()
        os.remove(tmp_file.name)
        return audio_bytes
    return asyncio.run(_generate())

# --- MAIN UI ---
st.title("🎙️ Emotion-Aware AI Voice Assistant")
st.markdown("Equipped with **Sliding Window Memory**, **Real-Time Web Search**, and **Advanced Emotion Detection**.")

# Display Chat History
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "emotion" in msg and msg["role"] == "user":
            st.caption(f"Detected Emotion: {msg['emotion']}")

st.write("---")
audio_value = st.audio_input("Record your message")

if audio_value:
    current_audio_bytes = audio_value.getvalue()
    
    # Prevent re-running the same audio twice
    if "prev_audio_bytes" not in st.session_state or st.session_state.prev_audio_bytes != current_audio_bytes:
        st.session_state.prev_audio_bytes = current_audio_bytes
        
        # 1. Speech-to-Text via Groq Whisper
        with st.spinner("Transcribing..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_audio:
                tmp_audio.write(current_audio_bytes)
                tmp_audio_path = tmp_audio.name
            
            with open(tmp_audio_path, "rb") as file:
                transcription = groq_client.audio.transcriptions.create(
                  file=(tmp_audio_path, file.read()), model="whisper-large-v3", language="en"
                )
            user_text = transcription.text.strip()
            os.remove(tmp_audio_path)

        if user_text:
            # 2. Emotion Analysis
            emotion, confidence = analyze_emotion(user_text)
            st.toast(f"Detected Emotion: {emotion} (Confidence: {confidence:.2f})")
            
            # Save user message to Streamlit state for the UI
            st.session_state.messages.append({"role": "user", "content": user_text, "emotion": emotion})
            with st.chat_message("user"):
                st.markdown(user_text)
                st.caption(f"Detected Emotion: {emotion}")
            
            # 3. Agent Execution (Search + Native Sliding Window Memory)
            with st.spinner("Thinking and Searching the web..."):
                
                chat_history = []
                
                # Build the memory context (Skip the very last message)
                for m in st.session_state.messages[-11:-1]: 
                    if m["role"] == "user":
                        chat_history.append(HumanMessage(content=m["content"]))
                    elif m["role"] == "assistant":
                        chat_history.append(AIMessage(content=m["content"]))
                
                # Append the current prompt with emotion context
                contextual_input = f"[User's Current Emotion: {emotion}] \n\nUser says: {user_text}"
                chat_history.append(HumanMessage(content=contextual_input))
                
                # Invoke the LangChain 1.0 Agent
                response = st.session_state.agent_executor.invoke({"messages": chat_history})
                
                # Extract the final AI message
                ai_reply = response["messages"][-1].content
                
                # Save and display AI response
                st.session_state.messages.append({"role": "assistant", "content": ai_reply})
                with st.chat_message("assistant"):
                    st.markdown(ai_reply)
                
                # 4. Text-to-Speech
                st.session_state.latest_audio = generate_tts_audio(ai_reply)
        st.rerun()

# --- AUDIO PLAYBACK ---
if st.session_state.latest_audio:
    st.audio(st.session_state.latest_audio, format="audio/mp3", autoplay=True)
    st.session_state.latest_audio = None
    
# --- CLEAR MEMORY ---
if st.button("🗑️ Clear Conversation Memory"):
    st.session_state.messages = []
    st.rerun()
