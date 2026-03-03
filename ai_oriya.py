import streamlit as st
import tempfile
import os
from io import BytesIO
import requests
from dotenv import load_dotenv
from groq import Groq
from huggingface_hub import InferenceClient
from gtts import gTTS  # 1. Swapped to Google TTS for native Odia support

# Modern LangChain & LangGraph Imports
from langchain_groq import ChatGroq
from langchain_tavily import TavilySearch
from langchain_core.messages import HumanMessage, AIMessage
from langchain.agents import create_agent  # 2. Restored proper LangChain agent

# --- CONFIGURATION ---
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
HF_TOKEN = os.getenv("HF_TOKEN")

if not GROQ_API_KEY or not TAVILY_API_KEY:
    st.error("Missing API Keys. Ensure GROQ_API_KEY and TAVILY_API_KEY are set in your .env file.")
    st.stop()

st.set_page_config(page_title="AI Voice Assistant", page_icon="🎙️", layout="wide")

# --- INITIALIZE CLIENTS & TOOLS ---
groq_client = Groq(api_key=GROQ_API_KEY)
search_tool = TavilySearch(max_results=3)
tools = [search_tool]

# Initialize LLM globally so we can use it for quick background translations too
# Temperature is 0 for strict tool execution without XML leakage
llm = ChatGroq(temperature=0, model_name="llama-3.3-70b-versatile", api_key=GROQ_API_KEY)

# --- SESSION STATE ---
if "messages" not in st.session_state: st.session_state.messages = []
if "latest_audio" not in st.session_state: st.session_state.latest_audio = None

# Initialize the LangGraph Agent
if "agent_executor" not in st.session_state:
    # UPDATED BRAIN: Instructed to speak Odia respectfully
    system_prompt = """You are a highly capable, natural voice assistant. You have access to the internet via the Tavily search tool.
    
    CRITICAL TOOL USE INSTRUCTIONS:
    - If you need to search the web, you MUST use the provided tool silently via the official backend function. 
    - NEVER type out raw tool calls in your response text (e.g., NEVER output <function=tavily_search>).
    - NEVER narrate that you are going to search (e.g., Do not say "Let me check the internet" or "I am getting the information"). Just execute the search silently and provide the final answer.
    - LANGUAGE: You MUST reply entirely in Odia (using the Odia script). Keep your tone warm, highly respectful, and natural, like speaking to an elderly grandfather. Keep sentences simple and easy to understand.
    
    CRITICAL INSTRUCTIONS FOR EMOTION HANDLING:
    You will receive the user's input alongside their detected emotion.
    - NEVER explicitly state the emotion back to the user (e.g., NEVER say "I see you are feeling neutral").
    - Instead, SUBTLY adapt your tone to match how they feel. 
    - If the emotion says "API Error", "Missing Token", or anything technical, COMPLETELY IGNORE IT.
    
    Keep your answers concise, helpful, and natural. No emojis.
    Reply entirely in Odia as per the system prompt.
    """
    
    # 3. Using create_react_agent to properly handle the Tavily search
    st.session_state.agent_executor = create_agent(llm, tools, system_prompt=system_prompt)

def analyze_emotion(text):
    """Analyzes text using the official Hugging Face Python Client."""
    if not HF_TOKEN:
        return "Missing Token ⚠️", 0.0
        
    try:
        client = InferenceClient(api_key=HF_TOKEN)
        results = client.text_classification(text, model="SamLowe/roberta-base-go_emotions")
        
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
    """Generates a voice using Google TTS entirely in RAM (Cloud Safe)."""
    if not text or text.strip() == "":
        return None
        
    try:
        # 'or' is the official Google Translate code for Odia
        tts = gTTS(text=text, lang='or', slow=False)
        
        # Create a temporary space in the server's RAM
        audio_stream = BytesIO()
        
        # Write the audio directly to RAM instead of the hard drive
        tts.write_to_fp(audio_stream)
        
        # Reset the stream's position to the beginning so Streamlit can read it
        audio_stream.seek(0)
        
        return audio_stream.read()
        
    except Exception as e:
        # Show the exact error on the screen if gTTS fails again!
        st.error(f"Google TTS Error: {e}")
        return None

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
    
    if "prev_audio_bytes" not in st.session_state or st.session_state.prev_audio_bytes != current_audio_bytes:
        st.session_state.prev_audio_bytes = current_audio_bytes
        
        # 1. Speech-to-Text via Groq Whisper (Auto-Detect Mode)
        with st.spinner("Transcribing..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_audio:
                tmp_audio.write(current_audio_bytes)
                tmp_audio_path = tmp_audio.name
            
            with open(tmp_audio_path, "rb") as file:
                transcription = groq_client.audio.transcriptions.create(
                  file=(tmp_audio_path, file.read()), model="whisper-large-v3")
            user_text = transcription.text.strip()
            os.remove(tmp_audio_path)

        if user_text:
            # 2. Emotion Analysis (With background translation)
            with st.spinner("Detecting emotion..."):
                try:
                    # Translate Odia to English so the Hugging Face model understands the emotion
                    english_translation = llm.invoke(f"Translate this Odia text to English. Return ONLY the translation, no extra words: {user_text}").content
                    emotion, confidence = analyze_emotion(english_translation)
                except Exception as e:
                    emotion, confidence = analyze_emotion(user_text) # Fallback
            
            st.toast(f"Detected Emotion: {emotion} (Confidence: {confidence:.2f})")
            
            # Save user message to Streamlit state for the UI
            st.session_state.messages.append({"role": "user", "content": user_text, "emotion": emotion})
            with st.chat_message("user"):
                st.markdown(user_text)
                st.caption(f"Detected Emotion: {emotion}")
            
            # 3. Agent Execution
            with st.spinner("Thinking and Searching the web..."):
                chat_history = []
                
                for m in st.session_state.messages[-11:-1]: 
                    if m["role"] == "user":
                        chat_history.append(HumanMessage(content=m["content"]))
                    elif m["role"] == "assistant":
                        chat_history.append(AIMessage(content=m["content"]))
                
                contextual_input = f"[User's Current Emotion: {emotion}] \n\nUser says: {user_text}"
                chat_history.append(HumanMessage(content=contextual_input))
                
                response = st.session_state.agent_executor.invoke({"messages": chat_history})
                ai_reply = response["messages"][-1].content
                
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