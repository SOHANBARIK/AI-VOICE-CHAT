from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import pipeline
import uvicorn

# 1. Initialize the FastAPI App
app = FastAPI(title="Local Emotion API", description="Serves the custom GoEmotions model.")

# 2. Define the expected input data format
class EmotionRequest(BaseModel):
    text: str

# 3. Load the model globally so it only loads into RAM ONCE
print("Loading model into memory... Please wait.")
try:
    classifier = pipeline("text-classification", model="./my-custom-goemotions-model", top_k=1)
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ Failed to load model. Did you extract the folder correctly? Error: {e}")
    classifier = None

# 4. Map the GoEmotions labels (The 'simplified' dataset has 28 labels)
go_emotions_labels = [
    "admiration", "amusement", "anger", "annoyance", "approval", "caring", 
    "confusion", "curiosity", "desire", "disappointment", "disapproval", 
    "disgust", "embarrassment", "excitement", "fear", "gratitude", "grief", 
    "joy", "love", "nervousness", "optimism", "pride", "realization", 
    "relief", "remorse", "sadness", "surprise", "neutral"
]

emoji_map = {
    "anger": "😠", "annoyance": "😠", "disapproval": "👎", "disgust": "🤢", 
    "fear": "😨", "nervousness": "😬", "joy": "😊", "amusement": "😂", 
    "approval": "👍", "excitement": "🤩", "gratitude": "🙏", "love": "❤️", 
    "optimism": "🌟", "relief": "😌", "admiration": "🤩", "neutral": "😐",
    "sadness": "😔", "disappointment": "😞", "grief": "😢", "remorse": "😔",
    "surprise": "😲", "confusion": "😕", "curiosity": "🤔", "realization": "💡",
    "caring": "🤗", "desire": "😍", "embarrassment": "😳", "pride": "🦁"
}

# 5. Create the API Endpoint
@app.post("/analyze")
async def analyze_text(request: EmotionRequest):
    if classifier is None:
        raise HTTPException(status_code=500, detail="Model failed to load on the server.")
        
    try:
        # Run inference
        results = classifier(request.text)
        top_prediction = results[0][0]
        
        # Handle Colab's default 'LABEL_X' output format
        if "LABEL_" in top_prediction['label']:
            idx = int(top_prediction['label'].split("_")[-1])
            emotion_name = go_emotions_labels[idx]
        else:
            emotion_name = top_prediction['label']

        score = float(top_prediction['score'])
        formatted_emotion = f"{emotion_name.capitalize()} {emoji_map.get(emotion_name, '✨')}"
        
        return {
            "emotion": formatted_emotion,
            "confidence": score,
            "raw_label": emotion_name
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)