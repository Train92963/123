import torch
import librosa
import os
import cv2
import streamlit as st
import noisereduce as nr
from deepface import DeepFace
from tempfile import NamedTemporaryFile
import whisper
import soundfile as sf
from functools import lru_cache
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TextClassificationPipeline, MarianMTModel, MarianTokenizer
import mistralai

# Whisper 模型載入
@st.cache_resource
def load_whisper_model():
    return whisper.load_model("medium")

# 翻譯模型載入（Helsinki-NLP 中文翻英文）
@st.cache_resource
def load_marian_translator():
    model_name = "Helsinki-NLP/opus-mt-zh-en"
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)
    return tokenizer, model

def translate_zh_to_en(text):
    tokenizer, model = load_marian_translator()
    inputs = tokenizer([text], return_tensors="pt", truncation=True, padding=True)
    translated = model.generate(**inputs)
    return tokenizer.decode(translated[0], skip_special_tokens=True)

# 語音轉文字並偵測語言
def transcribe_audio(wav_bytes):
    with NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(wav_bytes.read())
        tmp_path = tmp.name
    try:
        audio, rate = librosa.load(tmp_path, sr=16000)
        audio = nr.reduce_noise(y=audio, sr=rate)
        sf.write(tmp_path, audio, rate)
    except Exception as e:
        st.error(f"音頻處理錯誤: {e}")
        return "", ""

    model = load_whisper_model()
    try:
        result = model.transcribe(tmp_path)
        return result["text"], result["language"]
    except Exception as e:
        st.error(f"語音辨識錯誤: {e}")
        return "", ""

# 文本情緒分析（以英文為主）
@lru_cache(maxsize=1)
def load_text_pipeline():
    model_name = "bhadresh-savani/distilbert-base-uncased-emotion"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    return TextClassificationPipeline(model=model, tokenizer=tokenizer, return_all_scores=False)

def analyze_text_emotion(text, lang):
    original_text = text
    translated_text = text  # 預設就是英文
    if lang == "zh":
        translated_text = translate_zh_to_en(text)
    classifier = load_text_pipeline()
    result = classifier(translated_text)
    return sorted(result, key=lambda x: x['score'], reverse=True), translated_text


# 影像情緒分析
def analyze_image_emotion(image_file):
    with NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        tmp.write(image_file.read())
        tmp_path = tmp.name
    try:
        result = DeepFace.analyze(img_path=tmp_path, actions=['emotion'], enforce_detection=False)
        return result[0]['dominant_emotion']
    except Exception as e:
        st.error(f"影像情緒分析錯誤: {e}")
        return "neutral"

# 對應 DeepFace 到 Text Emotion 的類別
def map_deepface_to_distilbert_emotion(deepface_emotion):
    emotion_map = {
        "angry": "anger",
        "fear": "fear",
        "happy": "joy",
        "sad": "sadness",
        "surprise": "surprise",
        "neutral": "neutral",
    }
    return emotion_map.get(deepface_emotion.lower(), "neutral")

# 多模態情緒綜合判斷
def multimodal_decision(text_emotion, image_emotion):
    weights = {"text": 0.6, "image": 0.4}
    emotions = [text_emotion, image_emotion]
    scores = {e: weights['text'] if e == text_emotion else weights['image'] for e in emotions}
    return max(scores, key=scores.get)

# Mistral API 金鑰
os.environ["MISTRAL_API_KEY"] = "AUJXbBOsFXJYTxDMiRV8J2EwuBMkkhy7"

# AI 應對生成
def generate_response(text, emotion, lang):
    api_key = os.getenv("MISTRAL_API_KEY")
    if not api_key:
        return "❌ API 金鑰遺失"
    client = mistralai.Mistral(api_key=api_key)

    if lang == "zh":
        system_msg = "你是情感分析 AI，請用中文回應"
        prompt = f"[情緒: {emotion}] {text}\nAI 回應："
    else:
        system_msg = "You are an empathetic AI assistant. Respond in English."
        prompt = f"[Emotion: {emotion}] {text}\nAI response:"

    try:
        response = client.chat.complete(
            model="mistral-medium",
            messages=[{"role": "system", "content": system_msg},
                      {"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"❌ 回應錯誤: {e}"

# ---------- Streamlit UI ----------
st.title("🎧 AI 情緒小助手")
st.write("上傳語音與照片，AI 幫你分析心情，並試著讓你開心起來！")

# 上傳檔案
audio_file = st.file_uploader("請上傳語音檔 (.wav)", type=['wav','mp3'])
image_file = st.file_uploader("請上傳圖片檔", type=['jpg', 'jpeg', 'png'])

if audio_file:
    with st.spinner("語音辨識中..."):
        text, lang = transcribe_audio(audio_file)
    if not text:
        st.stop()

    lang_name = "中文" if lang == "zh" else "English"
    st.success(f"語音轉文字結果：{text}（語言偵測：{lang_name}）")

    with st.spinner("分析文字情緒中..."):
        emotion_result, translated_text = analyze_text_emotion(text, lang)
        text_emotion = emotion_result[0]['label']

    if lang == "zh":
        st.markdown(f"🈶 **翻譯後英文：** `{translated_text}`")

    st.info(f"📘 文本情緒：{text_emotion}")

    if image_file:
        with st.spinner("分析影像情緒中..."):
            image_emotion = analyze_image_emotion(image_file)
            mapped_image_emotion = map_deepface_to_distilbert_emotion(image_emotion)
        st.info(f"📷 影像情緒：{mapped_image_emotion}")
    else:
        mapped_image_emotion = "neutral"

    final_emotion = multimodal_decision(text_emotion, mapped_image_emotion)
    st.markdown(f"### 🤖 綜合情緒判斷：**{final_emotion}**")

    response = generate_response(text, final_emotion, lang)
    st.markdown("---")
    st.subheader("💬 AI 對你的回應：")
    st.write(response)

    # 🎁 情緒急救包
    if final_emotion in ["sadness", "fear", "anger"]:
        st.markdown("---")
        st.subheader("🎁 看起來你有點不開心，要不要來點療癒的？")
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("🎵 聽點音樂"):
                st.audio("https://www.soundhelix.com/examples/mp3/SoundHelix-Song-1.mp3")
        with col2:
            if st.button("🐱 療癒貓貓"):
                st.image("https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTGgWKxatJXccVp82cs2ILRlYAOZ45BBtAkRg&s", caption="貓貓：我陪你喔！")
        with col3:
            if st.button("🧠 再聊一下"):
                st.write(generate_response(text, final_emotion, lang))

