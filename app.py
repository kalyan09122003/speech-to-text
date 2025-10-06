import streamlit as st
import whisper
import torch
import os
import tempfile
from moviepy.editor import VideoFileClip

# ---------- Streamlit Page Setup ----------
st.set_page_config(page_title="Speech-to-Text & Language Detection", layout="centered")
st.title("🎙️ Audio/Video → Script")
st.caption("detects Telugu, Hindi, or English")

# ---------- Load Whisper Model ----------
@st.cache_resource
def load_model():
    return whisper.load_model("tiny")

model = load_model()
use_fp16 = torch.cuda.is_available()

LANG_MAP = {"en": "English", "hi": "Hindi", "te": "Telugu"}

# ---------- Upload File ----------
st.subheader("📂 Upload Audio or Video File")
uploaded_file = st.file_uploader("Choose a file", type=["mp3", "wav", "m4a", "mp4", "mov", "mkv"])

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=uploaded_file.name) as tmp_file:
        tmp_file.write(uploaded_file.read())
        temp_file_path = tmp_file.name

    # 🔧 Extract audio if it's a video
    audio_path = temp_file_path
    if uploaded_file.name.lower().endswith((".mp4", ".mkv", ".mov")):
        try:
            clip = VideoFileClip(temp_file_path)
            audio_path = tempfile.mktemp(suffix=".wav")
            clip.audio.write_audiofile(audio_path, verbose=False, logger=None)
            clip.close()
        except Exception as e:
            st.error(f"Error extracting audio from video: {e}")
            os.remove(temp_file_path)
            st.stop()

    # ---------- Transcribe ----------
    with st.spinner("Transcribing and detecting language... ⏳"):
        try:
            result = model.transcribe(audio_path, fp16=use_fp16)
        except Exception as e:
            st.error(f"❌ Error while transcribing: {e}")
            os.remove(temp_file_path)
            if audio_path != temp_file_path:
                os.remove(audio_path)
            st.stop()

    # ---------- Show Results ----------
    detected_lang = LANG_MAP.get(result["language"], "Other (Not Telugu/Hindi/English)")
    st.success(f"🌐 Detected Language: **{detected_lang}**")

    st.subheader("📝 Transcript")
    st.write(result["text"])

    # ---------- Download Script ----------
    transcript_text = f"Detected Language: {detected_lang}\n\nTranscript:\n{result['text']}"
    st.download_button(
        label="⬇️ Download Script (.txt)",
        data=transcript_text,
        file_name=f"transcript_{detected_lang.lower()}.txt",
        mime="text/plain"
    )

    # ---------- Cleanup ----------
    os.remove(temp_file_path)
    if audio_path != temp_file_path:
        os.remove(audio_path)

