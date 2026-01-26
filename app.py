import os
import streamlit as st
from PIL import Image
import pytesseract
import cv2
from faster_whisper import WhisperModel

from llama_index.core import Document, VectorStoreIndex
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.readers.file import PDFReader
from llama_index.core.node_parser import SentenceSplitter

# -------------------------
# LOAD AUDIO MODEL
# -------------------------
audio_model = WhisperModel("base", device="cpu", compute_type="int8")

def load_audio(path: str) -> str:
    segments, _ = audio_model.transcribe(path)
    return " ".join(s.text for s in segments).strip()

def ocr_keyframes(video_path: str, every_seconds=5, max_frames=12) -> str:
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    step = max(1, int(fps * every_seconds))

    texts, frame_idx, grabbed = [], 0, 0
    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx % step == 0:
            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            txt = pytesseract.image_to_string(img).strip()
            if txt:
                texts.append(txt)
            grabbed += 1
            if grabbed >= max_frames:
                break
        frame_idx += 1

    cap.release()
    return "\n".join(texts).strip()

def load_video(path: str) -> str:
    audio_text = load_audio(path)
    frame_text = ocr_keyframes(path)
    return "\n".join([t for t in [audio_text, frame_text] if t]).strip()

# -------------------------
# LOAD DOCUMENTS
# -------------------------
documents = []
pdf_reader = PDFReader()

for file in os.listdir("data"):
    path = os.path.join("data", file)
    low = file.lower()

    if low.endswith(".txt"):
        with open(path, "r", encoding="utf-8") as f:
            text = f.read().strip()
        if text:
            documents.append(Document(text=text))

    elif low.endswith(".pdf"):
        documents.extend(pdf_reader.load_data(path))

    elif low.endswith((".png", ".jpg", ".jpeg")):
        text = pytesseract.image_to_string(Image.open(path)).strip()
        if text:
            documents.append(Document(text=text))

    elif low.endswith((".wav", ".mp3", ".m4a", ".flac", ".ogg")):
        text = load_audio(path)
        if text:
            documents.append(Document(text=text))

    elif low.endswith((".mp4", ".mov", ".mkv", ".avi", ".webm")):
        text = load_video(path)
        if text:
            documents.append(Document(text=text))

# -------------------------
# BUILD INDEX
# -------------------------
embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
splitter = SentenceSplitter(chunk_size=512, chunk_overlap=100)

index = VectorStoreIndex.from_documents(
    documents,
    embed_model=embed_model,
    transformations=[splitter]
)

llm = Ollama(model="llama3", request_timeout=120.0)
query_engine = index.as_query_engine(llm=llm)

# -------------------------
# STREAMLIT UI
# -------------------------
st.set_page_config(page_title="RAG App", layout="wide")
st.title("📄 RAG Question Answering System")
st.write("Now supports: TXT, PDF, Images, Audio, Video")

query = st.text_input("Enter your question:")

if st.button("Ask"):
    if query.strip() == "":
        st.warning("Please enter a question.")
    else:
        with st.spinner("Thinking..."):
            response = query_engine.query(query)
        st.subheader("✅ Answer")
        st.write(response.response)
