import os
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
# AUDIO MODEL (ASR)
# -------------------------
audio_model = WhisperModel("base", device="cpu", compute_type="int8")

def load_audio(path: str) -> str:
    segments, _ = audio_model.transcribe(path)
    return " ".join(segment.text for segment in segments).strip()

# -------------------------
# IMAGE OCR
# -------------------------
def load_image(path: str) -> str:
    img = Image.open(path)
    return pytesseract.image_to_string(img).strip()

# -------------------------
# VIDEO FRAME OCR
# -------------------------
def ocr_keyframes(video_path: str, every_seconds: int = 5, max_frames: int = 12) -> str:
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    step = max(1, int(fps * every_seconds))

    texts = []
    frame_idx = 0
    grabbed = 0

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

# -------------------------
# LOAD VIDEO (AUDIO + OPTIONAL OCR)
# -------------------------
def load_video(path: str, do_frame_ocr: bool = True) -> str:
    parts = []

    audio_text = load_audio(path)
    if audio_text:
        parts.append(audio_text)

    if do_frame_ocr:
        frame_text = ocr_keyframes(path, every_seconds=5, max_frames=12)
        if frame_text:
            parts.append(frame_text)

    return "\n".join(parts).strip()

# -------------------------
# LOAD DOCUMENTS (5 DATA TYPES)
# -------------------------
documents = []
data_folder = "data"
pdf_reader = PDFReader()

for file in os.listdir(data_folder):
    path = os.path.join(data_folder, file)
    low = file.lower()

    #  TXT
    if low.endswith(".txt"):
        with open(path, "r", encoding="utf-8") as f:
            text = f.read().strip()

        if text:
            documents.append(Document(
                text=f"SOURCE: {file}\nTYPE: txt\n\n{text}",
                metadata={"source": file, "type": "txt"}
            ))

    #  PDF
    elif low.endswith(".pdf"):
        docs = pdf_reader.load_data(path)

        # create NEW Document objects (don't mutate d.text)
        for d in docs:
            pdf_text = (d.get_content() if hasattr(d, "get_content") else d.text).strip()
            if pdf_text:
                documents.append(Document(
                    text=f"SOURCE: {file}\nTYPE: pdf\n\n{pdf_text}",
                    metadata={"source": file, "type": "pdf"}
                ))

    #  IMAGE (OCR)
    elif low.endswith((".png", ".jpg", ".jpeg")):
        text = load_image(path)
        print(f"Image OCR: {file} chars={len(text)}")
        if text:
            documents.append(Document(
                text=f"SOURCE: {file}\nTYPE: image\n\n{text}",
                metadata={"source": file, "type": "image"}
            ))

    #  AUDIO (ASR)
    elif low.endswith((".wav", ".mp3", ".m4a", ".flac", ".ogg")):
        print(f"Transcribing audio: {file}")
        text = load_audio(path)
        print(f"[AUDIO] {file} chars={len(text)} preview={text[:120]!r}")
        if text:
            documents.append(Document(
                text=f"SOURCE: {file}\nTYPE: audio\n\n{text}",
                metadata={"source": file, "type": "audio"}
            ))
        else:
            print(f"⚠️ Skipping {file}: extracted audio text empty")

    #  VIDEO (ASR + optional OCR)
    elif low.endswith((".mp4", ".mov", ".mkv", ".avi", ".webm")):
        print(f"Processing video: {file}")
        text = load_video(path, do_frame_ocr=True)
        print(f"[VIDEO] {file} chars={len(text)} preview={text[:120]!r}")
        if text:
            documents.append(Document(
                text=f"SOURCE: {file}\nTYPE: video\n\n{text}",
                metadata={"source": file, "type": "video"}
            ))
        else:
            print(f"⚠️ Skipping {file}: extracted video text empty")

print(f"\nLoaded {len(documents)} documents.\n")

# -------------------------
# EMBEDDINGS
# -------------------------
embed_model = HuggingFaceEmbedding(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# -------------------------
# CHUNKING (IMPORTANT)
# -------------------------
splitter = SentenceSplitter(chunk_size=512, chunk_overlap=100)

# -------------------------
# VECTOR INDEX
# -------------------------
index = VectorStoreIndex.from_documents(
    documents,
    embed_model=embed_model,
    transformations=[splitter],
)

# -------------------------
# LLM + QUERY ENGINE
# -------------------------
llm = Ollama(model="llama3")
query_engine = index.as_query_engine(llm=llm)

# -------------------------
# MAIN LOOP (WITH RETRIEVAL DEBUG)
# -------------------------
print("LlamaIndex RAG System Ready!\n")

retriever = index.as_retriever(similarity_top_k=3)

while True:
    query = input("Ask a question (or exit): ")
    if query.lower() == "exit":
        break

    nodes = retriever.retrieve(query)
    print("\n--- RETRIEVED NODES ---")
    for i, n in enumerate(nodes, 1):
        md = getattr(n.node, "metadata", {})
        print(f"\n#{i} score={n.score:.3f} meta={md}")
        print(n.node.get_text()[:300], "...")

    response = query_engine.query(query)
    print("\n--- LLM With RAG ---")
    print(response)
