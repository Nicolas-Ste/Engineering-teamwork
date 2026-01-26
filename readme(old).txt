I downloaded using these commands;


installed Ollama
ollama pull llama3
ollama run llama3
(if it answers you are good)
pip install ollama

pip install numpy sentence-transformers torch pypdf pillow pytesseract ollama

https://github.com/UB-Mannheim/tesseract/wiki
download latest version and make sure path is correct

Pipeline now works like this

User Question
     ↓
Vector Search (Your DB)
     ↓
Retrieve relevant chunks
     ↓
Insert into prompt template
     ↓
Send to Ollama (LLaMA3 / Mistral)
     ↓
Final Answer

ollama list

==============================

Files → Text → Chunks → Embeddings → NumPy Vector DB → Cosine Search → Prompt → LLM

When you move to LangChain or LlamaIndex, you gain:

✅ Automatic document loading
✅ Automatic chunking
✅ Plug-and-play vector databases (FAISS, Chroma, etc.)
✅ Built-in retrievers
✅ Built-in RAG chains
✅ Easier scaling to millions of documents
✅ Easier swapping of models
✅ Industry-standard structure (important for jobs/research)

==================================

pip install llama-index llama-index-llms-ollama llama-index-embeddings-huggingface sentence-transformers pypdf


| Package                                   | Why You Need It               |

| `llama-index`                             | Core RAG framework            |
| `llama-index-llms-ollama`                 | Connects to your local Ollama |
| `llama-index-embeddings-huggingface`      | For MiniLM embeddings         |
| `sentence-transformers`                   | Used by the embedding model   |
| `pypdf`                                   | PDF loading                   |

what is the project purpose for the fall students 2025 ?

==================================
pip install streamlit

python -m streamlit run app.py

==================================

pip install -U openai-whisper

https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2

==================================

python -m pip install faster-whisper

===================================
Vector databases versus traditional databases Because they use high-dimensional vector embeddings, vector databases are better able to handle unstructured datasets. The nature of data has undergone a profound transformation. It's no longer confined to structured information easily stored in traditional databases.

In a
RAG (Retrieval-Augmented Generation) pipeline, indexing is the crucial setup phase where raw documents are processed, chunked, converted into numerical vector embeddings, and stored in a vector database to enable fast, semantic retrieval of relevant context for the LLM.

The dot product (or scalar product) of two vectors is a single number (scalar) found by multiplying their corresponding components and adding the results, or by multiplying their magnitudes by the cosine of the angle between them; it indicates how much the vectors point in the same direction, resulting in a large positive for similar directions, zero for perpendicular, and negative for opposite directions


python -m pip install opencv-python

OCR = optical character recognition

==================






