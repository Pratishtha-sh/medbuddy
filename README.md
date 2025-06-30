# 🩺 MedBuddy – A Healthcare Chatbot Using Generative AI

**MedBuddy** is an intelligent medical assistant chatbot powered by **LangChain**, **Hugging Face Transformers**, and **Mistral-7B-Instruct**. It answers user queries related to diseases and healthcare using context-aware responses retrieved from a trusted medical encyclopedia.

---

## 📌 Features

- 💬 Conversational chatbot using **Mistral-7B-Instruct** model
- 📚 Medical knowledge base parsed from **The Gale Encyclopedia of Medicine**
- 🔍 Accurate, context-based answers via **Retrieval-Augmented Generation (RAG)**
- ⚡ Fast search with **FAISS vector store**
- 🌐 Clean and interactive **Streamlit UI**
- 🛡️ Avoids hallucination by restricting answers to provided context

---

## 🧰 Tech Stack

| Component        | Technology                     |
|------------------|--------------------------------|
| Language Model   | `mistralai/Mistral-7B-Instruct-v0.3` (HuggingFace) |
| Framework        | [LangChain](https://www.langchain.com/) |
| Embeddings       | `sentence-transformers/all-MiniLM-L6-v2` |
| Vector Store     | FAISS                          |
| UI               | Streamlit                      |
| Backend          | Python                         |

---


---

## ⚙️ Setup Instructions

### 1. 🔧 Clone the repository
```bash
git clone https://github.com/Pratishtha-sh/medbuddy.git
cd medbuddy
### 2.Install Dependencied
pip install streamlit langchain huggingface_hub faiss-cpu python-dotenv

### Create a .env file
HF_TOKEN=your_huggingface_api_token

###Running the Chatbot
streamlit run medbuddy.py

### Example Queries
What are the symptoms of diabetes?

How is cervical cancer treated?

Give precautions for high blood pressure.

What are the side effects of chemotherapy?


