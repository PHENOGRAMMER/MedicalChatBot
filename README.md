# 🏥 MediChat — AI Medical Chatbot

A RAG-powered medical assistant built on the **Gale Encyclopedia of Medicine (2nd Edition)**.
Ask any medical question and get accurate, context-grounded answers.

## 🛠 Tech Stack
- **LLM:** LLaMA 3.1 8B via Groq API
- **Embeddings:** sentence-transformers/all-MiniLM-L6-v2
- **Vector DB:** Pinecone
- **Framework:** LangChain + Flask
- **Frontend:** Vanilla HTML/CSS/JS

## 🚀 Features
- Retrieval-Augmented Generation (RAG) pipeline
- Multi-turn chat memory with history-aware retrieval
- Structured responses with severity tags
- Right panel with resources and PubMed search
- Session-based conversation history

## 📚 Data
- Place your PDF(s) inside the `data/` folder before running `store_index.py`.
- This project uses the *Gale Encyclopedia of Medicine, 2nd Edition*.

## ⚙️ Setup
1. Clone the repo
```bash
   git clone https://github.com/PHENOGRAMMER/MedicalChatBot.git
   cd MedicalChatBot
```
2. Create virtual environment and install dependencies
```bash
   python -m venv venv
   venv\Scripts\activate  # Windows
   pip install -r requirements.txt
```
3. Add your API keys to `.env`
```
   PINECONE_API_KEY=your_key
   GROQ_API_KEY=your_key
```
4. Run the indexing script (once)
```bash
   python store_index.py
```
5. Start the app
```bash
   python app.py
```
6. Open `http://localhost:8080`

## ⚠️ Disclaimer
For informational purposes only. Not a substitute for professional medical advice.
