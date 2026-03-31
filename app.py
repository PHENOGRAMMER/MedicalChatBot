import os
import json
import sqlite3
import io
import re
from flask import Flask, render_template, jsonify, request, send_file
from dotenv import load_dotenv
from src.helper import download_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain_groq import ChatGroq
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from src.prompt import system_prompt, contextualize_q_system_prompt
try:
    from fpdf import FPDF
except ImportError:
    from fpdf2 import FPDF
from deep_translator import GoogleTranslator
from langchain.chains import create_history_aware_retriever

app = Flask(__name__)
app.secret_key = "medical-chatbot-secret"
load_dotenv()

PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["GROQ_API_KEY"] = GROQ_API_KEY

embeddings = download_embeddings()

index_name = "medical-chatbot"

docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)

retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3})


llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.4,
    max_tokens=600
)

# Contextualize Question
contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)
history_aware_retriever = create_history_aware_retriever(
    llm, retriever, contextualize_q_prompt
)

# Answer Question
qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)
question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

# Memory store for chat history (Simple global store for demo, ideally use DB)
chats = {}

def setup_db():
    conn = sqlite3.connect('medichat.db')
    cursor = conn.cursor()
    cursor.execute('''CREATE TABLE IF NOT EXISTS queries 
                      (id INTEGER PRIMARY KEY, session_id TEXT, question TEXT, answer TEXT, lang TEXT, timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)''')
    try:
        cursor.execute("ALTER TABLE queries ADD COLUMN chat_id TEXT")
    except sqlite3.OperationalError:
        pass
    conn.commit()
    conn.close()

setup_db()

def log_query(session_id, chat_id, question, answer, lang):
    conn = sqlite3.connect('medichat.db')
    cursor = conn.cursor()
    cursor.execute('INSERT INTO queries (session_id, chat_id, question, answer, lang) VALUES (?, ?, ?, ?, ?)', (session_id, chat_id, question, answer, lang))
    conn.commit()
    conn.close()


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/get", methods=["POST"])
def chat():
    msg = request.form["msg"]
    lang = request.form.get("lang", "en")
    chat_id = request.form.get("chat_id", "default")
    session_id = request.remote_addr
    chat_key = f"{session_id}_{chat_id}"
    
    # 1. Translate Incoming if not English
    msg_en = msg
    if lang != "en":
        try:
            msg_en = GoogleTranslator(source=lang, target='en').translate(msg)
        except: pass

    # --- Greeting Check ---
    greetings = ["hi", "hello", "hey", "howdy", "good morning", "good afternoon"]
    if msg_en.lower().strip() in greetings:
        ans_text = "Hello! I'm MedChat, your specialized AI medical assistant. I'm here to help you understand medical conditions, symptoms, and treatments from the Gale Encyclopedia of Medicine.\n\nHow can I help you today?"
        if lang != "en":
            try:
                ans_text = GoogleTranslator(source='en', target=lang).translate(ans_text)
            except: pass
        
        f1, f2, f3 = "What information can you provide?", "Tell me about common conditions.", "How do I use the body map?"
        if lang != "en":
            try:
                furl = GoogleTranslator(source='en', target=lang)
                f1, f2, f3 = furl.translate(f1), furl.translate(f2), furl.translate(f3)
            except: pass

        ans = f"SEVERITY: INFO\n{ans_text}\n\nFOLLOWUPS:\n- {f1}\n- {f2}\n- {f3}"
        log_query(session_id, chat_id, msg, ans, lang)
        return ans

    if chat_key not in chats:
        chats[chat_key] = []
    
    chat_history = chats[chat_key]
    
    response = rag_chain.invoke({"input": msg_en, "chat_history": chat_history})
    
    sources = list(set([
        os.path.basename(doc.metadata.get("source", "Gale Encyclopedia of Medicine"))
        for doc in response.get("context", [])
    ]))
    
    # 2. Extract Response Metadata
    ans_raw = response["answer"].strip()
    while ans_raw.endswith("**") or ans_raw.endswith("*"):
        ans_raw = ans_raw.rstrip("*").strip()
    
    severity = "INFO"
    if "SEVERITY:" in ans_raw:
        parts = ans_raw.split("SEVERITY:", 1)
        if len(parts) > 1:
            line = parts[1].split("\n", 1)[0].strip()
            severity = line.strip("[]")
            ans_raw = parts[0].strip() + "\n" + (parts[1].split("\n", 1)[1] if "\n" in parts[1] else "")
    
    followups = []
    if "FOLLOWUPS:" in ans_raw:
        parts = ans_raw.split("FOLLOWUPS:", 1)
        ans_raw = parts[0].strip()
        f_list = parts[1].strip().split("\n")
        followups = [f.strip("- ").strip() for f in f_list if f.strip().startswith("-")]

    sources_text = ""
    if sources:
        sources_text = "\n\nSOURCES:\n" + "\n".join(f"- {s}" for s in sources)
    
    main_ans = ans_raw.strip() + sources_text
    
    # 3. Translate BACK to user language if needed
    if lang != "en":
        try:
            translator = GoogleTranslator(source='en', target=lang)
            main_ans = translator.translate(main_ans)
            translated_followups = []
            for f in followups:
                translated_followups.append(translator.translate(f))
            followups = translated_followups
        except: pass

    final_answer = f"SEVERITY: {severity}\n{main_ans}"
    if followups:
        final_answer += "\n\nFOLLOWUPS:\n" + "\n".join(f"- {f}" for f in followups)

    # Log to DB
    log_query(session_id, chat_id, msg, final_answer, lang)

    # History
    chat_history.append(HumanMessage(content=msg_en))
    chat_history.append(AIMessage(content=ans_raw)) 
    if len(chat_history) > 10:
        chats[chat_key] = chat_history[-10:]
    
    return final_answer

from flask import Response, stream_with_context

@app.route("/stream", methods=["POST"])
def chat_stream():
    msg = request.form.get("msg", "")
    lang = request.form.get("lang", "en")
    chat_id = request.form.get("chat_id", "default")
    session_id = request.remote_addr
    chat_key = f"{session_id}_{chat_id}"
    
    msg_en = msg
    if chat_key not in chats:
        chats[chat_key] = []
    
    chat_history = chats[chat_key]
    
    f1, f2, f3 = "What information can you provide?", "Tell me about common conditions.", "How do I use the body map?"
    
    def generate():
        response_stream = rag_chain.stream({"input": msg_en, "chat_history": chat_history})
        ans_raw_accumulator = []
        context_docs = []
        has_yielded_context = False
        
        for chunk in response_stream:
            if "context" in chunk and not has_yielded_context:
                has_yielded_context = True
                context_docs = chunk["context"]
            
            if "answer" in chunk:
                ans_chunk = chunk["answer"]
                yield f"data: {json.dumps({'text': ans_chunk})}\n\n"
                ans_raw_accumulator.append(ans_chunk)
                
        # Send finalizing structure signal
        ans_str = "".join(ans_raw_accumulator)
        
        sources = list(set([os.path.basename(doc.metadata.get("source", "Gale Encyclopedia of Medicine")) for doc in context_docs]))
        s_text = ("\n\nSOURCES:\n" + "\n".join(f"- {s}" for s in sources)) if sources else ""
        
        if "FOLLOWUPS:" not in ans_str:
            s_text += f"\n\nFOLLOWUPS:\n- {f1}\n- {f2}\n- {f3}"
            
        final = ans_str + s_text
        # We also need to send the full complete formatted text block hidden.
        yield f"data: {json.dumps({'done': True, 'fullText': final})}\n\n"
        
        log_query(session_id, chat_id, msg, final, lang)
        chat_history.append(HumanMessage(content=msg_en))
        chat_history.append(AIMessage(content=ans_str))
        if len(chat_history) > 10:
            chats[chat_key] = chat_history[-10:]

    return Response(stream_with_context(generate()), mimetype="text/event-stream")

@app.route("/symptom-check", methods=["POST"])
def symptom_check():
    symptoms = request.json.get("symptoms", [])
    chat_id = request.json.get("chat_id", "default")
    sid = request.remote_addr
    query = f"Based on the provided medical encyclopedia, what conditions are associated with these symptoms: {', '.join(symptoms)}? Please categorize by likelihood and provide next steps."
    response = rag_chain.invoke({"input": query, "chat_history": []})
    ans = f"SEVERITY: CONSULT_DOCTOR\n" + response["answer"]
    log_query(sid, chat_id, f"SYMPTOM CHECK: {symptoms}", ans, "en")
    return jsonify({"answer": ans})

@app.route("/dashboard")
def dashboard():
    conn = sqlite3.connect('medichat.db')
    cursor = conn.cursor()
    # Top 10 queries
    cursor.execute('SELECT question, COUNT(*) as count FROM queries GROUP BY question ORDER BY count DESC LIMIT 10')
    top_q = cursor.fetchall()
    # Recent 10
    cursor.execute('SELECT question, lang, timestamp FROM queries ORDER BY timestamp DESC LIMIT 10')
    recent = cursor.fetchall()
    # Totals
    cursor.execute('SELECT COUNT(*) FROM queries')
    total = cursor.fetchone()[0]
    conn.close()
    return render_template("dashboard.html", top_queries=top_q, recent=recent, total=total)

@app.route("/export", methods=["POST"])
def export_chat():
    session_id = request.remote_addr
    chat_id = request.form.get("chat_id", "default")
    chat_key = f"{session_id}_{chat_id}"
    chat_history = chats.get(chat_key, [])
    
    # 1. Determine dynamic filename from first prompt
    file_prefix = "MedChat"
    for msg in chat_history:
        if isinstance(msg, HumanMessage):
            raw_text = getattr(msg, 'content', str(msg))
            clean_text = re.sub(r'[^a-zA-Z0-9_\- ]', '', raw_text).strip()
            if clean_text:
                file_prefix = clean_text[:30].replace(' ', '_')
                break
    
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 18)
    pdf.cell(0, 10, "MedChat Consultation Record", ln=True, align='C')
    pdf.ln(10)
    
    pdf.set_font("Arial", size=10)
    for msg in chat_history:
        role = "YOU" if isinstance(msg, HumanMessage) else "MEDICHAT"
        # Access content safely from LangChain message objects
        content = getattr(msg, 'content', str(msg))
        
        # Strip metadata from PDF
        if "SEVERITY:" in content: content = content.split("\n", 1)[1] if "\n" in content else content
        if "FOLLOWUPS:" in content: content = content.split("FOLLOWUPS:")[0]
        
        pdf.set_font("Arial", 'B', 10)
        pdf.cell(0, 8, f"{role}:", ln=True)
        pdf.set_font("Arial", size=10)
        pdf.multi_cell(0, 6, content.strip())
        pdf.ln(5)
    
    output = io.BytesIO()
    pdf.output(output)
    output.seek(0)
    
    return send_file(
        output,
        mimetype="application/pdf",
        as_attachment=True,
        download_name=f"{file_prefix}_Consultation.pdf"
    )

@app.route("/sessions", methods=["GET"])
def get_sessions():
    session_id = request.remote_addr
    conn = sqlite3.connect('medichat.db')
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT chat_id, MIN(timestamp), question FROM queries WHERE session_id = ? GROUP BY chat_id ORDER BY MIN(timestamp) DESC", (session_id,))
        rows = cursor.fetchall()
    except sqlite3.OperationalError:
        rows = []
    conn.close()
    
    sessions = []
    for r in rows:
        c_id = r[0] if r[0] else "legacy"
        if any(s['chat_id'] == c_id for s in sessions):
            continue
        title = r[2][:30] + '...' if len(r[2]) > 30 else r[2]
        if "SYMPTOM CHECK:" in title:
            title = "Symptom Check"
        sessions.append({"chat_id": c_id, "title": title})
            
    return jsonify({"sessions": sessions})

@app.route("/history/<chat_id>", methods=["GET"])
def get_history_by_id(chat_id):
    session_id = request.remote_addr
    chat_key = f"{session_id}_{chat_id}"
    conn = sqlite3.connect('medichat.db')
    cursor = conn.cursor()
    if chat_id == "legacy":
        cursor.execute('SELECT question, answer FROM queries WHERE session_id = ? AND chat_id IS NULL ORDER BY timestamp ASC', (session_id,))
    else:
        cursor.execute('SELECT question, answer FROM queries WHERE session_id = ? AND chat_id = ? ORDER BY timestamp ASC', (session_id, chat_id))
    rows = cursor.fetchall()
    conn.close()
    
    if chat_key not in chats and rows:
        history = []
        for q, a in rows:
            history.append(HumanMessage(content=q))
            if "SEVERITY:" in a:
                history.append(AIMessage(content=a.split("SEVERITY:", 1)[1].split("\n", 1)[-1]))
            else:
                history.append(AIMessage(content=a))
        chats[chat_key] = history[-10:]
        
    return jsonify({"history": [{"q": r[0], "a": r[1]} for r in rows]})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)