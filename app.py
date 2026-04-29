import os
import json
import sqlite3
import io
import re
import time
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

load_dotenv()  # Must run first so all os.environ.get() calls see .env values locally

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "medical-chatbot-secret-dev")

PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

# Safely set if they were missing from os.environ but present in .env
if PINECONE_API_KEY:
    os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
if GROQ_API_KEY:
    os.environ["GROQ_API_KEY"] = GROQ_API_KEY

# Global storage for lazy-loaded chain
_rag_chain = None

def get_rag_chain():
    global _rag_chain
    if _rag_chain is not None:
        return _rag_chain
        
    print("Initializing Lazy RAG Components...")
    try:
        embeddings = download_embeddings()
        index_name = "medical-chatbot"
        
        docsearch = PineconeVectorStore.from_existing_index(
            index_name=index_name,
            embedding=embeddings
        )
        
        retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3})
        
        llm = ChatGroq(
            model="llama-3.1-8b-instant",
            temperature=0.3,
            max_tokens=600
        )
        
        # Contextualize Question Prompt
        contextualize_q_prompt = ChatPromptTemplate.from_messages([
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])
        
        history_aware_retriever = create_history_aware_retriever(
            llm, retriever, contextualize_q_prompt
        )
        
        # QA Prompt
        qa_prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])
        
        question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
        
        # Store both components in a wrapper or handle logic in the route
        _rag_chain = {
            "retriever": retriever,
            "history_retriever": history_aware_retriever,
            "qa_chain": question_answer_chain,
            "full_chain": create_retrieval_chain(history_aware_retriever, question_answer_chain)
        }
        print("RAG Components Initialized Successfully!")
        return _rag_chain
    except Exception as e:
        print(f"ERROR: Failed to initialize RAG: {e}")
        return None

# Memory store for chat history (Simple global store for demo, ideally use DB)
chats = {}

def setup_db():
    conn = sqlite3.connect('medichat.db')
    cursor = conn.cursor()
    cursor.execute('''CREATE TABLE IF NOT EXISTS queries 
                      (id INTEGER PRIMARY KEY, session_id TEXT, question TEXT, answer TEXT, lang TEXT, 
                       timestamp DATETIME DEFAULT CURRENT_TIMESTAMP, chat_id TEXT, 
                       latency_ms REAL, q_words INTEGER, a_words INTEGER)''')
    try:
        cursor.execute("ALTER TABLE queries ADD COLUMN chat_id TEXT")
    except sqlite3.OperationalError: pass
    try:
        cursor.execute("ALTER TABLE queries ADD COLUMN latency_ms REAL")
    except sqlite3.OperationalError: pass
    try:
        cursor.execute("ALTER TABLE queries ADD COLUMN q_words INTEGER")
    except sqlite3.OperationalError: pass
    try:
        cursor.execute("ALTER TABLE queries ADD COLUMN a_words INTEGER")
    except sqlite3.OperationalError: pass
    conn.commit()
    conn.close()

setup_db()

def log_query(session_id, chat_id, question, answer, lang, latency_ms=0, q_words=0, a_words=0):
    try:
        conn = sqlite3.connect('medichat.db')
        cursor = conn.cursor()
        cursor.execute('''INSERT INTO queries (session_id, chat_id, question, answer, lang, latency_ms, q_words, a_words) 
                          VALUES (?, ?, ?, ?, ?, ?, ?, ?)''', 
                       (session_id, chat_id, question, answer, lang, latency_ms, q_words, a_words))
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"ERROR: Failed to log query to DB: {e}")

def finalize_streaming_response(start_time, msg_en, ans_str, context_docs, lang, session_id, chat_id, chat_history):
    sources = list(set([os.path.basename(doc.metadata.get("source", "Gale Encyclopedia of Medicine")) for doc in context_docs]))
    s_text = ("\n\nSOURCES:\n" + "\n".join(f"- {s}" for s in sources)) if sources else ""
    
    f1, f2, f3 = "What information can you provide?", "Tell me about common conditions.", "How do I use the body map?"
    if "FOLLOWUPS:" not in ans_str:
        s_text += f"\n\nFOLLOWUPS:\n- {f1}\n- {f2}\n- {f3}"
    
    final = ans_str + s_text
    latency = (time.time() - start_time) * 1000
    log_query(session_id, chat_id, msg_en, final, lang, latency_ms=latency, q_words=len(msg_en.split()), a_words=len(ans_str.split()))
    
    chat_history.append(HumanMessage(content=msg_en))
    chat_history.append(AIMessage(content=ans_str))
    return final


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
        log_query(session_id, chat_id, msg, ans, lang, latency_ms=0, q_words=len(msg.split()), a_words=len(ans.split()))
        return ans

    if chat_key not in chats:
        chats[chat_key] = []
    
    chat_history = chats[chat_key]
    
    chains = get_rag_chain()
    if not chains:
        return "SEVERITY: ERROR\nI encountered an issue connecting to the medical database. Please check your deployment logs."

    start_time = time.time()
    
    # PERFORMANCE OPTIMIZATION: If no history, bypass history_aware_retriever to save 1 LLM call
    if not chat_history:
        # Use basic retriever and direct QA chain
        docs = chains["retriever"].invoke(msg_en)
        ans_text = chains["qa_chain"].invoke({"input": msg_en, "chat_history": [], "context": docs})
        response = {"answer": ans_text, "context": docs}
    else:
        # Use full history-aware chain
        response = chains["full_chain"].invoke({"input": msg_en, "chat_history": chat_history})
    end_time = time.time()
    latency = (end_time - start_time) * 1000
    
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
    log_query(session_id, chat_id, msg, final_answer, lang, 
              latency_ms=latency, 
              q_words=len(msg.split()), 
              a_words=len(main_ans.split()))

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
        chains = get_rag_chain()
        if not chains:
            yield f"data: {json.dumps({'text': 'ERROR: The medical database is currently unavailable.'})}\n\n"
            return
            
        start_time = time.time()
        
        # PERFORMANCE OPTIMIZATION for FIRST QUERY
        if not chat_history:
            # Skip history-aware retriever
            docs = chains["retriever"].invoke(msg_en)
            response_stream = chains["qa_chain"].stream({"input": msg_en, "chat_history": [], "context": docs})
            context_docs = docs
            has_yielded_context = True # Already have them
            ans_str_list = []
            for chunk in response_stream:
                yield f"data: {json.dumps({'text': chunk})}\n\n"
                ans_str_list.append(chunk)
            
            # Send done signal
            final_ans = "".join(ans_str_list)
            # Finish processing normally
            full_final_text = finalize_streaming_response(start_time, msg_en, final_ans, context_docs, lang, session_id, chat_id, chat_history)
            yield f"data: {json.dumps({'done': True, 'fullText': full_final_text})}\n\n"
            return 
        else:
            response_stream = chains["full_chain"].stream({"input": msg_en, "chat_history": chat_history})
            context_docs = []
            has_yielded_context = False
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
        final = finalize_streaming_response(start_time, msg_en, ans_str, context_docs, lang, session_id, chat_id, chat_history)
        
        # We also need to send the full complete formatted text block hidden.
        yield f"data: {json.dumps({'done': True, 'fullText': final})}\n\n"
        
        if len(chat_history) > 10:
            chat_key = f"{session_id}_{chat_id}"
            chats[chat_key] = chat_history[-10:]

    return Response(stream_with_context(generate()), mimetype="text/event-stream")

@app.route("/symptom-check", methods=["POST"])
def symptom_check():
    symptoms = request.json.get("symptoms", [])
    chat_id = request.json.get("chat_id", "default")
    sid = request.remote_addr
    query = f"Based on the provided medical encyclopedia, what conditions are associated with these symptoms: {', '.join(symptoms)}? Please categorize by likelihood and provide next steps."
    chains = get_rag_chain()
    if not chains:
        return jsonify({"answer": "SEVERITY: ERROR\nThe medical database is currently unavailable."})

    start_time = time.time()
    response = chains["full_chain"].invoke({"input": query, "chat_history": []})
    latency = (time.time() - start_time) * 1000
    ans = f"SEVERITY: CONSULT_DOCTOR\n" + response["answer"]
    log_query(sid, chat_id, f"SYMPTOM CHECK: {symptoms}", ans, "en", latency_ms=latency, q_words=len(query.split()), a_words=len(ans.split()))
    return jsonify({"answer": ans})

@app.route("/dashboard")
def dashboard():
    conn = sqlite3.connect('medichat.db')
    cursor = conn.cursor()
    
    # 1. Basic Stats
    cursor.execute('SELECT COUNT(*) FROM queries')
    total = cursor.fetchone()[0]
    
    cursor.execute('SELECT AVG(latency_ms) FROM queries WHERE latency_ms > 0')
    avg_latency = cursor.fetchone()[0] or 0
    
    cursor.execute('SELECT AVG(q_words), AVG(a_words) FROM queries')
    avg_q_len, avg_a_len = cursor.fetchone()
    
    # 2. Languages
    cursor.execute('SELECT lang, COUNT(*) FROM queries GROUP BY lang')
    langs = cursor.fetchall()
    
    # 3. Top Queries
    cursor.execute('SELECT question, COUNT(*) as count FROM queries GROUP BY question ORDER BY count DESC LIMIT 10')
    top_q = cursor.fetchall()
    
    # 4. Recent Interactions
    cursor.execute('SELECT question, lang, timestamp, latency_ms FROM queries ORDER BY timestamp DESC LIMIT 10')
    recent = cursor.fetchall()
    
    conn.close()
    
    metrics = {
        "total": total,
        "avg_latency": f"{avg_latency:.2f}ms",
        "avg_q_len": f"{avg_q_len or 0:.1f}",
        "avg_a_len": f"{avg_a_len or 0:.1f}",
        "langs": langs
    }
    
    return render_template("dashboard.html", top_queries=top_q, recent=recent, metrics=metrics)

@app.route("/export", methods=["POST"])
def export_chat():
    try:
        session_id = request.remote_addr
        chat_id = request.form.get("chat_id", "default")
        chat_key = f"{session_id}_{chat_id}"
        chat_history = chats.get(chat_key, [])
        
        if not chat_history:
            return "No chat history found for this session to export.", 400

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
        
        # Try to set a common font
        try:
            pdf.set_font("Helvetica", 'B', 18)
        except:
            pdf.set_font("Arial", 'B', 18)
            
        pdf.cell(0, 10, "MedChat Consultation Record", ln=True, align='C')
        pdf.ln(10)
        
        for msg in chat_history:
            role = "YOU" if isinstance(msg, HumanMessage) else "MEDICHAT"
            content = getattr(msg, 'content', str(msg))
            
            # Metadata cleanup
            if "SEVERITY:" in content: content = content.split("\n", 1)[1] if "\n" in content else content
            if "FOLLOWUPS:" in content: content = content.split("FOLLOWUPS:")[0]
            if "SOURCES:" in content: content = content.split("SOURCES:")[0]
            
            # Clean text for Latin-1 (standard PDF fonts)
            # Replace smart quotes and other common symbols that crash FPDF
            content = content.replace('\u2013', '-').replace('\u2014', '-') \
                             .replace('\u2018', "'").replace('\u2019', "'") \
                             .replace('\u201c', '"').replace('\u201d', '"') \
                             .replace('\u2022', '*').replace('\u2026', '...')
            
            # Final fallback for any other non-latin-1 chars
            content = content.encode('latin-1', 'replace').decode('latin-1')

            pdf.set_font(pdf.font_family, 'B', 10)
            pdf.cell(0, 8, f"{role}:", ln=True)
            pdf.set_font(pdf.font_family, size=10)
            pdf.multi_cell(0, 6, content.strip())
            pdf.ln(5)
        
        pdf_bytes = pdf.output()
        
        return send_file(
            io.BytesIO(pdf_bytes),
            mimetype="application/pdf",
            as_attachment=True,
            download_name=f"{file_prefix}_Consultation.pdf"
        )
    except Exception as e:
        print(f"PDF Export Error: {e}")
        return f"Failed to generate PDF: {str(e)}", 500

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
    app.run(host="0.0.0.0", port=8080, debug=False)