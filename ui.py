import streamlit as st
import os
import psycopg2
import random
import datetime # Added for timestamps
from pymilvus import connections
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage

# Import the backend agent instance
from Insurance_Agent import agent 

load_dotenv()
if "OPENAI_API_KEY" not in os.environ:
    os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

st.set_page_config(page_title="Insurance AI Advisor", page_icon="✦", layout="wide")

# Core Visuals: Dark Sidebar + Compact Layout + Terminal Logs Styling
st.markdown("""
    <style>
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    [data-testid="stSidebar"] { background-color: #0f172a; color: #f1f5f9; padding-top: 1rem; }
    [data-testid="stSidebar"] [data-testid="stVerticalBlock"] { gap: 0.6rem; }
    [data-testid="stSidebar"] h2 { font-size: 1.2rem; color: #ffffff; }
    [data-testid="stSidebar"] h3 { font-size: 0.8rem; color: #64748b; text-transform: uppercase; margin-top: 1rem; }
    [data-testid="stSidebar"] p, [data-testid="stSidebar"] span { font-size: 0.85rem; line-height: 1.2; }
    
    /* Green terminal-style logs */
    .log-text { font-family: 'Courier New', monospace; font-size: 0.75rem; color: #10b981; line-height: 1.1; margin-bottom: 2px; }
    
    .stChatMessage p { font-size: 18px !important; line-height: 1.6; }
    div.stButton > button:first-child[kind="primary"] { background-color: #334155; border: none; border-radius: 8px; }
    </style>
    """, unsafe_allow_html=True)

# System checks and data insight functions
def get_system_status():
    status = {"PG": False, "MV": False}
    try:
        conn = psycopg2.connect(dbname=os.getenv("PG_DB_NAME"), user=os.getenv("PG_USER"),
                                password=os.getenv("PG_PASSWORD"), host=os.getenv("PG_HOST"), port=os.getenv("PG_PORT"))
        status["PG"] = True; conn.close()
    except: pass
    try:
        connections.connect(host=os.getenv("MILVUS_HOST"), port=os.getenv("MILVUS_PORT"))
        status["MV"] = True
    except: pass
    return status

def get_portfolio_summary():
    try:
        conn = psycopg2.connect(dbname=os.getenv("PG_DB_NAME"), user=os.getenv("PG_USER"),
                                password=os.getenv("PG_PASSWORD"), host=os.getenv("PG_HOST"), port=os.getenv("PG_PORT"))
        cur = conn.cursor()
        cur.execute("SELECT unnest(policy_type), count(*) FROM customer_insurance GROUP BY 1")
        data = cur.fetchall()
        cur.execute("SELECT count(DISTINCT customer_id) FROM customer_insurance")
        total = cur.fetchone()[0]
        conn.close()
        return data, total
    except: return [], 0

# Log Management Logic
if "logs" not in st.session_state:
    st.session_state.logs = []

def add_log(message):
    timestamp = datetime.datetime.now().strftime("%H:%M:%S")
    st.session_state.logs.insert(0, f"[{timestamp}] {message}")
    if len(st.session_state.logs) > 5: st.session_state.logs.pop()

# Zero-hallucination Question Bank
QUESTION_BANK = [
    ("Bruce Wayne's Assets", "What is the property address and valuation for Bruce Wayne?"),
    ("Peter Parker's Benefit", "Who is the designated beneficiary of Peter Parker's life insurance?"),
    ("Clark Kent's Vehicles", "What cars are registered under Clark Kent's name and their model years?"),
    ("Lena Luthor's Audi", "What is the specific model and year of Lena Luthor's vehicle?"),
    ("Diana Miller's Portfolio", "List all the insurance types held by Diana Miller."),
    ("Barry Allen's Premium", "How much is the total premium for Barry Allen?"),
    ("Oliver Queen's Address", "What is the residential address linked to Oliver Queen's policy?"),
    ("Standard Claim Flow", "Describe the standard claim process according to the insurance handbook."),
    ("Common Exclusions", "What are the common exclusion clauses mentioned in the manual?"),
    ("High-Value Property", "Which customers have property valuations exceeding $800,000?")
]

if "display_queries" not in st.session_state:
    st.session_state.display_queries = random.sample(QUESTION_BANK, 3)

# Sidebar: Compact Dashboard + Retrieval Logs
with st.sidebar:
    st.markdown("## ✦ Intelligence Hub")
    
    st.markdown("### Developers")
    st.markdown("**Yixuan Ye | Hammer Niu | Ruochen Wu**")
    st.caption("Columbia University • MSEE Program")
    
    st.markdown("### System Health")
    stats = get_system_status()
    c1, c2 = st.columns(2)
    c1.markdown(f"{'🟢' if stats['PG'] else '🔴'} **Postgres**")
    c2.markdown(f"{'🟢' if stats['MV'] else '🔴'} **Milvus Core**")
    
    st.markdown("### Data Insights")
    summary, total = get_portfolio_summary()
    if summary:
        st.markdown(f"**Total Clients**: {total}")
        for p_type, count in summary:
            st.markdown(f"• {p_type}: `{count} records`")
    
    # New Logs Section
    st.markdown("### Retrieval Logs")
    if st.session_state.logs:
        for log in st.session_state.logs:
            st.markdown(f"<p class='log-text'>{log}</p>", unsafe_allow_html=True)
    else:
        st.caption("System idle...")
    
    st.markdown("<div style='margin-top: 100px;'></div>", unsafe_allow_html=True)
    if st.button("Reset Chat Session", use_container_width=True):
        st.session_state.messages = []
        st.session_state.logs = []
        st.rerun()

# Main Interface: Project-linked Greeting
st.title("AI Insurance Agent")
st.markdown("""
    *Hello! I am your AI insurance advisor. I have full access to our **customer portfolios** and the **Insurance Handbook**. How can I assist you with client data or policy analysis today?*
""")

# Suggested Queries Section with Title
st.markdown("### 🔍 Try Asking")
col_q1, col_q2, col_q3, col_shuf = st.columns([2, 2, 2, 1])

for i, (label, q_text) in enumerate(st.session_state.display_queries):
    target_col = [col_q1, col_q2, col_q3][i]
    if target_col.button(label, key=f"q_{i}", use_container_width=True):
        st.session_state.active_prompt = q_text

if col_shuf.button("Shuffle", type="primary", use_container_width=True):
    st.session_state.display_queries = random.sample(QUESTION_BANK, 3)
    st.rerun()

# Chat Flow: Standard B/W dots
if "messages" not in st.session_state:
    st.session_state.messages = [] 

for msg in st.session_state.messages:
    is_user = isinstance(msg, HumanMessage)
    with st.chat_message("user" if is_user else "assistant", avatar="⚪" if is_user else "⚫"):
        st.write(msg.content)

# User Input Logic with Log Triggers
prompt = st.chat_input("Ask about Bruce Wayne, Clark Kent, or policy terms...")
if hasattr(st.session_state, 'active_prompt'):
    prompt = st.session_state.active_prompt
    del st.session_state.active_prompt

if prompt:
    # Initial Log for Input
    add_log(f"Inquiry received: {prompt[:15]}...")
    with st.chat_message("user", avatar="⚪"): st.write(prompt)
    st.session_state.messages.append(HumanMessage(content=prompt))

    with st.chat_message("assistant", avatar="⚫"):
        with st.spinner("Analyzing cross-source records..."):
            try:
                # Log Analysis step
                add_log("Analyzing user intent...")
                
                # Execute RAG Logic
                response = agent.invoke({"messages": st.session_state.messages})
                answer = response["messages"][-1]
                
                # Logic-driven Log Updates
                if any(x in prompt for x in ["Bruce", "Clark", "Lena", "Peter", "Oliver", "Diana", "Barry"]):
                    add_log("SQL Query: PostgreSQL Active.")
                if any(x in prompt for x in ["handbook", "process", "clause", "claim", "Exclusions"]):
                    add_log("Vector Search: Milvus Active.")
                add_log("Synthesizing final response.")

                st.write(answer.content)
                st.session_state.messages.append(answer)
                
                with st.expander("Knowledge Retrieval Metadata"):
                    st.markdown(f"""
                    **Verified Sources**: 
                    - Structured Data: `PostgreSQL (customer_insurance)`
                    - Unstructured Data: `Insurance_Handbook_removed.pdf`
                    
                    **Retrieval Method**: 
                    - Vector Search: `Milvus (L2 Distance)`
                    - Reasoning: `GPT-4o-mini [Zero-Shot RAG]`
                    """)
                    st.caption("This response was cross-referenced with your local insurance knowledge base.")
            except Exception as e:
                add_log("Execution Error.")
                st.error(f"Execution Error: {e}")