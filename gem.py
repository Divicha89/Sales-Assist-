# pitchmate_streamlit.py
import streamlit as st
import pandas as pd
import sqlite3
import google.generativeai as genai

# --- Page Config ---
st.set_page_config(page_title="PitchMate", page_icon="🪟", layout="wide")
st.image("Logo.png", width=300)

# --- CSS Styling ---
st.markdown("""<style>
body { background-color: #121212; }
[data-testid="stSidebar"] { background-color: #1e1e1e; color: white; }
.sidebar-title { font-size: 24px; font-weight: bold; padding-bottom: 10px; color: #00ffb3; }
.st-emotion-cache-1c7y2kd, .st-emotion-cache-br351g, .st-emotion-cache-16tyu1, .st-emotion-cache-1xulwhk { color: white; }
.faq { padding: 10px 0; font-size: 15px; border-bottom: 1px solid #333; }
.chat-box { background-color: #1f1f1f; padding: 30px; border-radius: 10px; color: white; }
.stApp { background-color: #1c1c1e; color: #f2f2f2; }
html, body, [class*="css"] { color: #7e5a9b !important; }
.st-emotion-cache-1y34ygi, .st-emotion-cache-6qob1r { background: #27272f; color: #ffffff; }
header, .css-18ni7ap { background-color: #121212 !important; color: #e0e0e0 !important; }
.stTextInput input { background-color: #2b2b2b; color: white; border-radius: 8px; }
.stButton > button { background-color: #00ffb3; color: black; border-radius: 8px; padding: 8px 20px; }
h1, h2, h3, h4, h5 { color: white; }
</style>""", unsafe_allow_html=True)

# --- Gemini API Key Setup ---
gemini_api_key = "AIzaSyC9o79E9lW0eiCAxgHc8Xe0oGUPRK7dNd8"
if not gemini_api_key:
    st.error("Please provide a Gemini API key.")
    st.stop()

genai.configure(api_key=gemini_api_key)
model = genai.GenerativeModel('gemini-2.5-flash')

# --- Sidebar UI ---
st.sidebar.image("Imarticus.png", use_container_width=True)

# --- FAQ Section ---
st.sidebar.markdown("Frequently Asked Prompts")
faq_questions = [
    "Who are the trainers from Hyderabad",
    "List all data science coaches",
    "What are all the companies hiring",
    "Are there any students placed as data engineer",
    "Which companies offer high ctc",
    "What are top placements",
    "List all dscc activities",
    "What DSCC activities are held monthly",
    "what are the projects are in the curriculum",
    "List all machine learning projects",
]
options = ["FAQs"] + faq_questions

def handle_faq_selection():
    selected = st.session_state.faq_selectbox
    if selected != "FAQs":
        st.session_state.user_input = selected
        st.session_state.process_faq = True

selected = st.sidebar.selectbox("FAQs", options, key="faq_selectbox", on_change=handle_faq_selection)

# --- SQLite DB Integration ---
DB_FILE = "courses_data.db"
@st.cache_resource
def init_db():
    conn = sqlite3.connect(DB_FILE)
    tables = ['trainers', 'placements', 'companies', 'dscc_activities', 'projects', 'courses']
    dataframes = {}
    for table in tables:
        try:
            df = pd.read_sql_query(f"SELECT * FROM {table}", conn)
            df.columns = [col.lower().strip() for col in df.columns]
            dataframes[table] = df
        except Exception as e:
            st.error(f"Error loading table '{table}': {str(e)}")
    conn.close()
    return dataframes

# --- Chat Functionality ---
def query_data_with_gemini(dataframes, query):
    query_lower = query.lower()

    mapping = {
        "trainers": ["trainer", "coach"],
        "placements": ["placement", "ctc"],
        "companies": ["company", "companies"],
        "dscc_activities": ["dscc"],
        "projects": ["project"]
    }
    selected_table = None
    for key, terms in mapping.items():
        if any(term in query_lower for term in terms):
            selected_table = key
            break

    if not selected_table:
        return "Please specify what you're looking for (trainer, project, company, placement, or DSCC activity)."

    df = dataframes.get(selected_table)
    if df is None or df.empty:
        return f"No data found for {selected_table}."

    possible_codes = dataframes['courses']['course_code'].tolist()
    found_code = next((code for code in possible_codes if code.lower() in query_lower), None)

    if found_code and 'course' in df.columns:
        df = df[df['course'].str.lower() == found_code.lower()]

    context = df.to_string(index=False)

    # --- Prompt Logic from gem.py ---
    if selected_table == 'trainers':
        prompt = f"""Based on the following trainer data:
{context}

Query: {query}

Please provide a response in the following structured format:

<h3 style="font-size:25px; font-weight:bold;">Trainer Profile: [Full Name]</h3>

| Field              | Details                                                                 |
|--------------------|-------------------------------------------------------------------------|
| Role               | [Trainer's Role]                                                        |
| Course             | [Course]                                                                |
| Experience         | [Experience summary]                                                    |
| Skill Set          | Expertise: [Key expertise areas]<br>Tools: [Key tools/languages used]  |
| Strengths          | [Trainer's strengths]                                                   |
| Location           | [Trainer's location]                                                    |
| LinkedIn Profile   | [Hyperlink if available, else write 'Not Available']                   |

2. Add a section about panel trainers at the end with below details and same format as above:
['Nikita Tandel', 'AVP, Data Science training','DSAI', 'Machine Learning, Statistical Analysis', 'Python, R, SQL, Tableau', 'Advanced statistical modeling and insights generation', 'Led 50+ ML projects, specialized in healthcare analytics', 'Pan-India', 'https://linkedin.com/in/nikita'],
['Karthik', 'VP, Head of Data Science','DSAI', 'Deep Learning, NLP', 'Python, TensorFlow, PyTorch', 'Advanced AI research and practical implementation', 'PhD in AI, published 25+ research papers, 10+ years industry experience', 'Pan-India', 'https://linkedin.com/in/karthik']

Please ensure:
- Use proper markdown table styling.
- Use <br> tags within the table to separate lines (as shown in Skill Set).
- Include only trainers relevant to the query.
- Use a simple and clean tone.
"""
    elif selected_table == 'placements':
        prompt = f"""Based on the following placement data:
{context}

Query: {query}

Please provide a well-formatted response that includes:
1. A clear answer/summary at the top (e.g., "Top CTC record: X LPA")
2. A table with Name, Course, Education Background, Company, Role, CTC, Location, Total Hires (sum of hires for each company)
3. Inspiring student stories at the end with LinkedIn links, testimonial prompts

Make it professional, structured, and motivational."""
    elif selected_table == 'companies':
        prompt = f"""You are helping students understand which companies are hiring.

Based on the following company recruitment data:
{context}

Query: {query}

Generate a detailed summary including:
1. A table with:
   - Company Name
   - Roles Offered
   - CTC Range
   - Hiring Frequency
   - Key Requirements
2. A short section titled "Know about the companies", showing a one line description of each company about tech stack, culture, or projects
3. A short section titled "Why This Info Matters for Learners", showing how students can use this insight for better placement preparation.

Make the summary professional, structured, and markdown formatted."""
    elif selected_table == 'projects':
        prompt = f"""Based on the following project data:
{context}

Query: {query}

Please provide a well-formatted response that includes:
1. A comprehensive table with Project Title, Domain, Technologies Used, and Skill Level
2. A section highlighting top problem statements explored by students
3. Information about how these projects enhance placement readiness

Make it professional and inspiring for students."""
    elif selected_table == 'dscc_activities':
        prompt = f"""Based on the following DSCC activities data:
{context}

Query: {query}

Please provide a well-formatted response that includes:
1. A comprehensive overview of DSCC activities
2. A table with Event Name, Key Highlights, and Timeline
3. A section explaining why these activities matter for students
4. Information about real-world exposure and career benefits

Make it engaging and motivational."""
    else:
        prompt = f"""Based on the following data:
{context}

Query: {query}

Please analyze the query and provide a comprehensive, well-formatted response based on the most relevant data. Use tables, bullet points, and clear sections as appropriate."""

    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error: {e}"

# --- UI Main ---
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

tab1, tab2 = st.tabs(["Main", "Chat History"])
with tab1:
    dataframes = init_db()
    query = st.text_input("What do you wanna know", key="user_input")
    if st.button("Ask"):
        st.session_state["last_query"] = query
        if query.strip():
            with st.spinner("Processing your query..."):
                response = query_data_with_gemini(dataframes, query)
            st.markdown(response, unsafe_allow_html=True)
            st.session_state.chat_history.append({"query": query, "response": response})
        else:
            st.warning("Please enter a query.")

with tab2:
    st.write("### Chat History")
    for i, entry in enumerate(reversed(st.session_state.chat_history), 1):
        with st.expander(f"Q{i}: {entry['query']}"):
            st.markdown(entry['response'], unsafe_allow_html=True)

if st.session_state.chat_history:
    if st.button("Clear Chat History"):
        st.session_state.chat_history = []
        st.rerun()
