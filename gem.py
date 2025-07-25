
# pitchmate_streamlit.py
import time
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
gemini_api_key = st.secrets["gemini"]["api_key"]
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
DB_FILE = "Database.db"
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
def query_data_with_gemini(dataframes, query):
    query_lower = query.lower()

    # Updated mapping with consistent table name and additional keywords
    mapping = {
        "trainers": ["trainer", "coach"],
        "placements": ["placement", "ctc"],
        "companies": ["company", "companies"],
        "dscc_activities": ["dscc"],
        "projects": ["project"],
        "courses": ["program", "course", "curriculum", "dsai", "cfa"]  # Added course codes
    }

    # Determine the selected table
    selected_table = None
    for key, terms in mapping.items():
        if any(term in query_lower for term in terms):
            selected_table = key
            break

    if not selected_table:
        return "Please specify what you're looking for (e.g., trainers, projects, companies, placements, DSCC activities, or Courses)."

    df = dataframes.get(selected_table)
    if df is None or df.empty:
        return f"No data found for {selected_table}."

    # Extract course code if present
    possible_codes = dataframes['courses']['course_code'].tolist() if 'courses' in dataframes else []
    
    found_code = None
    for code in possible_codes:
        if code and code.lower() in query_lower:
            found_code = code
            break
    if not found_code:
        st.warning(f"Did not detect any known course code in the query: {query}")

    # Filter courses table for specific course code queries
    if selected_table == 'courses' and found_code:
        if 'course_code' in df.columns:
            df = df[df['course_code'].str.lower() == found_code.lower()]
            if df.empty:
                return f"No course found with code {found_code}."
        else:
            return "The 'course_code' column is missing from the courses table."

    context = df.to_string(index=False)
    # Prompt Logic
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
        if "top" in query_lower:
            prompt = prompt = f"""You are analyzing top placement insights.

Based on the following placement data:
{context}

Query: {query}

Please return the response in the following format:
1. Top hiring companies like below:
### 📊 Top Hiring Companies (Based on Placement Frequency)

| Company Name | Number of Hires |
|--------------|------------------|
| TCS          | 12               |
| Infosys      | 9                |
| Wipro        | 6                |

(Use actual company names and frequencies from the data above)

---
2. Top students placed at those companies:
### ✅ Successful Hires at Top Hiring Company

Below are students placed in **[Top Company]**:

| Name | Course | Education Background | Role | CTC | Location |
|------|--------|----------------------|------|-----|----------|
| ...  | ...    | ...                  | ...  | ... | ...      |

Please ensure:
- Use markdown tables
- show only top 5 company names and all the students placed in those companies.

3.  *Why This Matters*
   - One paragraph on how this helps learners understand industry trends, set career goals, and prepare for high-opportunity roles.

4.  *Student Success Spotlights*
   - Highlight a few standout placed students with: Name, Role, Company, Short quote/testimonial, LinkedIn link

Keep the tone professional, encouraging, and use markdown formatting with tables and bold headings."""
        else:
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
2. Always remember, top company will be the most frequent hiring company not the top CTC range. when there is two roles for same company, give it in one row.
3. A short section titled "Know about the companies", showing a one line description of each company about tech stack, culture, or projects
4. A short section titled "Why This Info Matters for Learners", showing how students can use this insight for better placement preparation.

Make the summary professional, structured, and markdown formatted."""
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
2. Always remember, top company will be the most frequent hiring company not the top CTC range. when there is two roles for same company, give it in one row.
3. A short section titled "Know about the companies", showing a one line description of each company about tech stack, culture, or projects
4. A short section titled "Why This Info Matters for Learners", showing how students can use this insight for better placement preparation.

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
    elif selected_table == 'courses':
        # Enhanced prompt for course-specific queries
        if "who can enroll" in query_lower or "eligibility" in query_lower:
            prompt = f"""Based on the following course data:
{context}

Query: {query}

Generate a clear and concise response focusing on eligibility for the specified course in a horizontal markdown table so it is consumable.:
1. Title: Full course name
2. Markdown table (horizontal layout) with:
   - Course Code
   - Course Name
   - Description or Objective which you summarised from the data in excel, dont let it be exact.
   - Eligibility criteria (e.g., educational background, skills, or experience required)
   - if you could not find any specific "skills" make a little list from the program modules.
   - Duration
   - Technologies Covered - please done say not defined, just list some skills relevant to the course.
3. A note on who this course is best suited for, talk about past experience, past education and skills to have.
4. A breif catchy point with how the course can change the projection of the students career.
5. Use a professional tone and markdown formatting in a table where appropriate.
"""
        elif "explain" in query_lower or "tell me about" in query_lower or found_code:
            prompt = f"""Based on the following course data:
{context}

Query: {query}

Generate a detailed summary for the specified course(s):
1. If a specific course code (e.g., {found_code or 'any'}) is mentioned, provide:
   - 2. Markdown table (horizontal layout) with:
   - Course Code
   - Course Name
   - Description or Objective which you summarised from the data in excel, dont let it be exact.
   - Eligibility criteria (e.g., educational background, skills, or experience required)
   - if you could not find any specific "skills" make a little list from the program modules.
   - Duration
   - Technologies Covered - please done say not defined, just list some skills relevant to the course.
   - Who this course is best suited for
2. If no specific course is mentioned, provide a table summarizing all courses with the above fields.
3. Include a tip for students on how to choose the right course.

Make the summary professional, markdown formatted, and helpful for students comparing courses."""
        else:
            prompt = f"""Based on the following course data:
{context}

Query: {query}

Generate a clear summary that includes:
1. A table with:
   - course_Code
   - course_Name
   - Description or Objective (summarized, not copied verbatim)
   - Duration
   - Technologies Covered
2. A short section titled "Why This Course?" explaining who the course(s) is best suited for.
3. A tip for students on how to choose the right course.

Make the summary professional, markdown formatted, and helpful for students comparing courses."""
    else:
        prompt = f"""Based on the following data:
{context}

Query: {query}

Please analyze the query and provide a comprehensive, well-formatted response based on the most relevant data. Use tables, bullet points, and clear sections as appropriate."""

    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error processing query: {str(e)}"

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
                start_time = time.time()
                response = query_data_with_gemini(dataframes, query)
                end_time = time.time()

            response_time = round(end_time - start_time, 2)

            st.success(f"🕒 Response time: {response_time} seconds")
            st.markdown(response, unsafe_allow_html=True)
            

            st.session_state.chat_history.append({
                "query": query,
                "response": response
            })
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
