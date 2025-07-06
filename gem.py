import streamlit as st
import pandas as pd
import re
import google.generativeai as genai  # Import Google Generative AI library

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

# --- Sidebar UI ---
st.sidebar.image("Imarticus.png", use_container_width=True)
function = st.sidebar.selectbox("Select a Category", [
    "Trainer Details",
    "Placement Stats",
    "Company Info",
    "DSCC Activities",
    "Project Showcase"
])

# FAQ Section
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

selected = st.sidebar.selectbox(
    "FAQs", 
    options, 
    key="faq_selectbox",
    on_change=handle_faq_selection)

# --- Gemini API Key Setup ---
gemini_api_key = "AIzaSyDfuo-L0KOU1zZAsABl6Ltajy13EgvBpMc"  # Replace with your actual Gemini API key
if not gemini_api_key:
    st.error("Please provide a Gemini API key.")
    st.stop()

# Initialize Gemini client
genai.configure(api_key=gemini_api_key)
model = genai.GenerativeModel('gemini-1.5-flash')

# --- Excel File Setup ---
@st.cache_resource
def init_db():
    excel_files = {
        'trainers': 'trainers.xlsx',
        'placements': 'Placements.xlsx',
        'companies': 'companies.xlsx',
        'dscc_activities': 'dscc_activities.xlsx',
        'projects': 'projects.xlsx',
    }
    dataframes = {}
    for table, file_path in excel_files.items():
        try:
            df = pd.read_excel(file_path, engine='openpyxl')
            # Normalize column names to lowercase
            df.columns = [col.lower() for col in df.columns]

            if table == 'dscc_activities':
                df['activity'] = df.get('activity', pd.Series()).astype(str)
                df['agenda'] = df.get('agenda', pd.Series()).astype(str)
                df['cadence'] = df.get('cadence', pd.Series()).str.lower().str.strip()
            if table == 'projects':
                df['project_title'] = df.get('project_title', pd.Series()).astype(str)
                df['difficulty_level'] = df.get('difficulty_level', pd.Series()).str.lower().str.strip()
                df['topic'] = df.get('topic', pd.Series()).str.lower().str.strip()
            if table == 'companies':
                df['company_name'] = df.get('company_name', pd.Series()).astype(str)
                df['role'] = df.get('logos/role', pd.Series()).str.lower().str.strip()
                df['ctc_range'] = df.get('ctc_range', pd.Series()).str.extract(r'(\d+\.?\d*)').astype(float)
                df['hiring_frequency'] = df.get('hiring_frequency', pd.Series()).str.lower().str.strip()
            if table == 'trainers':
                df['name'] = df.get('name', pd.Series()).astype(str)
                df['location'] = df.get('location', pd.Series()).str.lower().str.strip()
                df['profile'] = df.get('profile', pd.Series()).str.lower().str.strip()
                df['skillset'] = df.get('skillset', pd.Series()).str.lower().str.strip()
            if table == 'placement_results':
                df['name'] = df.get('name', pd.Series()).astype(str)
                df['location'] = df.get('location', pd.Series()).str.lower().str.strip()
                df['role'] = df.get('role', pd.Series()).str.lower().str.strip()
                df['ctc'] = df.get('ctc', pd.Series()).str.extract(r'(\d+\.?\d*)').astype(float)
            dataframes[table] = df
        except FileNotFoundError as e:
            st.error(f"Missing file: {e}")
        except KeyError as e:
            st.error(f"Column error in {table} file: {e}. Please check column names.")
    return dataframes

# --- Updated Filters ---
def extract_filters(query):
    query = query.lower()
    patterns = {
        'domain': r"\b(data science|machine learning|nlp|visualization|social media|data analytics)\b",
        'frequency': r"\b(monthly|bimonthly|bi weekly|quarterly|half-yearly)\b",
        'location': r"from\s+(\w+)",
        'profile': r"\b(data science|machine learning|ai|artificial intelligence|python|analytics|coach|mentor|expert)\b",
        'hiring_frequency': r"\b(monthly|quarterly|bi weekly|bimonthly|half-yearly|bi-monthly)\b",
        'skillset': r"\b(skilled in|expertise|skilled with|for)\b",
        'role': r"\b(data scientist|ai engineer|machine learning engineer|business analyst|.*developer.*)\b",
        'highest_ctc': r"highest ctc is\s*(\d+\.?\d*)\s*lpa",
        'minimum_ctc': r"minimum ctc is\s*(\d+\.?\d*)\s*lpa"
    }
    
    filters = {}
    for key, pattern in patterns.items():
        match = re.search(pattern, query)
        if match:
            if key == 'ctc_range':
                filters['ctc_min'] = match.group(1)
                filters['ctc_max'] = match.group(2)
            else:
                filters[key] = match.group(1) if match.group(1) else match.group(2) if match.group(2) else None
    return filters

# --- Query Data for Non-LLM Tables ---
def query_data_non_llm(dataframes, table, filters, columns):
    df = dataframes.get(table, pd.DataFrame())
    if df.empty:
        st.write(f"Debug: {table} DataFrame is empty.")
        return []

    if table == 'trainers':
        if filters.get('location'):
            df = df[df['location'].str.contains(filters['location'], case=False, na=False)]
        if filters.get('profile'):
            df = df[df['profile'].str.contains(filters['profile'], case=False, na=False)]

    elif table == 'projects':
        if filters.get('difficulty_level') and 'difficulty_level' in df.columns:
            df = df[df['difficulty_level'].str.contains(filters['difficulty_level'], case=False, na=False)]
        if filters.get('topic') and 'topic' in df.columns:
            keyword = filters['topic'].lower()
            df = df[df['topic'].apply(lambda x: keyword in str(x).lower())]

    elif table == 'dscc_activities':
        if filters.get('hiring_frequency') == 'monthly' and 'cadence' in df.columns:
            df = df[df['cadence'] == 'monthly']

    return df[columns].values.tolist()

def query_data_with_llm(dataframes, table, query, columns):
    df = dataframes.get(table, pd.DataFrame())
    if df.empty:
        return "No data available to process."

    query_lower = query.lower()

    # ------------------------------
    # CASE 1: Placement Query → Full LLM
    # ------------------------------
    if "placement" in query_lower or "ctc" in query_lower:
        context = df.to_string(index=False)
        prompt = f"""Based on the following placement data:
{context}

Query: {query}

Please provide a well-formatted response that includes:
1. A clear answer/summary at the top (e.g., "Top CTC record: X LPA")
2. A table with Name, Education Background, Company, Role, CTC, Location
3. Inspiring student stories at the end with LinkedIn links, testimonial prompts

Make it professional, structured, and motivational."""
        response = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                max_output_tokens=1500,
                temperature=0.7
            )
        )
        return response.text

    # ------------------------------
    # CASE 2: Company Query → Filtered Table + LLM
    # ------------------------------
    elif "company" in query_lower or "companies" in query_lower:
        # Step 1: Extract filters
        filters = extract_filters(query)

        # Step 2: Apply filters manually
        if filters.get('role'):
            df = df[df['role'].str.contains(filters['role'], case=False, na=False)]
        if filters.get('hiring_frequency'):
            df = df[df['hiring_frequency'].str.contains(filters['hiring_frequency'], case=False, na=False)]
        if 'minimum_ctc' in filters:
            df = df[df['ctc_range'] >= float(filters['minimum_ctc'])]
        if 'highest_ctc' in filters:
            df = df[df['ctc_range'] <= float(filters['highest_ctc'])]

        if df.empty:
            return "No companies matched your filters."

        # Step 4: Pass filtered data to LLM
        context = df.to_string(index=False)
        prompt = f"""You are helping students understand which companies are hiring.

Based on the following company recruitment data:
{context}

Query: {query}
⚠️ VERY IMPORTANT:
- Display **all matching rows** in the table, **do not skip or summarize** any.
- The table should include **every filtered result**, even if there are many.
- Keep formatting clean and consistent.

Generate a detailed summary including:
1. A table with:
   - Company Name
   - Roles Offered
   - CTC Range
   - Hiring Frequency
   - Key Requirements
2. In a short section titled : *Know about the companies**, showing a one line description of each company descriptions about tech stack, culture, or projects
3. A short section titled: **Why This Info Matters for Learners**, showing how students can use this insight for better placement preparation.

Make the summary professional, structured, and markdown formatted."""
        response = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                max_output_tokens=1500,
                temperature=0.7
            )
        )
        return response.text

    # ------------------------------
    # Fallback
    # ------------------------------
    else:
        return "Query not recognized for LLM. Try asking about placements or companies."

# --- Formatting with ollama ---
def format_with_ollama(data, query, function):
    table_configs = {
        'DSCC Activities': {
            'fields': ['name of event', 'agenda', 'cadance'],  # Note: matching CSV column names exactly
            'prompt': """## 🎯 DSCC Activities: Transforming Learners into Industry-Ready Professionals

Here's a comprehensive overview of all DSCC activities designed to enhance your learning journey:

| Event Name | Key Highlights | Timeline |
|------------|----------------|----------|
{rows}

### 🌟 **Why These Activities Matter:**
- **Real-World Exposure**: Connect classroom learning with industry practices
- **Skill Enhancement**: Develop practical skills through hands-on competitions and projects
- **Networking**: Interact with industry experts and peers across India
- **Career Boost**: Previous participants have achieved 100% placement success with average packages of 6 LPA
- **Confidence Building**: Gain the confidence to tackle complex business problems

*These activities are your gateway to becoming industry-ready data science professionals!*"""
        },
        'Project Showcase': {
            'fields': ['project_title', 'domain', 'topic', 'difficulty_level'],
            'prompt': """#### Student Project Showcase

| Project Title | Domain | Technologies Used | Skill Level |
|---------------|--------|-------------------|-------------|
{rows}

 *These hands-on projects reflect students' ability to apply theoretical knowledge to solve real-world problems. They not only strengthen their technical foundation but also significantly enhance their placement readiness by building confidence, portfolio strength, and interview performance.*

 **Top Problem Statements Explored by Our Students**:
1. Predict Bitcoin prices using historical data (Crypto Price Prediction)
2. Analyze Twitter sentiment around a specific brand (Sentiment Analysis)
3. Extract relevant information from resumes in PDF format (Resume Parser)
4. Analyze customer purchase behavior using sales data (EDA)
5. Develop a system to detect credit card fraud (Financial Fraud Detection)
6. Create a sales dashboard to track revenue and customer growth (BI Dashboards)

*Each of these projects showcases our students’ readiness for real-world business challenges and plays a crucial role in shaping job-winning resumes.*
"""
        },    
        'Company Info': {
            'fields': ['company_name', 'role', 'ctc_range', 'hiring_frequency', 'requirements'],
            'prompt': """#### Companies Hiring Our Graduates

| Company Name | Roles Available | CTC Range | Hiring Frequency | Key Requirements |
|--------------|-----------------|-----------|------------------|------------------|
{rows}

---

🔍 Company Summaries  
{summaries}"""
        },
        'Trainer Details': {
            'fields': ['name', 'profile', 'strengths', 'skillset', 'experience', 'location', 'linkedin'],
            'prompt': """👨‍🏫 Trainer Profile: {0}

| Field | Details |
|-------|---------|
| Role | {1} |
| Experience | {4} |
| Skill Set | {3} |
| Strengths | {2} |
| Linkedin Profile | [Here]({6}) |""",
            'default_trainers': [
                ['Nikita Tandel', 'AVP, Data Science training', 'Machine Learning, Statistical Analysis', 'Python, R, SQL, Tableau', '5+ years', 'Hyderabad', 'https://linkedin.com/in/nikita'],
                ['Karthik', 'VP, Head of Data Science', 'Deep Learning, NLP', 'Python, TensorFlow, PyTorch', '7+ years', 'Bangalore', 'https://linkedin.com/in/karthik']
            ]
        },
        'Placement Results': {
            'fields': ['name', 'education', 'company', 'role', 'ctc', 'location'],
            'prompt': """🎓 Placement Results

| Name | Education | Company | Role | CTC | Location |
|------|-----------|---------|------|-----|----------|
{rows}

These results showcase the successful career paths of students across various domains and locations."""
        }
    }

    config = table_configs.get(function)
    if not config:
        return "Unsupported function."

    if function == 'Company Info':
        rows = [f"| {row[0]} | {row[1]} | {row[2]} LPA | {row[3]} | {row[4]} |" for row in data]
        return config['prompt'].format(rows='\n'.join(rows))

    elif function == 'Trainer Details':
        # Panel trainers to show at the end
        panel_trainers = config.get('default_trainers', [])
        
        # Show actual search results first, then panel trainers at the end (avoiding duplicates)
        formatted_results = [config['prompt'].format(*row) for row in data if row[0].lower() not in ['nikita', 'karthik']]
        formatted_panel = [config['prompt'].format(*row) for row in panel_trainers]
        
        result = ""
        
        # Show search results first if any
        if formatted_results:
            result += "\n\n" + '\n---\n\n'.join(formatted_results)
        
        # Always show panel trainers at the end
        if formatted_panel:
            if result:  # If there were search results, add separator
                result += "\n\n---\n\n"
            result += "## 🎯 Panel Trainers\n\n" + '\n---\n\n'.join(formatted_panel)
        
        return result

    elif function == 'Placement Results':
        rows = [f"| {' | '.join(map(str, row))} |" for row in data]
        return config['prompt'].format(rows='\n'.join(rows))

    elif function == 'Project Showcase':
        rows = [f"| {row[0]} | {row[1]} | {row[2]} | {row[3]} |" for row in data]
        return config['prompt'].format(rows='\n'.join(rows))

    elif function == 'DSCC Activities':
        rows = [f"| {row[0]} | {row[1]} | {row[2]} |" for row in data]
        return config['prompt'].format(rows='\n'.join(rows))

# --- Initialize Chat History ---
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# --- Main UI with Tabs ---
tab1, tab2 = st.tabs(["Main", "Chat History"])

with tab1:
    # --- Streamlit UI ---
    dataframes = init_db()
    query = st.text_input("What do you wanna know", key="user_input")
    
    if st.button("Ask"):
        st.session_state["last_query"] = query
        if query.strip():
            filters = extract_filters(query)
            
            # Enhanced query detection logic
            query_lower = query.lower()
            
            if "trainers" in query_lower or "coaches" in query_lower:
                table = 'trainers'
                columns = ['name', 'profile', 'strengths', 'skillset', 'experience', 'location', 'linkedin']
                function = 'Trainer Details'
                data = query_data_non_llm(dataframes, table, filters, columns)
            elif "project" in query_lower:
                table = 'projects'
                columns = ['project_title', 'domain', 'topic', 'difficulty_level']
                function = 'Project Showcase'
                data = query_data_non_llm(dataframes, table, filters, columns)
            elif "company" in query_lower or "companies" in query_lower:
                table = 'companies'
                columns = ['company_name', 'role', 'ctc_range', 'hiring_frequency', 'requirements']
                function = 'Company Info'
                data = query_data_with_llm(dataframes, table, query, columns)
            elif "placement" in query_lower or "ctc" in query_lower:
                table = 'placement_results'
                columns = ['name', 'education', 'company', 'role', 'ctc', 'location']
                function = 'Placement Results'
                data = query_data_with_llm(dataframes, table, query, columns)
            elif "dscc" in query_lower:
                table = 'dscc_activities'
                columns = ['activity', 'agenda', 'cadence']
                function = 'DSCC Activities'
                data = query_data_non_llm(dataframes, table, filters, columns)
            else:
                st.warning("Query not understood. Please ask about trainers, companies, projects, or DSCC activities.")
                st.stop()

            if data:
                df = pd.DataFrame(data, columns=columns) if isinstance(data, list) else None
                if df is not None:
                    with st.expander("🔍 View All Matches"):
                        st.dataframe(df, use_container_width=True)
                
                st.markdown(f"🔍 Detected Intent: {function}")
                
                formatted = format_with_ollama(data, query, function) if isinstance(data, list) else data
                st.markdown("### Response")
                st.markdown(formatted, unsafe_allow_html=True)
                
                st.session_state.chat_history.append({
                    "query": query, 
                    "response": formatted,
                    "function": function
                })
            else:
                response = "No results matched your filters."
                st.warning(response)
                st.session_state.chat_history.append({
                    "query": query, 
                    "response": response,
                    "function": "No Results"
                })
        else:
            st.warning("Please enter a query.")

with tab2:
    st.markdown("### Chat History")
    if st.session_state.chat_history:
        for i, entry in enumerate(reversed(st.session_state.chat_history), 1):
            with st.expander(f"Q{i}: {entry['query']}"):
                st.markdown(f"*Function:* {entry.get('function', 'Unknown')}")
                st.markdown("*Answer:*")
                st.markdown(entry['response'])
    else:
        st.info("No chat history yet. Start asking questions in the Main tab!")
