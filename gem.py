import streamlit as st
import pandas as pd
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
gemini_api_key = "AIzaSyDfuo-L0KOU1zZAsABl6Ltajy13EgvBpMc"  # Replace with your actual Gemini API key
if not gemini_api_key:
    st.error("Please provide a Gemini API key.")
    st.stop()

# Initialize Gemini client
genai.configure(api_key=gemini_api_key)
model = genai.GenerativeModel('gemini-1.5-flash')

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
            dataframes[table] = df
        except FileNotFoundError as e:
            st.error(f"Missing file: {e}")
        except KeyError as e:
            st.error(f"Column error in {table} file: {e}. Please check column names.")
    return dataframes

# --- Enhanced Filtering System ---
def extract_filters_advanced(query):
    """
    Extract various filters from the query for different data types
    """
    import re
    query_lower = query.lower()
    
    filters = {}
    
    # Location filters
    location_match = re.search(r'from\s+(\w+)|in\s+(\w+)|\b(hyderabad|bangalore|mumbai|delhi|pune|chennai)\b', query_lower)
    if location_match:
        filters['location'] = location_match.group(1) or location_match.group(2) or location_match.group(3)
    
    # Role/Profile filters
    role_patterns = [
        r'data scientist', r'data analyst', r'data engineer', r'ml engineer', 
        r'ai engineer', r'business analyst', r'software engineer', r'python developer',
        r'data science coach', r'trainer', r'mentor'
    ]
    for pattern in role_patterns:
        if re.search(pattern, query_lower):
            filters['role'] = pattern.replace(' ', '_')
            break
    
    # Company specific filters
    company_match = re.search(r'company\s+(\w+)|at\s+(\w+)', query_lower)
    if company_match:
        filters['company'] = company_match.group(1) or company_match.group(2)
    
    # CTC filters
    ctc_min_match = re.search(r'above\s+(\d+\.?\d*)\s*lpa|minimum\s+(\d+\.?\d*)\s*lpa|>\s*(\d+\.?\d*)', query_lower)
    if ctc_min_match:
        filters['ctc_min'] = float(ctc_min_match.group(1) or ctc_min_match.group(2) or ctc_min_match.group(3))
    
    ctc_max_match = re.search(r'below\s+(\d+\.?\d*)\s*lpa|maximum\s+(\d+\.?\d*)\s*lpa|<\s*(\d+\.?\d*)', query_lower)
    if ctc_max_match:
        filters['ctc_max'] = float(ctc_max_match.group(1) or ctc_max_match.group(2) or ctc_max_match.group(3))
    
    # Skill/Technology filters
    skill_match = re.search(r'skilled in\s+(\w+)|expertise in\s+(\w+)|\b(python|sql|tableau|powerbi|machine learning|deep learning|nlp)\b', query_lower)
    if skill_match:
        filters['skill'] = skill_match.group(1) or skill_match.group(2) or skill_match.group(3)
    
    # Frequency filters
    frequency_match = re.search(r'\b(monthly|quarterly|weekly|daily|bimonthly|half-yearly)\b', query_lower)
    if frequency_match:
        filters['frequency'] = frequency_match.group(1)
    
    # Domain filters
    domain_match = re.search(r'\b(machine learning|data science|nlp|visualization|analytics|ai|artificial intelligence)\b', query_lower)
    if domain_match:
        filters['domain'] = domain_match.group(1)
    
    # Difficulty level filters
    difficulty_match = re.search(r'\b(beginner|intermediate|advanced|expert)\b', query_lower)
    if difficulty_match:
        filters['difficulty'] = difficulty_match.group(1)
    
    return filters

def apply_filters_to_dataframe(df, filters, data_type):
    """
    Apply extracted filters to the dataframe based on data type
    """
    if df.empty:
        return df
    
    filtered_df = df.copy()
    
    # Apply location filter
    if 'location' in filters:
        location_cols = [col for col in df.columns if 'location' in col.lower()]
        if location_cols:
            filtered_df = filtered_df[filtered_df[location_cols[0]].str.contains(filters['location'], case=False, na=False)]
    
    # Apply role/profile filters
    if 'role' in filters:
        role_cols = [col for col in df.columns if any(x in col.lower() for x in ['role', 'profile', 'designation'])]
        if role_cols:
            role_pattern = filters['role'].replace('_', ' ')
            filtered_df = filtered_df[filtered_df[role_cols[0]].str.contains(role_pattern, case=False, na=False)]
    
    # Apply company filter
    if 'company' in filters:
        company_cols = [col for col in df.columns if 'company' in col.lower()]
        if company_cols:
            filtered_df = filtered_df[filtered_df[company_cols[0]].str.contains(filters['company'], case=False, na=False)]
    
    # Apply CTC filters
    if 'ctc_min' in filters or 'ctc_max' in filters:
        ctc_cols = [col for col in df.columns if 'ctc' in col.lower()]
        if ctc_cols:
            ctc_col = ctc_cols[0]
            # Extract numeric values from CTC column
            numeric_ctc = pd.to_numeric(filtered_df[ctc_col].astype(str).str.extract(r'(\d+\.?\d*)')[0], errors='coerce')
            
            if 'ctc_min' in filters:
                filtered_df = filtered_df[numeric_ctc >= filters['ctc_min']]
            if 'ctc_max' in filters:
                filtered_df = filtered_df[numeric_ctc <= filters['ctc_max']]
    
    # Apply skill filter
    if 'skill' in filters:
        skill_cols = [col for col in df.columns if any(x in col.lower() for x in ['skill', 'technology', 'tech'])]
        if skill_cols:
            filtered_df = filtered_df[filtered_df[skill_cols[0]].str.contains(filters['skill'], case=False, na=False)]
    
    # Apply frequency filter
    if 'frequency' in filters:
        freq_cols = [col for col in df.columns if any(x in col.lower() for x in ['frequency', 'cadence', 'schedule'])]
        if freq_cols:
            filtered_df = filtered_df[filtered_df[freq_cols[0]].str.contains(filters['frequency'], case=False, na=False)]
    
    # Apply domain filter
    if 'domain' in filters:
        domain_cols = [col for col in df.columns if 'domain' in col.lower() or 'topic' in col.lower()]
        if domain_cols:
            filtered_df = filtered_df[filtered_df[domain_cols[0]].str.contains(filters['domain'], case=False, na=False)]
    
    # Apply difficulty filter
    if 'difficulty' in filters:
        diff_cols = [col for col in df.columns if 'difficulty' in col.lower() or 'level' in col.lower()]
        if diff_cols:
            filtered_df = filtered_df[filtered_df[diff_cols[0]].str.contains(filters['difficulty'], case=False, na=False)]
    
    return filtered_df

# --- Enhanced LLM Query Function with Filtering ---
def query_data_with_gemini(dataframes, query):
    """
    Use filtering + Gemini LLM to process queries with proper data filtering
    """
    query_lower = query.lower()
    
    # Extract filters from query
    filters = extract_filters_advanced(query)
    
    # Determine which data to use and apply filters
    if "trainer" in query_lower or "coach" in query_lower:
        df = dataframes.get('trainers', pd.DataFrame())
        filtered_df = apply_filters_to_dataframe(df, filters, 'trainers')
        data_type = 'trainers'
    elif "project" in query_lower:
        df = dataframes.get('projects', pd.DataFrame())
        filtered_df = apply_filters_to_dataframe(df, filters, 'projects')
        data_type = 'projects'
    elif "company" in query_lower or "companies" in query_lower:
        df = dataframes.get('companies', pd.DataFrame())
        filtered_df = apply_filters_to_dataframe(df, filters, 'companies')
        data_type = 'companies'
    elif "placement" in query_lower or "ctc" in query_lower:
        df = dataframes.get('placements', pd.DataFrame())
        filtered_df = apply_filters_to_dataframe(df, filters, 'placements')
        data_type = 'placements'
    elif "dscc" in query_lower:
        df = dataframes.get('dscc_activities', pd.DataFrame())
        filtered_df = apply_filters_to_dataframe(df, filters, 'dscc_activities')
        data_type = 'dscc_activities'
    else:
        return "Please specify what you're looking for (trainers, companies, projects, placements, or DSCC activities)."
    
    # Check if we have filtered results
    if filtered_df.empty:
        return f"No results found matching your criteria. Applied filters: {filters}"
    
    # Convert filtered dataframe to string context
    context = f"=== FILTERED {data_type.upper()} DATA ===\n{filtered_df.to_string(index=False)}\n"
    
    # Add filter information to context
    filter_info = f"Filters applied: {filters}" if filters else "No specific filters applied"
    context += f"\n=== FILTER INFO ===\n{filter_info}\n"
    
    # Create appropriate prompt based on data type
    if data_type == 'trainers':
        prompt = f"""Based on the following trainer data:
{context}

Query: {query}

Please provide a well-formatted response that includes:
1. A clear summary at the top
2. Individual trainer profiles with:
   - Name and Role
   - Experience and Skills
   - Strengths and Location
   - LinkedIn profile if available
3. Add a section about panel trainers at the end

Make it professional, structured, and informative. Use markdown formatting with tables where appropriate."""

    elif data_type == 'placements':
        prompt = f"""Based on the following placement data:
{context}

Query: {query}

Please provide a well-formatted response that includes:
1. A clear answer/summary at the top (e.g., "Top CTC record: X LPA")
2. A table with Name, Education Background, Company, Role, CTC, Location
3. Inspiring student stories at the end with LinkedIn links, testimonial prompts

Make it professional, structured, and motivational."""

    elif data_type == 'companies':
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

    elif data_type == 'projects':
        prompt = f"""Based on the following project data:
{context}

Query: {query}

Please provide a well-formatted response that includes:
1. A comprehensive table with Project Title, Domain, Technologies Used, and Skill Level
2. A section highlighting top problem statements explored by students
3. Information about how these projects enhance placement readiness

Make it professional and inspiring for students."""

    elif data_type == 'dscc_activities':
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
        return f"Error generating response: {str(e)}"

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
            # Use unified LLM approach for all queries
            with st.spinner("Processing your query..."):
                response = query_data_with_gemini(dataframes, query)
            
            # Determine function type for chat history
            query_lower = query.lower()
            if "trainer" in query_lower or "coach" in query_lower:
                detected_function = "Trainer Details"
            elif "project" in query_lower:
                detected_function = "Project Showcase"
            elif "company" in query_lower or "companies" in query_lower:
                detected_function = "Company Info"
            elif "placement" in query_lower or "ctc" in query_lower:
                detected_function = "Placement Stats"
            elif "dscc" in query_lower:
                detected_function = "DSCC Activities"
            else:
                detected_function = "General Query"
            
            st.markdown(f"🔍 Detected Intent: {detected_function}")
            st.markdown("### Response")
            st.markdown(response, unsafe_allow_html=True)
            
            # Add to chat history
            st.session_state.chat_history.append({
                "query": query, 
                "response": response,
                "function": detected_function
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

# --- Clear Chat History Button ---
if st.session_state.chat_history:
    if st.button("Clear Chat History"):
        st.session_state.chat_history = []
        st.rerun()
