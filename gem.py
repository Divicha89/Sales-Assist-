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
gemini_api_key = "AIzaSyDsLZcDrZ2bC4fc7RVSMRP3Vl__vrVQwNM"  # Replace with your actual Gemini API key
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
            df.columns = [col.lower().strip() for col in df.columns]
            dataframes[table] = df
            print(f"Loaded {table} with columns: {df.columns.tolist()}")  # Debug print
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
    
    # Check if it's a general query first
    general_queries = [
        "who are data science trainers",
        "list all data science coaches",
        "show me trainers",
        "data science trainers",
        "all trainers"
    ]
    
    is_general = any(general_q in query_lower for general_q in general_queries)
    
    # Location filters - more comprehensive
    location_patterns = [
        r'from\s+(\w+)', r'in\s+(\w+)', r'at\s+(\w+)',
        r'\b(hyderabad|bangalore|mumbai|delhi|pune|chennai|kolkata|ahmedabad|noida|gurgaon)\b'
    ]
    for pattern in location_patterns:
        location_match = re.search(pattern, query_lower)
        if location_match:
            filters['location'] = location_match.group(1)
            break
    
    # Role/Profile filters - only apply for specific queries, not general ones
    if not is_general:
        role_patterns = [
            r'data scientist?s?', r'data analyst?s?', r'data engineer?s?', r'ml engineer?s?', 
            r'ai engineer?s?', r'business analyst?s?', r'software engineer?s?', r'python developer?s?',
            r'data science coach(?:es)?', r'trainer?s?', r'mentor?s?', r'instructor?s?'
        ]
        for pattern in role_patterns:
            if re.search(pattern, query_lower):
                filters['role'] = pattern.replace('?s?', '').replace('(?:es)?', '')
                break
    
    # Company specific filters
    company_match = re.search(r'company\s+(\w+)|at\s+(\w+)|working\s+at\s+(\w+)', query_lower)
    if company_match:
        filters['company'] = company_match.group(1) or company_match.group(2) or company_match.group(3)
    
    # CTC filters
    ctc_min_match = re.search(r'above\s+(\d+\.?\d*)\s*lpa|minimum\s+(\d+\.?\d*)\s*lpa|>\s*(\d+\.?\d*)|more than\s+(\d+\.?\d*)', query_lower)
    if ctc_min_match:
        filters['ctc_min'] = float(ctc_min_match.group(1) or ctc_min_match.group(2) or ctc_min_match.group(3) or ctc_min_match.group(4))
    
    ctc_max_match = re.search(r'below\s+(\d+\.?\d*)\s*lpa|maximum\s+(\d+\.?\d*)\s*lpa|<\s*(\d+\.?\d*)|less than\s+(\d+\.?\d*)', query_lower)
    if ctc_max_match:
        filters['ctc_max'] = float(ctc_max_match.group(1) or ctc_max_match.group(2) or ctc_max_match.group(3) or ctc_max_match.group(4))
    
    # Skill/Technology filters
    skill_match = re.search(r'skilled in\s+(\w+)|expertise in\s+(\w+)|\b(python|sql|tableau|powerbi|machine learning|deep learning|nlp|r|java|scala|spark|hadoop)\b', query_lower)
    if skill_match:
        filters['skill'] = skill_match.group(1) or skill_match.group(2) or skill_match.group(3)
    
    # Experience filters
    exp_match = re.search(r'(\d+)\s*(?:\+)?\s*years?\s*(?:of)?\s*experience|experience\s*(?:of)?\s*(\d+)\s*(?:\+)?\s*years?', query_lower)
    if exp_match:
        filters['experience'] = int(exp_match.group(1) or exp_match.group(2))
    
    # Frequency filters
    frequency_match = re.search(r'\b(monthly|quarterly|weekly|daily|bimonthly|half-yearly)\b', query_lower)
    if frequency_match:
        filters['frequency'] = frequency_match.group(1)
    
    # Domain filters
    domain_match = re.search(r'\b(machine learning|data science|nlp|visualization|analytics|ai|artificial intelligence|statistics|mathematics)\b', query_lower)
    if domain_match:
        filters['domain'] = domain_match.group(1)
    
    # Difficulty level filters
    difficulty_match = re.search(r'\b(beginner|intermediate|advanced|expert)\b', query_lower)
    if difficulty_match:
        filters['difficulty'] = difficulty_match.group(1)
    
    return filters

def find_column_matches(df, search_terms):
    """
    Find column names that match any of the search terms
    """
    matches = []
    for term in search_terms:
        matches.extend([col for col in df.columns if term in col.lower()])
    return list(set(matches))  # Remove duplicates

def apply_filters_to_dataframe(df, filters, data_type):
    """
    Apply extracted filters to the dataframe based on data type
    """
    if df.empty:
        return df
    
    filtered_df = df.copy()
    original_count = len(filtered_df)
    
    # Debug: Print available columns
    print(f"Available columns in {data_type}: {df.columns.tolist()}")
    print(f"Applying filters: {filters}")
    
    # Apply location filter
    if 'location' in filters:
        location_cols = find_column_matches(df, ['location', 'city', 'place', 'based'])
        if location_cols:
            location_col = location_cols[0]
            mask = filtered_df[location_col].astype(str).str.contains(filters['location'], case=False, na=False)
            filtered_df = filtered_df[mask]
            print(f"Applied location filter on column '{location_col}': {filters['location']}")
        else:
            print(f"No location column found for location filter: {filters['location']}")
    
    # Apply role/profile filters
    if 'role' in filters:
        role_cols = find_column_matches(df, ['role', 'profile', 'designation', 'title', 'position', 'job'])
        if role_cols:
            role_col = role_cols[0]
            role_pattern = filters['role'].replace('_', ' ')
            mask = filtered_df[role_col].astype(str).str.contains(role_pattern, case=False, na=False)
            filtered_df = filtered_df[mask]
            print(f"Applied role filter on column '{role_col}': {role_pattern}")
        else:
            print(f"No role column found for role filter: {filters['role']}")
    
    # Apply company filter
    if 'company' in filters:
        company_cols = find_column_matches(df, ['company', 'organization', 'firm', 'employer'])
        if company_cols:
            company_col = company_cols[0]
            mask = filtered_df[company_col].astype(str).str.contains(filters['company'], case=False, na=False)
            filtered_df = filtered_df[mask]
            print(f"Applied company filter on column '{company_col}': {filters['company']}")
        else:
            print(f"No company column found for company filter: {filters['company']}")
    
    # Apply CTC filters
    if 'ctc_min' in filters or 'ctc_max' in filters:
        ctc_cols = find_column_matches(df, ['ctc', 'salary', 'package', 'compensation'])
        if ctc_cols:
            ctc_col = ctc_cols[0]
            # Extract numeric values from CTC column
            numeric_ctc = pd.to_numeric(filtered_df[ctc_col].astype(str).str.extract(r'(\d+\.?\d*)')[0], errors='coerce')
            
            if 'ctc_min' in filters:
                mask = numeric_ctc >= filters['ctc_min']
                filtered_df = filtered_df[mask]
                print(f"Applied CTC min filter: {filters['ctc_min']}")
            if 'ctc_max' in filters:
                mask = numeric_ctc <= filters['ctc_max']
                filtered_df = filtered_df[mask]
                print(f"Applied CTC max filter: {filters['ctc_max']}")
        else:
            print(f"No CTC column found for CTC filters")
    
    # Apply skill filter
    if 'skill' in filters:
        skill_cols = find_column_matches(df, ['skill', 'technology', 'tech', 'expertise', 'specialization'])
        if skill_cols:
            skill_col = skill_cols[0]
            mask = filtered_df[skill_col].astype(str).str.contains(filters['skill'], case=False, na=False)
            filtered_df = filtered_df[mask]
            print(f"Applied skill filter on column '{skill_col}': {filters['skill']}")
        else:
            print(f"No skill column found for skill filter: {filters['skill']}")
    
    # Apply experience filter
    if 'experience' in filters:
        exp_cols = find_column_matches(df, ['experience', 'exp', 'years'])
        if exp_cols:
            exp_col = exp_cols[0]
            # Extract numeric values from experience column
            numeric_exp = pd.to_numeric(filtered_df[exp_col].astype(str).str.extract(r'(\d+)')[0], errors='coerce')
            mask = numeric_exp >= filters['experience']
            filtered_df = filtered_df[mask]
            print(f"Applied experience filter on column '{exp_col}': {filters['experience']}")
        else:
            print(f"No experience column found for experience filter: {filters['experience']}")
    
    # Apply frequency filter
    if 'frequency' in filters:
        freq_cols = find_column_matches(df, ['frequency', 'cadence', 'schedule'])
        if freq_cols:
            freq_col = freq_cols[0]
            mask = filtered_df[freq_col].astype(str).str.contains(filters['frequency'], case=False, na=False)
            filtered_df = filtered_df[mask]
            print(f"Applied frequency filter on column '{freq_col}': {filters['frequency']}")
        else:
            print(f"No frequency column found for frequency filter: {filters['frequency']}")
    
    # Apply domain filter
    if 'domain' in filters:
        domain_cols = find_column_matches(df, ['domain', 'topic', 'subject', 'area', 'field'])
        if domain_cols:
            domain_col = domain_cols[0]
            mask = filtered_df[domain_col].astype(str).str.contains(filters['domain'], case=False, na=False)
            filtered_df = filtered_df[mask]
            print(f"Applied domain filter on column '{domain_col}': {filters['domain']}")
        else:
            print(f"No domain column found for domain filter: {filters['domain']}")
    
    # Apply difficulty filter
    if 'difficulty' in filters:
        diff_cols = find_column_matches(df, ['difficulty', 'level'])
        if diff_cols:
            diff_col = diff_cols[0]
            mask = filtered_df[diff_col].astype(str).str.contains(filters['difficulty'], case=False, na=False)
            filtered_df = filtered_df[mask]
            print(f"Applied difficulty filter on column '{diff_col}': {filters['difficulty']}")
        else:
            print(f"No difficulty column found for difficulty filter: {filters['difficulty']}")
    
    print(f"Filtered dataframe shape: {filtered_df.shape} (from {original_count} original records)")
    
    # If all filters failed and we have no results, return original dataframe
    if len(filtered_df) == 0 and filters:
        print("All filters failed - returning original dataframe")
        return df
    
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
        data_type = 'trainers'
    elif "project" in query_lower:
        df = dataframes.get('projects', pd.DataFrame())
        data_type = 'projects'
    elif "company" in query_lower or "companies" in query_lower:
        df = dataframes.get('companies', pd.DataFrame())
        data_type = 'companies'
    elif "placement" in query_lower or "ctc" in query_lower:
        df = dataframes.get('placements', pd.DataFrame())
        data_type = 'placements'
    elif "dscc" in query_lower:
        df = dataframes.get('dscc_activities', pd.DataFrame())
        data_type = 'dscc_activities'
    else:
        return "Please specify what you're looking for (trainers, companies, projects, placements, or DSCC activities)."
    
    # Check if original dataframe exists and has data
    if df.empty:
        return f"No data found for {data_type}. Please check if the data file exists and contains records."
    
    # For trainer queries, be more lenient with filtering
    if data_type == 'trainers':
        # If it's a general query like "who are data science trainers", show all trainers
        general_trainer_queries = [
            "who are data science trainers",
            "list all data science coaches", 
            "show me trainers",
            "data science trainers",
            "all trainers"
        ]
        
        is_general_query = any(general_q in query_lower for general_q in general_trainer_queries)
        
        if is_general_query:
            # For general queries, apply minimal filtering or show all
            if 'location' in filters:
                filtered_df = apply_filters_to_dataframe(df, {'location': filters['location']}, 'trainers')
            else:
                filtered_df = df  # Show all trainers for general queries
        else:
            # Apply all filters for specific queries
            filtered_df = apply_filters_to_dataframe(df, filters, 'trainers')
    else:
        # Apply filters normally for other data types
        filtered_df = apply_filters_to_dataframe(df, filters, data_type)
    
    # Check if we have filtered results
    if filtered_df.empty:
        if filters:
            return f"No results found matching your criteria. Applied filters: {filters}\n\nTry broadening your search criteria or check if the data exists in the system."
        else:
            return f"No data found for {data_type}. Please check if the data file exists and contains records."
    
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
    
    # Debug: Show available data info
    if st.checkbox("Show Debug Info"):
        st.write("Available dataframes:")
        for name, df in dataframes.items():
            st.write(f"- {name}: {df.shape[0]} rows, {df.shape[1]} columns")
            st.write(f"  Columns: {df.columns.tolist()}")
    
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
                st.markdown(f"Function: {entry.get('function', 'Unknown')}")
                st.markdown("Answer:")
                st.markdown(entry['response'])
    else:
        st.info("No chat history yet. Start asking questions in the Main tab!")

# --- Clear Chat History Button ---
if st.session_state.chat_history:
    if st.button("Clear Chat History"):
        st.session_state.chat_history = []
        st.rerun()
