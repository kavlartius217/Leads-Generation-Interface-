import os
import streamlit as st
import pandas as pd
import time
from crewai import Agent, Task, Crew, Process, LLM
from crewai_tools import SerperDevTool
from crewai.tools import tool
from exa_py import Exa

# Page configuration
st.set_page_config(
    page_title="Lead Synapse Mark III",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .stApp {
        background-color: #f5f7f9;
    }
    .main-header {
        font-size: 2.5rem;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #3B82F6;
        margin-bottom: 1rem;
    }
    .card {
        background-color: white;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 20px;
    }
    .api-input {
        background-color: #EFF6FF;
        border-radius: 5px;
        padding: 10px;
        margin-bottom: 10px;
    }
    .status-container {
        padding: 15px;
        border-radius: 5px;
        margin-top: 10px;
    }
    .success {
        background-color: #D1FAE5;
        color: #065F46;
    }
    .loading {
        background-color: #FEF3C7;
        color: #92400E;
    }
    .error {
        background-color: #FEE2E2;
        color: #B91C1C;
    }
    </style>
    """, unsafe_allow_html=True)

# Header
st.markdown("<h1 class='main-header'>Lead Synapse Mark III</h1>", unsafe_allow_html=True)
st.markdown("<div class='card'><p>An AI-powered lead generation system that identifies relevant companies and key decision-makers for business development outreach.</p></div>", unsafe_allow_html=True)

# Sidebar for API keys
with st.sidebar:
    st.markdown("<h2 class='sub-header'>API Configuration</h2>", unsafe_allow_html=True)
    st.markdown("<div class='api-input'>", unsafe_allow_html=True)
    
    serper_api_key = st.text_input("Serper API Key", value=os.getenv("SERPER_API_KEY", ""), type="password")
    exa_api_key = st.text_input("Exa API Key", value=os.getenv("EXA_API_KEY", ""), type="password")
    
    llm_option = st.selectbox(
        "Select LLM Provider",
        ["OpenAI", "Groq"]
    )
    
    if llm_option == "OpenAI":
        openai_api_key = st.text_input("OpenAI API Key", value=os.getenv("OPENAI_API_KEY", ""), type="password")
        llm_model = st.selectbox("Select OpenAI Model", ["gpt-4o", "gpt-4o-mini", "gpt-3.5-turbo"])
    else:
        groq_api_key = st.text_input("Groq API Key", value=os.getenv("GROQ_API_KEY", ""), type="password")
        llm_model = st.selectbox("Select Groq Model", ["llama3-8b-8192", "llama3-70b-8192", "mixtral-8x7b-32768"])
    
    temperature = st.slider("LLM Temperature", min_value=0.0, max_value=1.0, value=0.2, step=0.1)
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Save API keys button
    if st.button("Save API Keys"):
        if serper_api_key:
            os.environ["SERPER_API_KEY"] = serper_api_key
        if exa_api_key:
            os.environ["EXA_API_KEY"] = exa_api_key
        if llm_option == "OpenAI" and openai_api_key:
            os.environ["OPENAI_API_KEY"] = openai_api_key
        elif llm_option == "Groq" and groq_api_key:
            os.environ["GROQ_API_KEY"] = groq_api_key
        
        st.success("API keys saved successfully!")

# Main form
st.markdown("<h2 class='sub-header'>Search Parameters</h2>", unsafe_allow_html=True)

with st.form("search_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        domain = st.text_input("Industry Domain", placeholder="e.g., healthcare, fintech, SaaS")
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        area = st.text_input("Geographic Area", placeholder="e.g., New York, London, Singapore")
        st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    col3, col4 = st.columns(2)
    
    with col3:
        num_companies = st.slider("Number of Companies to Find", min_value=5, max_value=30, value=15)
    
    with col4:
        contacts_per_company = st.slider("Contacts per Company", min_value=1, max_value=5, value=3)
    st.markdown("</div>", unsafe_allow_html=True)
    
    run_button = st.form_submit_button("Start Lead Generation")

# Tool definitions
@tool("Exa search and get contents")
def search_and_get_contents_tool(question: str) -> str:
    """Tool using Exa's Python SDK to run semantic search and return result highlights."""
    exa = Exa(os.getenv("EXA_API_KEY"))
    
    response = exa.search_and_contents(
        query=question,
        type="neural",
        num_results=30,
        highlights=True
    )
    
    parsed_result = '\n\n'.join([
        f"<Title id={idx}>{result.title}</Title>\n"
        f"<URL id={idx}>{result.url}</URL>\n"
        f"<Highlight id={idx}>{' | '.join(result.highlights)}</Highlight>"
        for idx, result in enumerate(response.results)
    ])
    
    return parsed_result

# Function to create agents and tasks
def setup_crew(domain, area, num_companies, contacts_per_company):
    # Configure LLM
    if llm_option == "OpenAI":
        llm = LLM(model=f"openai/{llm_model}", temperature=temperature)
    else:
        llm = LLM(model=f"groq/{llm_model}", temperature=temperature)
    
    # Create tools
    serper_dev_tool = SerperDevTool()
    exa_tool = search_and_get_contents_tool
    
    # Company finder agent
    company_finder_agent = Agent(
        role="Company Discovery Specialist",
        goal=f"Identify and extract a list of {num_companies} companies based on the {domain} industry domain in {area} for business development outreach.",
        backstory=(
            "You are a highly skilled research agent trained in identifying companies using real-time and semantic search tools. "
            "Your job is to find, evaluate, and compile a list of potential companies operating in a given sector within a specified region. "
            "Your output should be relevant, well-structured, and useful for the business development team to begin outreach."
        ),
        memory=True,
        verbose=True,
        llm=llm,
        tools=[serper_dev_tool]
    )
    
    company_finder_task = Task(
        description=(
            f"Use online tools to find and extract a comprehensive list of {num_companies} companies that operate in the **{domain}** domain "
            f"within the **{area}** region. You should use semantic and real-time search to ensure high relevance and accuracy.\n\n"
            "For each company, try to gather:\n"
            "1. Company Name\n"
            "2. Website URL\n"
            "3. Brief Description\n"
            "4. Industry tags or keywords\n"
            "5. Location (City/Country if available)\n"
            "6. Any public contact or LinkedIn URL (if accessible)\n\n"
            f"The list should contain {num_companies} companies that are relevant and active in the domain and location specified. "
            "Prioritize companies that are startups, scale-ups, or industry leaders."
        ),
        expected_output=(
            "A markdown file titled `companies.md` that contains a well-formatted list of companies matching the given domain and location. "
            "Each entry should include the company name, description, website, and any additional available metadata like location or contact info. "
            "The file should be structured with headings and bullet points for easy reading by the business development team."
        ),
        agent=company_finder_agent,
        output_file="companies.md"
    )
    
    # LinkedIn agent
    linkedin_agent = Agent(
        role="LinkedIn Prospector",
        goal=f"Find {contacts_per_company} professional profiles from each company identified",
        backstory="An expert in finding people on LinkedIn, able to search and extract names and profile URLs using web and semantic search tools.",
        tools=[exa_tool],
        memory=True,
        llm=llm,
        verbose=True
    )
    
    linkedin_task = Task(
        description=(
            f"For each company identified by the company_finder_task, research and identify {contacts_per_company} key decision-makers "
            "who would be ideal contacts for business development outreach. Focus on executives with authority to "
            "make partnership or purchasing decisions.\n\n"
            "Target roles should include: Founder, CEO, CTO, COO, CMO, VP/Director/Head of Business Development, "
            "Partnerships, Product, Sales, Marketing, or Growth. Verify that each person currently works at the company "
            "based on their LinkedIn profile information.\n\n"
            "For companies with fewer than 50 employees, prioritize C-level executives. For larger companies, "
            "focus on department heads or directors most relevant to your specific offering."
        ),
        expected_output=(
            "A markdown file named `people.md` with the following structure:\n\n"
            "**Company Name**\n\n"
            "- [Full Name](LinkedIn_URL) - Current Role\n"
            "- [Full Name](LinkedIn_URL) - Current Role\n"
            "- [Full Name](LinkedIn_URL) - Current Role\n\n"
            "**Next Company Name**\n\n"
            "- [Full Name](LinkedIn_URL) - Current Role\n"
            "- [Full Name](LinkedIn_URL) - Current Role\n\n"

            "Requirements:\n"
            "1. Include ONLY people with verified current employment at the company\n"
            "2. Format LinkedIn URLs as clickable markdown links with the person's name as the anchor text\n"
            "3. Ensure all LinkedIn URLs are valid and direct to the specific profile\n"
            "4. Use bold formatting for company names (with ** not as headers with #)\n"
            "5. Insert one blank line between each person's entry and two blank lines between companies\n"
            "6. Do not use any other markdown formatting elements like headers, bullet points, or code blocks\n"
            f"7. Include {contacts_per_company} contacts per company"
        ),
        agent=linkedin_agent,
        context=[company_finder_task],
        output_file="people.md"
    )
    
    # Create crew
    lead_synapse_crew = Crew(
        agents=[company_finder_agent, linkedin_agent],
        tasks=[company_finder_task, linkedin_task],
        process=Process.sequential
    )
    
    return lead_synapse_crew

# Display results function
def display_results(companies_md, people_md):
    st.markdown("<h2 class='sub-header'>Results</h2>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<h3>Companies Found</h3>", unsafe_allow_html=True)
        st.markdown(companies_md)
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<h3>Key Decision Makers</h3>", unsafe_allow_html=True)
        st.markdown(people_md)
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Convert to DataFrame for download
    try:
        companies_list = []
        current_company = {}
        
        for line in companies_md.split('\n'):
            if line.startswith('## '):
                if current_company:
                    companies_list.append(current_company)
                current_company = {'Company Name': line.replace('## ', '').strip()}
            elif line.startswith('- Website:'):
                current_company['Website'] = line.replace('- Website:', '').strip()
            elif line.startswith('- Description:'):
                current_company['Description'] = line.replace('- Description:', '').strip()
            elif line.startswith('- Location:'):
                current_company['Location'] = line.replace('- Location:', '').strip()
        
        if current_company:
            companies_list.append(current_company)
        
        companies_df = pd.DataFrame(companies_list)
        
        # Parse people
        people_list = []
        current_company = ""
        
        for line in people_md.split('\n'):
            if line.startswith('**') and line.endswith('**'):
                current_company = line.replace('**', '').strip()
            elif line.startswith('- ['):
                parts = line.split('](')
                name = parts[0].replace('- [', '').strip()
                url_role = parts[1].split(')')
                url = url_role[0].strip()
                role = url_role[1].replace('-', '').strip()
                
                people_list.append({
                    'Company Name': current_company,
                    'Full Name': name,
                    'Role': role,
                    'LinkedIn URL': url
                })
        
        people_df = pd.DataFrame(people_list)
        
        # Prepare for download
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.download_button(
                label="Download Companies CSV",
                data=companies_df.to_csv(index=False),
                file_name=f"companies_{domain}_{area}.csv",
                mime="text/csv"
            )
            st.markdown("</div>", unsafe_allow_html=True)
        
        with col2:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.download_button(
                label="Download Contacts CSV",
                data=people_df.to_csv(index=False),
                file_name=f"contacts_{domain}_{area}.csv",
                mime="text/csv"
            )
            st.markdown("</div>", unsafe_allow_html=True)
            
    except Exception as e:
        st.warning(f"Could not parse results to CSV: {str(e)}")
        
    # Combined report
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<h3>Generate Final Report</h3>", unsafe_allow_html=True)
    
    if st.button("Generate Comprehensive Report"):
        with st.spinner("Generating comprehensive report..."):
            # Combine data
            combined_md = f"""# Lead Generation Report: {domain} in {area}

## Summary
This report contains {len(companies_list)} companies in the {domain} industry located in {area}, with {len(people_list)} key decision-makers identified.

## Companies Overview
{companies_md}

## Key Decision Makers
{people_md}

## Next Steps
1. Prioritize outreach based on company size and relevance
2. Prepare personalized outreach templates for each decision-maker
3. Schedule follow-up cadence for non-responsive contacts
4. Track engagement metrics in CRM system
            """
            
            st.download_button(
                label="Download Complete Report",
                data=combined_md,
                file_name=f"lead_synapse_report_{domain}_{area}.md",
                mime="text/markdown"
            )
            
            st.success("Comprehensive report generated successfully!")
    st.markdown("</div>", unsafe_allow_html=True)

# Run the lead generation process
if run_button:
    if not (domain and area):
        st.error("Please provide both industry domain and geographic area")
    elif not (os.getenv("SERPER_API_KEY") and os.getenv("EXA_API_KEY")):
        st.error("API keys for Serper and Exa are required")
    elif (llm_option == "OpenAI" and not os.getenv("OPENAI_API_KEY")) or (llm_option == "Groq" and not os.getenv("GROQ_API_KEY")):
        st.error(f"API key for {llm_option} is required")
    else:
        # Create progress container
        progress_container = st.empty()
        progress_container.markdown("<div class='card status-container loading'>Starting lead generation process...</div>", unsafe_allow_html=True)
        
        try:
            # Set up crew
            crew = setup_crew(domain, area, num_companies, contacts_per_company)
            
            # Progress updates
            progress_container.markdown("<div class='card status-container loading'>Finding companies in the specified industry and location...</div>", unsafe_allow_html=True)
            
            # Start the process
            results = crew.kickoff({
                "domain": domain,
                "area": area
            })
            
            # Show success
            progress_container.markdown("<div class='card status-container success'>Lead generation completed successfully!</div>", unsafe_allow_html=True)
            
            # Read the output files
            try:
                with open("companies.md", "r") as f:
                    companies_md = f.read()
                
                with open("people.md", "r") as f:
                    people_md = f.read()
                
                # Display results
                display_results(companies_md, people_md)
                
            except FileNotFoundError:
                st.error("Output files were not generated. Please check the logs.")
                st.code(str(results))
                
        except Exception as e:
            progress_container.markdown(f"<div class='card status-container error'>Error: {str(e)}</div>", unsafe_allow_html=True)
            st.error(f"An error occurred: {str(e)}")

# Add documentation in an expander
with st.expander("How to use Lead Synapse"):
    st.markdown("""
    ### Getting Started
    
    1. **Configure API Keys**: In the sidebar, enter your API keys for the required services:
       - **Serper API Key**: Used for web search functionality
       - **Exa API Key**: Used for semantic search and content extraction
       - **OpenAI/Groq API Key**: Used for AI processing and analysis
    
    2. **Set Search Parameters**:
       - **Industry Domain**: The business sector you want to target (e.g., healthcare, fintech)
       - **Geographic Area**: The location where you want to find companies (e.g., New York, London)
       - **Number of Companies**: How many companies to identify
       - **Contacts per Company**: How many decision-makers to find at each company
    
    3. **Start Lead Generation**: Click the button to begin the process. This may take several minutes depending on your parameters.
    
    4. **Review and Download Results**: Once complete, you can view the companies and contacts found, and download them as CSV files or a comprehensive report.
    
    ### Best Practices
    
    - **Be Specific**: More specific industry domains will yield more relevant results
    - **Use Reasonable Numbers**: Finding more than 20 companies or 3 contacts per company will increase processing time
    - **Save Your Reports**: Download the generated reports for your CRM or follow-up systems
    
    ### Troubleshooting
    
    - If you encounter errors, check that all API keys are correct
    - For best results, use OpenAI's GPT-4 models or Groq's Llama3-70b model
    - If results are not relevant enough, try refining your industry domain with more specific terms
    """)

# Footer
st.markdown("""
<div style="text-align: center; margin-top: 30px; padding: 10px; background-color: #F3F4F6; border-radius: 5px;">
    <p>Lead Synapse Mark III © 2025 | Powered by CrewAI</p>
</div>
""", unsafe_allow_html=True)
