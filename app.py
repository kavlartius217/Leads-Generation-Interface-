import streamlit as st
import os
import time
import pandas as pd
from crewai import Agent, Task, Crew, Process
from crewai_tools import SerperDevTool
from crewai.tools import tool
from crewai import LLM
from exa_py import Exa

# Page configuration
st.set_page_config(
    page_title="Lead Synapse Mark III",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better appearance
st.markdown("""
<style>
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    h1, h2, h3 {
        color: #1E3A8A;
    }
    .stProgress .st-bo {
        background-color: #1E3A8A;
    }
    .css-18e3th9 {
        padding-top: 0;
    }
    .css-1d391kg {
        padding-top: 3.5rem;
    }
    .stButton button {
        background-color: #1E3A8A;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# App header
st.title("🔍 Lead Synapse Mark III")
st.subheader("AI-Powered Business Lead Generation")
st.markdown("---")

# Sidebar for configuration
with st.sidebar:
    st.header("Configuration")
    
    # API Keys section
    st.subheader("API Keys")
    st.info("API keys are loaded from Streamlit secrets. Make sure they're configured in your .streamlit/secrets.toml file.")
    
    # Check if API keys are available
    api_keys_status = {
        "Serper API": "✅ Configured" if "SERPER_API_KEY" in st.secrets else "❌ Missing",
        "Exa API": "✅ Configured" if "EXA_API_KEY" in st.secrets else "❌ Missing",
        "OpenAI API": "✅ Configured" if "OPENAI_API_KEY" in st.secrets else "❌ Missing"
    }
    
    for key, status in api_keys_status.items():
        st.text(f"{key}: {status}")
    
    # OpenAI model selection
    st.subheader("OpenAI Model Selection")
    model_option = st.selectbox(
        "Select Model",
        ["gpt-4o-mini", "gpt-4o", "gpt-4-turbo"],
        index=0
    )
    
    temperature = st.slider("Temperature", 0.0, 1.0, 0.1, 0.1)
    
    
    st.markdown("---")
    
    # About section
    st.subheader("About")
    st.markdown("""
    **Lead Synapse Mark III** is an AI-powered business lead generation tool that:
    1. Discovers relevant companies in a specific domain and location
    2. Identifies key decision-makers at those companies
    3. Generates comprehensive, organized lead data
    """)

# Main content area
col1, col2 = st.columns([3, 2])

with col1:
    st.header("Lead Generation Parameters")
    
    domain = st.text_input("Industry Domain", "healthcare")
    area = st.text_input("Geographic Area", "New York")
    
    with st.expander("Advanced Options", expanded=False):
        company_count = st.slider("Number of companies to find", 5, 30, 15)
        contacts_per_company = st.slider("Contacts per company", 1, 5, 3)
        
    st.markdown("---")
    
    start_button = st.button("Start Lead Generation", use_container_width=True)

with col2:
    st.header("How It Works")
    st.markdown("""
    **Process Flow:**
      🔍 Find Companies
      
      👥 Find Decision-Makers
      
      📊 Generate Report
""")
    
    st.markdown("""
    1. **Company Discovery**: Our AI agent identifies relevant companies based on your criteria
    2. **Contact Research**: We find key decision-makers at each company
    3. **Lead Compilation**: All information is compiled into a comprehensive report
    """)

# Main workflow function
def run_lead_synapse(domain, area, company_count=15, contacts_per_company=3):
    # Set API keys from Streamlit secrets
    os.environ['SERPER_API_KEY'] = st.secrets["SERPER_API_KEY"]
    os.environ['EXA_API_KEY'] = st.secrets["EXA_API_KEY"]
    os.environ['OPENAI_API_KEY'] = st.secrets["OPENAI_API_KEY"]
    
    # Progress tracking
    progress = st.progress(0)
    status_container = st.empty()
    status_container.info("Initializing Lead Synapse...")
    
    # Configure LLM
    llm = LLM(model=f'openai/{model_option}', temperature=temperature)
    
    # Tools configuration
    serper_dev_tool = SerperDevTool()
    
    # Exa tool
    @tool("Exa search and get contents")
    def search_and_get_contents_tool(question: str) -> str:
        """Tool using Exa's Python SDK to run semantic search and return result highlights."""
        exa = Exa(st.secrets["EXA_API_KEY"])
        
        response = exa.search_and_contents(
            query=question,
            type="neural",
            num_results=30,
            highlights=True
        )
        
        parsedResult = '\n\n'.join([
            f"<Title id={idx}>{result.title}</Title>\n"
            f"<URL id={idx}>{result.url}</URL>\n"
            f"<Highlight id={idx}>{' | '.join(result.highlights)}</Highlight>"
            for idx, result in enumerate(response.results)
        ])
        
        return parsedResult
    
    # Update progress
    progress.progress(10)
    status_container.info("Setting up AI agents...")
    
    # Create agents
    company_finder_agent = Agent(
        role="Company Discovery Specialist",
        goal=f"Identify {company_count} relevant companies in the {domain} industry within {area} for business development outreach.",
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
    
    linkedin_agent = Agent(
        role="LinkedIn Prospector",
        goal=f"Find {contacts_per_company} professional profiles from EACH company identified by the company finder agent",
        backstory="An expert in finding people on LinkedIn, able to search and extract names and profile URLs using web and semantic search tools.",
        tools=[search_and_get_contents_tool],
        memory=True,
        llm=llm,
        verbose=True
    )
    
    # Update progress
    progress.progress(20)
    status_container.info("Creating tasks...")
    
    # Define tasks
    company_finder_task = Task(
        description=(
            f"Use online tools to find and extract a comprehensive list of {company_count} companies that operate in the **{domain}** domain "
            f"within the **{area}** region. You should use semantic and real-time search to ensure high relevance and accuracy.\n\n"
            "For each company, try to gather:\n"
            "1. Company Name\n"
            "2. Website URL\n"
            "3. Brief Description\n"
            "4. Industry tags or keywords\n"
            "5. Location (City/Country if available)\n"
            "6. Any public contact or LinkedIn URL (if accessible)\n\n"
            f"The list should contain {company_count} companies that are relevant and active in the domain and location specified. "
            "Prioritize companies that are startups, scale-ups, or industry leaders."
        ),
        expected_output=(
            "A markdown file that contains a well-formatted list of companies matching the given domain and location. "
            "Each entry should include the company name, description, website, and any additional available metadata like location or contact info. "
            "The file should be structured with headings and bullet points for easy reading by the business development team."
        ),
        agent=company_finder_agent,
        output_file="companies.md"
    )
    
    linkedin_task = Task(
        description=(
            "For EACH AND EVERY company identified by the company_finder_task, research and identify key decision-makers "
            "who would be ideal contacts for business development outreach. Do not skip any companies. Make sure to find contacts "
            "for all companies in the list. Focus on executives with authority to make partnership or purchasing decisions.\n\n"
            "Target roles should include: Founder, CEO, CTO, COO, CMO, VP/Director/Head of Business Development, "
            "Partnerships, Product, Sales, Marketing, or Growth. Verify that each person currently works at the company "
            "based on their LinkedIn profile information.\n\n"
            "For companies with fewer than 50 employees, prioritize C-level executives. For larger companies, "
            "focus on department heads or directors most relevant to your specific offering."
        ),
        expected_output=(
            "A markdown file with the following structure:\n\n"
            "**Company Name**\n\n"
            "- [Full Name](LinkedIn_URL) - Current Role\n"
            "- [Full Name](LinkedIn_URL) - Current Role\n"
            "- [Full Name](LinkedIn_URL) - Current Role\n\n"
            "**Next Company Name**\n\n"
            "- [Full Name](LinkedIn_URL) - Current Role\n"
            "- [Full Name](LinkedIn_URL) - Current Role\n\n"
            "Requirements:\n"
            f"1. Include ONLY people with verified current employment at the company\n"
            f"2. Format LinkedIn URLs as clickable markdown links with the person's name as the anchor text\n"
            f"3. Ensure all LinkedIn URLs are valid and direct to the specific profile\n"
            f"4. Use bold formatting for company names (with ** not as headers with #)\n"
            f"5. Insert one blank line between each person's entry and two blank lines between companies\n"
            f"6. Do not use any other markdown formatting elements like headers, bullet points, or code blocks\n"
            f"7. Include {contacts_per_company} contacts per company (not more, not less)\n"
            f"8. IMPORTANT: Make sure to include contacts for ALL companies identified in the first task"
        ),
        agent=linkedin_agent,
        context=[company_finder_task],
        output_file="people.md"
    )
    
    # Create crew with appropriate process type
    lead_synapse_crew = Crew(
        agents=[company_finder_agent, linkedin_agent],
        tasks=[company_finder_task, linkedin_task],
        verbose=True
    )
    
    # Update progress
    progress.progress(30)
    status_container.info("Starting company discovery process...")
    
    # Execute crew
    result = lead_synapse_crew.kickoff(inputs={"area": area, "domain": domain})
    
    # Update progress at completion
    progress.progress(100)
    status_container.success("Lead generation completed!")
    
    return result

# Display results section
st.markdown("---")
results_container = st.container()

# Run the process when the button is clicked
if start_button:
    with results_container:
        st.header("Generated Leads")
        
        tabs = st.tabs(["Companies", "Contacts", "Combined Report"])
        
        with st.spinner("Generating leads... This may take several minutes."):
            try:
                result = run_lead_synapse(domain, area, company_count, contacts_per_company)
                
                # Read the output files created by the tasks
                try:
                    with open("companies.md", "r") as f:
                        companies_text = f.read()
                except FileNotFoundError:
                    companies_text = "No company data generated."
                
                try:
                    with open("people.md", "r") as f:
                        contacts_text = f.read()
                except FileNotFoundError:
                    contacts_text = "No contact data generated."
                
                # Display in tabs
                with tabs[0]:  # Companies tab
                    st.markdown(companies_text)
                    st.download_button(
                        "Download Companies List",
                        companies_text,
                        file_name="companies.md",
                        mime="text/markdown",
                        use_container_width=True
                    )
                
                with tabs[1]:  # Contacts tab
                    st.markdown(contacts_text)
                    st.download_button(
                        "Download Contacts List",
                        contacts_text,
                        file_name="people.md",
                        mime="text/markdown",
                        use_container_width=True
                    )
                
                with tabs[2]:  # Combined report tab
                    combined_report = f"# Lead Synapse Report\n\n## Domain: {domain}\n## Region: {area}\n\n## Companies\n\n{companies_text}\n\n## Key Contacts\n\n{contacts_text}"
                    st.markdown(combined_report)
                    
                    # Create Excel export
                    st.subheader("Export Options")
                    st.download_button(
                        "Download Full Report (Markdown)",
                        combined_report,
                        file_name="lead_synapse_report.md",
                        mime="text/markdown",
                        use_container_width=True
                    )
                    
                    st.write("Note: For Excel export functionality, additional parsing would be required.")
                
            except Exception as e:
                st.error(f"An error occurred during lead generation: {str(e)}")
                st.info("Please check your API keys and try again.")

# Footer
st.markdown("---")
st.caption("Lead Synapse Mark III © 2025 | Powered by CrewAI")
