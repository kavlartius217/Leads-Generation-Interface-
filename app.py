import streamlit as st
import os
from crewai import Agent, Task, Crew, Process, LLM
from crewai_tools import SerperDevTool
from crewai.tools import tool
from exa_py import Exa

# Page configuration
st.set_page_config(
    page_title="Lead Synapse Mark III",
    page_icon="🎯",
    layout="wide"
)

# Set environment variables from secrets
for key in ["SERPER_API_KEY", "EXA_API_KEY", "OPENAI_API_KEY", "GROQ_API_KEY"]:
    os.environ[key] = st.secrets[key]

# Title and description
st.title("🎯 Lead Synapse Mark III")
st.markdown("AI-Powered B2B Lead Generation System")

# LLM Configuration
llm_openai = LLM(model='openai/gpt-4o-mini', temperature=0)

# Tools
@tool("Exa search and get contents")
def search_and_get_contents_tool(question: str) -> str:
    """Tool using Exa's Python SDK to run semantic search and return result highlights."""

    exa = Exa(exa_api_key)

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

serper_dev_tool = SerperDevTool()

# Agents
company_finder_agent = Agent(
    role="Company Discovery Specialist",
    goal="Identify and extract a relevant list of companies based on a specific industry domain and geographic area for business development outreach.",
    backstory=(
        "You are a highly skilled research agent trained in identifying companies using real-time and semantic search tools. "
        "Your job is to find, evaluate, and compile a list of potential companies operating in a given sector within a specified region. "
        "Your output should be relevant, well-structured, and useful for the business development team to begin outreach."
    ),
    memory=True,
    verbose=True,
    llm=llm_openai,
    tools=[serper_dev_tool]
)

linkedin_agent = Agent(
    role="LinkedIn Prospector",
    goal="Find professional profiles from given companies",
    backstory="An expert in finding people on LinkedIn, able to search and extract names and profile URLs using web and semantic search tools.",
    tools=[search_and_get_contents_tool],
    memory=True,
    llm=llm_openai,
    verbose=True
)

# Tasks
company_finder_task = Task(
    description=(
        "Use online tools to find and extract a comprehensive list of companies that operate in the **{domain}** domain "
        "within the **{area}** region. You should use semantic and real-time search to ensure high relevance and accuracy.\n\n"
        "For each company, try to gather:\n"
        "1. Company Name\n"
        "2. Website URL\n"
        "3. Brief Description\n"
        "4. Industry tags or keywords\n"
        "5. Location (City/Country if available)\n"
        "6. Any public contact or LinkedIn URL (if accessible)\n\n"
        "The list should contain 15-20 companies that are relevant and active in the domain and location specified. "
        "Prioritize companies that are startups, scale-ups, or industry leaders."
    ),
    expected_output=(
        "A markdown file titled `companies.md` that contains a well-formatted list of 15-20 companies matching the given domain and location. "
        "Each entry should include the company name, description, website, and any additional available metadata like location or contact info. "
        "The file should be structured with headings and bullet points for easy reading by the business development team."
    ),
    agent=company_finder_agent,
    output_file="companies.md"
)

linkedin_task = Task(
    description=(
        "For each company identified by the company_finder_task, research and identify 2-3 key decision-makers "
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
    ),
    agent=linkedin_agent,
    context=[company_finder_task],
    output_file="people.md"
)

# Input Interface
col1, col2 = st.columns(2)
with col1:
    area = st.text_input("Target Area", value="New York")
with col2:
    domain = st.text_input("Industry Domain", value="healthcare")

# Run Button and Process
if st.button("Generate Leads", type="primary"):
    if not area or not domain:
        st.error("Please provide both area and domain!")
    else:
        try:
            with st.status("🔍 Lead Synapse is working...", expanded=True) as status:
                st.write("Phase 1: Researching companies...")
                
                # Create and run crew
                lead_synapse_crew = Crew(
                    agents=[company_finder_agent, linkedin_agent],
                    tasks=[company_finder_task, linkedin_task]
                )
                
                result = lead_synapse_crew.kickoff({
                    "area": area,
                    "domain": domain
                })
                
                status.update(label="✅ Research completed!", state="complete")

            # Display Results
            tabs = st.tabs(["📊 Companies", "👥 Contacts"])
            
            with tabs[0]:
                try:
                    with open("companies.md", "r") as f:
                        content = f.read()
                        st.markdown(content)
                        st.download_button(
                            "📥 Download Companies List",
                            content,
                            file_name="companies.md",
                            mime="text/markdown"
                        )
                except FileNotFoundError:
                    st.error("No companies data generated yet")
            
            with tabs[1]:
                try:
                    with open("people.md", "r") as f:
                        content = f.read()
                        st.markdown(content)
                        st.download_button(
                            "📥 Download Contacts List",
                            content,
                            file_name="people.md",
                            mime="text/markdown"
                        )
                except FileNotFoundError:
                    st.error("No contacts data generated yet")

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
