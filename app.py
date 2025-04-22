__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
# -*- coding: utf-8 -*-
import streamlit as st
import os
import traceback
from crewai import Agent, Task, Crew, Process, LLM
from crewai_tools import SerperDevTool
from crewai.tools import tool
from exa_py import Exa

# --- Page Configuration ---
st.set_page_config(page_title="Lead Synapse Mark III", layout="wide", initial_sidebar_state="expanded")
st.title("🚀 Lead Synapse Mark III")
st.markdown("Automated Lead Generation using AI Agents. Enter the domain and area, then click 'Generate Leads'.")

# --- API Key Handling using st.secrets ---
def load_api_keys():
    # ... (Keep the load_api_keys function exactly as before) ...
    keys = {}
    required_keys = ["SERPER_API_KEY", "EXA_API_KEY", "OPENAI_API_KEY"]
    optional_keys = ["GROQ_API_KEY"]
    missing_keys = []

    for key_name in required_keys:
        try:
            keys[key_name] = st.secrets[key_name]
        except KeyError:
            missing_keys.append(key_name)

    for key_name in optional_keys:
         try:
            keys[key_name] = st.secrets[key_name]
         except KeyError:
            st.warning(f"Optional API key '{key_name}' not found in st.secrets.")
            keys[key_name] = None

    return keys, missing_keys

api_keys, missing = load_api_keys()

if missing:
    st.error(f"Missing required API keys in st.secrets: {', '.join(missing)}. "
             "Please add them to your `.streamlit/secrets.toml` file and restart the app.")
    st.stop()
else:
    st.sidebar.success("API Keys loaded successfully!")
    os.environ['SERPER_API_KEY'] = api_keys["SERPER_API_KEY"]
    os.environ['EXA_API_KEY'] = api_keys["EXA_API_KEY"]
    os.environ['OPENAI_API_KEY'] = api_keys["OPENAI_API_KEY"]
    if api_keys.get("GROQ_API_KEY"):
      os.environ['GROQ_API_KEY'] = api_keys["GROQ_API_KEY"]

    openai_api_key = api_keys["OPENAI_API_KEY"]
    exa_api_key = api_keys["EXA_API_KEY"]

# --- CrewAI Components Definition ---

# LLM Configuration (Using the same fast model as before)
llm_openai = LLM(model='openai/gpt-4o-mini', temperature=0, api_key=openai_api_key)
# Consider uncommenting below if you have a Groq key and want to test its speed
# from crewai_groq import GroqLLM
# llm_groq = GroqLLM(api_key=api_keys.get("GROQ_API_KEY"), model='llama3-8b-8192')
# llm_to_use = llm_openai # or llm_groq

# Exa Tool Definition - OPTIMIZED
@tool("Exa search and get contents")
def search_and_get_contents_tool(question: str) -> str:
    """Tool using Exa's Python SDK to run semantic search and return result highlights."""
    try:
        exa = Exa(exa_api_key)
        response = exa.search_and_contents(
            query=question,
            type="neural",
            # --- SPEED OPTIMIZATION: Reduced number of results requested ---
            # Original was 30. Reducing to 10 should be sufficient to find
            # 2-3 contacts per company, significantly speeding up this tool call.
            num_results=10,
            # --- End Optimization ---
            highlights=True
        )
        # Limit highlights length slightly if needed (optional)
        parsedResult = '\n\n'.join([
            f"<Title id={idx}>{result.title}</Title>\n"
            f"<URL id={idx}>{result.url}</URL>\n"
            # Example: Limit highlights displayed/processed if they are excessively long
            # f"<Highlight id={idx}>{' | '.join(h[:500] for h in result.highlights)}</Highlight>"
            f"<Highlight id={idx}>{' | '.join(result.highlights)}</Highlight>"
            for idx, result in enumerate(response.results)
        ])
        return parsedResult
    except Exception as e:
        st.warning(f"Exa search tool encountered an error: {e}") # Show warning in UI
        return f"Error during Exa search: {e}" # Return error string to agent

exa_tools = search_and_get_contents_tool

# Serper Tool Definition
serper_dev_tool = SerperDevTool()

# Agent Definitions (Keep verbose=False for speed)
company_finder_agent = Agent(
    role="Company Discovery Specialist",
    goal="Identify and extract a relevant list of companies based on a specific industry domain and geographic area for business development outreach.",
    backstory=(
        "You are a highly skilled research agent trained in identifying companies using real-time and semantic search tools. "
        "Your job is to find, evaluate, and compile a list of potential companies operating in a given sector within a specified region. "
        "Your output should be relevant, well-structured, and useful for the business development team to begin outreach."
    ),
    memory=True,
    verbose=False, # Keep False for speed
    llm=llm_openai, # Use the chosen LLM
    tools=[serper_dev_tool],
    allow_delegation=False
)

linkedin_agent = Agent(
    role="LinkedIn Prospector",
    goal="Find professional profiles from given companies",
    backstory="An expert in finding people on LinkedIn, able to search and extract names and profile URLs using web and semantic search tools.",
    tools=[exa_tools],
    memory=True,
    llm=llm_openai, # Use the chosen LLM
    verbose=False, # Keep False for speed
    allow_delegation=False
)

# Task Definitions (Instructions unchanged)
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
        "For each company identified by the company_finder_task (access its output context), research and identify 2-3 key decision-makers "
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
        "7. Include 2-3 contacts per company (not more, not less)"
    ),
    agent=linkedin_agent,
    context=[company_finder_task],
    output_file="people.md"
)

# --- Streamlit User Interface Elements ---
st.sidebar.header("Lead Generation Inputs")
domain_input = st.sidebar.text_input("Target Industry Domain:", "Healthcare Technology")
area_input = st.sidebar.text_input("Target Geographic Area:", "New York City")

col1, col2 = st.columns(2)
with col1:
    st.subheader("🏢 Found Companies")
    companies_placeholder = st.empty()
    companies_placeholder.markdown("*(Results will appear here after generation)*")
with col2:
    st.subheader("👥 Identified Contacts")
    people_placeholder = st.empty()
    people_placeholder.markdown("*(Results will appear here after generation)*")

if st.sidebar.button("✨ Generate Leads"):
    if not domain_input or not area_input:
        st.sidebar.warning("Please enter both Domain and Area.")
    else:
        st.sidebar.info("Initializing Crew...")
        companies_placeholder.info("🔄 Task 1: Finding companies...")
        people_placeholder.markdown("*(Waiting for company list...)*")

        try:
            # Instantiate the Crew
            lead_synapse_crew = Crew(
                agents=[company_finder_agent, linkedin_agent],
                tasks=[company_finder_task, linkedin_task],
                process=Process.sequential,
                # Keep verbose off for speed in production
                # verbose=2 # Uncomment ONLY for deep debugging
            )

            inputs = {"area": area_input, "domain": domain_input}

            st.sidebar.info("🚀 Kicking off the Crew... This might take a few minutes.")
            status_indicator = st.sidebar.empty()
            status_indicator.write("⏳ Agents are working...")

            with st.spinner("Processing... Task 1 (Companies) -> Task 2 (Contacts)"):
                crew_result = lead_synapse_crew.kickoff(inputs=inputs)

            status_indicator.write("✅ Crew finished!")
            st.sidebar.success("Lead generation process completed!")

            # --- Display Results from Files ---
            results_displayed = False
            # Display Companies
            try:
                with open("companies.md", "r", encoding="utf-8") as f:
                    companies_md = f.read()
                companies_placeholder.markdown(companies_md)
                people_placeholder.info("🔄 Task 2: Finding contacts...")
                results_displayed = True
            except FileNotFoundError:
                companies_placeholder.error("❌ Error: `companies.md` file not found. Task 1 might have failed.")
                people_placeholder.empty()

            # Display People
            if os.path.exists("companies.md"): # Check if first task succeeded
                try:
                    # Add a small delay in case file writing takes a moment
                    # import time
                    # time.sleep(1)
                    with open("people.md", "r", encoding="utf-8") as f:
                        people_md = f.read()
                    people_placeholder.markdown(people_md)
                    results_displayed = True
                except FileNotFoundError:
                     people_placeholder.error("❌ Error: `people.md` file not found. Task 2 might have failed or produced no output.")
                except Exception as read_err: # Catch other potential read errors
                    people_placeholder.error(f"❌ Error reading `people.md`: {read_err}")


            if not results_displayed and crew_result:
                 st.subheader("Raw Crew Output (Debug Info)")
                 st.write(crew_result)


        except Exception as e:
            st.sidebar.error("An error occurred during the CrewAI process.")
            st.error(f"Error details: {e}")
            st.error(f"Traceback:\n```\n{traceback.format_exc()}\n```")
            companies_placeholder.error("Failed to generate company list due to an error.")
            people_placeholder.error("Failed to generate contact list due to an error.")
            if 'status_indicator' in locals():
                 status_indicator.write("❌ Error occurred.")

st.sidebar.markdown("---")
st.sidebar.markdown("Powered by [CrewAI](https://crewai.com/) & [Streamlit](https://streamlit.io/)")
