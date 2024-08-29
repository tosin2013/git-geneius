import os
import streamlit as st
import tempfile
from git import Repo

from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

def list_all_files(directory):
    file_list = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            file_list.append(os.path.join(root, file))
    return file_list


def preprocess_error_message(error_message):
    lines = error_message.split("\n")
    relevant_lines = []
    
    for line in lines:
        if "TypeError" in line:
            relevant_lines.append(line.strip())
        elif "File" not in line and "line" not in line:
            relevant_lines.append(line.strip())
    
    preprocessed_error = " ".join(relevant_lines)
    return preprocessed_error

from langchain.agents import initialize_agent, Tool
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.schema import HumanMessage, SystemMessage
from langchain_community.tools import DuckDuckGoSearchResults


def smartTextDebugger(error_message, langchain, file_details, openai_api_key):
    print(openai_api_key)
    
    prompt = PromptTemplate(
        input_variables=["error_message", "langchain", "file_details", "search_quality_reflection", "search_quality_score"],
        template="You are an AI assistant tasked with providing smart text output for debugging error messages with context on file interactions. Error message: {error_message}. Context about file interactions: {langchain}. File details: {file_details}. Relevant information from DuckDuckGo search: {search_quality_reflection}. Search quality score: {search_quality_score}. Please generate a smart text output for debugging."
    )
    
    search = DuckDuckGoSearchResults()
    
    tools = [
        Tool(
            name="Error Message Preprocessor",
            func=preprocess_error_message,
            description="Preprocesses the error message to extract relevant information for searching."
        ),
        Tool(
            name="DuckDuckGo Search",
            func=search.invoke,
            description="Searches DuckDuckGo for information related to the preprocessed error message."
        )
    ]
    
    llm = ChatOpenAI(
        model_name="gpt-3.5-turbo",
        openai_api_key=openai_api_key, 
        temperature=0.7
    )
    
    agent = initialize_agent(
        tools, 
        llm, 
        agent="zero-shot-react-description", 
        verbose=True,
        return_intermediate_steps=True,
        handle_parsing_errors=True  # Add this parameter
    )
    
    agent_result = agent(error_message)
    
    search_quality_reflection = agent_result["intermediate_steps"][1][1] if len(agent_result["intermediate_steps"][1]) > 1 else ""
    search_quality_score = agent_result["intermediate_steps"][1][2] if len(agent_result["intermediate_steps"][1]) > 2 else ""
    
    formatted_prompt = prompt.format(
        error_message=error_message,
        langchain=langchain,
        file_details=file_details,
        search_quality_reflection=search_quality_reflection,
        search_quality_score=search_quality_score
    )
    
    messages = [
        SystemMessage(content="You are a helpful assistant."),
        HumanMessage(content=formatted_prompt)
    ]
    
    result = llm(messages)
    
    return result

openai_api_key = st.sidebar.text_input(
    "OpenAI API Key",
    placeholder="sk-...",
    value=os.getenv("OPENAI_API_KEY", ""),
    type="password"
)

if "messages" not in st.session_state:
    st.session_state.messages = []

if "cloned_files" not in st.session_state:
    st.session_state['cloned_files'] = []

with st.form(key='form'):
    st.title('Git Genius')
    repo_url = st.text_input("Enter the git repository URL")
    repo_token = st.text_input("Enter the repository token (if private)", type="password")
    repo_username = st.text_input("Enter the repository username")
    
    clone_button = st.form_submit_button('Clone Repository')
    
    if clone_button:
        if not openai_api_key.startswith('sk-'):
            st.warning('Please enter your OpenAI API key!', icon='⚠️')
        elif repo_url:
            with tempfile.TemporaryDirectory() as tmp_dir:
                repo_url_with_token = repo_url if not repo_token else repo_url.replace("https://", f"https://{repo_username}:{repo_token}@")
                Repo.clone_from(repo_url_with_token, tmp_dir)
                st.session_state['cloned_files'] = list_all_files(tmp_dir)
                st.write("Files in the repository:", st.session_state['cloned_files'])
    
    error_message = st.text_area("Paste the error message here")
    
    if st.session_state['cloned_files']:
        files_to_debug = st.multiselect("Select files for debugging", st.session_state['cloned_files'])
        if files_to_debug:
            file_details_message = "Error is occurring in the following files: " + ', '.join(files_to_debug)
            langchain_info = st.text_input("Provide the langchain information", value=file_details_message)
        else:
            langchain_info = st.text_input("Provide the langchain information")
    else:
        st.warning("Please clone a repository first to select files for debugging.")
    
    submit_button = st.form_submit_button('Submit Debug')

if submit_button and files_to_debug and error_message and langchain_info:
    debug_results = []
    for file in files_to_debug:
        debug_output = smartTextDebugger(error_message, langchain_info, ', '.join(files_to_debug), openai_api_key)
        debug_results.append((file, debug_output))
        st.write(f"File: {file}")
        st.write("LangChain Debug Output:", debug_output)
        st.write("debug_results:", debug_output.content)
