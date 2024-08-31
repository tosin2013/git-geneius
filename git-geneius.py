import os
import streamlit as st
import tempfile
from git import Repo
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.agents import initialize_agent, Tool
from langchain_community.tools import DuckDuckGoSearchResults
from langchain.schema import HumanMessage, SystemMessage
from langchain.memory import ConversationBufferMemory

# Function to list all files in the cloned repository
def list_all_files(directory):
    file_list = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            file_list.append(os.path.join(root, file))
    return file_list

# Function to preprocess the error message
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

# Initialize memory
memory = ConversationBufferMemory()

# Function to handle debugging using LangChain and OpenAI
def smartTextDebugger(error_message, langchain_info, file_details, openai_api_key):
    prompt = PromptTemplate(
        input_variables=["error_message", "langchain", "file_details", "search_quality_reflection", "search_quality_score"],
        template=(
            "You are an AI assistant tasked with providing smart text output for debugging error messages with context on file interactions. "
            "Error message: {error_message}. "
            "Context about file interactions: {langchain}. "
            "File details: {file_details}. "
            "Relevant information from DuckDuckGo search: {search_quality_reflection}. "
            "Search quality score: {search_quality_score}. "
            "Please generate a smart text output for debugging."
        )
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
    
    # Initialize the agent with memory
    agent = initialize_agent(
        tools, 
        llm, 
        agent="zero-shot-react-description", 
        verbose=True,
        memory=memory,  # Add memory to the agent
        return_intermediate_steps=True,
        handle_parsing_errors=True
    )
    
    agent_result = agent(error_message)
    
    # Safely access intermediate steps to avoid IndexError
    search_quality_reflection = ""
    search_quality_score = ""
    
    if "intermediate_steps" in agent_result and len(agent_result["intermediate_steps"]) > 1:
        intermediate_step = agent_result["intermediate_steps"][1]
        if len(intermediate_step) > 1:
            search_quality_reflection = intermediate_step[1]
        if len(intermediate_step) > 2:
            search_quality_score = intermediate_step[2]
    
    formatted_prompt = prompt.format(
        error_message=error_message,
        langchain=langchain_info,
        file_details=file_details,
        search_quality_reflection=search_quality_reflection,
        search_quality_score=search_quality_score
    )
    
    messages = [
        SystemMessage(content="You are a helpful assistant."),
        HumanMessage(content=formatted_prompt)
    ]
    
    result = llm(messages)
    
    return result, search_quality_reflection, search_quality_score

# Streamlit Interface
st.title('Interactive Debugging Chatbot')

# Sidebar for OpenAI API Key and Clear Chat
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

# Initialize files_to_debug
files_to_debug = []

# Clear Chat Button
clear_button = st.sidebar.button('Clear Chat')
if clear_button:
    st.session_state.messages = []
    memory.clear()  # Clear the memory
    st.experimental_rerun()

# Form for cloning repository and submitting debug
with st.form(key='form'):
    st.header("Repository Setup")
    repo_url = st.text_input("Enter the git repository URL")
    repo_token = st.text_input("Enter the repository token (if private)", type="password")
    repo_username = st.text_input("Enter the repository username")
    
    clone_button = st.form_submit_button('Clone Repository')
    
    if clone_button:
        if not openai_api_key.startswith('sk-'):
            st.warning('Please enter your OpenAI API key!', icon='⚠️')
        elif repo_url:
            try:
                with tempfile.TemporaryDirectory() as tmp_dir:
                    # Insert token into repo URL if provided
                    if repo_token:
                        # Assumes HTTPS URL
                        if repo_url.startswith("https://"):
                            repo_url_with_token = repo_url.replace(
                                "https://", f"https://{repo_username}:{repo_token}@"
                            )
                        else:
                            repo_url_with_token = repo_url
                    else:
                        repo_url_with_token = repo_url
                    
                    Repo.clone_from(repo_url_with_token, tmp_dir)
                    st.session_state['cloned_files'] = list_all_files(tmp_dir)
                    st.success("Repository cloned successfully!")
                    st.write("Files in the repository:", st.session_state['cloned_files'])
            except Exception as e:
                st.error(f"Error cloning repository: {e}")
    
    st.header("Debugging Setup")
    error_message = st.text_area("Paste the error message here")
    
    if st.session_state['cloned_files']:
        files_to_debug = st.multiselect("Select files for debugging", st.session_state['cloned_files'])
        if files_to_debug:
            file_details_message = "Error is occurring in the following files: " + ', '.join(files_to_debug)
            langchain_info = st.text_input("Provide the langchain information", value=file_details_message)
        else:
            langchain_info = st.text_input("Provide the langchain information")
    else:
        langchain_info = st.text_input("Provide the langchain information")
        st.warning("Please clone a repository first to select files for debugging.")
    
    submit_button = st.form_submit_button('Submit Debug')

# Handle Debug Submission
if submit_button and error_message and langchain_info:
    if files_to_debug:
        # Append user error message to chat history
        st.session_state.messages.append({"role": "user", "content": error_message})
        
        # Get debug output
        debug_output, search_reflection, search_score = smartTextDebugger(
            error_message, 
            langchain_info, 
            ', '.join(files_to_debug), 
            openai_api_key
        )
        
        # Prepare explanation
        explanation = f"**Explanation of the Answer:**\n\n- **Error message preprocessed:** {preprocess_error_message(error_message)}"
        if search_reflection:
            explanation += f"\n- **Relevant information found online:** {search_reflection}\n- **Search quality score:** {search_score}"
        else:
            explanation += "\n- **No online search was necessary for this query.**"
        
        # Append assistant's debug output and explanation to chat history
        st.session_state.messages.append({"role": "assistant", "content": debug_output.content})
        st.session_state.messages.append({"role": "assistant", "content": explanation})
    else:
        st.warning("Please select at least one file for debugging.")

# Display chat messages from the session state
# Display chat messages from the session state
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Input box for new messages to continue the conversation
user_input = st.text_input("You:", key="user_input")
if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    debug_output, search_reflection, search_score = smartTextDebugger(
        user_input, 
        langchain_info, 
        ', '.join(files_to_debug), 
        openai_api_key
    )
    
    explanation = f"**Explanation of the Answer:**\n\n- **Error message preprocessed:** {preprocess_error_message(user_input)}"
    if search_reflection:
        explanation += f"\n- **Relevant information found online:** {search_reflection}\n- **Search quality score:** {search_score}"
    else:
        explanation += "\n- **No online search was necessary for this query.**"
    
    st.session_state.messages.append({"role": "assistant", "content": debug_output.content})
    st.session_state.messages.append({"role": "assistant", "content": explanation})
    
