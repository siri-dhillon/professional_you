#!/usr/bin/env python
# coding: utf-8

# # 🤖 Professionally You: AI Portfolio Agent
# ### Developed by Sirpreet Dhillon
# 
# This project builds a professional "AI Alter-Ego" using **RAG (Retrieval-Augmented Generation)** and **Agentic Tool-Use**. 
# 
# **Core Functionality:**
# * **Academic & Professional RAG:** Leverages a ChromaDB vector store containing my SFU courses, LinkedIn profile, and resume to advocate for my skills.
# * **Live GitHub Integration:** Uses a GraphQL tool to fetch and display my pinned repositories in real-time.
# * **Lead Capture:** Includes a Pushover-integrated tool to capture visitor inquiries and send push notifications directly to my device.
# * **Self-Evaluating Logic:** Features an "Evaluator Agent" that reviews responses for professional tone before delivery.

# In[43]:


from dotenv import load_dotenv
from openai import OpenAI
import json
import os
import requests
from pypdf import PdfReader
import gradio as gr
from github import Github

# RAG specific imports
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma


# In[44]:


load_dotenv(override=True)
openai = OpenAI()

# Professional Identity
name = "Sirpreet Dhillon"


# In[45]:


# For pushover

pushover_user = os.getenv("PUSHOVER_USER")
pushover_token = os.getenv("PUSHOVER_TOKEN")
pushover_url = "https://api.pushover.net/1/messages.json"


# In[46]:


documents_to_load = [
    {"path": "me/linkedin.pdf", "type": "pdf"},
    {"path": "me/resume.pdf", "type": "pdf"},
    {"path": "me/projects.pdf", "type": "pdf"},
    {"path": "me/courses.pdf", "type": "pdf"},
    {"path": "me/summary.txt", "type": "txt"}
]


# In[47]:


all_docs = []
for doc in documents_to_load:
    if os.path.exists(doc["path"]):
        loader = PyPDFLoader(doc["path"]) if doc["type"] == "pdf" else TextLoader(doc["path"])
        all_docs.extend(loader.load())

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(all_docs)

vectorstore = Chroma.from_documents(
    documents=splits, 
    embedding=OpenAIEmbeddings(),
    persist_directory="./chroma_db"
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

def get_rag_context(query):
    relevant_docs = retriever.invoke(query)
    return "\n\n".join([doc.page_content for doc in relevant_docs])


# In[48]:


def fetch_github_projects():
    """Fetches real-time project data from GitHub."""
    try:
        g = Github(os.getenv("GITHUB_TOKEN"))
        repos = [{"name": r.name, "url": r.html_url, "stars": r.stargazers_count} 
                 for r in g.get_user().get_repos(sort="updated")[:3]]
        return json.dumps(repos)
    except Exception as e:
        return f"Error fetching GitHub: {str(e)}"


# In[49]:


def fetch_pinned_repos():
    """Fetches my pinned repositories using GraphQL to show featured work."""
    token = os.getenv("GITHUB_TOKEN")
    username = "siri-dhillon" 
    url = "https://api.github.com/graphql"

    query = """
    {
      user(login: "%s") {
        pinnedItems(first: 6, types: REPOSITORY) {
          nodes {
            ... on Repository {
              name
              description
              url
              stargazerCount
              primaryLanguage { name }
            }
          }
        }
      }
    }
    """ % username

    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    try:
        response = requests.post(url, json={'query': query}, headers=headers)
        if response.status_code == 200:
            return response.json()['data']['user']['pinnedItems']['nodes']
        return f"Error: {response.status_code}"
    except Exception as e:
        return str(e)


# In[50]:


def send_lead_to_pushover(user_name, user_email, message):
    """Sends user contact info to Sirpreet's phone via Pushover."""
    url = "https://api.pushover.net/1/messages.json"
    formatted_msg = f"New Lead!\nName: {user_name}\nEmail: {user_email}\nMessage: {message}"

    data = {
        "token": os.getenv("PUSHOVER_TOKEN"),
        "user": os.getenv("PUSHOVER_USER"),
        "message": formatted_msg
    }
    response = requests.post(url, data=data)
    return "Message sent to Sirpreet successfully!" if response.status_code == 200 else "Failed to send."


# In[51]:


tools = [
    {
        "type": "function",
        "function": {
            "name": "fetch_github_projects",
            "description": "Get real-time info about my latest GitHub repositories."
        }
    },
    {
        "type": "function",
        "function": {
            "name": "fetch_pinned_repos",
            "description": "Get real-time info about Sirpreet's featured/pinned GitHub repositories using GraphQL."
        }
    },
    {
        "type": "function",
        "function": {
            "name": "send_lead_to_pushover",
            "description": "Call this when a user wants to contact Sirpreet or leave their details.",
            "parameters": {
                "type": "object",
                "properties": {
                    "user_name": {"type": "string", "description": "The user's name"},
                    "user_email": {"type": "string", "description": "The user's email address"},
                    "message": {"type": "string", "description": "The message they want to leave"}
                },
                "required": ["user_name", "user_email", "message"]
            }
        }
    }
]


# In[52]:


def handle_tool_calls(tool_calls):
    results = []
    for tool_call in tool_calls:
        tool_name = tool_call.function.name
        # The arguments come in as a JSON string from the LLM
        arguments = json.loads(tool_call.function.arguments)
        print(f"Tool called: {tool_name}", flush=True)

        # Routing to your specific project tools
        if tool_name == "fetch_pinned_repos": 
            # Calls your new GraphQL function for pinned items
            result = fetch_pinned_repos()

        elif tool_name == "send_lead_to_pushover":
            # Unpacks user_name, user_email, and message into the function
            result = send_lead_to_pushover(**arguments)

        elif tool_name == "record_user_details":
            # Original tool from your lab files
            result = record_user_details(**arguments)

        elif tool_name == "record_unknown_question":
            # Original tool from your lab files
            result = record_unknown_question(**arguments)
        else:
            result = {"error": f"Tool {tool_name} not found."}

        # Append the result in the format OpenAI expects
        results.append({
            "role": "tool",
            "content": json.dumps(result),
            "tool_call_id": tool_call.id
        })
    return results


# In[53]:


def evaluate_response(user_query, ai_response):

    eval_prompt = f"""
    You are a professional brand manager for {name}. 
    Review this draft to ensure it sounds like {name} speaking directly:

    Draft: {ai_response}

    Criteria:
    1. Does it use "I" and "me"? (Critical)
    2. Is it professional and accurate to the provided records?
    3. Does it avoid sounding like a 'chatbot'?

    Respond 'PASS' or 'CORRECTION: [instruction]'."""


    response = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "system", "content": eval_prompt}]
    )
    return response.choices[0].message.content


# In[54]:

def record_user_details(email, name="Name not provided", notes="not provided"):
    return send_lead_to_pushover(name, email, notes)

def record_unknown_question(question):
    # Sends a notification so you know a question went unanswered
    return send_lead_to_pushover("System", "N/A", f"Unanswered Question: {question}")


def chat(message, history):
    # 1. Fetch relevant data from your PDFs (LinkedIn, Resume, Projects)
    context = get_rag_context(message)

    # 2. System Prompt with RAG Context
    system_message = f"""
        I am {name}. I am a Computer Engineering graduate from Simon Fraser University.
        CONTEXT FROM MY RECORDS (including my Resume, Projects, and SFU Coursework):{context}

        My Voice & Rules:
        - I speak in the first person ("I," "me," "my").
        - I am professional, technical, and grounded in the facts from my history. look into it or chat more. Would you like to leave your email so I can follow up?"
        - Use my SFU coursework to answer technical questions even if I haven't used a specific tool in a professional job yet.
        - If someone asks 'What did you learn about [Topic]?', use the descriptions from my courses to provide a detailed answer.
        - If I'm asked for a skill I don't explicitly have, I will look for transferable skills in my records. I will explain how my background in SFU or my other projects makes me capable of learning it quickly. I will advocate for myself before offering to take an email.
        - Only use 'send_lead_to_pushover' if the user explicitly wants to contact me or after I've tried my best to answer.
        - If the user wants to see my code, I use 'fetch_github_projects' to show my work.
        - If they want to reach me, I use 'send_lead_to_pushover' to take a message.
        """

    messages = [{"role": "system", "content": system_message}] + history + [{"role": "user", "content": message}]

    # --- Tool Execution Loop ---
    while True:
        response = openai.chat.completions.create(
            model="gpt-4o-mini", 
            messages=messages, 
            tools=tools
        )
        choice = response.choices[0].message

        if not choice.tool_calls:
            break

        messages.append(choice)
        for tool_call in choice.tool_calls:
            function_name = tool_call.function.name
            args = json.loads(tool_call.function.arguments)

            if function_name == "fetch_github_projects":
                result = fetch_github_projects()
            elif function_name == "fetch_pinned_repos": 
                result = fetch_pinned_repos()
            elif function_name == "send_lead_to_pushover":
                result = send_lead_to_pushover(**args)

            messages.append({
                "role": "tool", 
                "tool_call_id": tool_call.id, 
                "content": result
            })

    # --- Evaluator & Final Output ---
    draft = response.choices[0].message.content
    eval_result = evaluate_response(message, draft)

    if "PASS" in eval_result:
        return draft
    else:
        messages.append({"role": "user", "content": f"Correction needed: {eval_result}"})
        final = openai.chat.completions.create(model="gpt-4o-mini", messages=messages)
        return final.choices[0].message.content


# In[56]:


intro_blurb = f"""
### 👋 Welcome! I am the AI version of {name}.
I can discuss my professional experience, education, and technical skills in detail. 
I also have live access to my featured GitHub repositories.

**Try asking me:**
* "What are your top pinned projects on GitHub?"
* "Do you have experience with [specific technology]?"
* "I'd like to leave a message for Sirpreet."
"""

gr.ChatInterface(
    fn=chat, 
    type="messages", 
    title=f"{name}'s Professional AI Portfolio",
    description=intro_blurb, # This adds your blurb to the UI
).launch()


# In[ ]:




