---
title: Professional_Alter_Ego
app_file: app.py
sdk: gradio
sdk_version: 5.49.1
---
# Professional AI Alter Ego

Welcome to my AI-powered portfolio. This application is a digital version of my professional identity, capable of discussing my engineering background, academic coursework at SFU, and coding projects.

## 🚀 Features

- **Knowledge Retrieval (RAG):** The agent answers questions based on my official Resume, LinkedIn profile, and SFU course descriptions.
- **Live Project Fetching:** Ask the bot about my "pinned projects" to see real-time data fetched from my GitHub via GraphQL.
- **Direct Contact:** If you'd like to reach the "real" me, the agent can collect your contact details and send them to my phone instantly via Pushover.
- **Advocacy Engine:** Designed to find transferable skills; the agent doesn't just look for keywords—it advocates for my engineering potential.

## 🛠️ Tech Stack

- **LLM:** OpenAI GPT-4o-mini
- **Frameworks:** LangChain, Gradio
- **Vector Store:** ChromaDB
- **APIs:** GitHub GraphQL, Pushover

## 📂 Project Structure

- `app.py`: Main application logic.
- `me/`: Directory containing source PDFs (Resume, LinkedIn, SFU Courses).
- `chroma_db/`: Persistent vector store for fast retrieval.
- `requirements.txt`: Necessary Python dependencies.

## 📥 Getting in Touch

Feel free to interact with the chatbot to learn more about my work. If you have a specific inquiry, leave your details with the agent, and I will get back to you shortly!
