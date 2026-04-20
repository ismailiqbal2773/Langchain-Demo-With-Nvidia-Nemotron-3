import streamlit as st
from dotenv import load_dotenv
import os
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI 

load_dotenv()

# Call OpenRouter key
OpenRouter_API_Key = os.getenv("OpenRouter_API_Key")

# LLM Configuration (For NVIDIA)
llm = ChatOpenAI(
    openai_api_base="https://openrouter.ai/api/v1",
    openai_api_key=OpenRouter_API_Key,
    model_name="nvidia/nemotron-3-super-120b-a12b:free"
)

# Professional prompt variations for experimentation
PROMPT_VARIANTS = {
    "Professional & Concise": {
        "system": "You are a professional AI assistant. Provide clear, concise, and accurate responses. Use proper formatting and structure your answers logically.",
        "human": "Question: {question}"
    },
    "Expert Consultant": {
        "system": "You are an expert consultant with deep domain knowledge. Provide detailed, well-reasoned responses with relevant examples and best practices. Use professional language.",
        "human": "User Query: {question}"
    },
    "Technical Specialist": {
        "system": "You are a technical specialist. Provide accurate, detailed technical explanations. Include relevant context, explain complex concepts clearly, and provide practical examples.",
        "human": "Technical Question: {question}"
    },
    "Formal Assistant": {
        "system": "You are a formal business assistant. Provide professional, polished responses suitable for business communication. Be thorough but concise.",
        "human": "Request: {question}"
    }
}

st.title('Langchain Demo With Nvidia Nemotron-3')

# Sidebar for prompt selection
st.sidebar.header("🔧 Experimentation Settings")
selected_variant = st.sidebar.selectbox(
    "Choose Prompt Style:",
    list(PROMPT_VARIANTS.keys()),
    index=0
)

# Display the selected prompt for learning
st.sidebar.subheader("📋 Active Prompt:")
st.sidebar.code(
    f"System: {PROMPT_VARIANTS[selected_variant]['system']}\n\nHuman: {PROMPT_VARIANTS[selected_variant]['human']}",
    language="text"
)

# Create prompt with selected variant
prompt = ChatPromptTemplate.from_messages([
    ("system", PROMPT_VARIANTS[selected_variant]["system"]),
    ("human", PROMPT_VARIANTS[selected_variant]["human"])
])

input_text = st.text_input("Enter your question here")

output_parser = StrOutputParser()
chain = prompt | llm | output_parser

if input_text:
    st.subheader(f"Response ({selected_variant}):")
    st.write(chain.invoke({'question': input_text}))