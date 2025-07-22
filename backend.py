from openai import OpenAI
import os
from dotenv import load_dotenv
import re
import json
import pandas as pd
import chainlit as cl

load_dotenv()

client = OpenAI(base_url="https://api.groq.com/openai/v1", api_key=os.environ.get("GROQ_API_KEY"))

chat_history = [
    {'role': 'system', 'content': "You are a helpful assistant."}
]

data = pd.read_csv('data.csv')

async def handle_general_query(user_query):
    general_prompt = """
    You are a friendly, knowledgeable, and polite finance chatbot assistant. Your primary role is to assist users with finance-related questions—especially mutual fund recommendations based on their investment goals, risk appetite, and time horizon.

    Respond clearly, concisely, and respectfully to all queries. Do not answer to general question, ask user for help about personalized fund recommendation. 

    If the user asks something unrelated to finance, respond politely and gently redirect them back to your area of expertise. You can say something like:
    "I'm here to help with financial topics—especially mutual fund recommendations and investment-related questions. If you’d like assistance with your investments, feel free to share your goals or risk profile!"

    Always keep a warm, approachable tone and guide users toward making informed financial decisions.
    """
    response = client.chat.completions.create(
        model="moonshotai/kimi-k2-instruct",
        messages= chat_history + [
            {"role": "system", "content": general_prompt},
            {"role": "user", "content": user_query}
        ],
        temperature=0.7
    )
    return response.choices[0].message.content.strip()

@cl.step(type="tool")
async def classify_user_query(user_query):
    routing_prompt = """
    You are a smart assistant that classifies user queries for a finance chatbot.

    Based on the user message, classify it as one of the following:
    - "investment_query" → if the user is providing or intending to provide financial details for investment, such as amount, goal, risk, duration, SIP/lumpsum, etc.
    - "general_query" → if the user is asking general questions, greetings, or anything not related to providing investment input.

    Respond with ONLY one of: "investment_query" or "general_query". Do NOT explain anything.
    """
    response = client.chat.completions.create(
        model="moonshotai/kimi-k2-instruct",
        messages= chat_history + [
            {"role": "system", "content": routing_prompt},
            {"role": "user", "content": user_query}
        ],
        temperature=0
    )
    return response.choices[0].message.content.strip().strip('"').lower()

@cl.on_message
async def main(message: cl.Message):
    user_query = message.content.strip().lower()
    print("chat_history", chat_history)
    
    classification = await classify_user_query(user_query)
    classification = re.sub(r'<think>[\s\S]*?<\/think>', '', classification)
    classification = classification.strip().lower()

    if classification == "general_query":
        response = await handle_general_query(user_query)   
        response = re.sub(r'<think>[\s\S]*?<\/think>', '', response)
        await cl.Message(content=response).send()
        return
