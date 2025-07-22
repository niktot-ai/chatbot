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

@cl.step(type="tool")
async def extract_fields(user_query):
    FIELD_EXTRACTION_SYSTEM_PROMPT = """
    You are an intelligent assistant that extracts structured investment information from user messages and performs all necessary calculations.

    Extract the following fields and perform calculations:
    - transaction_type: "lumpsum" or "sip" 
    - known_amount: "investment" or "goal" (if it's goal-oriented then "goal", otherwise "investment")
    - investment_amount: float (rupees) — Calculate the required investment amount
    - goal_amount: float (rupees) — Calculate the inflation-adjusted goal amount
    - number_of_years: integer
    - risk_profile: "conservative", "moderate", or "aggressive"
    - expected_return: float — If not mentioned, use default based on risk profile
    - inflation_rate: float — default to 0.0 unless user specifies a rate
    - monthly_sip: float — Only if user explicitly mentions a SIP amount

    CALCULATION RULES:
    1. **For goal-oriented investments with existing assets:**
       - Step 1: Calculate inflation-adjusted goal = current_cost × (1 + inflation_rate)^years
       - Step 2: Calculate future value of existing assets = existing_assets × (1 + existing_growth_rate)^years  
       - Step 3: Calculate additional investment needed = inflation_adjusted_goal - future_existing_assets
       - Set goal_amount = inflation_adjusted_goal
       - Set investment_amount = additional_investment_needed

    2. **For goal-oriented investments without existing assets:**
       - Calculate inflation-adjusted goal = current_cost × (1 + inflation_rate)^years
       - Set goal_amount = inflation_adjusted_goal
       - Calculate investment_amount based on transaction_type

    3. **Risk Profile Inference:**
       - If expected_return ≤ 0.08 → conservative
       - If expected_return > 0.08 and ≤ 0.10 → moderate  
       - If expected_return > 0.10 → aggressive

    4. **Default Expected Returns:**
       - Conservative → 0.08, Moderate → 0.10, Aggressive → 0.12
    
    5. If `goal_amount` is provided and `inflation_rate` is also provided:
   - Calculate adjusted_goal = round(goal_amount × (1 + inflation_rate) ^ number_of_years, 2)
   - Set goal_amount = adjusted_goal

    6. If `goal_amount` is known (after inflation adjustment if applicable) and `investment_amount` is null:
    - For `transaction_type = "lumpsum"`:
    - investment_amount = round(goal_amount / ((1 + expected_return) ^ number_of_years), 2)

   - For `transaction_type = "sip"`:
     - r = expected_return / 12
     - n = number_of_years * 12
     - factor = ((1 + r) ^ n - 1) / r
     - monthly_sip = round(goal_amount / factor, 2)

7. Only perform the above calculations if the field is **null**. If the field already has a valid value, do not modify or overwrite it.

    EXAMPLE CALCULATION for house buying scenario:
    - Current house cost: ₹90,00,000
    - Inflation rate: 7 percent for 25 years
    - Existing assets: ₹10,00,000 growing at 10 percent annually
    
    Calculations:
    - Inflation-adjusted goal = 9000000 * (1.07)^25 = 48734972
    - Future existing assets = 1000000 * (1.10)^25 = 10834706  
    - Additional investment needed = 48734972 - 10834706 = 37900266
    
    Result should be:
    - goal_amount: 48734972
    - investment_amount: 37900266

    Additional Rules:
    - Perform ALL calculations within this extraction step
    - Only extract what is clearly stated or calculable
    - Do NOT include explanation, only return JSON
    - Be precise with mathematical calculations

    Respond in strict JSON format:
    {
    "transaction_type": "lumpsum",
    "known_amount": "goal",
    "investment_amount": 37900266,
    "goal_amount": 48734972,
    "number_of_years": 25,
    "risk_profile": "moderate",
    "expected_return": 0.1,
    "inflation_rate": 0.07,
    "monthly_sip": null
    }
    """
    
    response = client.chat.completions.create(
        model = "moonshotai/kimi-k2-instruct",
        messages= chat_history + [
            {"role": "system", "content": FIELD_EXTRACTION_SYSTEM_PROMPT},
            {"role": "user", "content": user_query}
        ],
        temperature=0
    )
    content = response.choices[0].message.content.strip()
    
    cleaned_content = re.sub(r'<think>[\s\S]*?<\/think>', '', content)  
    print(cleaned_content)
    match = re.search(r'({[\s\S]*?})', cleaned_content)
    if match:
        json_str = match.group(1)
    else:
        json_str = cleaned_content
    try:
        parsed_json = json.loads(json_str)
    except Exception as e:
        print("Failed to parse LLM output:", content)
        raise e
    print("parsed_json", parsed_json)
    return parsed_json

@cl.step(type="tool")
async def generate_followup_question(fields):
    prompt = f"""
        You are a smart and polite assistant. Based on the available fields, either return a follow-up question OR return the existing field values AS-IS.
    You MUST detect the user's language based on the message content and respond in the same language if possible.
    
    CRITICAL RULES:
    1. DO NOT recalculate or modify any existing field values
    2. DO NOT override any non-null values
    3. Only fill in missing (null) fields that can be derived
    4. If all required fields are present, return the fields exactly as provided
    
    Rules:
    1. If no follow-up question is needed and all required values are either given or can be derived, return the field values as JSON EXACTLY AS PROVIDED.
    2. If only investment amount is given then no need to ask for the goal amount, first clarify the known_amount it is investment type or goal type
    3. Derive `expected_return` based on `risk_profile` ONLY if and only if expected_return is null:
       - Conservative → 0.08
       - Moderate → 0.10
       - Aggressive → 0.12
    4. If a field is missing and cannot be derived, return a **single natural-language follow-up question** to ask the user for that missing field.
    
    Required fields to check:
    - transaction_type (must not be null)
    - known_amount (must not be null) 
    - investment_amount OR monthly_sip (depending on transaction_type)
    - number_of_years (must not be null)
    - risk_profile (must not be null)
    - expected_return (can be derived from risk_profile if null)
    
    Your response must be:
    - If follow-up needed → return just the follow-up question
    - If all is complete → return the EXACT same JSON with all fields preserved
    - DO NOT explain anything
    - DO NOT return both question and JSON
    - DO NOT modify any existing values

    fields:
    {fields}
    """

    response = client.chat.completions.create(
        model = "deepseek-r1-distill-llama-70b",
        messages= [
            {"role": "system", "content": prompt}
        ],
        temperature=0
    )
    content =  response.choices[0].message.content.strip()
    cleaned_content = re.sub(r'<think>[\s\S]*?<\/think>', '', content)
    return cleaned_content


async def get_all_fields(user_query):
    fields = await extract_fields(user_query)
    followup_question = await generate_followup_question(fields)   
    match = re.search(r'({[\s\S]*?})', followup_question)
    if match:
        json_str = match.group(1)
    else:
        json_str = followup_question
        chat_history.append({"role": "assistant", "content":json_str})
        await cl.Message(content=json_str).send()
    try:
        parsed_json = json.loads(json_str)
    except Exception as e:
        print("Failed to parse LLM output:", json_str)
        raise e
    print("json string:", json_str)    
    print(parsed_json)
    return parsed_json

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

    chat_history.append({"role": "user", "content":user_query})
    print("chat_history", chat_history)
    fields = await get_all_fields(user_query)
    print("all the fields",fields)
