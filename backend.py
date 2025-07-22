from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict, Any, List
import uuid
import json
import re
import pandas as pd
from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="Mutual Fund Chatbot API", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this based on your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize OpenAI client
client = OpenAI(base_url="https://api.groq.com/openai/v1", api_key=os.environ.get("GROQ_API_KEY"))

# Load data
data = pd.read_csv('data.csv')

# In-memory storage for chat sessions (use Redis or database in production)
chat_sessions: Dict[str, List[Dict[str, str]]] = {}

# Pydantic models for request/response
class ChatRequest(BaseModel):
    session_id: Optional[str] = None
    message: str

class ChatResponse(BaseModel):
    session_id: str
    response: Any  # Can be string (followup question) or dict (recommendations)
    response_type: str  # "followup_question" or "recommendations" or "general_response"

class SessionResponse(BaseModel):
    session_id: str
    message: str

# Helper functions (keeping your existing logic)
def get_time_horizon(years):
    if years <= 4:
        return 'short'
    elif years <= 10:
        return 'medium'
    else:
        return 'long'

def get_allocation_percentage(risk_profile, time_horizon):
    matrix = {
        "aggressive": {"long": {"equity": 80, "hybrid": 10, "debt": 10},
                       "medium": {"equity": 60, "hybrid": 20, "debt": 20},
                       "short": {"equity": 30, "hybrid": 30, "debt": 40}},
        "moderate": {"long": {"equity": 80, "hybrid": 10, "debt": 10},
                     "medium": {"equity": 40, "hybrid": 20, "debt": 40},
                     "short": {"equity": 20, "hybrid": 40, "debt": 40}},
        "conservative": {"long": {"equity": 40, "hybrid": 30, "debt": 30},
                         "medium": {"equity": 20, "hybrid": 30, "debt": 50},
                         "short": {"equity": 10, "hybrid": 30, "debt": 60}}
    }
    return matrix[risk_profile][time_horizon]

def get_return_column(transaction_type, number_of_years):
    if number_of_years < 2:
        return 'OneYearReturns' if transaction_type == 'lumpsum' else 'SYRET1'
    elif number_of_years <= 4:
        return 'ThreeYearReturns' if transaction_type == 'lumpsum' else 'SYRET3'
    else:
        return 'FiveYearReturns' if transaction_type == 'lumpsum' else 'SYRET5'

def filter_funds(equity, debt, hybrid, allocation, number_of_years):
    if number_of_years <= 4:
        equity = equity[~equity['Category'].isin(['Mid Cap', 'Small Cap'])]

    if allocation.get('hybrid', 0) == 20:
        hybrid = hybrid[hybrid['Category'].isin(['Dynamic Asset Allocation', 'Multi Asset Allocation'])]

    debt = debt[debt['Category'].isin([
        'Corporate Bond', 'Short Duration', 'Banking and PSU', 'Liquid/Overnight'
    ])]

    return equity, debt, hybrid

def get_top_funds(equity, debt, hybrid):
    return (
        equity.groupby('Category').head(1),
        debt.groupby('Category').head(1),
        hybrid.groupby('Category').head(1)
    )

async def extract_fields(user_query, chat_history):
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
        model="moonshotai/kimi-k2-instruct",
        messages=chat_history + [
            {"role": "system", "content": FIELD_EXTRACTION_SYSTEM_PROMPT},
            {"role": "user", "content": user_query}
        ],
        temperature=0
    )
    content = response.choices[0].message.content.strip()
    
    cleaned_content = re.sub(r'<think>[\s\S]*?</think>', '', content)
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
    
    return parsed_json

async def handle_general_query(user_query, chat_history):
    general_prompt = """
    You are a friendly, knowledgeable, and polite finance chatbot assistant. Your primary role is to assist users with finance-related questions—especially mutual fund recommendations based on their investment goals, risk appetite, and time horizon.

    Respond clearly, concisely, and respectfully to all queries. Do not answer to general question, ask user for help about personalized fund recommendation. 

    If the user asks something unrelated to finance, respond politely and gently redirect them back to your area of expertise. You can say something like:
    "I'm here to help with financial topics—especially mutual fund recommendations and investment-related questions. If you'd like assistance with your investments, feel free to share your goals or risk profile!"

    Always keep a warm, approachable tone and guide users toward making informed financial decisions.
    """
    response = client.chat.completions.create(
        model="moonshotai/kimi-k2-instruct",
        messages=chat_history + [
            {"role": "system", "content": general_prompt},
            {"role": "user", "content": user_query}
        ],
        temperature=0.7
    )
    return response.choices[0].message.content.strip()

async def classify_user_query(user_query, chat_history):
    routing_prompt = """
    You are a smart assistant that classifies user queries for a finance chatbot.

    Based on the user message, classify it as one of the following:
    - "investment_query" → if the user is providing or intending to provide financial details for investment, such as amount, goal, risk, duration, SIP/lumpsum, etc.
    - "general_query" → if the user is asking general questions, greetings, or anything not related to providing investment input.

    Respond with ONLY one of: "investment_query" or "general_query". Do NOT explain anything.
    """
    response = client.chat.completions.create(
        model="moonshotai/kimi-k2-instruct",
        messages=chat_history + [
            {"role": "system", "content": routing_prompt},
            {"role": "user", "content": user_query}
        ],
        temperature=0
    )
    return response.choices[0].message.content.strip().strip('"').lower()

async def generate_followup_question(fields, chat_history):
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
        model="deepseek-r1-distill-llama-70b",
        messages=[
            {"role": "system", "content": prompt}
        ],
        temperature=0.1
    )
    content = response.choices[0].message.content.strip()
    cleaned_content = re.sub(r'<think>[\s\S]*?<\/think>', '', content)
    return cleaned_content

async def get_all_fields(user_query, chat_history):
    fields = await extract_fields(user_query, chat_history)
    followup_question = await generate_followup_question(fields, chat_history)
    
    match = re.search(r'({[\s\S]*?})', followup_question)
    if match:
        json_str = match.group(1)
        try:
            parsed_json = json.loads(json_str)
            return {"type": "complete", "data": parsed_json}
        except Exception as e:
            print("Failed to parse LLM output:", json_str)
            return {"type": "followup", "data": followup_question}
    else:
        return {"type": "followup", "data": followup_question}

def call_ai_for_recommendation(equity, debt, hybrid, allocation, allocated_amount, transaction_type, number_of_years, time_horizon, chat_history):
    prompt = f"""
    # MUTUAL FUND RECOMMENDATION SYSTEM

## CORE REQUIREMENTS
- Recommend ONLY funds from the provided 'mutual_funds equity, debt and hybrid data'

## FUND SELECTION RULES
### Portfolio Constraints:
- Maximum 4 equity subnatures
- Maximum 2 schemes per AMC (Asset management company)  
- One scheme per subnature
- NO Small Cap/Mid Cap for short-term goals
- NO ELSS unless explicitly requested
- Prioritize diversification and AMC rotation

### Subnature Guidelines:
- **Equity**: Prefer Multi Cap for moderate+ profiles with long-term goals
- **Hybrid (≥20%)**: Split between Balanced Advantage Fund (BAF) and Multi Asset
- **Debt**: Use Corporate Bond, Short Duration, Banking & PSU, Liquid based on horizon

### SIP Constraints (if applicable):
- Minimum ₹1,000 per scheme
- Must comply with fund's SIPMinAmt and SIPMaxAmt
- Increments must follow StepUpMulAmt

### Minimum SIP Requirements:
SIP Allocation Matrix
**Apply ONLY when SIP is specifically requested:**

**Conservative Risk:**
- Short-term: Min ₹3,000 SIP, 3 funds
- Medium-term: Min ₹4,000 SIP, 4 funds
- Long-term: Min ₹5,000 SIP, 5 funds

**Moderate Risk:**
- Short-term: Min ₹4,000 SIP, 4 funds
- Medium-term: Min ₹5,000 SIP, 5 funds
- Long-term: Min ₹6,000 SIP, 6 funds

**Aggressive Risk:**
- Short-term: Min ₹4,000 SIP, 4 funds
- Medium-term: Min ₹5,000 SIP, 5 funds
- Long-term: Min ₹6,000 SIP, 6 funds

**Matrix Rules:**
- These are minimums - scale up if goal requires more
- If user's amount is below minimum, explain insufficiency and suggest adjustments
- Maintain fund count consistency with guidelines

## AMOUNT VALIDATION
### Lumpsum Rules:
- All amounts must be multiples of fund's `MultipleInvestment`
- Formula: `round(calculated_amount / MultipleInvestment) * MultipleInvestment`

### SIP Rules:
- Amount must be within SIPMinAmt to SIPMaxAmt range
- Must be multiples of StepUpMulAmt
- Formula: `max(SIPMinAmt, round(calculated_sip / StepUpMulAmt) * StepUpMulAmt)`

## DATA SCHEMA REFERENCE
**Key Fields:**
- `Encrypt_SchemeCode`: Unique fund ID
- `SchemeName`: Full fund name
- `Nature`: Equity/Debt/Hybrid
- `MinimumInvestment`, `MultipleInvestment`: Lumpsum constraints
- `SIPMinAmt`, `SIPMaxAmt`, `StepUpMulAmt`: SIP constraints
- `OneYearReturns`, `ThreeYearReturns`, `FiveYearReturns`: Lumpsum returns
- `SYRET1`, `SYRET3`, `SYRET5`: SIP returns

## CRITICAL RULES
1. **Cumulative Investment**: Add new investments to existing ones (never replace)
2. **Mathematical Accuracy**: All allocations must sum to exactly ₹total_investment:, (100%)
3. **Compliance First**: Adjust amounts to meet fund constraints, then rebalance proportionally

## RESPONSE FORMAT
Respond in this exact JSON structure and use appropriate fields and only give the json response:

{{
  "goal_oriented": true/false,
  "goal_name": "<goal_name_if_applicable>",
  "transaction_type": "{transaction_type}",
  "time_horizon": "{time_horizon}",
  "number_of_years": {number_of_years},
  "allocation": {{
    "equity": {allocation['equity']},
    "hybrid": {allocation['hybrid']}, 
    "debt": {allocation['debt']}
  }},
  "mutual_fund_recommendations": [
    {{
      "scheme_no": <serial_number>,
      "scheme_name": "<exact_fund_name_from_data>",
      "asset_class": "<equity/hybrid/debt>",
      "allocation_percent": <percent>,
      "allocation_amount": <compliant_amount>,
      "Encrypt_SchemeCode": "<scheme_code>",
      "Category": "<category>",
      "Nature": "<nature>",
      "MinimumInvestment": <min_investment>,
      "MultipleInvestment": <multiple_investment>,
      "SIPMinAmt": <sip_min>,
      "SIPMaxAmt": <sip_max>,
      "StepUpMulAmt": <step_up_multiple>,
      "OneYearReturns": <returns>,
      "ThreeYearReturns": <returns>,
      "FiveYearReturns": <returns>,
      "SYRET1": <sip_returns>,
      "SYRET3": <sip_returns>,
      "SYRET5": <sip_returns>,
      "rationale": "<selection_reason>",
      "amount_adjustment": "<compliance_adjustments>"
    }}
  ],
  "validation_summary": {{
    "total_allocated": <sum_of_allocations>,
    "compliance_check": "<validation_status>",
    "adjustments_made": "<summary_of_changes>"
  }}
}}
allocation amount: {allocated_amount}
allocation percentage: {allocation}

equity funds: 
{equity}

debt funds: 
{debt}

hybrid funds: 
{hybrid}
    """
    response = client.chat.completions.create(
        model="meta-llama/llama-4-scout-17b-16e-instruct",
        messages= [
            {"role": "system", "content": prompt}
        ],
        temperature=0.1,
    )
    
    extracted_json = response.choices[0].message.content.strip()
    return extracted_json

async def recommend_funds(fields, chat_history):
    transaction_type = fields['transaction_type']
    number_of_years = fields['number_of_years']
    
    if transaction_type == 'sip' and fields.get('monthly_sip'):
        amount = fields['monthly_sip']
    else:
        amount = fields['investment_amount']
    
    risk_profile = fields['risk_profile']

    time_horizon = get_time_horizon(number_of_years)
    allocation = get_allocation_percentage(risk_profile, time_horizon)
    allocated_amount = {category: (percentage / 100) * amount for category, percentage in allocation.items()}

    return_column = get_return_column(transaction_type, number_of_years)

    equity = data[data['Nature'] == 'Equity'].sort_values(by=return_column, ascending=False)
    hybrid = data[data['Nature'] == 'Hybrid'].sort_values(by=return_column, ascending=False)
    debt = data[data['Nature'] == 'Debt'].sort_values(by=return_column, ascending=False)

    equity, debt, hybrid = filter_funds(equity, debt, hybrid, allocation, number_of_years)
    equity, debt, hybrid = get_top_funds(equity, debt, hybrid)
    
    equity_csv = equity.to_csv(index=False)
    debt_csv = debt.to_csv(index=False)
    hybrid_csv = hybrid.to_csv(index=False)
    
    return call_ai_for_recommendation(equity_csv, debt_csv, hybrid_csv, allocation, allocated_amount, transaction_type, number_of_years, time_horizon, chat_history)

# API Endpoints
@app.post("/chat/start", response_model=SessionResponse)
async def start_chat_session():
    """Start a new chat session"""
    session_id = str(uuid.uuid4())
    chat_sessions[session_id] = [
        {'role': 'system', 'content': "You are a helpful assistant."}
    ]
    return SessionResponse(session_id=session_id, message="Chat session started")

@app.post("/chat/message", response_model=ChatResponse)
async def send_message(request: ChatRequest):
    """Send a message to the chatbot"""
    # Handle session management
    if request.session_id and request.session_id in chat_sessions:
        session_id = request.session_id
    else:
        session_id = str(uuid.uuid4())
        chat_sessions[session_id] = [
            {'role': 'system', 'content': "You are a helpful assistant."}
        ]
    
    chat_history = chat_sessions[session_id]
    user_query = request.message.strip().lower()
    
    try:
        # Classify the query
        classification = await classify_user_query(user_query, chat_history)
        classification = re.sub(r'<think>[\s\S]*?<\/think>', '', classification)
        classification = classification.strip().lower()

        # Add user message to history
        chat_history.append({"role": "user", "content": user_query})
        
        if classification == "general_query":
            response = await handle_general_query(user_query, chat_history)
            response = re.sub(r'<think>[\s\S]*?<\/think>', '', response)
            chat_history.append({"role": "assistant", "content": response})
            
            return ChatResponse(
                session_id=session_id,
                response=response,
                response_type="general_response"
            )
        
        # Handle investment query
        result = await get_all_fields(user_query, chat_history)
        
        if result["type"] == "followup":
            # Return followup question
            chat_history.append({"role": "assistant", "content": result["data"]})
            return ChatResponse(
                session_id=session_id,
                response=result["data"],
                response_type="followup_question"
            )
        
        # Generate recommendations
        fields = result["data"]
        recommendation_result = await recommend_funds(fields, chat_history)
        
        # Parse the recommendation result
        match = re.search(r"```json(.*?)```", recommendation_result, re.DOTALL)
        if match:
            json_str = match.group(1).strip()
        else:
            json_str = recommendation_result.strip()
        
        try:
            recommendation_json = json.loads(json_str)
            chat_history.append({"role": "assistant", "content": json.dumps(recommendation_json)})
            
            return ChatResponse(
                session_id=session_id,
                response=recommendation_json,
                response_type="recommendations"
            )
        except json.JSONDecodeError:
            # If JSON parsing fails, return the raw result
            chat_history.append({"role": "assistant", "content": recommendation_result})
            return ChatResponse(
                session_id=session_id,
                response=recommendation_result,
                response_type="recommendations"
            )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing message: {str(e)}")

@app.get("/chat/history/{session_id}")
async def get_chat_history(session_id: str):
    """Get chat history for a session"""
    if session_id not in chat_sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    return {"session_id": session_id, "history": chat_sessions[session_id]}

@app.delete("/chat/session/{session_id}")
async def delete_chat_session(session_id: str):
    """Delete a chat session"""
    if session_id not in chat_sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    del chat_sessions[session_id]
    return {"message": "Session deleted successfully"}

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
