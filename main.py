import sqlite3
import datetime
import os
import json
from typing import Optional

from fastapi import FastAPI, HTTPException, Header, Depends, status
from pydantic import BaseModel
from dotenv import load_dotenv
from google import genai
from google.genai import types

# --- CONFIGURATION ---
load_dotenv()
GEMINI_KEY = os.getenv("GEMINI_API_KEY")
SERVER_SECRET = os.getenv("DONTFORGET_SECRET_KEY")
DB_PATH = "dontforget.db"
MODEL_ID = "gemini-2.0-flash" 

if not GEMINI_KEY:
    raise ValueError("GEMINI_API_KEY missing in .env")
if not SERVER_SECRET:
    raise ValueError("DONTFORGET_SECRET_KEY missing in .env")

app = FastAPI(title="DontForget - Private Memory API")
client = genai.Client(api_key=GEMINI_KEY)

# --- SECURITY ---
async def verify_api_key(x_api_key: str = Header(..., description="Your Server Secret")):
    if x_api_key != SERVER_SECRET:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API Key. You are not the owner.",
        )
    return x_api_key

# --- DATA MODELS ---
class ThoughtRequest(BaseModel):
    text: str

class QueryRequest(BaseModel):
    question: str

# --- DATABASE ---
def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE VIRTUAL TABLE IF NOT EXISTS memory 
        USING fts5(text, tags, timestamp, UNINDEXED);
    """)
    conn.commit()
    conn.close()

init_db()

def execute_sql(sql_query: str):
    """TOOL: Executes SQL query on 'memory' table."""
    try:
        if any(x in sql_query.upper() for x in ["DELETE", "DROP", "UPDATE", "INSERT"]):
            return "Error: Read-only access allowed."
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row 
        cursor = conn.execute(sql_query)
        rows = [dict(row) for row in cursor.fetchall()]
        conn.close()
        if not rows: return "No results found."
        return json.dumps(rows, default=str)
    except Exception as e:
        return f"SQL Error: {e}"

# --- ENDPOINTS (Protected) ---

@app.get("/health", dependencies=[Depends(verify_api_key)])
def health():
    return {"status": "online", "system": "DontForget"}

@app.post("/remember", dependencies=[Depends(verify_api_key)])
def remember(request: ThoughtRequest):
    try:
        # 1. AI Analysis
        prompt = f"Analyze this thought for a database. Input: '{request.text}'. Output JSON keys: 'tags' (csv strings)."
        resp = client.models.generate_content(
            model=MODEL_ID, contents=prompt,
            config=types.GenerateContentConfig(response_mime_type="application/json")
        )
        tags = json.loads(resp.text).get("tags", "general")

        # 2. Save
        ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute("INSERT INTO memory (text, tags, timestamp) VALUES (?, ?, ?)", 
                         (request.text, tags, ts))
        
        return {"status": "saved", "tags": tags}
    except Exception as e:
        raise HTTPException(500, str(e))

@app.post("/remind", dependencies=[Depends(verify_api_key)])
def remind(request: QueryRequest):
    try:
        today = datetime.datetime.now().strftime("%Y-%m-%d")
        sys_instruct = f"""
        You are 'DontForget', a memory assistant. Date: {today}.
        Database: 'memory' table (text, tags, timestamp).
        Goal: Answer user question by writing/executing SQL using 'execute_sql' tool.
        Use FTS5 matching or LIKE for dates. Self-correct if SQL fails.
        """
        
        chat = client.chats.create(
            model=MODEL_ID,
            config=types.GenerateContentConfig(tools=[execute_sql], system_instruction=sys_instruct, temperature=0.1)
        )
        
        response = chat.send_message(request.question)
        
        # ReAct Loop
        while response.candidates[0].content.parts[0].function_call:
            part = response.candidates[0].content.parts[0]
            query = part.function_call.args["sql_query"]
            print(f"🤖 Running SQL: {query}")
            
            result = execute_sql(query)
            
            response = chat.send_message(
                types.Part(function_response=types.FunctionResponse(
                    name="execute_sql", response={"result": result}
                ))
            )
            
        return {"answer": response.text}
    except Exception as e:
        print(e)
        raise HTTPException(500, "Memory retrieval failed.")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
