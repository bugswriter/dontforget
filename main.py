import sqlite3
import datetime
import os
import json
import re
from typing import List, Optional
from fastapi import FastAPI, HTTPException, Header, Depends
from pydantic import BaseModel
from dotenv import load_dotenv
from google import genai
from google.genai import types

# --- CONFIGURATION ---
load_dotenv()
GEMINI_KEY = os.getenv("GEMINI_API_KEY")
SERVER_SECRET = os.getenv("DONTFORGET_SECRET_KEY")
DB_PATH = "memory.db"
MODEL_ID = "gemini-2.5-flash" 

if not GEMINI_KEY or not SERVER_SECRET:
    raise ValueError("Missing keys in .env file.")

app = FastAPI(title="DontForget - Bulletproof")
client = genai.Client(api_key=GEMINI_KEY)

# --- SECURITY ---
async def verify_api_key(x_api_key: str = Header(..., description="Server Secret")):
    if x_api_key != SERVER_SECRET:
        raise HTTPException(status_code=401, detail="Unauthorized")
    return x_api_key

# --- MODELS ---
class ThoughtRequest(BaseModel):
    text: str

class QueryRequest(BaseModel):
    question: str

# --- DATABASE (The Fix) ---
def init_db():
    conn = sqlite3.connect(DB_PATH)
    # We use ONE table. FTS5 tables can store data directly.
    # Columns: raw_text, ai_tags, timestamp
    # Note: FTS5 tables are virtual. "UNINDEXED" means we can store it but not fuzzy search it (saves speed),
    # but for simplicity, we will just make it all searchable or normal columns.
    # To keep it bulletproof: Standard Table for data, FTS for index.
    
    conn.execute("""
        CREATE TABLE IF NOT EXISTS memories (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            raw_text TEXT,
            ai_tags TEXT
        );
    """)
    
    # FTS Index linked to the table (Contentless or Content-managed is complex, we use triggers for safety)
    conn.execute("""
        CREATE VIRTUAL TABLE IF NOT EXISTS memories_idx 
        USING fts5(raw_text, ai_tags, content='memories', content_rowid='id');
    """)

    # Triggers to keep them in sync automatically
    conn.execute("""
        CREATE TRIGGER IF NOT EXISTS memories_ai AFTER INSERT ON memories BEGIN
          INSERT INTO memories_idx(rowid, raw_text, ai_tags) VALUES (new.id, new.raw_text, new.ai_tags);
        END;
    """)
    conn.execute("""
        CREATE TRIGGER IF NOT EXISTS memories_ad AFTER DELETE ON memories BEGIN
          INSERT INTO memories_idx(memories_idx, rowid, raw_text, ai_tags) VALUES('delete', old.id, old.raw_text, old.ai_tags);
        END;
    """)
    
    conn.commit()
    conn.close()

init_db()

# --- HELPER FUNCTIONS ---

def estimate_tokens(text: str) -> int:
    return len(str(text)) // 4

def execute_fuzzy_search(keywords: List[str]):
    """
    Search mechanism.
    1. Tries to match ALL keywords first (AND).
    2. If fail, tries ANY keyword (OR).
    """
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    
    # Clean keywords to prevent syntax errors
    clean_keys = [re.sub(r'[^a-zA-Z0-9]', '', k) for k in keywords if k]
    if not clean_keys: return []

    # Strategy 1: AND (Specific)
    query_and = " AND ".join(f'"{k}"' for k in clean_keys)
    sql = f"SELECT rowid, * FROM memories_idx WHERE memories_idx MATCH ? ORDER BY rank LIMIT 30"
    
    rows = []
    try:
        cursor = conn.execute(sql, (query_and,))
        rows = [dict(row) for row in cursor.fetchall()]
    except:
        pass

    # Strategy 2: OR (Broad) - If AND returned too few results
    if len(rows) < 5:
        query_or = " OR ".join(f'"{k}"' for k in clean_keys)
        try:
            sql = f"SELECT rowid, * FROM memories_idx WHERE memories_idx MATCH ? ORDER BY rank LIMIT 30"
            cursor = conn.execute(sql, (query_or,))
            new_rows = [dict(row) for row in cursor.fetchall()]
            # Merge logic (simple dict update by rowid to dedup)
            existing_ids = {r['rowid'] for r in rows}
            for nr in new_rows:
                if nr['rowid'] not in existing_ids:
                    rows.append(nr)
        except:
            pass

    # Fetch timestamps from main table (since FTS doesn't have it easily)
    final_results = []
    if rows:
        ids = [r['rowid'] for r in rows]
        id_list = ",".join(str(i) for i in ids)
        # Get full data including timestamp
        sql_full = f"SELECT id, raw_text, ai_tags, timestamp FROM memories WHERE id IN ({id_list})"
        cursor = conn.execute(sql_full)
        final_results = [dict(row) for row in cursor.fetchall()]

    conn.close()
    return final_results

def delete_by_ids(ids: List[int]):
    conn = sqlite3.connect(DB_PATH)
    placeholders = ','.join('?' * len(ids))
    conn.execute(f"DELETE FROM memories WHERE id IN ({placeholders})", ids)
    conn.commit()
    conn.close()

# --- ENDPOINTS ---

@app.post("/remember", dependencies=[Depends(verify_api_key)])
def remember(request: ThoughtRequest):
    try:
        # 1. AI Tagging (Just add metadata to help search find it)
        prompt = f"""
        Generate 5 search tags for this thought.
        Input: "{request.text}"
        Format: JSON {{ "tags": ["tag1", "tag2"] }}
        """
        resp = client.models.generate_content(
            model=MODEL_ID, 
            contents=prompt,
            config=types.GenerateContentConfig(response_mime_type="application/json")
        )
        tags = json.loads(resp.text).get("tags", [])
        tags_str = ", ".join(tags)

        # 2. Store
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute(
                "INSERT INTO memories (raw_text, ai_tags) VALUES (?, ?)", 
                (request.text, tags_str)
            )
            
        print(f"🧠 Saved: {request.text[:30]}... [Tags: {tags_str}]")
        return {"status": "saved", "tags": tags} # Returns list for CLI
        
    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(500, str(e))

@app.post("/remind", dependencies=[Depends(verify_api_key)])
def remind(request: QueryRequest):
    try:
        today = datetime.datetime.now().strftime("%Y-%m-%d %A")
        
        # 1. Extract Keywords
        prompt_strat = f"""
        User Query: "{request.question}"
        Extract 3-5 keywords to search the database.
        Format: JSON {{ "keywords": ["word1", "word2"] }}
        """
        resp_strat = client.models.generate_content(
            model=MODEL_ID, 
            contents=prompt_strat,
            config=types.GenerateContentConfig(response_mime_type="application/json")
        )
        keywords = json.loads(resp_strat.text).get("keywords", [])
        print(f"🕵️ Keywords: {keywords}")

        # 2. Search DB
        rows = execute_fuzzy_search(keywords)
        
        # 3. Contextualize
        context_str = ""
        for r in rows:
            context_str += f"[ID:{r['id']}] [{r['timestamp']}] {r['raw_text']}\n"

        token_count = estimate_tokens(context_str)
        if token_count > 6000: context_str = context_str[:24000] + "\n...[TRUNCATED]"

        # 4. Answer
        final_prompt = f"""
        You are a Memory Assistant.
        Date: {today}
        User Query: "{request.question}"
        
        Relevant Memories:
        {context_str}
        
        Task: 
        1. Answer the question using ONLY the memories above.
        2. If asking for "Today" or "Last week", check the timestamps.
        3. If no relevant memories found, say "No relevant info found."
        """
        
        resp_final = client.models.generate_content(model=MODEL_ID, contents=final_prompt)
        
        return {
            "answer": resp_final.text,
            "stats": {"found": len(rows), "tokens": token_count}
        }

    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(500, str(e))

@app.post("/delete", dependencies=[Depends(verify_api_key)])
def delete_endpoint(request: QueryRequest):
    try:
        # Same search logic
        keywords = request.question.split()
        rows = execute_fuzzy_search(keywords)
        
        if not rows: return {"answer": "No items found."}

        # Ask AI which ID matches best
        context = "\n".join([f"ID:{r['id']} Text:{r['raw_text']}" for r in rows])
        prompt = f"""
        User wants to delete: "{request.question}"
        Which IDs match?
        Options:
        {context}
        Return JSON {{ "ids": [1] }} or {{ "ids": [] }}
        """
        resp = client.models.generate_content(
            model=MODEL_ID, 
            contents=prompt,
            config=types.GenerateContentConfig(response_mime_type="application/json")
        )
        ids = json.loads(resp.text).get("ids", [])
        
        if ids:
            delete_by_ids(ids)
            return {"answer": f"Deleted {len(ids)} items."}
        else:
            return {"answer": "Found items, but none matched exactly."}

    except Exception as e:
        raise HTTPException(500, str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
