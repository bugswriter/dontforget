# 🧠 DontForget: The AI Second Brain

**DontForget** is a local, "God Mode" memory engine for your terminal. It allows you to dump raw thoughts, tasks, and ideas into a database and retrieve them later using natural language, powered by Google Gemini AI and SQLite.

Unlike complex RAG systems that require vector databases and embeddings, DontForget uses a **"Lazy Storage, Brute-Force Retrieval"** architecture. It stores raw text with smart AI tags and uses SQLite's Full-Text Search (FTS5) to hunt down information, feeding the results back to the LLM for synthesis.

## ✨ Features

* **⚡ Zero-Friction Capture:** Just type `mem r "anything..."`. The AI automatically generates searchable tags and summaries.
* **🔍 "God Mode" Retrieval:** Ask questions like *"What tasks did I have for Project Cyoni last week?"*. The system uses fuzzy search + time-filtering + AI analysis to find the exact answer.
* **🛠️ Bulletproof Architecture:** Uses a single SQLite table with an FTS5 index. No sync issues, no complex vector math, no "missing ID" bugs.
* **📊 Cost-Aware:** Every response shows you the token usage and record count, so you know exactly how much "brain power" you used.
* **🔐 Private & Secure:** Self-hosted on your machine. Protected by a Secret Key.
* **📝 Editor Support:** Automatically opens Vim/Nano for long notes.

## 🚀 Installation

### 1. Prerequisites

* Python 3.10+
* `uv` (for fast package management) or `pip`
* A Google Gemini API Key (Free tier works great)

### 2. Setup Server

```bash
# Clone or create directory
mkdir dontforget && cd dontforget

# Initialize project
uv init
uv add fastapi uvicorn python-dotenv google-genai pydantic

# Create .env file
echo 'GEMINI_API_KEY="your_gemini_key"' >> .env
echo 'DONTFORGET_SECRET_KEY="your_secret_password"' >> .env

```

### 3. Run Server

```bash
uv run main.py
# Server runs on http://0.0.0.0:8000

```

### 4. Setup CLI Tool (`mem`)

1. Copy the `mem` script to `/usr/local/bin/mem`.
2. Make it executable: `chmod +x /usr/local/bin/mem`.
3. Add your secret key and API URL to your shell config (`~/.bashrc` or `~/.zshrc`):
```bash
export DONTFORGET_SECRET_KEY="your_secret_password"
export DONTFORGET_API_URL="0.0.0.0:8000" # By default
```



---

## 📖 Usage

### Remember (Input)

Dump anything. The AI will tag it concepts (e.g., "debt", "finance") rather than just words.

```bash
mem r "Paid 432 rs to Akash for dinner"
# 🧠 Saved! [Tags: finance, debt, akash, dinner]

mem r "Fix the login bug on Cyoni project"
# 🧠 Saved! [Tags: project-cyoni, bug, urgent]

```

**Pro Tip:** Type `mem r` without arguments to open Vim for pasting long lists or code snippets.

### Remind (Query)

Ask naturally. You can filter by project, person, or time.

```bash
mem q "How much do I owe Akash?"
# Output: "You owe Akash 432 rs for dinner."

mem q "What are my pending tasks for Cyoni?"
# Output: "1. Fix login bug..."

```

### Delete (Forget)

Delete memories by describing them. The AI finds the best match.

```bash
mem d "That note about Akash"
# Output: "Deleted 1 item."

```

---

## 🏗️ Architecture

1. **Ingestion:**
* User sends text -> AI generates `tags` (Concepts) -> Stored in SQLite (Raw Table + FTS Index).


2. **Retrieval ("The Hunter"):**
* User asks question -> AI extracts `keywords` -> FTS5 performs a fuzzy search (Broad Match).
* **Context Stuffing:** The system retrieves the top 30 relevant rows and dumps them into the AI's context window.
* **Synthesis:** The AI reads the raw rows, filters out irrelevant noise (e.g., ignoring old dates if you asked for "today"), and answers.



## 🛡️ Troubleshooting

* **"Search Error"**: Usually means the database schema is out of sync. Delete `dontforget.db` and restart the server to rebuild cleanly.
* **"Connection Refused"**: Ensure the server is running (`uv run main.py`) and port 8000 is open.

---

**License:** GPL-3
**Author:** Suraj Kushwah
