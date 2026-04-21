# 🎬 AutoStream AI Agent
### Social-to-Lead Agentic Workflow | ServiceHive × Inflx Assignment

An agentic conversational AI built with **LangGraph + Claude Haiku** that converts social media conversations into qualified business leads for AutoStream — a fictional SaaS video editing platform.

---

## 📁 Project Structure

```
autostream-agent/
├── agent.py                        # Core agent logic (LangGraph graph)
├── requirements.txt                # Python dependencies
├── README.md                       # This file
└── knowledge_base/
    └── autostream_kb.json          # RAG knowledge base (pricing, policies, FAQs)
```

---

## 🚀 How to Run Locally

### Prerequisites
- Python 3.9+
- An [groq API key](https://console.groq.com//)

### Step 1 — Clone & Install

```bash
git clone https://github.com/YOUR_USERNAME/autostream-agent.git
cd autostream-agent

python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

### Step 2 — Set Your API Key

```bash
export GROQ_API_KEY="sk-ant-..."    # Linux / macOS
# OR
set GROQ_API_KEY=sk-ant-...         # Windows CMD
```

Or create a `.env` file:
```
GROQ_API_KEY=sk-ant-...
```

### Step 3 — Run the Agent

```bash
python agent.py
```

You'll see an interactive CLI session:

```
============================================================
  🎬  AutoStream AI Assistant — Powered by Inflx/ServiceHive
============================================================

Aria: Hi! 👋 I'm Aria, your AutoStream assistant...

You: Hi, tell me about your pricing.
Aria: ...
```

### Example Conversation Flow

```
You: Hi, tell me about your pricing.
Aria: [Retrieves pricing from KB — Basic $29/mo, Pro $79/mo]

You: What's the refund policy?
Aria: [Retrieves policy — No refunds after 7 days]

You: That sounds great, I want to try the Pro plan for my YouTube channel.
Aria: [Detects HIGH INTENT] Fantastic! To get started, could you share your full name?

You: Alex Johnson
Aria: Great to meet you, Alex! What's the best email address to reach you at?

You: alex@example.com
Aria: Almost there! Which platform do you primarily create content for?

You: YouTube
Aria: 🎉 You're all set, Alex! ...
[System: ✅ Lead captured — Alex Johnson | alex@example.com | YouTube]
```

---

## 🏗 Architecture Explanation (~200 words)

### Why LangGraph?

LangGraph was chosen over AutoGen because it provides **explicit, inspectable state management** via a typed `StateGraph`. For a lead-qualification workflow, the progression `greeting → inquiry → high-intent → lead collection → tool execution` maps naturally to a directed graph with conditional edges. LangGraph makes this state machine transparent and debuggable, while AutoGen's multi-agent conversation model adds unnecessary complexity for a single-agent task.

### How State is Managed

The `AgentState` TypedDict holds the entire session in one place:

| Field | Purpose |
|---|---|
| `messages` | Full conversation history (LangGraph `add_messages` reducer handles appends) |
| `intent` | Classified intent for the current turn |
| `lead_name / email / platform` | Progressive lead field collection |
| `awaiting_field` | Tracks which detail we're waiting for next |
| `lead_captured` | Boolean flag preventing duplicate tool calls |

Every graph turn, the full state is passed through three possible nodes: `detect_intent` → (conditional router) → `generate_response` OR `collect_lead`. The router checks `awaiting_field` first, ensuring mid-collection turns always continue collection regardless of re-classified intent. The `mock_lead_capture()` tool is only called once all three fields are verified — preventing premature execution.

### RAG Pipeline

Product knowledge is stored in `knowledge_base/autostream_kb.json` and loaded once at startup into a formatted context string injected into the LLM system prompt. This is a lightweight RAG approach (no vector DB required for this scale) that gives the model grounded, accurate knowledge without hallucination.

---

## 📱 WhatsApp Deployment via Webhooks

### Overview

To deploy this agent on WhatsApp, use the **WhatsApp Business API** (Meta Cloud API) with a webhook-based architecture:

### Architecture

```
WhatsApp User
     │  (sends message)
     ▼
Meta WhatsApp Business API
     │  (HTTP POST webhook event)
     ▼
Your Webhook Server (FastAPI / Flask)
     │
     ├── Verify webhook (GET challenge handshake)
     │
     └── Handle message (POST)
              │
              ├── Extract phone_number + message_body
              │
              ├── Load session state from Redis/DB (keyed by phone_number)
              │
              ├── Run LangGraph agent.invoke(state)
              │
              ├── Save updated state back to Redis/DB
              │
              └── POST reply to Meta Send Message API
                        │
                        ▼
                  WhatsApp User receives reply
```

### Key Implementation Steps

**1. Register a Webhook (FastAPI example)**
```python
from fastapi import FastAPI, Request
import httpx

app = FastAPI()

@app.get("/webhook")
async def verify(request: Request):
    # Meta sends a challenge token — echo it back to verify ownership
    params = dict(request.query_params)
    if params.get("hub.verify_token") == "MY_VERIFY_TOKEN":
        return int(params["hub.challenge"])

@app.post("/webhook")
async def receive_message(request: Request):
    body = await request.json()
    phone = body["entry"][0]["changes"][0]["value"]["messages"][0]["from"]
    text  = body["entry"][0]["changes"][0]["value"]["messages"][0]["text"]["body"]

    # Load state from Redis
    state = redis_client.get(phone) or default_state()

    # Run agent
    state["messages"].append(HumanMessage(content=text))
    state = graph.invoke(state)

    # Save state
    redis_client.set(phone, state)

    # Get reply
    reply = [m.content for m in reversed(state["messages"]) if isinstance(m, AIMessage)][0]

    # Send reply via Meta API
    await send_whatsapp_message(phone, reply)
```

**2. Send Reply via Meta Cloud API**
```python
async def send_whatsapp_message(phone: str, text: str):
    async with httpx.AsyncClient() as client:
        await client.post(
            f"https://graph.facebook.com/v19.0/{PHONE_NUMBER_ID}/messages",
            headers={"Authorization": f"Bearer {META_ACCESS_TOKEN}"},
            json={
                "messaging_product": "whatsapp",
                "to": phone,
                "type": "text",
                "text": {"body": text}
            }
        )
```

**3. State Persistence**
- Use **Redis** with phone number as key (TTL ~24 hours for session expiry)
- Serialize `AgentState` to JSON before storing; deserialize on load

**4. Production Considerations**
- Host webhook on HTTPS (required by Meta) — use Render, Railway, or AWS Lambda
- Use async FastAPI for handling concurrent WhatsApp users
- Implement rate limiting per phone number
- Add logging and monitoring (e.g., Datadog, Sentry)

---

## 🧠 Evaluation Checklist

| Criterion | Implementation |
|---|---|
| Intent Detection | `detect_intent` node classifies each turn via LLM |
| RAG | KB loaded from JSON, injected into system prompt |
| State Management | LangGraph `AgentState` TypedDict with `add_messages` |
| Tool Calling Logic | `mock_lead_capture()` called only after all 3 fields collected |
| Code Clarity | Modular nodes, clear routing, typed state |
| Real-world Deployability | WhatsApp webhook architecture documented above |

---

## 🛠 Tech Stack

| Component | Technology |
|---|---|
| LLM | Claude Haiku (claude-haiku-4-5-20251001) |
| Framework | LangGraph 0.4+ |
| Language | Python 3.9+ |
| Knowledge Base | JSON (local RAG) |
| Lead Tool | Mock function (CRM-ready) |
