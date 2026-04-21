"""
AutoStream Conversational AI Agent
Built with LangGraph + Llama 3.3 70B via Groq
Social-to-Lead Agentic Workflow for ServiceHive / Inflx Assignment
"""

import json
import os
import re
from typing import Annotated, TypedDict, Literal
from pathlib import Path

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages


# ─────────────────────────────────────────────
# 1. KNOWLEDGE BASE LOADER (RAG)
# ─────────────────────────────────────────────

def load_knowledge_base() -> str:
    """Load and format the knowledge base into a readable context string."""
    kb_path = Path(__file__).parent / "knowledge_base" / "autostream_kb.json"
    with open(kb_path, "r") as f:
        kb = json.load(f)

    context = f"""
COMPANY: {kb['company']['name']} — {kb['company']['tagline']}
{kb['company']['description']}

=== PRICING PLANS ===
"""
    for plan in kb['pricing']['plans']:
        context += f"\n{plan['name']} — ${plan['price_monthly']}/month\n"
        for feature in plan['features']:
            context += f"  • {feature}\n"
        context += f"  Best for: {plan['ideal_for']}\n"

    context += "\n=== COMPANY POLICIES ===\n"
    context += f"Refund Policy: {kb['policies']['refund_policy']['description']} {kb['policies']['refund_policy']['details']}\n"
    context += f"Support — Basic Plan: {kb['policies']['support_policy']['basic_plan']}\n"
    context += f"Support — Pro Plan: {kb['policies']['support_policy']['pro_plan']}\n"
    context += f"Cancellation: {kb['policies']['cancellation']}\n"

    context += "\n=== FREQUENTLY ASKED QUESTIONS ===\n"
    for faq in kb['faqs']:
        context += f"Q: {faq['question']}\nA: {faq['answer']}\n\n"

    return context


KNOWLEDGE_BASE = load_knowledge_base()


# ─────────────────────────────────────────────
# 2. MOCK LEAD CAPTURE TOOL
# ─────────────────────────────────────────────

def mock_lead_capture(name: str, email: str, platform: str) -> str:
    """
    Mock API function to capture a qualified lead.
    In production this would POST to a CRM like HubSpot or Salesforce.
    """
    print(f"\n{'='*50}")
    print(f"✅  LEAD CAPTURED SUCCESSFULLY")
    print(f"    Name     : {name}")
    print(f"    Email    : {email}")
    print(f"    Platform : {platform}")
    print(f"{'='*50}\n")
    return f"Lead captured successfully: {name}, {email}, {platform}"


# ─────────────────────────────────────────────
# 3. LANGGRAPH STATE DEFINITION
# ─────────────────────────────────────────────

class AgentState(TypedDict):
    # Full conversation history (managed by LangGraph add_messages reducer)
    messages: Annotated[list, add_messages]

    # Intent classification for current turn
    intent: Literal["greeting", "product_inquiry", "high_intent", "unknown"]

    # Lead collection fields (filled progressively)
    lead_name: str | None
    lead_email: str | None
    lead_platform: str | None

    # Whether the lead has been successfully captured
    lead_captured: bool

    # Which field we're currently waiting for
    awaiting_field: Literal["name", "email", "platform", "none"] | None


# ─────────────────────────────────────────────
# 4. LLM INITIALIZATION
# ─────────────────────────────────────────────

def get_llm():
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise EnvironmentError("GROQ_API_KEY environment variable not set.")
    return ChatGroq(
        model="llama-3.3-70b-versatile",
        groq_api_key=api_key,
        temperature=0.3,
        max_tokens=1024,
    )


# ─────────────────────────────────────────────
# 5. INTENT DETECTION NODE
# ─────────────────────────────────────────────

INTENT_SYSTEM_PROMPT = """You are an intent classifier for AutoStream, a video editing SaaS.
Classify the user's latest message into EXACTLY one of these labels:

- greeting        → Simple hello, hi, hey, or social opener with no product question
- product_inquiry → Questions about features, pricing, plans, policies, or how AutoStream works
- high_intent     → User expresses desire to sign up, try, buy, subscribe, or start using the product
- unknown         → None of the above

Respond with ONLY the label. No explanation. No punctuation."""


def detect_intent(state: AgentState) -> AgentState:
    """Classify the intent of the latest user message."""
    llm = get_llm()
    last_user_message = next(
        (m.content for m in reversed(state["messages"]) if isinstance(m, HumanMessage)),
        ""
    )

    response = llm.invoke([
        SystemMessage(content=INTENT_SYSTEM_PROMPT),
        HumanMessage(content=last_user_message)
    ])

    raw_intent = response.content.strip().lower()

    # Normalize
    valid_intents = {"greeting", "product_inquiry", "high_intent", "unknown"}
    intent = raw_intent if raw_intent in valid_intents else "unknown"

    return {**state, "intent": intent}


# ─────────────────────────────────────────────
# 6. RESPONSE GENERATION NODE
# ─────────────────────────────────────────────

AGENT_SYSTEM_PROMPT = f"""You are Aria, the friendly and knowledgeable sales assistant for AutoStream — a SaaS platform offering automated video editing tools for content creators.

Your personality: warm, concise, helpful, and professional. You never make up information.

=== YOUR KNOWLEDGE BASE ===
{KNOWLEDGE_BASE}
=== END KNOWLEDGE BASE ===

RULES:
1. Answer ALL product/pricing questions strictly from the knowledge base above.
2. If a user expresses interest in signing up, gently guide them to share their details.
3. Do NOT ask for name/email/platform unless the user has shown genuine high intent.
4. Collect one piece of information at a time — do not ask for everything at once.
5. Never reveal you are built on an AI model. You are simply "Aria from AutoStream."
6. Keep responses concise (2–4 sentences max for general queries).
"""


def generate_response(state: AgentState) -> AgentState:
    """Generate the agent's reply using full conversation context."""
    llm = get_llm()

    messages_for_llm = [SystemMessage(content=AGENT_SYSTEM_PROMPT)] + state["messages"]
    response = llm.invoke(messages_for_llm)

    return {**state, "messages": [AIMessage(content=response.content)]}


# ─────────────────────────────────────────────
# 7. LEAD COLLECTION NODE
# ─────────────────────────────────────────────

def collect_lead_info(state: AgentState) -> AgentState:
    """
    Progressively collect name → email → platform.
    Parses the user's latest message to extract the awaited field.
    """
    last_user_msg = next(
        (m.content for m in reversed(state["messages"]) if isinstance(m, HumanMessage)),
        ""
    ).strip()

    new_state = dict(state)

    # If we're awaiting a specific field, try to fill it
    awaiting = state.get("awaiting_field", "none")

    if awaiting == "name" and not state.get("lead_name"):
        new_state["lead_name"] = last_user_msg
        awaiting = "email"
    elif awaiting == "email" and not state.get("lead_email"):
        # Basic email validation
        if re.search(r"[^@]+@[^@]+\.[^@]+", last_user_msg):
            new_state["lead_email"] = last_user_msg
            awaiting = "platform"
        else:
            # Ask again
            new_state["messages"] = new_state["messages"] + [
                AIMessage(content="That doesn't look like a valid email address. Could you double-check and share it again?")
            ]
            new_state["awaiting_field"] = "email"
            return new_state
    elif awaiting == "platform" and not state.get("lead_platform"):
        new_state["lead_platform"] = last_user_msg
        awaiting = "none"

    new_state["awaiting_field"] = awaiting

    # Decide what to say next
    if not new_state.get("lead_name"):
        new_state["awaiting_field"] = "name"
        reply = "Fantastic! I'd love to set up your account. To get started, could you share your full name?"
    elif not new_state.get("lead_email"):
        new_state["awaiting_field"] = "email"
        reply = f"Great to meet you, {new_state['lead_name']}! What's the best email address to reach you at?"
    elif not new_state.get("lead_platform"):
        new_state["awaiting_field"] = "platform"
        reply = "Almost there! Which platform do you primarily create content for? (e.g., YouTube, Instagram, TikTok)"
    else:
        # All fields collected — trigger the lead capture tool
        result = mock_lead_capture(
            name=new_state["lead_name"],
            email=new_state["lead_email"],
            platform=new_state["lead_platform"],
        )
        new_state["lead_captured"] = True
        new_state["awaiting_field"] = "none"
        reply = (
            f"🎉 You're all set, {new_state['lead_name']}! "
            f"Our team will reach out to {new_state['lead_email']} shortly to get your AutoStream Pro account activated. "
            f"We're excited to help you supercharge your {new_state['lead_platform']} content!"
        )

    new_state["messages"] = new_state["messages"] + [AIMessage(content=reply)]
    return new_state


# ─────────────────────────────────────────────
# 8. ROUTING LOGIC
# ─────────────────────────────────────────────

def route_after_intent(state: AgentState) -> str:
    """Route to the correct node based on detected intent and lead collection state."""
    # If we're mid-collection, keep collecting
    if state.get("awaiting_field") and state["awaiting_field"] != "none":
        return "collect_lead"

    # If lead is already captured, just generate a normal response
    if state.get("lead_captured"):
        return "generate_response"

    intent = state.get("intent", "unknown")

    if intent == "high_intent":
        return "collect_lead"
    elif intent in ("greeting", "product_inquiry", "unknown"):
        return "generate_response"
    else:
        return "generate_response"


# ─────────────────────────────────────────────
# 9. BUILD LANGGRAPH GRAPH
# ─────────────────────────────────────────────

def build_graph():
    graph = StateGraph(AgentState)

    # Add nodes
    graph.add_node("detect_intent", detect_intent)
    graph.add_node("generate_response", generate_response)
    graph.add_node("collect_lead", collect_lead_info)

    # Entry point
    graph.add_edge(START, "detect_intent")

    # Conditional routing after intent detection
    graph.add_conditional_edges(
        "detect_intent",
        route_after_intent,
        {
            "generate_response": "generate_response",
            "collect_lead": "collect_lead",
        }
    )

    # Both terminal nodes end the graph turn
    graph.add_edge("generate_response", END)
    graph.add_edge("collect_lead", END)

    return graph.compile()


# ─────────────────────────────────────────────
# 10. CONVERSATION RUNNER (CLI)
# ─────────────────────────────────────────────

def run_agent():
    """Interactive CLI loop for the AutoStream agent."""
    print("\n" + "="*60)
    print("  🎬  AutoStream AI Assistant — Powered by Inflx/ServiceHive")
    print("="*60)
    print("  Type your message and press Enter.")
    print("  Type 'quit' or 'exit' to end the session.\n")

    graph = build_graph()

    # Initialize state
    state: AgentState = {
        "messages": [],
        "intent": "unknown",
        "lead_name": None,
        "lead_email": None,
        "lead_platform": None,
        "lead_captured": False,
        "awaiting_field": "none",
    }

    # Opening message from agent
    opening = (
        "Hi! 👋 I'm Aria, your AutoStream assistant. "
        "I can help you with pricing, features, or getting you signed up. "
        "What can I do for you today?"
    )
    print(f"Aria: {opening}\n")
    state["messages"].append(AIMessage(content=opening))

    while True:
        user_input = input("You: ").strip()
        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit"):
            print("\nAria: Thanks for chatting! Have a great day. 🎬\n")
            break

        # Add user message to state
        state["messages"].append(HumanMessage(content=user_input))

        # Run the graph
        state = graph.invoke(state)

        # Print the last AI message
        last_ai_msg = next(
            (m.content for m in reversed(state["messages"]) if isinstance(m, AIMessage)),
            ""
        )
        print(f"\nAria: {last_ai_msg}\n")

        # Show lead status summary if captured
        if state.get("lead_captured") and state.get("lead_name"):
            print(f"[System: Lead captured — {state['lead_name']} | {state['lead_email']} | {state['lead_platform']}]\n")


if __name__ == "__main__":
    run_agent()
