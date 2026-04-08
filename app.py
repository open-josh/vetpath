"""
VetPath — AI-Guided VR&E Self-Employment Platform
MVP: Stages 1-3 (Discovery → Validation → Plan Construction)
"""
import os
import json
import requests
from flask import Flask, render_template, request, jsonify, session
from datetime import datetime as dt
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", "vetpath-dev-key-change-in-production")
app.config["SESSION_TYPE"] = "filesystem"

MINIMAX_API_KEY = os.environ.get("MINIMAX_API_KEY", "")
MINIMAX_BASE_URL = "https://api.minimax.io/v1"
MINIMAX_MODEL = "MiniMax-M2"


# ─── AI Call Helper ───────────────────────────────────────────────────────────

def call_minimax(system_prompt, user_prompt, temperature=0.7, max_tokens=4000):
    """Call MiniMax chat API and return the response text."""
    if not MINIMAX_API_KEY:
        raise ValueError("MiniMax API key not configured.")

    url = f"{MINIMAX_BASE_URL}/text/chatcompletion_v2"
    headers = {
        "Authorization": f"Bearer {MINIMAX_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": MINIMAX_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": temperature,
        "max_completion_tokens": max_tokens,
    }
    resp = requests.post(url, headers=headers, json=payload, timeout=60)
    if resp.status_code != 200:
        raise RuntimeError(f"MiniMax error {resp.status_code}: {resp.text[:200]}")
    data = resp.json()
    choices = data.get("choices", [])
    if not choices:
        raise ValueError("Empty response from MiniMax")
    return choices[0]["message"]["content"]


# ─── Stage 1: Discovery ─────────────────────────────────────────────────────

DISCOVERY_SYSTEM_PROMPT = """You are a veteran vocational counselor helping veterans discover self-employment opportunities through the VA VR&E (Veteran Readiness and Employment) program under 38 CFR § 21.252.

Your job is to guide the veteran through a brief discovery conversation. Ask ONE question at a time. Be warm, conversational, and encouraging. Veterans have earned this benefit through their service.

After collecting enough information, generate 5-8 specific business concepts tailored to this veteran's profile.

Collect ALL of the following before generating concepts:
1. Service-connected condition(s) and how it limits work
2. Location (city/state or ZIP code)
3. Available startup budget (personal funds)
4. Skills, work experience, and military background
5. Preferred business type (local service, online, mobile, etc.)
6. Income goal (monthly)
7. How much time they can commit weekly

Rules:
- Ask ONE question at a time. Wait for a response before asking the next.
- Keep questions short and specific.
- Acknowledge each answer briefly (1 sentence) before moving to the next.
- After all 7 items are collected, generate 5-8 business concepts.
- Each concept must include: name, why it fits the veteran's disability, estimated startup cost, and monthly revenue potential.
- Flag any concept that may face VR&E rejection risks.

When ready to generate concepts, output a JSON block:
{"__type": "concepts", "concepts": [{"name": "...", "why_fits": "...", "startup_cost": "...", "monthly_revenue": "...", "vre_risks": "..."}]}

Keep the warm conversation going alongside the JSON."""


@app.route("/")
def index():
    """Landing page."""
    return render_template("index.html")


@app.route("/journey")
def journey():
    """The guided VetPath journey — Discovery, Validation, Plan."""
    return render_template("journey.html")


@app.route("/api/discovery/chat", methods=["POST"])
def api_discovery_chat():
    """Conversational discovery — one question at a time."""
    if not MINIMAX_API_KEY:
        return jsonify({"success": False, "error": "API not configured"}), 500

    data = request.get_json()
    messages = data.get("messages", [])
    collected = data.get("collected", {})

    # Build prompt from collected data + new messages
    context = ""
    if collected:
        context += "Veteran profile so far:\n"
        for k, v in collected.items():
            context += f"- {k}: {v}\n"
        context += "\n"

    user_content = context
    for m in messages:
        user_content += f"\nVeteran: {m.get('content', '')}"

    try:
        reply = call_minimax(DISCOVERY_SYSTEM_PROMPT, user_content, temperature=0.8, max_tokens=3000)
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

    # Check for concepts JSON block
    concepts = None
    import re
    m = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*"__type"\s*:\s*"concepts"[^\}]*\}', reply, re.DOTALL)
    if m:
        try:
            concepts = json.loads(m.group())["concepts"]
        except Exception:
            pass

    return jsonify({
        "success": True,
        "reply": reply,
        "concepts": concepts,
    })


# ─── Stage 2: Validation ────────────────────────────────────────────────────

VALIDATION_SYSTEM_PROMPT = """You are an expert in VA VR&E self-employment feasibility analysis under 38 CFR § 21.257.

Given a selected business concept and a veteran's profile, conduct a rigorous adversarial analysis:

1. Is this business type allowed under VR&E self-employment rules (§ 21.252)?
2. Is the startup cost realistic and adequately funded?
3. Is there sufficient non-VA financing documented?
4. Is the market demand evidence-based or vague?
5. Does the veteran have or have a plan to acquire necessary skills?
6. Is the disability-vocation link clear and compelling?
7. What are the top 3 rejection risks a VRC would raise?

Also conduct a basic market reality check:
- Estimate the local market size (population/income/demand signals)
- Identify 2-3 actual competitor businesses (use real data where possible)
- Assess pricing viability

Return a JSON block:
{
  "__type": "validation",
  "verdict": "VIABLE|NEEDS_WORK|UNLIKELY",
  "confidence": "HIGH|MEDIUM|LOW",
  "strengths": ["..."],
  "weaknesses": ["..."],
  "rejection_risks": [{"risk": "...", "mitigation": "..."}],
  "market_notes": "...",
  "recommended_adjustments": ["..."]
}

Then give a 2-3 sentence plain-English verdict the veteran can understand."""


@app.route("/api/validate", methods=["POST"])
def api_validate():
    """Validate a selected concept against CFR feasibility requirements."""
    if not MINIMAX_API_KEY:
        return jsonify({"success": False, "error": "API not configured"}), 500

    data = request.get_json()
    concept = data.get("concept", {})
    veteran_profile = data.get("veteran_profile", {})

    user_prompt = f"""Veteran's profile:
{json.dumps(veteran_profile, indent=2)}

Selected business concept:
{json.dumps(concept, indent=2)}

Conduct full adversarial feasibility analysis."""

    try:
        reply = call_minimax(VALIDATION_SYSTEM_PROMPT, user_prompt, temperature=0.3, max_tokens=4000)
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

    # Extract JSON
    import re
    validation = None
    m = re.search(r'\{.*?"__type"\s*:\s*"validation".*?\}', reply, re.DOTALL)
    if m:
        try:
            validation = json.loads(m.group())
        except Exception:
            pass

    return jsonify({
        "success": True,
        "reply": reply,
        "validation": validation,
    })


# ─── Stage 3: Plan Construction ─────────────────────────────────────────────

PLAN_SYSTEM_PROMPT = """You are an expert VR&E business plan writer for the VA Vocational Rehabilitation program under 38 CFR § 21.257.

Generate a complete, submission-ready VR&E self-employment business plan.

The plan MUST include ALL of the following CFR § 21.257(f) required elements:
1. Economic Viability — market demand, revenue projections, sustainability
2. Cost Analysis — itemized startup costs, VR&E funding requested, personal contribution
3. Market Analysis — target customer, competitors, pricing, local market data
4. Non-VA Financing — personal funds, SBA, bank loans, family assistance
5. SBA Coordination — SCORE mentorship, SBA counseling, § 8 special consideration
6. Site Location — physical/online/home-based, costs, zoning, accessibility
7. Training Plan — how the veteran will learn to operate this business

Plus:
- Disability-Vocation Link (§ 21.257(b)): why this business suits the veteran's specific limitations
- Rejection Risk Analysis: address the 10 most common VR&E rejection reasons
- Executive Summary
- Declaration section

Format the plan with clear section headers. Write at a 6th-grade reading level. Use plain English.
Include specific numbers, local market data (use estimates if real data unavailable), and concrete details.

IMPORTANT: The plan must be specific to the veteran's actual profile — use the data provided, not generic filler."""

PLAN_USER_PROMPT_TEMPLATE = """Generate a complete VR&E business plan for this veteran and business concept:

Veteran Profile:
{profile}

Business Concept:
{concept}

Validation Notes (use these to strengthen the plan):
{validation_notes}
"""


@app.route("/api/generate-plan", methods=["POST"])
def api_generate_plan():
    """Generate the full CFR § 21.257-compliant business plan."""
    if not MINIMAX_API_KEY:
        return jsonify({"success": False, "error": "API not configured"}), 500

    data = request.get_json()
    veteran_profile = data.get("veteran_profile", {})
    concept = data.get("concept", {})
    validation_notes = data.get("validation_notes", "")

    user_prompt = PLAN_USER_PROMPT_TEMPLATE.format(
        profile=json.dumps(veteran_profile, indent=2),
        concept=json.dumps(concept, indent=2),
        validation_notes=validation_notes or "None provided — use standard feasibility analysis."
    )

    try:
        plan = call_minimax(PLAN_SYSTEM_PROMPT, user_prompt, temperature=0.5, max_tokens=8000)
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

    return jsonify({
        "success": True,
        "plan": plan,
    })


# ─── Email Capture ──────────────────────────────────────────────────────────

@app.route("/api/capture", methods=["POST"])
def api_capture():
    """Capture veteran email for follow-up."""
    data = request.get_json()
    email = data.get("email", "").strip()
    stage = data.get("stage", "unknown")
    if not email or "@" not in email:
        return jsonify({"success": False, "error": "Invalid email"}), 400

    leads_file = os.path.join(os.path.dirname(__file__), "leads.json")
    leads = []
    if os.path.exists(leads_file):
        try:
            with open(leads_file) as f:
                leads = json.load(f)
        except Exception:
            leads = []

    leads.append({"email": email, "stage": stage, "ts": dt.now().isoformat()})
    with open(leads_file, "w") as f:
        json.dump(leads, f)

    return jsonify({"success": True})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
