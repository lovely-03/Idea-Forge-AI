# =============================================================================
#  IdeaForge AI – Smart Business Idea Generator
#  Powered by IBM watsonx.ai Studio & IBM Granite Models
# =============================================================================
#
#  Agentic AI Architecture:
#    Agent 1 – Knowledge Discovery Agent
#    Agent 2 – Idea Generation Agent
#    Agent 3 – Trend Forecasting Agent
#    Agent 4 – Feasibility Analysis Agent
#    Agent 5 – Innovation Strategy Agent
#    Master   – Orchestrator Agent (coordinates all agents)
#
#  Tech Stack: Python · Flask · IBM Granite (ibm-granite/granite-3-3-8b-instruct)
#
#  Setup:
#    pip install flask requests python-dotenv
#
#    Set environment variables (or create a .env file):
#      WATSONX_API_KEY    = <your IBM Cloud API key>
#      WATSONX_PROJECT_ID = <your watsonx.ai project ID>
#      WATSONX_URL        = https://us-south.ml.cloud.ibm.com
#
#  Run:
#    python app.py
# =============================================================================

import os
import json
import time
import requests
from datetime import datetime
from flask import Flask, request, jsonify, render_template_string
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Load environment variables from a .env file if present
# ---------------------------------------------------------------------------
load_dotenv()

app = Flask(__name__)

# =============================================================================
#  IBM watsonx.ai Configuration
# =============================================================================
WATSONX_API_KEY    = os.getenv("WATSONX_API_KEY", "your-api-key-here")
WATSONX_PROJECT_ID = os.getenv("WATSONX_PROJECT_ID", "your-project-id-here")
WATSONX_URL        = os.getenv("WATSONX_URL", "https://us-south.ml.cloud.ibm.com")

# IBM Granite model identifier used across all agents
GRANITE_MODEL_ID = "ibm/granite-3-3-8b-instruct"

# ---------------------------------------------------------------------------
# Token cache – avoids requesting a new IAM token on every call
# ---------------------------------------------------------------------------
_token_cache = {"token": None, "expires_at": 0}


# =============================================================================
#  IBM watsonx.ai Helper – IAM Token
# =============================================================================
def get_iam_token() -> str:
    """
    Obtain (and cache) an IBM Cloud IAM bearer token.
    The token is reused until 5 minutes before its expiry.
    """
    now = time.time()
    if _token_cache["token"] and now < _token_cache["expires_at"]:
        return _token_cache["token"]

    url  = "https://iam.cloud.ibm.com/identity/token"
    data = {
        "grant_type": "urn:ibm:params:oauth:grant-type:apikey",
        "apikey": WATSONX_API_KEY,
    }
    headers = {"Content-Type": "application/x-www-form-urlencoded"}

    try:
        response = requests.post(url, data=data, headers=headers, timeout=30)
        response.raise_for_status()
        token_data = response.json()
        _token_cache["token"]      = token_data["access_token"]
        _token_cache["expires_at"] = now + token_data.get("expires_in", 3600) - 300
        return _token_cache["token"]
    except Exception as exc:
        print(f"[IAM] Token error: {exc}")
        return ""


# =============================================================================
#  IBM watsonx.ai Helper – Core Inference Function
# =============================================================================
def generate_response(prompt: str, max_tokens: int = 900) -> str:
    """
    Send a prompt to IBM Granite via the watsonx.ai text-generation endpoint
    and return the generated text.

    IBM watsonx.ai Studio integration point
    Model: ibm/granite-3-3-8b-instruct
    """
    token = get_iam_token()
    if not token:
        return _fallback_response(prompt)

    url     = f"{WATSONX_URL}/ml/v1/text/generation?version=2023-05-29"
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type":  "application/json",
        "Accept":        "application/json",
    }
    payload = {
        "model_id": GRANITE_MODEL_ID,
        "input":    prompt,
        "parameters": {
            "decoding_method":  "greedy",
            "max_new_tokens":    max_tokens,
            "min_new_tokens":    50,
            "stop_sequences":    [],
            "repetition_penalty": 1.1,
        },
        "project_id": WATSONX_PROJECT_ID,
    }

    try:
        response = requests.post(url, headers=headers, json=payload, timeout=60)
        response.raise_for_status()
        result = response.json()
        generated = result.get("results", [{}])[0].get("generated_text", "").strip()
        return generated if generated else _fallback_response(prompt)
    except Exception as exc:
        print(f"[Granite] Inference error: {exc}")
        return _fallback_response(prompt)


# =============================================================================
#  Fallback – when watsonx.ai credentials are not yet configured
# =============================================================================
def _fallback_response(prompt: str) -> str:
    """
    Return a structured demo response so the UI remains fully functional
    even before real IBM credentials are provided.
    """
    return (
        "IBM Granite Model Response (Demo Mode): "
        "Configure WATSONX_API_KEY and WATSONX_PROJECT_ID environment variables "
        "to enable live IBM Granite inference. "
        "This response demonstrates the agent workflow and dashboard UI. "
        "The full agentic pipeline — Knowledge Discovery → Idea Generation → "
        "Trend Forecasting → Feasibility Analysis → Innovation Strategy — "
        "is operational and will process real AI responses once credentials are set."
    )


# =============================================================================
#  AGENT 1 – Knowledge Discovery Agent
# =============================================================================
def knowledge_discovery_agent(user_input: str, industry: str, voice_note: str = "") -> dict:
    """
    Agent 1: Knowledge Discovery Agent
    -----------------------------------
    Responsibility: Collect and organise contextual knowledge.
    - Identifies market opportunities
    - Detects unmet needs
    - Surfaces emerging technology trends
    - Produces an Opportunity Snapshot
    Powered by IBM Granite via watsonx.ai Studio.
    """
    context = f"Industry: {industry}" if industry else ""
    voice   = f"\nVoice Note: {voice_note}" if voice_note else ""

    prompt = f"""You are a Knowledge Discovery Agent specializing in market research and opportunity identification.

Analyze the following input and provide structured insights:
User Interest: {user_input}
{context}{voice}

Provide a comprehensive knowledge discovery report with:
1. OPPORTUNITY SNAPSHOT: 3 key market opportunities (each with title and 2-sentence description)
2. MARKET SIGNALS: 4 current market signals or trends driving this space
3. EMERGING TECHNOLOGIES: 3 emerging technologies relevant to this domain
4. UNMET NEEDS: 3 specific unmet needs in this market
5. INDUSTRY RELEVANCE: Brief assessment of industry timing and readiness

Be specific, data-informed, and forward-looking. Format clearly."""

    raw = generate_response(prompt, max_tokens=800)

    return {
        "agent":       "Knowledge Discovery Agent",
        "agent_id":    1,
        "icon":        "🔍",
        "purpose":     "Collect and organize contextual information about market opportunities",
        "status":      "completed",
        "raw_output":  raw,
        "metadata": {
            "industry":   industry or "General",
            "voice_note": bool(voice_note),
            "timestamp":  datetime.now().strftime("%H:%M:%S"),
        },
    }


# =============================================================================
#  AGENT 2 – Idea Generation Agent
# =============================================================================
def idea_generation_agent(user_input: str, opportunities: str, image_desc: str = "") -> dict:
    """
    Agent 2: Idea Generation Agent
    --------------------------------
    Responsibility: Generate innovative business and project ideas.
    - Creates startup, research, product, app, and social impact ideas
    - Each idea includes: Name, Problem, Solution, Target Users
    Powered by IBM Granite via watsonx.ai Studio.
    """
    image_context = f"\nImage/Sketch Description: {image_desc}" if image_desc else ""

    prompt = f"""You are an Idea Generation Agent — a creative powerhouse for innovative business ideas.

Based on the following context, generate 5 distinct, innovative ideas:
User Interest: {user_input}
Market Opportunities: {opportunities[:400]}
{image_context}

For EACH of the 5 ideas provide:
IDEA [N]: [Creative Idea Name]
- CATEGORY: (Startup / Research / Product / App / Social Impact)
- PROBLEM SOLVED: One clear sentence describing the problem
- PROPOSED SOLUTION: Two sentences describing the innovative solution
- TARGET USERS: Specific audience (e.g., "Urban farmers aged 25-45 in developing nations")
- UNIQUE VALUE: What makes this different from existing solutions
- IMPACT SCORE: Rate impact potential 1-10

Generate ideas that are creative, practical, and future-ready."""

    raw = generate_response(prompt, max_tokens=900)

    return {
        "agent":      "Idea Generation Agent",
        "agent_id":   2,
        "icon":       "💡",
        "purpose":    "Generate innovative business and project ideas",
        "status":     "completed",
        "raw_output": raw,
        "metadata": {
            "image_context": bool(image_desc),
            "timestamp":     datetime.now().strftime("%H:%M:%S"),
        },
    }


# =============================================================================
#  AGENT 3 – Trend Forecasting Agent
# =============================================================================
def trend_forecasting_agent(user_input: str, ideas: str) -> dict:
    """
    Agent 3: Trend Forecasting Agent
    ----------------------------------
    Responsibility: Predict future relevance of generated ideas.
    - Analyses technology trends, market evolution, consumer behaviour
    - Produces Trend Score, Growth Potential, Future Demand Prediction
    Powered by IBM Granite via watsonx.ai Studio.
    """
    prompt = f"""You are a Trend Forecasting Agent with expertise in market dynamics and technology evolution.

Analyze the following ideas for future relevance and market potential:
Original Interest: {user_input}
Generated Ideas Summary: {ideas[:500]}

Provide a comprehensive trend forecast:
1. OVERALL TREND SCORE: Rate the idea space 1-100 with justification
2. GROWTH POTENTIAL: (High Growth / Medium Growth / Emerging / Saturated) with explanation
3. FUTURE DEMAND PREDICTION: 3-5 year outlook in 3 bullet points
4. KEY DRIVERS: 4 macro trends accelerating this space
5. RISK FACTORS: 3 potential headwinds or disruptions
6. BEST TIMING: Optimal market entry window (Now / 1-2 Years / 3-5 Years)
7. COMPARABLE MARKETS: 2 analogous markets that followed a similar trajectory

Be analytical, specific, and cite industry patterns where relevant."""

    raw = generate_response(prompt, max_tokens=750)

    return {
        "agent":      "Trend Forecasting Agent",
        "agent_id":   3,
        "icon":       "📈",
        "purpose":    "Predict future relevance and market trajectory of ideas",
        "status":     "completed",
        "raw_output": raw,
        "metadata":   {"timestamp": datetime.now().strftime("%H:%M:%S")},
    }


# =============================================================================
#  AGENT 4 – Feasibility Analysis Agent
# =============================================================================
def feasibility_analysis_agent(user_input: str, ideas: str) -> dict:
    """
    Agent 4: Feasibility Analysis Agent
    -------------------------------------
    Responsibility: Evaluate practicality of ideas.
    - Assesses technical feasibility, business viability, resource requirements
    - Produces Feasibility Score, Difficulty, Risk Factors, Required Skills
    Powered by IBM Granite via watsonx.ai Studio.
    """
    prompt = f"""You are a Feasibility Analysis Agent with deep expertise in business strategy and technical assessment.

Evaluate the practicality of implementing ideas in this space:
User Interest: {user_input}
Ideas to Evaluate: {ideas[:500]}

Deliver a thorough feasibility assessment:
1. OVERALL FEASIBILITY SCORE: Rate 1-100 with brief rationale
2. ESTIMATED DIFFICULTY: (Low / Medium / High / Very High) with explanation
3. TECHNICAL REQUIREMENTS: 4 key technical capabilities needed
4. BUSINESS VIABILITY: Revenue potential and business model fit assessment
5. RESOURCE REQUIREMENTS: Team size, budget range, time to MVP
6. RISK FACTORS: Top 4 risks with mitigation strategies
7. REQUIRED SKILLS: 5 key skill sets needed for the founding team
8. COMPETITIVE LANDSCAPE: Brief assessment of competition intensity

Be realistic and actionable in your assessment."""

    raw = generate_response(prompt, max_tokens=750)

    return {
        "agent":      "Feasibility Analysis Agent",
        "agent_id":   4,
        "icon":       "⚖️",
        "purpose":    "Evaluate technical and business feasibility of ideas",
        "status":     "completed",
        "raw_output": raw,
        "metadata":   {"timestamp": datetime.now().strftime("%H:%M:%S")},
    }


# =============================================================================
#  AGENT 5 – Innovation Strategy Agent
# =============================================================================
def innovation_strategy_agent(user_input: str, ideas: str, feasibility: str) -> dict:
    """
    Agent 5: Innovation Strategy Agent
    ------------------------------------
    Responsibility: Convert ideas into actionable innovation plans.
    - Produces MVP Suggestions, Revenue Models, Go-To-Market Strategy
    - Generates Startup Roadmap and Success Metrics
    Powered by IBM Granite via watsonx.ai Studio.
    """
    prompt = f"""You are an Innovation Strategy Agent — a world-class startup advisor and product strategist.

Create a comprehensive innovation strategy for the best idea in this space:
User Interest: {user_input}
Top Ideas: {ideas[:400]}
Feasibility Context: {feasibility[:300]}

Generate a complete innovation roadmap:
1. TOP RECOMMENDED IDEA: Name and one-line pitch
2. MVP PLAN: 3 core features for the Minimum Viable Product
3. REVENUE MODELS: 3 viable monetization strategies with brief descriptions
4. GO-TO-MARKET STRATEGY: Step-by-step launch approach (5 steps)
5. STARTUP ROADMAP:
   - Month 1-3 (Discovery & Validation)
   - Month 4-6 (Build & Test)
   - Month 7-12 (Launch & Scale)
6. SUCCESS METRICS: 5 KPIs to track progress
7. FUTURE ENHANCEMENTS: 3 Phase-2 features or expansions
8. FUNDING STRATEGY: Best funding path (Bootstrapped / Angel / VC / Grants)

Make this actionable, specific, and investor-ready."""

    raw = generate_response(prompt, max_tokens=900)

    return {
        "agent":      "Innovation Strategy Agent",
        "agent_id":   5,
        "icon":       "🚀",
        "purpose":    "Convert ideas into actionable innovation plans and roadmaps",
        "status":     "completed",
        "raw_output": raw,
        "metadata":   {"timestamp": datetime.now().strftime("%H:%M:%S")},
    }


# =============================================================================
#  MASTER ORCHESTRATOR AGENT
# =============================================================================
def orchestrator_agent(user_input: str, industry: str = "",
                       image_desc: str = "", voice_note: str = "") -> dict:
    """
    Master Orchestrator Agent
    --------------------------
    The central brain of IdeaForge AI.
    Coordinates all five specialized agents in a defined pipeline:
      1. Knowledge Discovery  → surfaces opportunities
      2. Idea Generation      → creates ideas from opportunities
      3. Trend Forecasting    → evaluates market timing
      4. Feasibility Analysis → assesses practicality
      5. Innovation Strategy  → builds actionable plans
    Combines all outputs into a unified Innovation Report.
    """
    start_time = time.time()
    workflow   = []

    # ── Agent 1: Knowledge Discovery ──────────────────────────────────────────
    workflow.append({"step": 1, "agent": "Knowledge Discovery", "status": "running"})
    agent1_result = knowledge_discovery_agent(user_input, industry, voice_note)
    workflow[-1]["status"] = "completed"

    # ── Agent 2: Idea Generation ───────────────────────────────────────────────
    workflow.append({"step": 2, "agent": "Idea Generation", "status": "running"})
    agent2_result = idea_generation_agent(
        user_input, agent1_result["raw_output"], image_desc
    )
    workflow[-1]["status"] = "completed"

    # ── Agent 3: Trend Forecasting ─────────────────────────────────────────────
    workflow.append({"step": 3, "agent": "Trend Forecasting", "status": "running"})
    agent3_result = trend_forecasting_agent(user_input, agent2_result["raw_output"])
    workflow[-1]["status"] = "completed"

    # ── Agent 4: Feasibility Analysis ─────────────────────────────────────────
    workflow.append({"step": 4, "agent": "Feasibility Analysis", "status": "running"})
    agent4_result = feasibility_analysis_agent(user_input, agent2_result["raw_output"])
    workflow[-1]["status"] = "completed"

    # ── Agent 5: Innovation Strategy ──────────────────────────────────────────
    workflow.append({"step": 5, "agent": "Innovation Strategy", "status": "running"})
    agent5_result = innovation_strategy_agent(
        user_input, agent2_result["raw_output"], agent4_result["raw_output"]
    )
    workflow[-1]["status"] = "completed"

    # ── Final Synthesis ───────────────────────────────────────────────────────
    synthesis_prompt = f"""You are the Master Orchestrator for IdeaForge AI.

Synthesize insights from all five AI agents into a concise Executive Summary:
- User Interest: {user_input}
- Industry: {industry or 'General'}

Agent Outputs Summary:
Knowledge Discovery: {agent1_result['raw_output'][:250]}
Generated Ideas: {agent2_result['raw_output'][:250]}
Trend Forecast: {agent3_result['raw_output'][:200]}
Feasibility: {agent4_result['raw_output'][:200]}
Strategy: {agent5_result['raw_output'][:200]}

Create a 4-paragraph Executive Summary covering:
1. The core opportunity and why it matters now
2. The single best idea recommended and why
3. Key risks and how to mitigate them
4. Immediate next steps for the entrepreneur

Be decisive, inspiring, and actionable."""

    synthesis = generate_response(synthesis_prompt, max_tokens=600)
    elapsed   = round(time.time() - start_time, 2)

    return {
        "orchestrator": "IdeaForge AI Master Orchestrator",
        "model":        GRANITE_MODEL_ID,
        "user_input":   user_input,
        "industry":     industry or "General",
        "timestamp":    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "processing_time_sec": elapsed,
        "workflow":     workflow,
        "agents": {
            "agent1_knowledge":   agent1_result,
            "agent2_ideas":       agent2_result,
            "agent3_trends":      agent3_result,
            "agent4_feasibility": agent4_result,
            "agent5_strategy":    agent5_result,
        },
        "executive_summary": synthesis,
    }


# =============================================================================
#  Flask Routes
# =============================================================================

@app.route("/")
def index():
    """Render the main single-page application."""
    return render_template_string(HTML_TEMPLATE)


@app.route("/api/generate", methods=["POST"])
def api_generate():
    """
    POST /api/generate
    Body (JSON):
      user_input  – business interest or challenge (required)
      industry    – industry focus (optional)
      image_desc  – image / sketch description (optional)
      voice_note  – transcribed voice note (optional)
    Returns the full orchestration result as JSON.
    """
    data       = request.get_json(force=True) or {}
    user_input = data.get("user_input", "").strip()
    industry   = data.get("industry", "").strip()
    image_desc = data.get("image_desc", "").strip()
    voice_note = data.get("voice_note", "").strip()

    if not user_input:
        return jsonify({"error": "user_input is required"}), 400

    try:
        result = orchestrator_agent(user_input, industry, image_desc, voice_note)
        return jsonify(result)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/api/health", methods=["GET"])
def health():
    """Health check endpoint."""
    return jsonify({
        "status":    "healthy",
        "app":       "IdeaForge AI",
        "model":     GRANITE_MODEL_ID,
        "watsonx":   WATSONX_URL,
        "timestamp": datetime.now().isoformat(),
    })


# =============================================================================
#  HTML Template – Full Single-Page Application
# =============================================================================
HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>IdeaForge AI – Smart Business Idea Generator</title>
  <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css" rel="stylesheet"/>
  <link href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.3/font/bootstrap-icons.css" rel="stylesheet"/>
  <style>
    :root {
      --primary:   #1a56db;
      --secondary: #7e3af2;
      --accent:    #0e9f6e;
      --danger:    #e74c3c;
      --warning:   #f59e0b;
      --dark:      #0f172a;
      --surface:   #f8fafc;
      --card-bg:   #ffffff;
      --border:    #e2e8f0;
      --text:      #1e293b;
      --muted:     #64748b;
    }

    * { box-sizing: border-box; }

    body {
      font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
      background: var(--surface);
      color: var(--text);
      margin: 0;
    }

    /* ── Navigation ─────────────────────────────────────────── */
    .navbar-brand-custom {
      background: linear-gradient(135deg, var(--primary), var(--secondary));
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
      font-weight: 800;
      font-size: 1.45rem;
      letter-spacing: -0.5px;
    }

    .navbar {
      border-bottom: 1px solid var(--border);
      backdrop-filter: blur(8px);
      background: rgba(255,255,255,0.95) !important;
    }

    /* ── Hero ───────────────────────────────────────────────── */
    .hero-section {
      background: linear-gradient(135deg, #0f172a 0%, #1e3a8a 50%, #4c1d95 100%);
      color: #fff;
      padding: 3.5rem 0 2.5rem;
      position: relative;
      overflow: hidden;
    }

    .hero-section::before {
      content: '';
      position: absolute;
      inset: 0;
      background: url("data:image/svg+xml,%3Csvg width='60' height='60' viewBox='0 0 60 60' xmlns='http://www.w3.org/2000/svg'%3E%3Cg fill='none' fill-rule='evenodd'%3E%3Cg fill='%23ffffff' fill-opacity='0.04'%3E%3Cpath d='M36 34v-4h-2v4h-4v2h4v4h2v-4h4v-2h-4zm0-30V0h-2v4h-4v2h4v4h2V6h4V4h-4zM6 34v-4H4v4H0v2h4v4h2v-4h4v-2H6zM6 4V0H4v4H0v2h4v4h2V6h4V4H6z'/%3E%3C/g%3E%3C/g%3E%3C/svg%3E");
    }

    .hero-badge {
      display: inline-block;
      background: rgba(255,255,255,0.15);
      border: 1px solid rgba(255,255,255,0.3);
      border-radius: 50px;
      padding: 4px 16px;
      font-size: 0.78rem;
      letter-spacing: 1px;
      text-transform: uppercase;
      margin-bottom: 1rem;
    }

    /* ── Cards ──────────────────────────────────────────────── */
    .card {
      border: 1px solid var(--border);
      border-radius: 14px;
      background: var(--card-bg);
      box-shadow: 0 1px 4px rgba(0,0,0,0.06);
      transition: box-shadow 0.2s;
    }
    .card:hover { box-shadow: 0 4px 16px rgba(0,0,0,0.1); }

    .card-header-custom {
      background: linear-gradient(135deg, var(--primary), var(--secondary));
      color: #fff;
      border-radius: 13px 13px 0 0 !important;
      padding: 1rem 1.25rem;
      font-weight: 700;
    }

    /* ── Agent Cards ────────────────────────────────────────── */
    .agent-card {
      border-left: 4px solid var(--primary);
      border-radius: 12px;
      background: var(--card-bg);
      padding: 1.2rem;
      margin-bottom: 1rem;
      border: 1px solid var(--border);
      border-left: 4px solid var(--primary);
      transition: all 0.3s;
    }
    .agent-card.active {
      border-left-color: var(--accent);
      background: #f0fdf4;
    }
    .agent-card.pending { opacity: 0.5; }

    .agent-icon {
      width: 46px; height: 46px;
      border-radius: 12px;
      display: flex; align-items: center; justify-content: center;
      font-size: 1.4rem;
      flex-shrink: 0;
    }
    .agent-badge {
      font-size: 0.7rem;
      padding: 2px 10px;
      border-radius: 50px;
      font-weight: 600;
    }

    /* ── Input Form ─────────────────────────────────────────── */
    .form-control, .form-select {
      border-radius: 10px;
      border: 1.5px solid var(--border);
      padding: 0.7rem 1rem;
      font-size: 0.95rem;
      transition: border-color 0.2s;
    }
    .form-control:focus, .form-select:focus {
      border-color: var(--primary);
      box-shadow: 0 0 0 3px rgba(26,86,219,0.12);
    }

    .btn-forge {
      background: linear-gradient(135deg, var(--primary), var(--secondary));
      border: none;
      color: #fff;
      border-radius: 10px;
      padding: 0.75rem 2rem;
      font-weight: 700;
      font-size: 1rem;
      letter-spacing: 0.3px;
      transition: opacity 0.2s, transform 0.1s;
    }
    .btn-forge:hover { opacity: 0.92; color: #fff; }
    .btn-forge:active { transform: scale(0.98); }

    /* ── Progress Pipeline ──────────────────────────────────── */
    .pipeline {
      display: flex;
      align-items: center;
      gap: 0;
      overflow-x: auto;
      padding: 0.5rem 0;
    }
    .pipeline-step {
      display: flex;
      flex-direction: column;
      align-items: center;
      flex: 1;
      min-width: 100px;
    }
    .pipeline-dot {
      width: 40px; height: 40px;
      border-radius: 50%;
      display: flex; align-items: center; justify-content: center;
      font-weight: 700;
      font-size: 1rem;
      background: #e2e8f0;
      color: var(--muted);
      border: 2px solid var(--border);
      transition: all 0.4s;
      position: relative;
      z-index: 1;
    }
    .pipeline-dot.done {
      background: var(--accent);
      color: #fff;
      border-color: var(--accent);
    }
    .pipeline-dot.active {
      background: var(--primary);
      color: #fff;
      border-color: var(--primary);
      animation: pulse 1.2s infinite;
    }
    .pipeline-line {
      flex: 1;
      height: 2px;
      background: var(--border);
      margin: 0 -1px;
      margin-top: -24px;
      transition: background 0.4s;
    }
    .pipeline-line.done { background: var(--accent); }
    .pipeline-label { font-size: 0.7rem; margin-top: 6px; text-align: center; color: var(--muted); }

    @keyframes pulse {
      0%, 100% { box-shadow: 0 0 0 0 rgba(26,86,219,0.4); }
      50%       { box-shadow: 0 0 0 8px rgba(26,86,219,0); }
    }

    /* ── Output Sections ────────────────────────────────────── */
    .output-section { display: none; }
    .output-section.visible { display: block; }

    .result-text {
      background: #f8fafc;
      border: 1px solid var(--border);
      border-radius: 10px;
      padding: 1.1rem 1.25rem;
      font-size: 0.88rem;
      line-height: 1.75;
      white-space: pre-wrap;
      color: var(--text);
      max-height: 360px;
      overflow-y: auto;
    }

    .metric-card {
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 1.2rem;
      text-align: center;
    }
    .metric-value {
      font-size: 2rem;
      font-weight: 800;
      background: linear-gradient(135deg, var(--primary), var(--secondary));
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
    }

    /* ── Idea Map ───────────────────────────────────────────── */
    #ideaMapContainer {
      background: #0f172a;
      border-radius: 14px;
      min-height: 380px;
      overflow: hidden;
      position: relative;
    }

    /* ── Tabs ───────────────────────────────────────────────── */
    .nav-tabs .nav-link {
      border-radius: 10px 10px 0 0;
      color: var(--muted);
      font-weight: 500;
      border: 1px solid transparent;
    }
    .nav-tabs .nav-link.active {
      color: var(--primary);
      font-weight: 700;
      border-color: var(--border) var(--border) #fff;
    }

    /* ── Loading Overlay ─────────────────────────────────────── */
    .loading-overlay {
      display: none;
      position: fixed; inset: 0;
      background: rgba(15,23,42,0.7);
      z-index: 9999;
      align-items: center;
      justify-content: center;
      flex-direction: column;
      gap: 1rem;
    }
    .loading-overlay.active { display: flex; }

    .spinner-ring {
      width: 64px; height: 64px;
      border: 5px solid rgba(255,255,255,0.2);
      border-top-color: #7e3af2;
      border-radius: 50%;
      animation: spin 0.8s linear infinite;
    }
    @keyframes spin { to { transform: rotate(360deg); } }

    .loading-text { color: #fff; font-size: 1rem; font-weight: 500; }
    .loading-step { color: rgba(255,255,255,0.6); font-size: 0.85rem; }

    /* ── Summary Card ───────────────────────────────────────── */
    .summary-card {
      background: linear-gradient(135deg, #0f172a, #1e3a8a);
      color: #fff;
      border-radius: 14px;
      padding: 1.8rem;
      border: none;
    }

    /* ── Footer ─────────────────────────────────────────────── */
    footer {
      background: var(--dark);
      color: rgba(255,255,255,0.5);
      text-align: center;
      padding: 1.8rem 0;
      font-size: 0.82rem;
      margin-top: 3rem;
      border-top: 1px solid rgba(255,255,255,0.08);
    }

    /* ── Responsive ─────────────────────────────────────────── */
    @media (max-width: 768px) {
      .pipeline-label { display: none; }
      .hero-section   { padding: 2rem 0 1.5rem; }
    }

    .tag {
      display: inline-block;
      background: rgba(26,86,219,0.1);
      color: var(--primary);
      border-radius: 50px;
      padding: 2px 12px;
      font-size: 0.75rem;
      font-weight: 600;
      margin: 2px;
    }
  </style>
</head>
<body>

<!-- ═══════════════════ LOADING OVERLAY ═══════════════════ -->
<div class="loading-overlay" id="loadingOverlay">
  <div class="spinner-ring"></div>
  <div class="loading-text">IdeaForge AI is thinking…</div>
  <div class="loading-step" id="loadingStep">Activating Knowledge Discovery Agent</div>
</div>

<!-- ═══════════════════ NAVBAR ═══════════════════════════════ -->
<nav class="navbar navbar-expand-lg sticky-top">
  <div class="container">
    <a class="navbar-brand" href="#">
      <span class="navbar-brand-custom">⚡ IdeaForge AI</span>
    </a>
    <div class="d-flex align-items-center gap-3">
      <span class="badge text-bg-primary rounded-pill px-3">IBM Granite</span>
      <span class="badge text-bg-secondary rounded-pill px-3">watsonx.ai</span>
      <span class="badge text-bg-success rounded-pill px-3">Agentic AI</span>
    </div>
  </div>
</nav>

<!-- ═══════════════════ HERO ══════════════════════════════════ -->
<section class="hero-section">
  <div class="container text-center position-relative">
    <div class="hero-badge">IBM watsonx.ai Studio · 5-Agent System</div>
    <h1 class="fw-800 mb-3" style="font-size:2.6rem; font-weight:800;">
      IdeaForge AI
    </h1>
    <p class="lead mb-2" style="opacity:.85; max-width:620px; margin:0 auto;">
      Smart Business Idea Generator powered by <strong>IBM Granite Models</strong>
    </p>
    <p style="opacity:.6; font-size:.9rem; margin-top:.5rem;">
      Knowledge Discovery · Idea Generation · Trend Forecasting · Feasibility Analysis · Innovation Strategy
    </p>
  </div>
</section>

<!-- ═══════════════════ MAIN CONTENT ═════════════════════════ -->
<div class="container py-5">

  <!-- Input Form -->
  <div class="row justify-content-center mb-5">
    <div class="col-lg-9">
      <div class="card shadow-sm">
        <div class="card-header-custom">
          <i class="bi bi-lightning-charge-fill me-2"></i>
          Forge Your Innovation
        </div>
        <div class="card-body p-4">

          <!-- Main Input -->
          <div class="mb-3">
            <label class="form-label fw-semibold">
              <i class="bi bi-lightbulb text-warning me-1"></i>
              Business Interest / Innovation Challenge *
            </label>
            <textarea id="userInput" class="form-control" rows="3"
              placeholder="e.g. I want to build an AI solution for sustainable agriculture in developing countries…"></textarea>
          </div>

          <div class="row g-3 mb-3">
            <!-- Industry -->
            <div class="col-md-4">
              <label class="form-label fw-semibold">
                <i class="bi bi-building me-1"></i> Industry Focus
              </label>
              <select id="industrySelect" class="form-select">
                <option value="">Select industry…</option>
                <option>Agriculture &amp; FoodTech</option>
                <option>HealthTech &amp; MedTech</option>
                <option>FinTech &amp; InsurTech</option>
                <option>EdTech &amp; E-Learning</option>
                <option>CleanTech &amp; Sustainability</option>
                <option>Retail &amp; E-Commerce</option>
                <option>Smart Cities &amp; Infrastructure</option>
                <option>AI &amp; Automation</option>
                <option>Logistics &amp; Supply Chain</option>
                <option>Entertainment &amp; Media</option>
                <option>Travel &amp; Hospitality</option>
                <option>Real Estate &amp; PropTech</option>
              </select>
            </div>

            <!-- Image Description -->
            <div class="col-md-4">
              <label class="form-label fw-semibold">
                <i class="bi bi-image me-1"></i> Image / Sketch Description
              </label>
              <input id="imageDesc" type="text" class="form-control"
                placeholder="e.g. UI wireframe for a mobile app…"/>
            </div>

            <!-- Voice Note -->
            <div class="col-md-4">
              <label class="form-label fw-semibold">
                <i class="bi bi-mic me-1"></i> Voice Note (transcribed)
              </label>
              <input id="voiceNote" type="text" class="form-control"
                placeholder="e.g. I want to help rural farmers…"/>
            </div>
          </div>

          <!-- Example prompts -->
          <div class="mb-4">
            <small class="text-muted fw-semibold">Quick Examples:</small>
            <div class="mt-2 d-flex flex-wrap gap-2">
              <button class="btn btn-sm btn-outline-primary rounded-pill example-btn"
                data-text="AI-powered crop disease detection for smallholder farmers"
                data-industry="Agriculture &amp; FoodTech">
                🌾 Smart Agriculture
              </button>
              <button class="btn btn-sm btn-outline-secondary rounded-pill example-btn"
                data-text="Personalized mental health support platform using AI"
                data-industry="HealthTech &amp; MedTech">
                🧠 Mental Health AI
              </button>
              <button class="btn btn-sm btn-outline-success rounded-pill example-btn"
                data-text="Blockchain-based micro-lending for unbanked populations"
                data-industry="FinTech &amp; InsurTech">
                💳 FinTech Inclusion
              </button>
              <button class="btn btn-sm btn-outline-warning rounded-pill example-btn"
                data-text="Gamified learning platform for K-12 STEM education"
                data-industry="EdTech &amp; E-Learning">
                📚 EdTech Gamification
              </button>
              <button class="btn btn-sm btn-outline-danger rounded-pill example-btn"
                data-text="Carbon footprint tracker and offset marketplace"
                data-industry="CleanTech &amp; Sustainability">
                🌿 Carbon Tracking
              </button>
            </div>
          </div>

          <div class="d-grid">
            <button class="btn btn-forge" id="generateBtn" onclick="runOrchestrator()">
              <i class="bi bi-magic me-2"></i> Generate Innovation Report
            </button>
          </div>
        </div>
      </div>
    </div>
  </div>

  <!-- ═══ Agent Pipeline Visualization ═══ -->
  <div class="row justify-content-center mb-4">
    <div class="col-lg-10">
      <div class="card">
        <div class="card-body py-3 px-4">
          <div class="d-flex align-items-center mb-3">
            <h6 class="mb-0 fw-bold me-auto">
              <i class="bi bi-diagram-3 me-2 text-primary"></i>
              Agentic Workflow Pipeline
            </h6>
            <span class="badge text-bg-light" id="pipelineStatus">Waiting for input</span>
          </div>
          <div class="pipeline" id="pipelineViz">
            <div class="pipeline-step">
              <div class="pipeline-dot" id="dot1">1</div>
              <div class="pipeline-label">Knowledge<br>Discovery</div>
            </div>
            <div class="pipeline-line" id="line12"></div>
            <div class="pipeline-step">
              <div class="pipeline-dot" id="dot2">2</div>
              <div class="pipeline-label">Idea<br>Generation</div>
            </div>
            <div class="pipeline-line" id="line23"></div>
            <div class="pipeline-step">
              <div class="pipeline-dot" id="dot3">3</div>
              <div class="pipeline-label">Trend<br>Forecast</div>
            </div>
            <div class="pipeline-line" id="line34"></div>
            <div class="pipeline-step">
              <div class="pipeline-dot" id="dot4">4</div>
              <div class="pipeline-label">Feasibility<br>Analysis</div>
            </div>
            <div class="pipeline-line" id="line45"></div>
            <div class="pipeline-step">
              <div class="pipeline-dot" id="dot5">5</div>
              <div class="pipeline-label">Innovation<br>Strategy</div>
            </div>
          </div>
        </div>
      </div>
    </div>
  </div>

  <!-- ═══ Results Section ═══ -->
  <div id="resultsSection" class="output-section">

    <!-- Executive Summary -->
    <div class="row justify-content-center mb-4">
      <div class="col-lg-10">
        <div class="summary-card">
          <h5 class="fw-bold mb-3">
            <i class="bi bi-stars me-2 text-warning"></i>
            Executive Summary
            <small class="ms-2 fw-normal opacity-50" style="font-size:.75rem;" id="summaryMeta"></small>
          </h5>
          <p id="executiveSummary" style="opacity:.9; line-height:1.8; font-size:.95rem;"></p>
        </div>
      </div>
    </div>

    <!-- Dashboard Tabs -->
    <div class="row justify-content-center">
      <div class="col-lg-10">
        <ul class="nav nav-tabs mb-0" id="dashTabs">
          <li class="nav-item">
            <button class="nav-link active" data-bs-toggle="tab" data-bs-target="#tabAgents">
              <i class="bi bi-robot me-1"></i> Agents
            </button>
          </li>
          <li class="nav-item">
            <button class="nav-link" data-bs-toggle="tab" data-bs-target="#tabIdeas">
              <i class="bi bi-lightbulb me-1"></i> Ideas
            </button>
          </li>
          <li class="nav-item">
            <button class="nav-link" data-bs-toggle="tab" data-bs-target="#tabTrends">
              <i class="bi bi-graph-up me-1"></i> Trends
            </button>
          </li>
          <li class="nav-item">
            <button class="nav-link" data-bs-toggle="tab" data-bs-target="#tabFeasibility">
              <i class="bi bi-clipboard-check me-1"></i> Feasibility
            </button>
          </li>
          <li class="nav-item">
            <button class="nav-link" data-bs-toggle="tab" data-bs-target="#tabStrategy">
              <i class="bi bi-rocket me-1"></i> Strategy
            </button>
          </li>
          <li class="nav-item">
            <button class="nav-link" data-bs-toggle="tab" data-bs-target="#tabMap">
              <i class="bi bi-diagram-2 me-1"></i> Idea Map
            </button>
          </li>
        </ul>

        <div class="tab-content border border-top-0 rounded-bottom p-4 bg-white shadow-sm">

          <!-- ── Tab: Agents ──────────────────────────────── -->
          <div class="tab-pane fade show active" id="tabAgents">
            <h6 class="fw-bold mb-3 text-primary">
              <i class="bi bi-diagram-3 me-2"></i>Multi-Agent Collaboration Output
            </h6>
            <div id="agentCards"></div>
          </div>

          <!-- ── Tab: Ideas ───────────────────────────────── -->
          <div class="tab-pane fade" id="tabIdeas">
            <h6 class="fw-bold mb-3 text-primary">
              <i class="bi bi-lightbulb me-2"></i>Generated Business Ideas
            </h6>
            <div id="ideasContent" class="result-text"></div>
          </div>

          <!-- ── Tab: Trends ──────────────────────────────── -->
          <div class="tab-pane fade" id="tabTrends">
            <h6 class="fw-bold mb-3 text-primary">
              <i class="bi bi-graph-up-arrow me-2"></i>Trend Forecast & Market Analysis
            </h6>
            <div class="row g-3 mb-3">
              <div class="col-md-4">
                <div class="metric-card">
                  <div class="metric-value" id="trendScore">–</div>
                  <div class="fw-semibold mt-1">Trend Score</div>
                  <small class="text-muted">Out of 100</small>
                </div>
              </div>
              <div class="col-md-4">
                <div class="metric-card">
                  <div id="growthLabel" class="fw-bold fs-5 text-success">–</div>
                  <div class="fw-semibold mt-1">Growth Potential</div>
                  <small class="text-muted">Market trajectory</small>
                </div>
              </div>
              <div class="col-md-4">
                <div class="metric-card">
                  <div id="timingLabel" class="fw-bold fs-5 text-primary">–</div>
                  <div class="fw-semibold mt-1">Best Entry Timing</div>
                  <small class="text-muted">Market window</small>
                </div>
              </div>
            </div>
            <div id="trendsContent" class="result-text"></div>
          </div>

          <!-- ── Tab: Feasibility ─────────────────────────── -->
          <div class="tab-pane fade" id="tabFeasibility">
            <h6 class="fw-bold mb-3 text-primary">
              <i class="bi bi-clipboard-data me-2"></i>Feasibility Assessment
            </h6>
            <div class="row g-3 mb-3">
              <div class="col-md-4">
                <div class="metric-card">
                  <div class="metric-value" id="feasScore">–</div>
                  <div class="fw-semibold mt-1">Feasibility Score</div>
                  <small class="text-muted">Out of 100</small>
                </div>
              </div>
              <div class="col-md-4">
                <div class="metric-card">
                  <div id="difficultyLabel" class="fw-bold fs-5 text-warning">–</div>
                  <div class="fw-semibold mt-1">Difficulty Level</div>
                  <small class="text-muted">Implementation complexity</small>
                </div>
              </div>
              <div class="col-md-4">
                <div class="metric-card">
                  <div id="competitionLabel" class="fw-bold fs-5 text-danger">–</div>
                  <div class="fw-semibold mt-1">Competition</div>
                  <small class="text-muted">Market intensity</small>
                </div>
              </div>
            </div>
            <div id="feasibilityContent" class="result-text"></div>
          </div>

          <!-- ── Tab: Strategy ────────────────────────────── -->
          <div class="tab-pane fade" id="tabStrategy">
            <h6 class="fw-bold mb-3 text-primary">
              <i class="bi bi-map me-2"></i>Innovation Roadmap & Strategy
            </h6>
            <div id="strategyContent" class="result-text"></div>
          </div>

          <!-- ── Tab: Idea Map ─────────────────────────────── -->
          <div class="tab-pane fade" id="tabMap">
            <h6 class="fw-bold mb-3 text-primary">
              <i class="bi bi-share me-2"></i>Visual Idea Map
            </h6>
            <div id="ideaMapContainer">
              <svg id="ideaMapSvg" width="100%" height="400"></svg>
            </div>
          </div>

        </div><!-- /tab-content -->
      </div>
    </div><!-- /row -->
  </div><!-- /resultsSection -->
</div>

<!-- ═══════════════════ FOOTER ════════════════════════════════ -->
<footer>
  <div class="container">
    <strong style="color:rgba(255,255,255,0.8);">IdeaForge AI</strong>
    &nbsp;·&nbsp; Powered by <strong style="color:#7e3af2;">IBM Granite Models</strong>
    &nbsp;·&nbsp; IBM watsonx.ai Studio
    &nbsp;·&nbsp; Agentic AI Architecture
    <br/><span style="opacity:.4; font-size:.75rem;">
      Built for IBM SkillsBuild · Hackathons · Academic Projects · Startup Showcases
    </span>
  </div>
</footer>

<script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/js/bootstrap.bundle.min.js"></script>
<script>
// ===========================================================
//  IdeaForge AI – Front-end Controller
// ===========================================================

// ── Example prompt fill ──────────────────────────────────────
document.querySelectorAll('.example-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    document.getElementById('userInput').value = btn.dataset.text;
    const sel = document.getElementById('industrySelect');
    for (let opt of sel.options) {
      if (opt.text.replace('&amp;','&') === btn.dataset.industry.replace('&amp;','&')) {
        sel.value = opt.value;
        break;
      }
    }
  });
});

// ── Pipeline helpers ─────────────────────────────────────────
function setDot(n, state) {
  const dot = document.getElementById('dot' + n);
  dot.className = 'pipeline-dot' + (state === 'done' ? ' done' : state === 'active' ? ' active' : '');
  if (state === 'done') dot.innerHTML = '<i class="bi bi-check-lg" style="font-size:.8rem"></i>';
}

function setLine(id, done) {
  const line = document.getElementById(id);
  if (line) line.className = 'pipeline-line' + (done ? ' done' : '');
}

function resetPipeline() {
  for (let i = 1; i <= 5; i++) setDot(i, 'pending');
  ['line12','line23','line34','line45'].forEach(l => setLine(l, false));
  document.getElementById('pipelineStatus').textContent = 'Running…';
}

// ── Loading step messages ────────────────────────────────────
const stepMessages = [
  'Agent 1: Discovering market opportunities…',
  'Agent 2: Generating innovative ideas…',
  'Agent 3: Forecasting market trends…',
  'Agent 4: Analysing feasibility…',
  'Agent 5: Building innovation strategy…',
  'Orchestrator: Synthesizing final report…',
];
let stepIndex = 0;
let stepTimer = null;

function startLoadingCycle() {
  stepIndex = 0;
  document.getElementById('loadingStep').textContent = stepMessages[0];
  stepTimer = setInterval(() => {
    stepIndex = (stepIndex + 1) % stepMessages.length;
    document.getElementById('loadingStep').textContent = stepMessages[stepIndex];
  }, 3200);
}

function stopLoadingCycle() {
  if (stepTimer) clearInterval(stepTimer);
}

// ── Extract a number from text (e.g. "Score: 78") ────────────
function extractNumber(text, keywords) {
  for (const kw of keywords) {
    const re = new RegExp(kw + '[:\\s]*(\\d+)', 'i');
    const m  = text.match(re);
    if (m) return m[1];
  }
  // try standalone number near keyword
  const re2 = /(\d{2,3})\s*\/\s*100/i;
  const m2  = text.match(re2);
  if (m2) return m2[1];
  return '–';
}

function extractLabel(text, keywords, fallback = '–') {
  for (const kw of keywords) {
    const re = new RegExp(kw + '[:\\s]*([A-Za-z ]+)', 'i');
    const m  = text.match(re);
    if (m) return m[1].trim().split('\\n')[0].slice(0, 25);
  }
  return fallback;
}

// ── Build Agent Cards ─────────────────────────────────────────
function buildAgentCards(agents) {
  const colours = ['#1a56db','#7e3af2','#0e9f6e','#f59e0b','#e74c3c'];
  const whys = [
    'Activated first to map the opportunity landscape and surface market signals.',
    'Activated to transform discovered opportunities into concrete, creative ideas.',
    'Activated to evaluate long-term market viability and timing of generated ideas.',
    'Activated to assess technical and business feasibility before committing resources.',
    'Activated last to convert the best idea into a fully actionable launch plan.',
  ];

  const keys = ['agent1_knowledge','agent2_ideas','agent3_trends','agent4_feasibility','agent5_strategy'];
  let html = '';

  keys.forEach((key, idx) => {
    const ag = agents[key];
    if (!ag) return;
    const colour = colours[idx];
    html += `
      <div class="agent-card mb-3" style="border-left-color:${colour}">
        <div class="d-flex align-items-start gap-3">
          <div class="agent-icon" style="background:${colour}20; color:${colour}; font-size:1.5rem; width:48px; height:48px; border-radius:12px; display:flex; align-items:center; justify-content:center; flex-shrink:0;">
            ${ag.icon}
          </div>
          <div class="flex-grow-1">
            <div class="d-flex align-items-center justify-content-between mb-1">
              <strong style="color:${colour};">Agent ${ag.agent_id}: ${ag.agent}</strong>
              <span class="agent-badge text-white" style="background:${colour}; font-size:.7rem; padding:2px 12px; border-radius:50px;">✓ Completed</span>
            </div>
            <p class="text-muted mb-2" style="font-size:.82rem;">${ag.purpose}</p>
            <div class="mb-2">
              <span class="tag" style="background:${colour}10; color:${colour};">
                <i class="bi bi-question-circle me-1"></i> Why activated?
              </span>
              <small class="text-muted ms-1" style="font-size:.8rem;">${whys[idx]}</small>
            </div>
            <details>
              <summary style="cursor:pointer; font-size:.82rem; color:${colour}; font-weight:600;">
                View Agent Output ▾
              </summary>
              <div class="result-text mt-2" style="max-height:240px; font-size:.82rem;">${escHtml(ag.raw_output)}</div>
            </details>
          </div>
        </div>
      </div>`;
  });

  document.getElementById('agentCards').innerHTML = html;
}

function escHtml(s) {
  return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}

// ── Build Visual Idea Map (SVG) ──────────────────────────────
function buildIdeaMap(topic, industry) {
  const svg = document.getElementById('ideaMapSvg');
  const W = svg.parentElement.offsetWidth || 700;
  const H = 400;
  svg.setAttribute('viewBox', `0 0 ${W} ${H}`);

  const cx = W / 2, cy = H / 2;
  const nodes = [
    { label: topic.slice(0, 22) || 'Your Idea', r: 52, fill: '#1a56db', textColor: '#fff', cx, cy },
    { label: '💡 Business Ideas', r: 38, fill: '#7e3af2', textColor: '#fff', cx: cx - 210, cy: cy - 110 },
    { label: '📈 Market Trends', r: 38, fill: '#0e9f6e', textColor: '#fff', cx: cx + 210, cy: cy - 110 },
    { label: '👥 Target Users', r: 38, fill: '#f59e0b', textColor: '#fff', cx: cx - 230, cy: cy + 100 },
    { label: '💰 Revenue Models', r: 38, fill: '#e74c3c', textColor: '#fff', cx: cx + 230, cy: cy + 100 },
    { label: '🚀 MVP Plan', r: 34, fill: '#0891b2', textColor: '#fff', cx: cx, cy: cy - 160 },
  ];

  let svgContent = `<rect width="${W}" height="${H}" fill="#0f172a" rx="14"/>`;
  // Lines from center
  for (let i = 1; i < nodes.length; i++) {
    svgContent += `<line x1="${cx}" y1="${cy}" x2="${nodes[i].cx}" y2="${nodes[i].cy}"
      stroke="rgba(255,255,255,0.15)" stroke-width="1.5" stroke-dasharray="5,4"/>`;
  }
  // Circles + labels
  nodes.forEach(n => {
    svgContent += `
      <circle cx="${n.cx}" cy="${n.cy}" r="${n.r}" fill="${n.fill}" fill-opacity="0.9"/>
      <text x="${n.cx}" y="${n.cy}" text-anchor="middle" dominant-baseline="middle"
        fill="${n.textColor}" font-size="${n.r > 40 ? 11 : 10}" font-weight="600"
        font-family="Segoe UI,system-ui,sans-serif">${n.label}</text>`;
  });

  // Industry label
  if (industry) {
    svgContent += `<text x="${W/2}" y="${H - 18}" text-anchor="middle"
      fill="rgba(255,255,255,0.35)" font-size="11" font-family="Segoe UI,system-ui,sans-serif">
      Industry: ${escHtml(industry)}</text>`;
  }

  svg.innerHTML = svgContent;
}

// ── Main orchestration call ──────────────────────────────────
async function runOrchestrator() {
  const userInput = document.getElementById('userInput').value.trim();
  if (!userInput) {
    alert('Please describe your business interest or innovation challenge.');
    return;
  }

  const industry   = document.getElementById('industrySelect').value;
  const imageDesc  = document.getElementById('imageDesc').value.trim();
  const voiceNote  = document.getElementById('voiceNote').value.trim();

  // Show loading
  document.getElementById('loadingOverlay').classList.add('active');
  document.getElementById('generateBtn').disabled = true;
  startLoadingCycle();
  resetPipeline();

  // Animate pipeline dots while waiting
  const dotDelay = [0, 3200, 6400, 9600, 12800];
  dotDelay.forEach((d, i) => {
    setTimeout(() => {
      if (i > 0) { setDot(i, 'done'); setLine(['','line12','line23','line34','line45'][i], true); }
      if (i < 5) setDot(i + 1, 'active');
    }, d);
  });

  try {
    const resp = await fetch('/api/generate', {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify({ user_input: userInput, industry, image_desc: imageDesc, voice_note: voiceNote }),
    });

    if (!resp.ok) throw new Error(`Server error ${resp.status}`);
    const data = await resp.json();

    // Mark all done
    for (let i = 1; i <= 5; i++) setDot(i, 'done');
    ['line12','line23','line34','line45'].forEach(l => setLine(l, true));
    document.getElementById('pipelineStatus').textContent =
      `Completed in ${data.processing_time_sec}s`;

    renderResults(data);

  } catch (err) {
    document.getElementById('pipelineStatus').textContent = 'Error – see console';
    alert('Error: ' + err.message);
    console.error(err);
  } finally {
    stopLoadingCycle();
    document.getElementById('loadingOverlay').classList.remove('active');
    document.getElementById('generateBtn').disabled = false;
  }
}

// ── Render all results ────────────────────────────────────────
function renderResults(data) {
  const agents = data.agents;

  // Executive Summary
  document.getElementById('executiveSummary').textContent = data.executive_summary;
  document.getElementById('summaryMeta').textContent =
    `${data.timestamp}  ·  Model: ${data.model}  ·  ${data.processing_time_sec}s`;

  // Agent cards
  buildAgentCards(agents);

  // Ideas tab
  document.getElementById('ideasContent').textContent =
    agents.agent2_ideas?.raw_output || '';

  // Trends tab
  const trendText = agents.agent3_trends?.raw_output || '';
  document.getElementById('trendsContent').textContent = trendText;
  document.getElementById('trendScore').textContent =
    extractNumber(trendText, ['TREND SCORE','OVERALL TREND SCORE','score']) || '–';
  document.getElementById('growthLabel').textContent =
    extractLabel(trendText, ['GROWTH POTENTIAL'], 'High Growth');
  document.getElementById('timingLabel').textContent =
    extractLabel(trendText, ['BEST TIMING','TIMING'], 'Now');

  // Feasibility tab
  const feasText = agents.agent4_feasibility?.raw_output || '';
  document.getElementById('feasibilityContent').textContent = feasText;
  document.getElementById('feasScore').textContent =
    extractNumber(feasText, ['FEASIBILITY SCORE','OVERALL FEASIBILITY']) || '–';
  document.getElementById('difficultyLabel').textContent =
    extractLabel(feasText, ['ESTIMATED DIFFICULTY','DIFFICULTY'], 'Medium');
  document.getElementById('competitionLabel').textContent =
    extractLabel(feasText, ['COMPETITIVE LANDSCAPE','COMPETITION'], 'Moderate');

  // Strategy tab
  document.getElementById('strategyContent').textContent =
    agents.agent5_strategy?.raw_output || '';

  // Idea Map
  buildIdeaMap(data.user_input, data.industry);

  // Show results section and scroll
  const section = document.getElementById('resultsSection');
  section.classList.add('visible');
  section.scrollIntoView({ behavior: 'smooth', block: 'start' });
}
</script>
</body>
</html>"""


# =============================================================================
#  Application Entry Point
# =============================================================================
if __name__ == "__main__":
    print("=" * 65)
    print("  IdeaForge AI – Smart Business Idea Generator")
    print("  Powered by IBM Granite Models & watsonx.ai Studio")
    print("=" * 65)
    print(f"  Model     : {GRANITE_MODEL_ID}")
    print(f"  Endpoint  : {WATSONX_URL}")
    print(f"  Project ID: {WATSONX_PROJECT_ID[:8]}..." if len(WATSONX_PROJECT_ID) > 8 else f"  Project ID: {WATSONX_PROJECT_ID}")
    print("=" * 65)
    print("  Open in browser → http://localhost:5000")
    print("=" * 65)
    app.run(debug=True, host="0.0.0.0", port=5000)
