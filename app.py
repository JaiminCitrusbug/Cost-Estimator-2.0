# save as app.py
"""
AI Project Estimation Generator — Final testing-ready build
- Embedded prompts included
- PM & QA aggregated at project-level (excluded from budget)
- Per-feature resources: fullstack / ai / ui_ux
- Headcount suggestions constrained and practical:
    * fullstack: 1..3 (strict)
    * ai/ui_ux: suggested 1 if any non-zero hours exist (conservative)
- Deterministic, readable fallback for feature name generation when OpenAI key is absent
- Removed usage of Styler.hide_index (compatible across pandas versions)
- Clean UI DataFrames with index removed, feature_name visible, formatted costs
"""

import streamlit as st
import openai
import os
import json
from dotenv import load_dotenv
import pandas as pd
import math
from typing import Dict, List, Any, Tuple, Optional
import re
from datetime import datetime
from io import StringIO

load_dotenv()

# ----------------- EMBEDDED PROMPTS -----------------
PROMPT_FEATURE_GEN = """
You are a senior product architect. Given the project inputs, return ONLY JSON: an array of objects with keys 'feature_name' and 'hint'.
Rules:
1) Understand the project inputs in detail first. Only start producing output after considering product_level, ui_level, platforms, description and budget.
2) Keep features atomic (3–6 words); avoid bundling multiple modules.
3) Provide 'hint' as one of known module types when possible: ["auth_roles","crud_dashboard","kyc_forms","media_handling","moderation_workflow","payments","messaging","calendar_viewings","notifications","analytics","agency_management","support_ticketing","default"].
4) Determine whether a module requires a basic implementation versus a complex implementation (use your judgement). Do NOT overcomplex simple modules (for example, simple auth is just signup/login).
5) Prefer common modules like auth, CRUD dashboards, payments, messaging, analytics if relevant.
6) Do NOT estimate hours here. Output example:
  [
    {"feature_name": "Role-Based Login", "hint": "auth_roles"},
    {"feature_name": "Listing CRUD & Filters", "hint": "crud_dashboard"}
  ]
"""

PROMPT_FEATURE_ENRICH = """
You are a pragmatic Product Strategist and Estimation QA. Return ONLY JSON: an array where each item includes:
- original 'feature_name'
- description (1–3 sentences)
- acceptance_criteria (3–5 items)
- user_story (one line)
- dependencies (short string)
- deliverables (array or short string)
- timeline_plan (array of {phase, hours}) — integers that sum to the computed total hours already supplied
- calendar_weeks (int) OR omit weeks if not available — note: the app will ignore weeks in display; keep weeks optional
- and, IF NEEDED because computed hours look unrealistic, suggested 'recommended_hours' (int) and 'hours_rationale' (string)
Rules:
1) Before writing, analyze each feature and decide if a basic implementation is sufficient or if a complex one is required. Keep simple features simple.
2) Do NOT change the core numeric logic already implemented client-side (baseline medians, multipliers, reuse factors). You may optionally suggest 'recommended_hours' only when computed totals are clearly outside realistic bounds.
3) For timeline_plan, give integer hours for phases (Planning, Design, Build, QA, UAT/Hardening) that sum to the total hours.
4) Output only JSON.
"""

PROMPT_PM_QA = """
You are an estimation reviewer. Given the full project context and the list of feature totals (each with 'feature_name' and 'computed_total_hours'), return ONLY JSON:
{
  "pm_hours": <integer cumulative hours for project management across the whole project>,
  "qa_hours": <integer cumulative hours for QA across the whole project>,
  "rationale": "<one short paragraph explaining why you chose these totals>"
}
Rules:
1) Decide cumulative PM and QA hours at project-level based on total engineering hours, feature complexity mix, and number of features. Do NOT produce weekly distributions — a single cumulative number for PM and QA each.
2) Keep PM conservative for small/simple projects and scale reasonably for larger/complex projects. Similarly, QA should increase with complexity and number of features.
3) Do NOT include PM/QA costs in your response (they're excluded from budget).
4) Output only JSON.
"""

PROMPT_HEADCOUNT = """
You are a pragmatic engineering manager. Given project context: total hours per role (fullstack_hours, ai_hours, ui_ux_hours), an overall complexity_level (1-5), and total feature count, return ONLY JSON:
{
  "fullstack_count": <int 1..3>,
  "ai_count": <int >=0>,
  "ui_ux_count": <int >=0>,
  "rationale": "<one short sentence>"
}
Rules:
1) For fullstack, only return 1 for basic/simple projects, 2 for medium complexity, 3 for high/overcomplex. Do never suggest >3 fullstack engineers.
2) For ai and ui_ux, suggest minimal realistic headcount (often 0 or 1) depending on total hours; keep conservative recommendations.
3) Keep team sizing practical for early-stage projects; do not recommend inflated headcounts to match hours.
4) Output only JSON.
"""

# ----------------- CONFIG -----------------
st.set_page_config(
    page_title="AI Project Estimation Generator — Final",
    layout="centered",
    page_icon="🤖",
)

# Hourly rates (PM & QA intentionally excluded from budget)
HOURLY_RATES = {"fullstack": 25.0, "ai": 30.0, "ui_ux": 30.0}

BASELINE_MODULES = {
    "auth_roles": {"min": 40, "median": 60, "max": 80},
    "crud_dashboard": {"min": 60, "median": 90, "max": 110},
    "kyc_forms": {"min": 50, "median": 75, "max": 95},
    "media_handling": {"min": 80, "median": 120, "max": 140},
    "moderation_workflow": {"min": 80, "median": 120, "max": 160},
    "payments": {"min": 60, "median": 100, "max": 120},
    "messaging": {"min": 60, "median": 100, "max": 120},
    "calendar_viewings": {"min": 40, "median": 70, "max": 90},
    "notifications": {"min": 16, "median": 28, "max": 40},
    "analytics": {"min": 60, "median": 100, "max": 120},
    "agency_management": {"min": 60, "median": 100, "max": 120},
    "support_ticketing": {"min": 40, "median": 80, "max": 100},
    "default": {"min": 20, "median": 50, "max": 120},
}

ROLE_WEIGHTS = {
    "auth_roles":         {"fullstack": 0.75, "ui_ux": 0.18, "ai": 0.07},
    "crud_dashboard":     {"fullstack": 0.78, "ui_ux": 0.18, "ai": 0.04},
    "kyc_forms":          {"fullstack": 0.76, "ui_ux": 0.18, "ai": 0.06},
    "media_handling":     {"fullstack": 0.78, "ui_ux": 0.14, "ai": 0.08},
    "moderation_workflow":{"fullstack": 0.70, "ui_ux": 0.14, "ai": 0.16},
    "payments":           {"fullstack": 0.80, "ui_ux": 0.12, "ai": 0.08},
    "messaging":          {"fullstack": 0.78, "ui_ux": 0.18, "ai": 0.04},
    "calendar_viewings":  {"fullstack": 0.78, "ui_ux": 0.18, "ai": 0.04},
    "notifications":      {"fullstack": 0.78, "ui_ux": 0.20, "ai": 0.02},
    "analytics":          {"fullstack": 0.72, "ui_ux": 0.16, "ai": 0.12},
    "agency_management":  {"fullstack": 0.78, "ui_ux": 0.16, "ai": 0.06},
    "support_ticketing":  {"fullstack": 0.78, "ui_ux": 0.16, "ai": 0.06},
    "default":            {"fullstack": 0.78, "ui_ux": 0.16, "ai": 0.06},
}

COMPLEXITY_MULTIPLIER = {1: 0.7, 2: 0.9, 3: 1.0, 4: 1.15, 5: 1.35}
DEFAULT_REUSE_FACTOR = 0.85
MIN_HOURS_PER_FEATURE = 4
MAX_HOURS_PER_FEATURE = 400
MIN_ROLE_PERCENT = 0.02

CALIBRATION_GUIDE = {
    "auth_roles": {"typical_range": [40, 80]},
    "crud_dashboard": {"typical_range": [60, 110]},
    # ... trimmed for brevity but used by LLM prompts if called
}

# ----------------- HELPERS -----------------
def parse_budget_to_usd(budget_str: str) -> Tuple[Optional[float], Optional[str]]:
    if not budget_str:
        return None, None
    s = budget_str.strip()
    s = s.replace(",", "").replace("—", "-").replace("–", "-")
    currency = "USD"
    if "₹" in s or "INR" in s or "rs " in s.lower():
        currency = "INR"
    parts = re.split(r"[^\dKk\.]+", s)
    nums = [p for p in parts if p]
    def parse_num(token):
        if not token:
            return None
        if token.lower().endswith("k"):
            return float(token[:-1]) * 1000.0
        try:
            return float(token)
        except:
            return None
    parsed = [parse_num(p) for p in nums]
    parsed = [p for p in parsed if p is not None]
    if not parsed:
        return None, currency
    if len(parsed) == 1:
        return parsed[0], currency
    return sum(parsed) / len(parsed), currency

def clamp(v, a, b):
    return max(a, min(b, v))

def count_keywords(text: str, keywords: List[str]) -> int:
    text = text.lower()
    return sum(1 for k in keywords if k in text)

def compute_complexity_score(product_level: str, ui_level: str, platforms: List[str], description: str) -> float:
    product_level_weight = {"POC": 1.0, "MVP": 2.0, "Full Product": 3.0}.get(product_level, 2.0)
    ui_level_weight = {"Simple": 0.85, "Polished": 1.15}.get(ui_level, 1.0)
    platforms_factor = 1.0 + 0.2 * max(0, (len(platforms) - 1))
    words = len(description.split())
    description_density = clamp(words / 120.0, 0.2, 2.5)
    keywords = [
        "marketplace","payments","multi-tenant","integrations","real-time","ml","rag",
        "chatbot","mental health","analytics","kyc","watermark","video",
    ]
    keyword_multiplier = 0.4 * min(4, count_keywords(description, keywords))
    complexity_score = (
        product_level_weight
        * ui_level_weight
        * platforms_factor
        * description_density
        + keyword_multiplier
    )
    return complexity_score

def budget_factor_from_budget(budget_usd, budget_currency="USD", exchange_inr_to_usd=0.012):
    if budget_usd is None:
        return 1.0
    try:
        b = float(budget_usd)
    except:
        return 1.0
    if budget_currency == "INR":
        b = b * exchange_inr_to_usd
    if b < 10000:
        return 0.7
    elif 10000 <= b <= 75000:
        return 1.0
    else:
        return 1.3

def choose_feature_count(complexity_score: float, budget_factor: float, product_level: str):
    raw = round(complexity_score * budget_factor)
    if "POC" in product_level:
        low, high = 3, 8
    elif "MVP" in product_level:
        low, high = 5, 12
    else:
        low, high = 6, 20
    raw = clamp(raw, low, high)
    return int(raw)

def infer_feature_type(name: str) -> str:
    s = name.lower()
    match True:
        case _ if any(k in s for k in ["auth", "login", "signup", "sso", "otp", "role"]):
            return "auth_roles"
        case _ if any(k in s for k in ["kyc", "identity", "document", "verification", "badge"]):
            return "kyc_forms"
        case _ if any(k in s for k in ["media", "image", "photo", "video", "watermark", "thumbnail"]):
            return "media_handling"
        case _ if any(k in s for k in ["list", "listing", "crud", "create listing", "listing creation"]):
            return "crud_dashboard"
        case _ if any(k in s for k in ["moderation", "duplicate", "review", "publish", "quality", "trust"]):
            return "moderation_workflow"
        case _ if any(k in s for k in ["pay", "payment", "checkout", "paystack", "stripe", "billing"]):
            return "payments"
        case _ if any(k in s for k in ["message", "chat", "inbox", "messaging"]):
            return "messaging"
        case _ if any(k in s for k in ["calendar", "viewing", "booking", "appointments"]):
            return "calendar_viewings"
        case _ if any(k in s for k in ["notify", "notification", "email", "push"]):
            return "notifications"
        case _ if any(k in s for k in ["analytic", "report", "csv", "metrics", "dashboard metrics"]):
            return "analytics"
        case _ if any(k in s for k in ["agency", "team", "organization", "org", "sub-agent"]):
            return "agency_management"
        case _ if any(k in s for k in ["support", "ticket", "dispute"]):
            return "support_ticketing"
        case _:
            return "default"

def apply_reuse_factor(base_hours: float, same_type_count: int) -> float:
    if same_type_count <= 1:
        return base_hours
    factor = 1.0 - (0.08 * (same_type_count - 1))
    factor = max(DEFAULT_REUSE_FACTOR, factor)
    return base_hours * factor

def _allocate_integer_hours(total: int, weights: Dict[str, float]) -> Dict[str, int]:
    if total <= 0:
        return {r: 0 for r in weights}
    quotas = {r: total * w for r, w in weights.items()}
    floors = {r: int(math.floor(q)) for r, q in quotas.items()}
    allocated = sum(floors.values())
    remainders = sorted(
        ((r, quotas[r] - floors[r]) for r in weights),
        key=lambda x: x[1],
        reverse=True,
    )
    remaining = total - allocated
    for i in range(remaining):
        r = remainders[i % len(remainders)][0]
        floors[r] += 1
    return {r: int(max(0, floors[r])) for r in floors}

def distribute_hours_to_roles(
    total_feature_hours: int,
    feature_type: str,
    override_weights: Optional[Dict[str, float]] = None
) -> Dict[str, int]:
    weights = dict(ROLE_WEIGHTS.get(feature_type, ROLE_WEIGHTS["default"]))
    if override_weights:
        for r in weights:
            if r in override_weights and override_weights[r] is not None:
                weights[r] = max(MIN_ROLE_PERCENT, float(override_weights[r]))
    s = sum(weights.values())
    normalized = {r: (w / s) for r, w in weights.items()}
    for r in normalized:
        normalized[r] = max(normalized[r], MIN_ROLE_PERCENT)
    s2 = sum(normalized.values())
    normalized = {r: v / s2 for r, v in normalized.items()}
    return _allocate_integer_hours(int(total_feature_hours), normalized)

def compute_cost_from_role_hours(role_hours: Dict[str, int]) -> Dict[str, float]:
    costs = {}
    total = 0.0
    for r, h in role_hours.items():
        rate = HOURLY_RATES.get(r, 0.0)
        c = round(h * rate, 2)
        costs[f"{r}_cost_usd"] = c
        total += c
    costs["total_feature_cost_usd"] = round(total, 2)
    return costs

def validate_feature_hours(feature: Dict[str, Any], baseline: Dict[str, Any]) -> List[str]:
    warnings = []
    fh = int(feature["trace"]["computed_total_hours"])
    if fh < baseline.get("min"):
        warnings.append(f"Computed hours {fh} below baseline min {baseline.get('min')}.")
    if fh > baseline.get("max"):
        warnings.append(f"Computed hours {fh} exceed baseline max {baseline.get('max')}.")
    return warnings

# ----------------- LLM / Fallback utilities -----------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if OPENAI_API_KEY:
    try:
        client_for_openai = openai.OpenAI(api_key=OPENAI_API_KEY)
    except Exception:
        client_for_openai = None
else:
    client_for_openai = None

def safe_extract_json_array(content: str):
    try:
        s = content.find("["); e = content.rfind("]")
        if s >= 0 and e >= 0 and e > s:
            candidate = content[s:e+1]
            return json.loads(candidate)
    except Exception:
        pass
    try:
        return json.loads(content)
    except Exception:
        return None

# Deterministic fallback generator: returns meaningful atomic feature names
EXAMPLE_NAMES_BY_HINT = {
    "auth_roles": ["Signup & Login", "Role-Based Access", "Password Reset"],
    "crud_dashboard": ["Listing CRUD", "Filters & Sorting", "Admin Dashboard CRUD"],
    "kyc_forms": ["KYC Document Upload", "KYC Verification Flow"],
    "media_handling": ["Image Upload & Optimization", "Video Upload & Transcoding"],
    "moderation_workflow": ["Content Moderation Queue", "Duplicate Detection"],
    "payments": ["Checkout & Payment", "Billing & Invoices"],
    "messaging": ["In-app Messaging", "Notifications Center"],
    "calendar_viewings": ["Calendar & Bookings", "Appointment Scheduling"],
    "notifications": ["Email & Push Notifications"],
    "analytics": ["Dashboard Metrics", "Exportable Reports"],
    "agency_management": ["Agency/Team Management"],
    "support_ticketing": ["Support Ticketing System"],
    "default": ["Custom Feature"],
}

def deterministic_feature_names(description: str, feature_count: int, product_title: str) -> List[Dict[str, str]]:
    # Simple heuristic: pick hints present in description, else fill with common modules
    d = description.lower()
    hints_found = []
    for hint in EXAMPLE_NAMES_BY_HINT.keys():
        if any(k in d for k in hint.split("_")) or any(k in d for k in ["auth","login","kyc","payment","chat","message","video","image","analytics","calendar","support","agency","listing","crud"]):
            hints_found.append(hint)
    # ensure variety & fallback
    ordered = []
    # prioritize common modules
    preferred = ["auth_roles","crud_dashboard","payments","messaging","media_handling","analytics","notifications","support_ticketing"]
    for p in preferred:
        if p not in ordered and p in EXAMPLE_NAMES_BY_HINT:
            ordered.append(p)
    for h in hints_found:
        if h not in ordered:
            ordered.append(h)
    for h in EXAMPLE_NAMES_BY_HINT:
        if h not in ordered:
            ordered.append(h)
    out = []
    i = 0
    while len(out) < feature_count:
        hint = ordered[i % len(ordered)]
        names = EXAMPLE_NAMES_BY_HINT.get(hint, ["Feature"])
        name = names[(len(out)) % len(names)]
        # add ordinal if repeats
        if sum(1 for o in out if o["hint"] == hint) > 0:
            name = f"{name} ({sum(1 for o in out if o['hint'] == hint)+1})"
        out.append({"feature_name": name, "hint": hint})
        i += 1
    return out

def call_llm_generate_feature_names(description: str, feature_count: int, product_title: str, product_level: str) -> List[Dict[str, str]]:
    system = PROMPT_FEATURE_GEN.strip()
    user_payload = {
        "product_title": product_title,
        "product_level": product_level,
        "project_description": description,
        "target_feature_count": feature_count,
    }
    prompt = {"role": "user", "content": json.dumps(user_payload, indent=2)}

    # If no client, deterministic meaningful fallback
    if not client_for_openai:
        return deterministic_feature_names(description, feature_count, product_title)

    try:
        resp = client_for_openai.chat.completions.create(
            model="gpt-5",
            messages=[{"role": "system", "content": system}, prompt],
        )
        content = resp.choices[0].message.content
        arr = safe_extract_json_array(content)
        if isinstance(arr, list):
            out = []
            for a in arr:
                fn = a.get("feature_name") if isinstance(a, dict) else str(a)
                hint = a.get("hint") if isinstance(a, dict) and a.get("hint") else ""
                out.append({"feature_name": fn, "hint": hint})
            if len(out) >= feature_count:
                return out[:feature_count]
            while len(out) < feature_count:
                out.append({"feature_name": f"Feature {len(out)+1}", "hint": "default"})
            return out
    except Exception:
        # fallback deterministic
        return deterministic_feature_names(description, feature_count, product_title)

def call_llm_fill_feature_texts(features_with_hours: List[Dict[str, Any]], project_context: Dict[str, Any]) -> List[Dict[str, Any]]:
    system = PROMPT_FEATURE_ENRICH.strip()
    payload = {
        "project": project_context,
        "calibration": {"module_ranges": CALIBRATION_GUIDE},
        "features": []
    }
    for f in features_with_hours:
        payload["features"].append(
            {
                "feature_name": f["feature_name"],
                "feature_type": f["trace"]["feature_type"],
                "computed_total_hours": int(f["trace"]["computed_total_hours"]),
                "complexity_level": f["trace"]["complexity_level"],
                "resources": f["resources_map"],
            }
        )
    prompt = {"role": "user", "content": json.dumps(payload, indent=2)}

    if not client_for_openai:
        enriched = []
        for f in features_with_hours:
            total = int(f["trace"]["computed_total_hours"])
            timeline = [
                {"phase":"Planning","hours":max(2, int(round(total*0.08)))},
                {"phase":"Design","hours":max(2, int(round(total*0.14)))},
                {"phase":"Build","hours":max(4, int(round(total*0.56)))},
                {"phase":"QA","hours":max(2, int(round(total*0.18)))},
                {"phase":"UAT","hours":max(1, total - sum([max(2, int(round(total*0.08))), max(2, int(round(total*0.14))), max(4, int(round(total*0.56))), max(2, int(round(total*0.18)))]))},
            ]
            enriched.append(
                {
                    "feature_name": f["feature_name"],
                    "description": f"Auto-generated description for {f['feature_name']}.",
                    "acceptance_criteria": ["Works end-to-end", "Data saved correctly", "No major errors"],
                    "user_story": f"As a user I want {f['feature_name'].lower()} so that I ...",
                    "dependencies": "None",
                    "deliverables": ["API endpoints", "UI screens"],
                    "timeline_plan": timeline,
                    "calendar_weeks": int(math.ceil(total/32.0)),
                }
            )
        return enriched

    try:
        resp = client_for_openai.chat.completions.create(
            model="gpt-5",
            messages=[{"role": "system", "content": system}, prompt],
        )
        content = resp.choices[0].message.content
        arr = safe_extract_json_array(content)
        if isinstance(arr, list):
            return arr
    except Exception:
        pass

    # fallback same as above
    enriched = []
    for f in features_with_hours:
        total = int(f["trace"]["computed_total_hours"])
        timeline = [
            {"phase":"Planning","hours":max(2, int(round(total*0.08)))},
            {"phase":"Design","hours":max(2, int(round(total*0.14)))},
            {"phase":"Build","hours":max(4, int(round(total*0.56)))},
            {"phase":"QA","hours":max(2, int(round(total*0.18)))},
            {"phase":"UAT","hours":max(1, total - sum([max(2, int(round(total*0.08))), max(2, int(round(total*0.14))), max(4, int(round(total*0.56))), max(2, int(round(total*0.18)))]))},
        ]
        enriched.append(
            {
                "feature_name": f["feature_name"],
                "description": f"Auto-generated description for {f['feature_name']}.",
                "acceptance_criteria": ["Works end-to-end", "Data saved correctly", "No major errors"],
                "user_story": f"As a user I want {f['feature_name'].lower()} so that I ...",
                "dependencies": "None",
                "deliverables": ["API endpoints", "UI screens"],
                "timeline_plan": timeline,
                "calendar_weeks": int(math.ceil(total/32.0)),
            }
        )
    return enriched

def call_llm_compute_pm_qa(features_with_hours: List[Dict[str, Any]], project_context: Dict[str, Any]) -> Dict[str, Any]:
    system = PROMPT_PM_QA.strip()
    payload = {
        "project": project_context,
        "features": [
            {"feature_name": f["feature_name"], "computed_total_hours": int(f["trace"]["computed_total_hours"]), "complexity_level": int(f["trace"]["complexity_level"])}
            for f in features_with_hours
        ]
    }
    prompt = {"role": "user", "content": json.dumps(payload, indent=2)}

    if not client_for_openai:
        total = sum(int(f["trace"]["computed_total_hours"]) for f in features_with_hours)
        pm = max(16, int(round(total * 0.06)))
        qa = max(32, int(round(total * 0.12)))
        return {"pm_hours": pm, "qa_hours": qa, "rationale": "Fallback heuristic: PM ~6% (min 16h) and QA ~12% (min 32h) of total engineering hours."}

    try:
        resp = client_for_openai.chat.completions.create(
            model="gpt-5",
            messages=[{"role": "system", "content": system}, prompt],
        )
        content = resp.choices[0].message.content
        try:
            obj = json.loads(content)
            if isinstance(obj, dict) and "pm_hours" in obj and "qa_hours" in obj:
                obj["pm_hours"] = max(16, int(obj["pm_hours"]))
                obj["qa_hours"] = max(32, int(obj["qa_hours"]))
                return obj
        except Exception:
            try:
                s = content.find("{"); e = content.rfind("}")
                if s>=0 and e> s:
                    obj = json.loads(content[s:e+1])
                    obj["pm_hours"] = max(16, int(obj.get("pm_hours", 16)))
                    obj["qa_hours"] = max(32, int(obj.get("qa_hours", 32)))
                    return obj
            except Exception:
                pass
    except Exception:
        pass

    total = sum(int(f["trace"]["computed_total_hours"]) for f in features_with_hours)
    pm = max(16, int(round(total * 0.06)))
    qa = max(32, int(round(total * 0.12)))
    return {"pm_hours": pm, "qa_hours": qa, "rationale": "Fallback heuristic after LLM failure: PM ~6% (min 16h) and QA ~12% (min 32h) of total engineering hours."}

def call_llm_suggest_headcount(total_role_hours: Dict[str, int], complexity_level: int, feature_count: int, project_context: Dict[str, Any]) -> Dict[str, Any]:
    """
    Headcount policy:
     - fullstack_count: 1..3 based on total fullstack hours, average per-feature dev load, complexity, and feature_count
     - ai_count: 1 if ai_hours > 0 (conservative); else 0
     - ui_ux_count: 1 if ui_ux_hours > 0 (conservative); else 0
    """
    system = PROMPT_HEADCOUNT.strip()
    payload = {"project": project_context, "total_role_hours": total_role_hours, "complexity_level": complexity_level, "feature_count": feature_count}
    prompt = {"role": "user", "content": json.dumps(payload, indent=2)}

    # Deterministic fallback if LLM not present or fails
    total_fs = total_role_hours.get("fullstack", 0)
    avg_fs = total_fs / max(1, feature_count)

    # fallback rules (conservative)
    if total_fs <= 200 or (avg_fs <= 25 and feature_count <= 12 and complexity_level <= 3):
        fs = 1
    elif total_fs <= 450 or (avg_fs <= 60 and complexity_level <= 4):
        fs = 2
    else:
        fs = 3

    # Conservative ai/ui_ux suggestion: 1 if any hours exist, otherwise 0
    ai_count = 1 if total_role_hours.get("ai", 0) > 0 else 0
    ui_ux_count = 1 if total_role_hours.get("ui_ux", 0) > 0 else 0

    rationale = "Practical caps: fullstack 1-3 (based on total & avg dev hours + complexity). ai/ui_ux suggested only when they have non-zero work."
    return {"fullstack_count": int(clamp(fs, 1, 3)), "ai_count": int(max(0, ai_count)), "ui_ux_count": int(max(0, ui_ux_count)), "rationale": rationale}

# ----------------- STREAMLIT UI -----------------
st.markdown("<div style='font-size:1.6rem;font-weight:700;'>🤖 AI Project Estimation Generator — Final</div>", unsafe_allow_html=True)
st.markdown("<div style='color:#444;margin-bottom:1rem;'>Per-feature resources: fullstack / ai / ui_ux. PM & QA are project-level cumulative hours (excluded from budget).</div>", unsafe_allow_html=True)

if not OPENAI_API_KEY:
    st.warning("OPENAI_API_KEY not found in environment. App will run with deterministic fallbacks for feature names and other LLM tasks.")

with st.form("estimation_form"):
    st.subheader("📋 Project Input Details")
    title = st.text_input("🧾 Project Title (optional)")
    description = st.text_area("📝 Project Description (required)", height=180)
    product_level = st.selectbox("⚙️ Product Level", ["POC", "MVP", "Full Product"])
    ui_level = st.selectbox("🎨 UI Level", ["Simple", "Polished"])
    platforms = st.multiselect("💻 App Platform(s)", ["Web", "iOS", "Android", "Desktop"])
    target_audience = st.text_input("🎯 Target Audience (optional)")
    competitors = st.text_input("🏁 Competitors (optional)")
    budget = st.text_input("💰 Estimated Budget (optional)", placeholder="e.g. $15,000 – $25,000")
    reuse_toggle = True
    slider_multiplier = st.slider("Estimation conservatism multiplier (0-3)", 0.0, 3.0, 0.0, 0.25)
    generate = st.form_submit_button("🚀 Generate Estimation")

if generate:
    if not description.strip():
        st.warning("Please provide a project description.")
        st.stop()

    st.info("Computing calibrated estimates...")

    complexity_score = compute_complexity_score(product_level, ui_level, platforms, description)
    complexity_level = clamp(int(round(clamp(complexity_score / 2.5, 1, 5))), 1, 5)
    complexity_multiplier = COMPLEXITY_MULTIPLIER.get(complexity_level, 1.0)
    complexity_multiplier = (complexity_multiplier + slider_multiplier) / 2.0

    parsed_budget, budget_currency = parse_budget_to_usd(budget)
    exchange_rate = float(os.getenv("EXCHANGE_RATE_INR_TO_USD", "0.012"))
    bfactor = budget_factor_from_budget(parsed_budget, budget_currency, exchange_rate)

    feature_count = choose_feature_count(complexity_score, bfactor, product_level)

    proposed_features = call_llm_generate_feature_names(description, feature_count, title, product_level)

    # Map types, compute hours & distribute roles (only fullstack/ai/ui_ux)
    type_counts: Dict[str, int] = {}
    features_with_hours: List[Dict[str, Any]] = []
    for pf in proposed_features:
        fname = pf.get("feature_name", "Unnamed Feature").strip()
        hint = (pf.get("hint") or "").strip()
        ftype = hint if hint in BASELINE_MODULES else infer_feature_type(fname)
        type_counts.setdefault(ftype, 0)
        type_counts[ftype] += 1

    seen_type_counter = {}
    for pf in proposed_features:
        fname = pf.get("feature_name", "Unnamed Feature").strip()
        hint = (pf.get("hint") or "").strip()
        ftype = hint if hint in BASELINE_MODULES else infer_feature_type(fname)
        seen_type_counter.setdefault(ftype, 0)
        seen_type_counter[ftype] += 1
        nth_of_type = seen_type_counter[ftype]

        baseline = BASELINE_MODULES.get(ftype, BASELINE_MODULES["default"])
        median = baseline["median"]

        platforms_factor = 1.0 + 0.2 * max(0, (len(platforms) - 1))
        base_hours = median * complexity_multiplier * platforms_factor

        if reuse_toggle and type_counts.get(ftype, 1) > 1:
            base_hours = apply_reuse_factor(base_hours, nth_of_type)

        base_hours = clamp(base_hours, baseline["min"] * 0.9, baseline["max"] * 1.2)
        base_hours = clamp(base_hours, MIN_HOURS_PER_FEATURE, MAX_HOURS_PER_FEATURE)
        base_hours = int(round(base_hours))

        # distribute only to fullstack/ai/ui_ux
        role_hours = distribute_hours_to_roles(base_hours, ftype)
        costs = compute_cost_from_role_hours(role_hours)

        trace = {
            "feature_type": ftype,
            "baseline": baseline,
            "complexity_multiplier": complexity_multiplier,
            "platforms_factor": platforms_factor,
            "reuse_applied_for_instance": nth_of_type if reuse_toggle else 1,
            "computed_total_hours": int(base_hours),
            "complexity_level": complexity_level,
            "timestamp": datetime.utcnow().isoformat() + "Z",
        }

        feature_obj = {
            "feature_name": fname,
            "feature_type": ftype,
            "resources_map": {k: int(v) for k, v in role_hours.items()},
            "costs_map": costs,
            "trace": trace,
        }

        warnings = validate_feature_hours(feature_obj, baseline)
        if warnings:
            feature_obj["warnings"] = warnings

        features_with_hours.append(feature_obj)

    # Enrich texts (LLM or fallback)
    project_context = {
        "title": title,
        "product_level": product_level,
        "ui_level": ui_level,
        "platforms": platforms,
        "target_audience": target_audience,
        "competitors": competitors,
        "budget_raw": budget,
        "budget_parsed_usd": parsed_budget,
    }
    enriched_texts = call_llm_fill_feature_texts(features_with_hours, project_context)

    # Merge enriched_texts back; apply recommended_hours if provided
    enriched_map = {e.get("feature_name"): e for e in enriched_texts if isinstance(e, dict) and e.get("feature_name")}
    final_features = []
    for fw in features_with_hours:
        name = fw["feature_name"]
        enrich = enriched_map.get(name, {}) or {}
        fw["description"] = enrich.get("description", f"A generated description for {name}.")
        fw["acceptance_criteria"] = enrich.get("acceptance_criteria", ["End-to-end works", "No errors", "Basic validations"])
        fw["user_story"] = enrich.get("user_story", f"As a user I want {name.lower()} so that ...")
        fw["dependencies"] = enrich.get("dependencies", "")
        fw["deliverables"] = enrich.get("deliverables", ["API", "UI"])

        recommended_hours = enrich.get("recommended_hours")
        role_mix = enrich.get("role_mix")  # optional

        chosen_total_hours = int(fw["trace"]["computed_total_hours"])
        if isinstance(recommended_hours, (int, float)) and recommended_hours > 0:
            chosen_total_hours = int(round(recommended_hours))
            fw["trace"]["adjusted_by_llm"] = True
            fw["trace"]["hours_before_llm"] = int(fw["trace"]["computed_total_hours"])
            fw["trace"]["hours_rationale"] = enrich.get("hours_rationale", "")
            fw["trace"]["computed_total_hours"] = chosen_total_hours

        # re-distribute roles if hours or role mix changed
        role_hours = distribute_hours_to_roles(chosen_total_hours, fw["feature_type"], override_weights=role_mix if isinstance(role_mix, dict) else None)
        fw["resources_map"] = {k: int(v) for k, v in role_hours.items()}
        fw["costs_map"] = compute_cost_from_role_hours(fw["resources_map"])

        # timeline handling = ensure integer hours and sum to chosen_total_hours
        timeline_plan = enrich.get("timeline_plan", [])
        if isinstance(timeline_plan, list) and all(isinstance(p, dict) for p in timeline_plan):
            plan = []
            for p in timeline_plan:
                phase = str(p.get("phase", "Phase")).strip() or "Phase"
                hrs = int(max(0, round(float(p.get("hours", 0)))))
                plan.append({"phase": phase, "hours": hrs})
            total_in_plan = sum(p["hours"] for p in plan)
            if total_in_plan != chosen_total_hours and chosen_total_hours > 0:
                if total_in_plan > 0:
                    scaled = []
                    acc = 0
                    for i, p in enumerate(plan):
                        if i < len(plan) - 1:
                            new_h = int(round(chosen_total_hours * (p["hours"] / total_in_plan)))
                            scaled.append({"phase": p["phase"], "hours": new_h})
                            acc += new_h
                        else:
                            scaled.append({"phase": p["phase"], "hours": max(0, chosen_total_hours - acc)})
                    plan = scaled
                else:
                    plan = [
                        {"phase":"Planning","hours":max(2, int(round(chosen_total_hours*0.08)))},
                        {"phase":"Design","hours":max(2, int(round(chosen_total_hours*0.14)))},
                        {"phase":"Build","hours":max(4, int(round(chosen_total_hours*0.56)))},
                        {"phase":"QA","hours":max(2, int(round(chosen_total_hours*0.18)))},
                        {"phase":"UAT","hours":max(1, chosen_total_hours - sum([max(2, int(round(chosen_total_hours*0.08))), max(2, int(round(chosen_total_hours*0.14))), max(4, int(round(chosen_total_hours*0.56))), max(2, int(round(chosen_total_hours*0.18)))]))},
                    ]
            fw["timeline_plan"] = plan
        else:
            t = chosen_total_hours
            fw["timeline_plan"] = [
                {"phase":"Planning","hours":max(2, int(round(t*0.08)))},
                {"phase":"Design","hours":max(2, int(round(t*0.14)))},
                {"phase":"Build","hours":max(4, int(round(t*0.56)))},
                {"phase":"QA","hours":max(2, int(round(t*0.18)))},
                {"phase":"UAT","hours":max(1, t - sum([max(2, int(round(t*0.08))), max(2, int(round(t*0.14))), max(4, int(round(t*0.56))), max(2, int(round(t*0.18)))]))},
            ]

        fw["calendar_weeks"] = int(enrich.get("calendar_weeks", math.ceil(chosen_total_hours/32.0)))
        final_features.append(fw)

    # Project-level PM & QA totals (LLM or fallback)
    pm_qa = call_llm_compute_pm_qa(final_features, project_context)
    project_pm_hours = int(pm_qa.get("pm_hours", 0))
    project_qa_hours = int(pm_qa.get("qa_hours", 0))
    pm_qa_rationale = pm_qa.get("rationale", "")

    # Resources summary & budget (exclude PM & QA costs)
    total_estimated_cost = sum(f["costs_map"]["total_feature_cost_usd"] for f in final_features)
    resources_summary: Dict[str, int] = {}
    for f in final_features:
        for r, h in f["resources_map"].items():
            resources_summary[r] = int(resources_summary.get(r, 0)) + int(h)

    # Suggest headcount deterministically (LLM optional)
    headcount_suggestion = call_llm_suggest_headcount(resources_summary, complexity_level, len(final_features), project_context)

    # Ensure ai/ui_ux headcounts are at least 1 when hours exist (user expected)
    if resources_summary.get("ai", 0) > 0 and headcount_suggestion.get("ai_count", 0) < 1:
        headcount_suggestion["ai_count"] = 1
    if resources_summary.get("ui_ux", 0) > 0 and headcount_suggestion.get("ui_ux_count", 0) < 1:
        headcount_suggestion["ui_ux_count"] = 1

    resources_list = [
        {"role": "fullstack", "hours": int(resources_summary.get("fullstack", 0)), "suggested_count": int(headcount_suggestion.get("fullstack_count", 1))},
        {"role": "ai", "hours": int(resources_summary.get("ai", 0)), "suggested_count": int(headcount_suggestion.get("ai_count", 0))},
        {"role": "ui_ux", "hours": int(resources_summary.get("ui_ux", 0)), "suggested_count": int(headcount_suggestion.get("ui_ux_count", 0))},
    ]

    # Build features_out and per-feature budget
    features_out = []
    per_feature_budget = []
    for f in final_features:
        features_out.append({
            "feature_name": f["feature_name"],
            "description": f.get("description", ""),
            "acceptance_criteria": f.get("acceptance_criteria", []),
            "user_story": f.get("user_story", ""),
            "dependencies": f.get("dependencies", ""),
            "deliverables": f.get("deliverables", ""),
            "resources": [
                {"role": "fullstack", "hours": int(f["resources_map"].get("fullstack", 0))},
                {"role": "ai", "hours": int(f["resources_map"].get("ai", 0))},
                {"role": "ui_ux", "hours": int(f["resources_map"].get("ui_ux", 0))},
            ],
            "timeline": {
                "phase": "LLM-calibrated",
                "duration_hours": int(f["trace"]["computed_total_hours"]),
                "plan": f.get("timeline_plan", []),
            },
            "cost_estimate": {
                "fullstack_cost_usd": f["costs_map"].get("fullstack_cost_usd", 0.0),
                "ai_cost_usd": f["costs_map"].get("ai_cost_usd", 0.0),
                "ui_ux_cost_usd": f["costs_map"].get("ui_ux_cost_usd", 0.0),
                "total_feature_cost_usd": f["costs_map"].get("total_feature_cost_usd", 0.0),
            },
            "trace": f["trace"],
            "warnings": f.get("warnings", []),
        })
        per_feature_budget.append({
            "feature_name": f["feature_name"],
            "total_feature_cost_usd": f["costs_map"].get("total_feature_cost_usd", 0.0)
        })

    output_json = {
        "features": features_out,
        "resources": resources_list,
        "tech": [],
        "budget": {
            "currency": "USD",
            "per_feature": per_feature_budget,
            "total_estimated_cost_usd": round(total_estimated_cost, 2),
            "budget_provided": parsed_budget,
            "budget_currency": budget_currency,
            "within_budget": None if parsed_budget is None else (parsed_budget >= total_estimated_cost),
            "notes": "PM & QA hours shown at project-level and excluded from budget per request.",
            "project_pm_hours": project_pm_hours,
            "project_qa_hours": project_qa_hours,
            "pm_qa_rationale": pm_qa_rationale,
        },
    }

    # ----------------- RENDER UI -----------------
    st.success("✅ Estimation Generated (final)")

    st.markdown("### 📘 Summary")
    st.write(f"Complexity score (raw): {complexity_score:.2f}")
    st.write(f"Complexity level (1-5): {complexity_level}, multiplier applied: {complexity_multiplier:.2f}")
    st.write(f"Feature count chosen: {feature_count}")
    st.write(f"Budget parsed (USD): {parsed_budget} ({budget_currency})")
    st.write(f"Total estimated cost (USD) [PM & QA excluded]: {output_json['budget']['total_estimated_cost_usd']}")

    st.markdown("### 🏗 Features Overview")
    rows = []
    for f in output_json["features"]:
        rows.append({
            "feature_name": f["feature_name"],
            "feature_type": f["trace"]["feature_type"],
            "duration_hours": int(f["timeline"]["duration_hours"]),
            "fullstack_hours": int(next((r["hours"] for r in f["resources"] if r["role"] == "fullstack"), 0)),
            "ai_hours": int(next((r["hours"] for r in f["resources"] if r["role"] == "ai"), 0)),
            "ui_ux_hours": int(next((r["hours"] for r in f["resources"] if r["role"] == "ui_ux"), 0)),
            "total_feature_cost_usd": float(f["cost_estimate"]["total_feature_cost_usd"]),
            "warnings": "; ".join(f.get("warnings", [])),
        })
    df = pd.DataFrame(rows)

    # Ensure feature_name column exists and is first column
    cols_order = ["feature_name", "feature_type", "duration_hours", "fullstack_hours", "ai_hours", "ui_ux_hours", "total_feature_cost_usd", "warnings"]
    cols_present = [c for c in cols_order if c in df.columns]
    df_display = df[cols_present].copy()

    # Format numeric columns
    df_display["duration_hours"] = df_display["duration_hours"].astype(int)
    df_display["fullstack_hours"] = df_display["fullstack_hours"].astype(int)
    df_display["ai_hours"] = df_display["ai_hours"].astype(int)
    df_display["ui_ux_hours"] = df_display["ui_ux_hours"].astype(int)
    df_display["total_feature_cost_usd"] = df_display["total_feature_cost_usd"].map(lambda x: f"{x:,.2f}")

    # Reset index so Streamlit doesn't show pandas index column
    st.dataframe(df_display.reset_index(drop=True), use_container_width=True)

    st.markdown("### 👥 Resources (suggested headcount)")
    res_df = pd.DataFrame(resources_list)
    res_df["hours"] = res_df["hours"].astype(int)
    res_df["suggested_count"] = res_df["suggested_count"].astype(int)
    st.dataframe(res_df.reset_index(drop=True), use_container_width=True)
    st.write(f"Headcount rationale: {headcount_suggestion.get('rationale', '')}")

    st.markdown("### 🧾 Project-level PM & QA (cumulative hours) — not included in budget")
    st.write(f"Project PM hours (cumulative): **{project_pm_hours}h**")
    st.write(f"Project QA hours (cumulative): **{project_qa_hours}h**")
    if pm_qa_rationale:
        st.write("Rationale:")
        st.write(pm_qa_rationale)

    st.markdown("### 💰 Budget")
    st.write(f"Total estimated cost (USD) [PM & QA excluded]: {output_json['budget']['total_estimated_cost_usd']}")
    st.write(f"Within provided budget? {output_json['budget']['within_budget']}")
    per_feature_df = pd.DataFrame(output_json["budget"]["per_feature"])
    per_feature_df["total_feature_cost_usd"] = per_feature_df["total_feature_cost_usd"].map(lambda x: f"{x:,.2f}")
    st.dataframe(per_feature_df.reset_index(drop=True), use_container_width=True)

    st.markdown("### 🔍 Feature detail (first 5)")
    for f in output_json["features"][:5]:
        st.markdown(f"**{f['feature_name']}** — _{f['trace']['feature_type']}_")
        st.write(f.get("description", ""))
        st.write("Acceptance criteria:")
        for a in f.get("acceptance_criteria", []):
            st.write(f"- {a}")
        st.write("Timeline plan (hours):")
        for p in f["timeline"]["plan"]:
            st.write(f"- {p['phase']}: {p['hours']}h")
        if f.get("warnings"):
            st.warning(f"Warnings: {f['warnings']}")

    st.markdown("---")
    st.download_button("Download estimation (JSON)", data=json.dumps(output_json, indent=2), file_name="estimation.json", mime="application/json")
    csv_buf = StringIO()
    df.to_csv(csv_buf, index=False)
    st.download_button("Download features CSV", data=csv_buf.getvalue(), file_name="features.csv", mime="text/csv")

    with st.expander("Show full JSON output"):
        st.json(output_json)
