# save as app.py 
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

# ----------------- CONFIG -----------------
st.set_page_config(
    page_title="AI Project Estimation Generator (Refactored)",
    layout="centered",
    page_icon="🤖",
)

# Hourly rates (single source of truth)
HOURLY_RATES = {"fullstack": 25.0, "ai": 30.0, "ui_ux": 30.0, "pm": 40.0, "qa": 20.0}

# Baseline module table (min, median, max) - hours (total feature hours)
BASELINE_MODULES = {
    "auth_roles": {"min": 40, "median": 60, "max": 80},          # tightened
    "crud_dashboard": {"min": 60, "median": 90, "max": 110},      # tightened top
    "kyc_forms": {"min": 50, "median": 75, "max": 95},            # reduced UI inflation
    "media_handling": {"min": 80, "median": 120, "max": 140},
    "moderation_workflow": {"min": 80, "median": 120, "max": 160},
    "payments": {"min": 60, "median": 100, "max": 120},
    "messaging": {"min": 60, "median": 100, "max": 120},
    "calendar_viewings": {"min": 40, "median": 70, "max": 90},
    "notifications": {"min": 16, "median": 28, "max": 40},        # tightened low module
    "analytics": {"min": 60, "median": 100, "max": 120},
    "agency_management": {"min": 60, "median": 100, "max": 120},
    "support_ticketing": {"min": 40, "median": 80, "max": 100},
    # fallback
    "default": {"min": 20, "median": 50, "max": 120},
}

# Role weight templates by feature_type (balanced, within reasonable bounds)
ROLE_WEIGHTS = {
    "auth_roles":         {"fullstack": 0.58, "ui_ux": 0.12, "pm": 0.1,  "qa": 0.18, "ai": 0.02},
    "crud_dashboard":     {"fullstack": 0.50, "ui_ux": 0.22, "pm": 0.1,  "qa": 0.16, "ai": 0.02},
    "kyc_forms":          {"fullstack": 0.54, "ui_ux": 0.14, "pm": 0.1,  "qa": 0.20, "ai": 0.02},
    "media_handling":     {"fullstack": 0.62, "ui_ux": 0.12, "pm": 0.1,  "qa": 0.12, "ai": 0.04},
    "moderation_workflow":{"fullstack": 0.50, "ui_ux": 0.12, "pm": 0.1,  "qa": 0.18, "ai": 0.10},
    "payments":           {"fullstack": 0.58, "ui_ux": 0.12, "pm": 0.1,  "qa": 0.18, "ai": 0.02},
    "messaging":          {"fullstack": 0.58, "ui_ux": 0.18, "pm": 0.1,  "qa": 0.12, "ai": 0.02},
    "calendar_viewings":  {"fullstack": 0.58, "ui_ux": 0.18, "pm": 0.1,  "qa": 0.12, "ai": 0.02},
    "notifications":      {"fullstack": 0.50, "ui_ux": 0.22, "pm": 0.1,  "qa": 0.16, "ai": 0.02},
    "analytics":          {"fullstack": 0.58, "ui_ux": 0.12, "pm": 0.1,  "qa": 0.18, "ai": 0.02},
    "agency_management":  {"fullstack": 0.58, "ui_ux": 0.12, "pm": 0.1,  "qa": 0.18, "ai": 0.02},
    "support_ticketing":  {"fullstack": 0.58, "ui_ux": 0.12, "pm": 0.1,  "qa": 0.18, "ai": 0.02},
    "default":            {"fullstack": 0.58, "ui_ux": 0.15, "pm": 0.1,  "qa": 0.15, "ai": 0.02},
}

# Complexity score mapping (1..5)
COMPLEXITY_MULTIPLIER = {1: 0.7, 2: 0.9, 3: 1.0, 4: 1.15, 5: 1.35}  # slightly narrowed to avoid blowups

# Reuse multiplier when many same-type features exist
DEFAULT_REUSE_FACTOR = 0.85  # modest savings to avoid drastic drops

# Caps
MIN_HOURS_PER_FEATURE = 4
MAX_HOURS_PER_FEATURE = 400
MIN_ROLE_PERCENT = 0.02  # min share per role (2%)

# Calibration guide for prompts (used only in LLM instructions)
CALIBRATION_GUIDE = {
    "auth_roles": {"typical_range": [40, 80]},
    "crud_dashboard": {"typical_range": [60, 110]},
    "kyc_forms": {"typical_range": [50, 95]},
    "media_handling": {"typical_range": [80, 140]},
    "moderation_workflow": {"typical_range": [80, 160]},
    "payments": {"typical_range": [60, 120]},
    "messaging": {"typical_range": [60, 120]},
    "calendar_viewings": {"typical_range": [40, 90]},
    "notifications": {"typical_range": [16, 40]},
    "analytics": {"typical_range": [60, 120]},
    "agency_management": {"typical_range": [60, 120]},
    "support_ticketing": {"typical_range": [40, 100]},
    "default": {"typical_range": [20, 120]},
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


def compute_complexity_score(
    product_level: str,
    ui_level: str,
    platforms: List[str],
    description: str,
) -> float:
    product_level_weight = {"POC": 1.0, "MVP": 2.0, "Full Product": 3.0}.get(
        product_level, 2.0
    )
    ui_level_weight = {"Simple": 0.85, "Polished": 1.15}.get(ui_level, 1.0)
    platforms_factor = 1.0 + 0.2 * max(0, (len(platforms) - 1))  # slightly softer
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


def budget_factor_from_budget(budget_usd):
    if budget_usd is None:
        return 1.0
    try:
        b = float(budget_usd)
    except:
        return 1.0
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
        # only accept known roles, ignore others, never let any role be 0
        for r in weights:
            if r in override_weights and override_weights[r] is not None:
                weights[r] = max(MIN_ROLE_PERCENT, float(override_weights[r]))
    # normalize
    s = sum(weights.values())
    normalized = {r: (w / s) for r, w in weights.items()}
    # enforce mins
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
    fh_dev = feature["resources_map"].get("fullstack", 0)
    fh_qa = feature["resources_map"].get("qa", 0)
    if fh_dev > 0 and (fh_qa / max(1.0, fh_dev)) < 0.08 and feature["trace"]["complexity_level"] >= 3:
        warnings.append("QA hours seem low (<8% of dev) for a complexity >= 3 feature.")
    return warnings


# ----------------- LLM WRAPPER -----------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if OPENAI_API_KEY:
    try:
        client_for_openai = openai.OpenAI(api_key=OPENAI_API_KEY)
    except Exception as e:
        client_for_openai = None
        st.warning(f"OpenAI client initialization failed: {e}")
else:
    client_for_openai = None


def call_llm_generate_feature_names(description: str, feature_count: int, product_title: str, product_level: str) -> List[Dict[str, str]]:
    """
    Generates concise, atomic features with realistic module hints.
    Prompt is tuned to avoid oversized 'generic' features and to map to known modules.
    """
    system = (
        "You are a senior product architect. "
        "Given a product description, return ONLY JSON: an array of objects with keys 'feature_name' and 'hint'. "
        "Rules:\n"
        "1) Keep features atomic (3–6 words); avoid bundling multiple modules.\n"
        "2) Provide 'hint' as one of known module types when possible: "
        f"{list(BASELINE_MODULES.keys())}.\n"
        "3) Prefer common modules like auth, CRUD dashboards, payments, messaging, analytics if relevant.\n"
        "4) Avoid exotic/niche features unless explicitly stated.\n"
        "5) Do NOT estimate hours here."
    )
    user_payload = {
        "product_title": product_title,
        "product_level": product_level,
        "project_description": description,
        "target_feature_count": feature_count,
        "instructions": (
            "Return an array of objects [{'feature_name':'...', 'hint':'short module type'}]. "
            "Example: {'feature_name':'Role-Based Login','hint':'auth_roles'}"
        )
    }
    prompt = {"role": "user", "content": json.dumps(user_payload, indent=2)}

    if not client_for_openai:
        return [{"feature_name": f"Feature {i+1}", "hint": "default"} for i in range(feature_count)]

    try:
        resp = client_for_openai.chat.completions.create(
            model="gpt-5",
            messages=[{"role": "system", "content": system}, prompt],
            max_tokens=800,
            temperature=0.2,
        )
        content = resp.choices[0].message.content
    except Exception:
        try:
            resp = client_for_openai.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "system", "content": system}, prompt],
                max_tokens=800,
                temperature=0.2,
            )
            content = resp.choices[0].message.content
        except Exception:
            return [{"feature_name": f"Feature {i+1}", "hint": "default"} for i in range(feature_count)]

    try:
        s = content.find("["); e = content.rfind("]")
        arr = json.loads(content[s:e+1])
        out = []
        for a in arr:
            fn = a.get("feature_name") if isinstance(a, dict) else str(a)
            hint = a.get("hint") if isinstance(a, dict) and a.get("hint") else ""
            out.append({"feature_name": fn, "hint": hint})
        return out
    except Exception:
        return [{"feature_name": f"Feature {i+1}", "hint": "default"} for i in range(feature_count)]


def call_llm_fill_feature_texts(features_with_hours: List[Dict[str, Any]], project_context: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Enrich features with description, AC, user story, dependencies, deliverables,
    and provide a sanity-checked TIMELINE + (if needed) recommended hours & role mix
    aligned to calibrated ranges for common modules.
    """
    # Compact payload for the model
    payload = {
        "project": project_context,
        "calibration": {
            "module_ranges": CALIBRATION_GUIDE,
            "role_mix_bounds": {
                "ui_ux": [0.10, 0.25],
                "qa": [0.10, 0.22],
                "pm": [0.08, 0.12],
                "ai": [0.00, 0.12],
                "fullstack": [0.40, 0.70]
            },
            "rules": [
                "If computed_total_hours falls OUTSIDE typical_range by >20%, suggest 'recommended_hours' within a realistic band and explain why.",
                "If role mix seems off (UI/UX inflated for simple forms; QA too low for complex flows), suggest 'role_mix' percentages that sum to 1.0 within bounds.",
                "Build a practical 'timeline_plan' with phases (Planning, Design, Build, QA, UAT/Hardening) with integer hour allocations summing to the chosen total hours.",
                "Assume ~32 productive hours/week per engineer; output an estimated 'calendar_weeks' (ceil(total_hours/32))."
            ]
        },
        "features": []
    }
    for f in features_with_hours:
        payload["features"].append(
            {
                "feature_name": f["feature_name"],
                "feature_type": f["feature_type"],
                "computed_total_hours": int(f["trace"]["computed_total_hours"]),
                "complexity_level": f["trace"]["complexity_level"],
                "resources": f["resources_map"],
            }
        )

    system = (
        "You are a pragmatic Product Strategist and Estimation QA. "
        "Return ONLY JSON: an array where each item includes the original 'feature_name' plus: "
        "description (1–3 sentences), acceptance_criteria (3–5 items), user_story (one line), "
        "dependencies (short string), deliverables (array or short string), "
        "timeline_plan (array of {phase, hours}), calendar_weeks (int), "
        "and, IF NEEDED due to unrealistic totals, recommended_hours (int), hours_rationale (string), "
        "and role_mix (object of role->percent, sum to 1.0 within given bounds). "
        "Do NOT invent exotic work; reflect realistic scope for common modules."
    )
    prompt = {"role": "user", "content": json.dumps(payload, indent=2)}

    if not client_for_openai:
        # Fallback simple text + neutral timeline
        enriched = []
        for f in features_with_hours:
            total = int(f["trace"]["computed_total_hours"])
            timeline = [
                {"phase":"Planning","hours":max(2, int(round(total*0.08)))},
                {"phase":"Design","hours":max(2, int(round(total*0.14)))},
                {"phase":"Build","hours":max(4, int(round(total*0.56)))},
                {"phase":"QA","hours":max(2, int(round(total*0.18)))},
                {"phase":"UAT","hours":max(1, total - sum([
                    max(2, int(round(total*0.08))),
                    max(2, int(round(total*0.14))),
                    max(4, int(round(total*0.56))),
                    max(2, int(round(total*0.18))),
                ]))},
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
            max_tokens=2200,
            temperature=0.2,
        )
        content = resp.choices[0].message.content
    except Exception:
        try:
            resp = client_for_openai.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "system", "content": system}, prompt],
                max_tokens=2200,
                temperature=0.2,
            )
            content = resp.choices[0].message.content
        except Exception:
            # Same fallback as above
            enriched = []
            for f in features_with_hours:
                total = int(f["trace"]["computed_total_hours"])
                timeline = [
                    {"phase":"Planning","hours":max(2, int(round(total*0.08)))},
                    {"phase":"Design","hours":max(2, int(round(total*0.14)))},
                    {"phase":"Build","hours":max(4, int(round(total*0.56)))},
                    {"phase":"QA","hours":max(2, int(round(total*0.18)))},
                    {"phase":"UAT","hours":max(1, total - sum([
                        max(2, int(round(total*0.08))),
                        max(2, int(round(total*0.14))),
                        max(4, int(round(total*0.56))),
                        max(2, int(round(total*0.18))),
                    ]))},
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
        s = content.find("["); e = content.rfind("]")
        arr = json.loads(content[s:e+1])
        return arr
    except Exception:
        return []


# ----------------- STREAMLIT UI -----------------
# CSS (kept concise)
st.markdown(
    """
    <style>
        .main-title {font-size:1.8rem; font-weight:700;}
        .subtitle {color:#555;}
        .form-card {background:#f9fafb; padding:1.5rem; border-radius:10px; border:1px solid #e5e7eb;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown("<div class='main-title'>🤖 AI Project Estimation Generator — Refactored</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Genuine timelines & calibrated estimates. LLM used for text + sanity-check.</div>", unsafe_allow_html=True)

# API Key check
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    st.warning("OPENAI_API_KEY not found in environment. The app will run in degraded local-only mode (LLM calls limited).")

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
    generate = st.form_submit_button("🚀 Generate Estimation")

if generate:
    if not description.strip():
        st.warning("Please provide a project description.")
        st.stop()

    st.info("Computing calibrated estimates...")

    # 1) compute complexity
    complexity_score = compute_complexity_score(product_level, ui_level, platforms, description)
    slider_multiplier = 3  # fixed stand-in
    complexity_level = clamp(int(round(clamp(complexity_score / 2.5, 1, 5))), 1, 5)
    complexity_multiplier = COMPLEXITY_MULTIPLIER.get(complexity_level, 1.0)
    complexity_multiplier = (complexity_multiplier + slider_multiplier) / 2.0

    # 2) parse budget
    parsed_budget, budget_currency = parse_budget_to_usd(budget)
    bfactor = budget_factor_from_budget(parsed_budget)

    # 3) determine feature_count
    feature_count = choose_feature_count(complexity_score, bfactor, product_level)

    # 4) LLM proposes features
    proposed_features = call_llm_generate_feature_names(description, feature_count, title, product_level)

    # 5) Map, compute hours & distribute roles
    type_counts: Dict[str, int] = {}
    features_with_hours = []
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
        base_hours = int(round(base_hours))  # integer total

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

    # 6) Ask LLM to fill texts + sanity-check hours & role mix
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

    # Merge enriched_texts back; APPLY recommended hours & role_mix if provided
    enriched_map = {e.get("feature_name"): e for e in enriched_texts if isinstance(e, dict) and e.get("feature_name")}
    final_features = []
    for fw in features_with_hours:
        name = fw["feature_name"]
        enrich = enriched_map.get(name, {}) or {}
        # textuals
        fw["description"] = enrich.get("description", f"A generated description for {name}.")
        fw["acceptance_criteria"] = enrich.get("acceptance_criteria", ["End-to-end works", "No errors", "Basic validations"])
        fw["user_story"] = enrich.get("user_story", f"As a user I want {name.lower()} so that ...")
        fw["dependencies"] = enrich.get("dependencies", "")
        fw["deliverables"] = enrich.get("deliverables", ["API", "UI"])

        recommended_hours = enrich.get("recommended_hours")
        role_mix = enrich.get("role_mix")  # optional dict of percents

        chosen_total_hours = int(fw["trace"]["computed_total_hours"])
        applied_override = False
        if isinstance(recommended_hours, (int, float)) and recommended_hours > 0:
            chosen_total_hours = int(round(recommended_hours))
            fw["trace"]["adjusted_by_llm"] = True
            fw["trace"]["hours_before_llm"] = int(fw["trace"]["computed_total_hours"])
            fw["trace"]["hours_rationale"] = enrich.get("hours_rationale", "")
            fw["trace"]["computed_total_hours"] = chosen_total_hours
            applied_override = True

        # re-distribute roles if hours or role mix changed
        role_hours = distribute_hours_to_roles(chosen_total_hours, fw["feature_type"], override_weights=role_mix if isinstance(role_mix, dict) else None)
        fw["resources_map"] = {k: int(v) for k, v in role_hours.items()}
        fw["costs_map"] = compute_cost_from_role_hours(fw["resources_map"])

        # timeline handling
        timeline_plan = enrich.get("timeline_plan", [])
        # ensure integer hours and sum to chosen_total_hours (fix if needed)
        if isinstance(timeline_plan, list) and all(isinstance(p, dict) for p in timeline_plan):
            # sanitize
            plan = []
            for p in timeline_plan:
                phase = str(p.get("phase", "Phase")).strip() or "Phase"
                hrs = int(max(0, round(float(p.get("hours", 0)))))
                plan.append({"phase": phase, "hours": hrs})
            total_in_plan = sum(p["hours"] for p in plan)
            if total_in_plan != chosen_total_hours and chosen_total_hours > 0:
                # scale proportions
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
                    # default simple split
                    plan = [
                        {"phase":"Planning","hours":max(2, int(round(chosen_total_hours*0.08)))},
                        {"phase":"Design","hours":max(2, int(round(chosen_total_hours*0.14)))},
                        {"phase":"Build","hours":max(4, int(round(chosen_total_hours*0.56)))},
                        {"phase":"QA","hours":max(2, int(round(chosen_total_hours*0.18)))},
                        {"phase":"UAT","hours":max(1, chosen_total_hours - sum([
                            max(2, int(round(chosen_total_hours*0.08))),
                            max(2, int(round(chosen_total_hours*0.14))),
                            max(4, int(round(chosen_total_hours*0.56))),
                            max(2, int(round(chosen_total_hours*0.18))),
                        ]))},
                    ]
            fw["timeline_plan"] = plan
        else:
            # fallback plan
            t = chosen_total_hours
            fw["timeline_plan"] = [
                {"phase":"Planning","hours":max(2, int(round(t*0.08)))},
                {"phase":"Design","hours":max(2, int(round(t*0.14)))},
                {"phase":"Build","hours":max(4, int(round(t*0.56)))},
                {"phase":"QA","hours":max(2, int(round(t*0.18)))},
                {"phase":"UAT","hours":max(1, t - sum([
                    max(2, int(round(t*0.08))),
                    max(2, int(round(t*0.14))),
                    max(4, int(round(t*0.56))),
                    max(2, int(round(t*0.18))),
                ]))},
            ]

        fw["calendar_weeks"] = int(enrich.get("calendar_weeks", math.ceil(chosen_total_hours/32.0)))
        final_features.append(fw)

    # 7) resources summary & budget
    total_estimated_cost = sum(f["costs_map"]["total_feature_cost_usd"] for f in final_features)
    resources_summary: Dict[str, int] = {}
    for f in final_features:
        for r, h in f["resources_map"].items():
            resources_summary[r] = int(resources_summary.get(r, 0)) + int(h)

    resources_list = [{"role": r, "count": math.ceil(int(v) / 160)} for r, v in resources_summary.items()]  # ~1 FTE=160h

    # 8) Build output JSON
    features_out = []
    per_feature_budget = []
    for f in final_features:
        features_out.append(
            {
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
                    {"role": "pm", "hours": int(f["resources_map"].get("pm", 0))},
                    {"role": "qa", "hours": int(f["resources_map"].get("qa", 0))},
                ],
                "timeline": {
                    "phase": "LLM-calibrated",
                    "duration_hours": int(f["trace"]["computed_total_hours"]),
                    "calendar_weeks": int(f.get("calendar_weeks", math.ceil(int(f["trace"]["computed_total_hours"])/32.0))),
                    "plan": f.get("timeline_plan", []),
                },
                "cost_estimate": {
                    "fullstack_cost_usd": f["costs_map"].get("fullstack_cost_usd", 0.0),
                    "ai_cost_usd": f["costs_map"].get("ai_cost_usd", 0.0),
                    "ui_ux_cost_usd": f["costs_map"].get("ui_ux_cost_usd", 0.0),
                    "pm_cost_usd": f["costs_map"].get("pm_cost_usd", 0.0),
                    "qa_cost_usd": f["costs_map"].get("qa_cost_usd", 0.0),
                    "total_feature_cost_usd": f["costs_map"].get("total_feature_cost_usd", 0.0),
                },
                "trace": f["trace"],
                "warnings": f.get("warnings", []),
            }
        )
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
            "notes": "Estimates calibrated by baseline + LLM sanity-check. Timelines reflect realistic ranges for common modules.",
        },
    }

    # 9) Render
    st.success("✅ Estimation Generated (calibrated)")

    st.markdown("### 📘 Summary")
    st.write(f"Complexity score (raw): {complexity_score:.2f}")
    st.write(f"Complexity level (1-5): {complexity_level}, multiplier applied: {complexity_multiplier:.2f}")
    st.write(f"Feature count chosen: {feature_count}")
    st.write(f"Budget parsed (USD): {parsed_budget} ({budget_currency})")
    st.write(f"Total estimated cost (USD): {output_json['budget']['total_estimated_cost_usd']}")

    st.markdown("### 🏗 Features Overview")
    rows = []
    for f in output_json["features"]:
        rows.append(
            {
                "feature_name": f["feature_name"],
                "feature_type": f["trace"]["feature_type"],
                "duration_hours": int(f["timeline"]["duration_hours"]),
                "calendar_weeks": int(f["timeline"]["calendar_weeks"]),
                "fullstack_hours": int(next((r["hours"] for r in f["resources"] if r["role"] == "fullstack"), 0)),
                "ai_hours": int(next((r["hours"] for r in f["resources"] if r["role"] == "ai"), 0)),
                "ui_ux_hours": int(next((r["hours"] for r in f["resources"] if r["role"] == "ui_ux"), 0)),
                "pm_hours": int(next((r["hours"] for r in f["resources"] if r["role"] == "pm"), 0)),
                "qa_hours": int(next((r["hours"] for r in f["resources"] if r["role"] == "qa"), 0)),
                "total_feature_cost_usd": f["cost_estimate"]["total_feature_cost_usd"],
                "warnings": "; ".join(f.get("warnings", [])),
            }
        )
    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True)

    st.markdown("### 👥 Resources (headcount estimate)")
    st.dataframe(pd.DataFrame(output_json["resources"]), use_container_width=True)

    st.markdown("### 💰 Budget")
    st.write(f"Total estimated cost (USD): {output_json['budget']['total_estimated_cost_usd']}")
    st.write(f"Within provided budget? {output_json['budget']['within_budget']}")
    st.dataframe(pd.DataFrame(output_json["budget"]["per_feature"]), use_container_width=True)

    st.markdown("### 🔍 Feature detail (first 5)")
    for f in output_json["features"][:5]:
        st.markdown(f"**{f['feature_name']}** — _{f['trace']['feature_type']}_")
        st.write(f.get("description", ""))
        st.write("Acceptance criteria:")
        for a in f.get("acceptance_criteria", []):
            st.write(f"- {a}")
        st.write("Timeline plan:")
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
