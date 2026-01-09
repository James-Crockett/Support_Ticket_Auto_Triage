import json
import requests
import os
import streamlit as st

st.set_page_config(page_title="Support Ticket Auto-Triage Demo", layout="centered")

API_BASE = os.getenv("TRIAGE_API_URL", "http://localhost:8000").rstrip("/")
PREDICT_URL = f"{API_BASE}/predict"
HEALTH_URL = f"{API_BASE}/health"

st.title("Support Ticket Auto-Triage")
st.caption("Enter a ticket subject + body. The model predicts the best support queue/team.")

# --- Sidebar: API status and settings ---
st.sidebar.header("API")
st.sidebar.write(f"Base URL: `{API_BASE}`")

api_ok = False
health_detail = None
try:
    r = requests.get(HEALTH_URL, timeout=3)
    api_ok = r.ok
    health_detail = r.json() if r.headers.get("content-type", "").startswith("application/json") else r.text
except Exception as e:
    health_detail = str(e)

if api_ok:
    st.sidebar.success("API reachable")
else:
    st.sidebar.error("API not reachable")
    st.sidebar.write("Start your FastAPI server or update `TRIAGE_API_URL`.")
st.sidebar.caption("Health response:")
st.sidebar.code(json.dumps(health_detail, indent=2) if isinstance(health_detail, dict) else str(health_detail))

# --- Main inputs ---
EXAMPLES = [
    {
        "subject": "Double charge on my credit card",
        "body": "I see two transactions for the same order. Please help me resolve this.",
    },
    {
        "subject": "Can’t reset my password",
        "body": "The password reset link says it expired. I tried multiple times but I still can’t log in.",
    },
    {
        "subject": "Return request for an order",
        "body": "I want to return my item. What’s the process and how long does it take for a refund?",
    },
    {
        "subject": "Service outage?",
        "body": "Our dashboard is down and we’re getting 500 errors. Is there an outage right now?",
    },
]

col_a, col_b = st.columns([1, 1])
with col_a:
    if st.button("Load example"):
        ex = EXAMPLES[0]
        st.session_state["subject"] = ex["subject"]
        st.session_state["body"] = ex["body"]
with col_b:
    ex_choice = st.selectbox("Example picker", options=list(range(len(EXAMPLES))), format_func=lambda i: EXAMPLES[i]["subject"])
    if st.button("Load selected"):
        ex = EXAMPLES[ex_choice]
        st.session_state["subject"] = ex["subject"]
        st.session_state["body"] = ex["body"]

subject = st.text_input("Subject", key="subject", placeholder="e.g., Double charge on my credit card")
body = st.text_area("Body", key="body", height=180, placeholder="Describe the issue…")

# --- Predict ---
st.divider()

predict = st.button("Predict queue", type="primary", disabled=not api_ok)

if predict:
    payload = {"subject": subject.strip(), "body": body.strip()}
    if not payload["subject"] and not payload["body"]:
        st.warning("Please enter a subject or body.")
    else:
        with st.spinner("Calling model…"):
            try:
                resp = requests.post(PREDICT_URL, json=payload, timeout=10)
                if resp.ok:
                    data = resp.json()
                    st.success("Prediction complete")

                    # Try common keys; fall back to showing the whole JSON.
                    predicted = (
                        data.get("predicted_queue")
                        or data.get("predicted_label")
                        or data.get("label")
                        or data.get("prediction")
                    )
                    confidence = data.get("confidence") or data.get("score") or data.get("probability")

                    if predicted:
                        st.subheader("Result")
                        st.write(f"**Predicted queue:** `{predicted}`")
                    if confidence is not None:
                        try:
                            st.write(f"**Confidence:** `{float(confidence):.3f}`")
                        except Exception:
                            st.write(f"**Confidence:** `{confidence}`")

                    st.subheader("Raw JSON")
                    st.json(data)
                else:
                    st.error(f"API error: {resp.status_code}")
                    # Show response body for debugging
                    try:
                        st.code(resp.text)
                    except Exception:
                        pass
            except requests.RequestException as e:
                st.error(f"Request failed: {e}")
