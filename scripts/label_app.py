"""Streamlit UI for hand-labeling the golden set."""
import pandas as pd
import streamlit as st
from src.utils import get_paths

st.set_page_config(page_title="Golden Set Labeler", layout="wide")
st.title("AppleSupport Golden Set Labeler")

paths = get_paths()
csv_path = paths["golden"] / "golden_set.csv"

if not csv_path.exists():
    st.error("Run `python -m scripts.sample_for_labeling` first.")
    st.stop()

df = pd.read_csv(csv_path).fillna("")

INTENTS = [
    "device_issue", "account_access", "billing_refund", "software_update",
    "order_delivery", "warranty_repair", "data_privacy",
    "general_inquiry", "complaint_escalation", "out_of_scope",
]

unlabeled = df[df["labeled_intent"].astype(str).str.len() == 0]
total = len(df)
remaining = len(unlabeled)

st.markdown(f"### Progress: **{total - remaining} / {total}** labeled — {remaining} remaining")

if remaining == 0:
    st.success("All labeled. Run `python -m scripts.run_pipeline` next.")
    st.stop()

with st.sidebar:
    st.header("Options")
    jump = st.number_input("Jump to id", min_value=0, max_value=total - 1, value=int(unlabeled.index[0]))
    if st.button("Go"):
        st.session_state["jump_to"] = int(jump)

idx = int(st.session_state.get("jump_to", unlabeled.index[0]))
row = df.loc[idx]

if row["labeled_intent"]:
    idx = int(unlabeled.index[0])
    row = df.loc[idx]

st.markdown(f"#### Example id = **{idx}**")
st.info(row["customer_message"])
st.caption(f"Historical AppleSupport reply: {row['historical_reply']}")

col1, col2, col3 = st.columns(3)
with col1:
    intent = st.selectbox("Intent", INTENTS, key=f"intent_{idx}")
with col2:
    esc = st.selectbox("Should escalate?", ["False", "True"], key=f"esc_{idx}")
with col3:
    q = st.slider("Historical reply quality (1-5)", 1, 5, 3, key=f"q_{idx}")

notes = st.text_input("Notes (optional, e.g. 'fragment of multi-turn thread')", key=f"n_{idx}")

if st.button("Save & Next", type="primary"):
    df.loc[idx, "labeled_intent"] = intent
    df.loc[idx, "should_escalate"] = esc
    df.loc[idx, "reply_quality_1to5"] = q
    df.loc[idx, "notes"] = notes
    df.to_csv(csv_path, index=False)
    st.session_state.pop("jump_to", None)
    st.rerun()
