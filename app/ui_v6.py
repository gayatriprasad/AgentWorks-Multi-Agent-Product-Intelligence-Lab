import streamlit as st
from app.orchestrator_v6 import SimpleOrchestrator
from app.orchestrator_v6_llm import LLMOrchestrator

st.set_page_config(page_title="V6 — Simple Multi-Agent", page_icon="🕹️", layout="centered")
st.title("V6 — Simple Multi-Agent")

query = st.text_input("Ask something:", value="Find a budget phone under 12k, compare prices, and show top reviews")
mode = st.radio("Mode", ["Rule-Based (Option 1)", "LLM-Driven (Option 2)"])

if st.button("Run"):
    if mode.startswith("Rule"):
        orch = SimpleOrchestrator()
        out = orch.run(query)
        st.subheader("Final Answer")
        st.markdown(out["final"])
        with st.expander("Trace"):
            st.json(out["trace"])
    else:
        orch = LLMOrchestrator()
        out = orch.run(query)
        st.subheader("Plan")
        st.json(out["plan"])
        st.subheader("Steps")
        st.json(out["steps"])
