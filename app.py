import streamlit as st
import pandas as pd

# ----------------------------
# Retro CSS
# ----------------------------

st.markdown("""
<style>
    .stApp {
        background-color: #F4F1EC;
    }

    section[data-testid="stVerticalBlock"] > div:has(> div[data-testid="stContainer"]) {
        background-color: #1e1e1e;
        padding: 1rem;
        border-radius: 12px;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)


pages = [
        st.Page("pages/main.py", title="Main", icon="🏎️"),
        st.Page("pages/detailed.py", title="Detailed", icon="🏎️"),

]

pg = st.navigation(pages, position="hidden")
pg.run()