import streamlit as st
import pandas as pd
from pages.tech import colored_progress



st.set_page_config(layout="centered")

col1, col2 = st.columns([3,1])

display_to_col = {
    "Race Score": "race_score",
    "Dynamics": "race_dynamics_score_scaled",
    "Strategy": "strategy_score_scaled",
    "Chaos": "chaos_score_scaled"
}

with col1:
    st.title("2025 F1 GP Ranking")



with col2:
    st.space(size="small")
    with st.popover(':material/filter_alt:'):
        selected_display = st.selectbox(
            "Rank by",
            list(display_to_col.keys()),
            index=0
        )
        sort_col = display_to_col[selected_display]

        sort_order = st.radio(
            "Order",
            ["descending", "ascending"],
            horizontal=True
        )

ascending = sort_order == "ascending"



df = pd.read_csv("data/processed/2025/all_data.csv")


def render_gp_card(row):
    with st.container(border=True, height ="content"):

        # --- Header ---
        st.image(row.circuit_image, width="stretch", )
        st.markdown(f":{row.flag_emoji}: **{row.meeting_name_short}**", text_alignment='center')

        # --- HERO SCORE ---
        st.markdown(f"""<div style="font-size:36px; font-weight:700; text-align:center;">
                    {row.race_score:.2f}
                </div>""",
            unsafe_allow_html=True)
        
        st.markdown(f"""<div style="opacity:0.6; letter-spacing:1px; text-align:center;">
                    RACE SCORE
                </div>""",
            unsafe_allow_html=True)

        # --- Supporting metrics ---
        colored_progress(row.race_dynamics_score_scaled, "Race Dynamics", "#C0392B")
        colored_progress(row.strategy_score_scaled, "Strategy", "#D4A017") 
        colored_progress(row.chaos_score_scaled, "Chaos", "#1E5631") 
        #colored_progress(row.championship_impact_score_scaled, "Championship", "#2C5F7C")

        st.text("")

        st.page_link("pages/detailed.py", label="More details ▶︎", width="content", query_params={"session_key": row.session_key})


df_sorted = df.sort_values(
    by=sort_col,
    ascending=ascending
)

cols = st.columns(3)

for i, row in df_sorted.iterrows():
    with cols[i % 3]:
        render_gp_card(row)