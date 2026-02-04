import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from pages.tech import plot_race_timeline, plot_score, plot_bump_chart, plot_stint_strategy_sankey, colored_metric_by_quantile, plot_first_pit, plot_strategy_outcome

st.page_link("pages/main.py", label="◀︎ Back", width="content")
tab1, tab2, tab3 = st.tabs(["Race Dynamics", "Strategy", "Chaos"])

def colored_divider(color="#cccccc", height=2, margin="16px 0"):
    st.markdown(
        f"""
        <hr style="
            border: none;
            height: {height}px;
            background-color: {color};
            margin: {margin};
        ">
        """,
        unsafe_allow_html=True
    )

params = st.query_params
session_key =  int(params.get("session_key"))
df = pd.read_csv("data/processed/2025/all_data.csv")
row = df.loc[df["session_key"] == session_key].iloc[0]



with tab1:


    ########################### Race Dynamics Score
    st.header(f":{row.flag_emoji}: **{row.meeting_name_short}**: Race Dynamics Score", anchor="RDS")
    plot_score(df, "race_dynamics_score", session_key, "Race Dynamics Score", "#C0392B")
    st.text("Metric shows how often positions change during the race:")
    plot_bump_chart(row)

    colored_divider("#C0392B",4) 


with tab2:

    ########################### Strategy
    st.header(f":{row.flag_emoji}: **{row.meeting_name_short}**: Strategy Score", anchor="Strategy")
    plot_score(df, "strategy_score", session_key, "Strategy Score", "#D4A017" )
    st.text("Metric shows how different strategies were used (and how evenly) and did strategy really affect the final result:")



    #plot_score(df, "strategy_entropy_normalized", session_key, "strategy_entropy_normalized", "#D4A017" )
    with st.expander(f"Strategy Entropy: {colored_metric_by_quantile(df, "strategy_entropy_normalized", session_key)}", width="stretch"):
        plot_stint_strategy_sankey(row)

    #plot_score(df, "outcome_sens", session_key, "outcome_sens", "#D4A017" )
    with st.expander(f"Strategy Outcome Sensitivity: {colored_metric_by_quantile(df, "outcome_sens", session_key)}", width="stretch"):
        plot_strategy_outcome(row)

    #plot_score(df, "first_pit_spread_normalized", session_key, "first_pit_spread_normalized", "#D4A017" )
    with st.expander(f"First Pit Spread: {colored_metric_by_quantile(df, "first_pit_spread_normalized", session_key)}", width="stretch"):
        plot_first_pit(row)

    colored_divider("#D4A017", 4) 


with tab3:

    ########################### Chaos
    st.header(f":{row.flag_emoji}: **{row.meeting_name_short}**: Chaos Score", anchor="Chaos")
    plot_score(df, "chaos_score", session_key, "Chaos Score", "#1E5631" )
    plot_race_timeline(row)

    colored_divider("#1E5631", 4) 


