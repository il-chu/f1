import plotly.graph_objects as go
import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np

def plot_score(df, column_name, session_key, metric_name, color):
    # Извлекаем значения метрики и сортируем по ней
    scores = df[column_name].round(2)
    y = [0] * len(scores)

    highlight_idx = df.index[df["session_key"] == session_key][0]

    df_sorted = df.sort_values(column_name).reset_index(drop=True)
    current_idx = df_sorted.index[df_sorted["session_key"] == session_key].tolist()[0]
    prev_idx = df_sorted.iloc[max(0, current_idx - 1)]
    next_idx = df_sorted.iloc[min(len(df_sorted) - 1, current_idx + 1)]

    # Индексы максимальной и минимальной точки
    max_idx = df[column_name].idxmax()
    min_idx = df[column_name].idxmin()

    # Подписи для точек: только макс и мин
    texts = [""] * len(df)
    texts[max_idx] = f"{df.loc[max_idx, 'meeting_name_short']}: {scores[max_idx]}"
    texts[min_idx] = f"{df.loc[min_idx, 'meeting_name_short']}: {scores[min_idx]}"

    # Вычисляем расширенный диапазон
    x_min = scores.min()
    x_max = scores.max()
    x_range_padding = (x_max - x_min) * 0.3  # 20% с каждой стороны
    x_range = [x_min - x_range_padding, x_max + x_range_padding]

    fig = go.Figure()

    # Все точки (серые) + подписи мин/макс
    fig.add_trace(go.Scatter(
        x=scores,
        y=[0]*len(scores),
        mode="markers+text",
        marker=dict(size=10, color="#B0B0B0", opacity=0.7),
        text=texts,
        textposition="top center",
        hovertemplate="%{customdata[0]}: %{x}<extra></extra>",
        customdata=list(zip(
            df["meeting_name_short"],
            df["session_key"]
        )),
        showlegend=False
    ))

    # Подсвеченная точка
    fig.add_trace(go.Scatter(
        x=[scores.loc[highlight_idx]],
        y=[0],
        mode="markers",
        marker=dict(size=14, color=color),
        hovertemplate="<b>Selected</b><br>%{customdata[0]}: %{x}<extra></extra>",
        customdata=[(
            df.loc[highlight_idx, "meeting_name_short"],
            df.loc[highlight_idx, "session_key"]
        )],
        showlegend=False
    ))

    # Линия-ось
    fig.add_shape(
        type="line",
        x0=x_range[0],
        x1=x_range[1],
        y0=0,
        y1=0,
        line=dict(color="#888", width=2)
    )

    fig.update_layout(
        height=76,
        margin=dict(l=20, r=20, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )

    # Расширяем ось X
    fig.update_xaxes(range=x_range, showgrid=False, zeroline=False, ticks="outside")
    fig.update_yaxes(visible=False)

    # Streamlit layout
    col1, col2 = st.columns([1, 3])
    with col1:
        st.metric(metric_name, f"{df.loc[highlight_idx, column_name]:.2f}", border=True)
    with col2:
        with st.container(border=True):
            col3, col4, col5 = st.columns([1, 5, 1], vertical_alignment="center")
            with col3:
                st.page_link("pages/detailed.py", label=":material/arrow_back_ios:", icon=":material/arrow_back_ios:", icon_position="left", width="stretch", query_params={"session_key": prev_idx["session_key"]})

            with col4:
                st.plotly_chart(fig, use_container_width=True)

            with col5:
                st.page_link("pages/detailed.py", label=":material/arrow_forward_ios:", icon=":material/arrow_forward_ios:", icon_position="right", width="stretch", query_params={"session_key": next_idx["session_key"]})


def plot_bump_chart(row):
    laps_df = pd.read_csv(f"data/raw/2025/{row.country_code} - {row.circuit_short_name}/laps_v2.csv")
    drivers = pd.read_csv(f"data/raw/2025/{row.country_code} - {row.circuit_short_name}/drivers.csv")
    laps = laps_df.merge(drivers, left_on ="DriverNumber", right_on="driver_number", how="left")

    df_sorted = laps.sort_values(["driver_number", "LapNumber"])
    fig = go.Figure()
    for driver, d in df_sorted.groupby("driver_number"):
        color = "#"+d["team_colour"].iloc[0]
        name = d["name_acronym"].iloc[0]

        fig.add_trace(go.Scatter(
            x=d["LapNumber"],
            y=d["Position"],
            mode="lines",
            name=name,                     # 👈 подпись в легенде
            line=dict(width=3, color=color),
            marker=dict(size=6, color=color),
            hovertemplate=(
                f"{name}<br>"
                "Lap: %{x}<br>"
                "Position: %{y}<extra></extra>"
            )
        ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=20, r=20, t=40, b=20)
    )


    fig.update_layout(
        #title="Bump chart — позиции по кругам",
        xaxis_title="Lap",
        yaxis_title="Position",
        yaxis=dict(
            autorange="reversed",
            tickmode="linear",
            dtick=1
        ),
        template="plotly_white",
        legend_title_text="Driver"
    )

    with st.container(border=True):
        st.plotly_chart(fig, width='stretch')


def plot_stint_strategy_sankey(row):

    stint_df = pd.read_csv(f"data/raw/2025/{row.country_code} - {row.circuit_short_name}/stints.csv")
    drivers = pd.read_csv(f"data/raw/2025/{row.country_code} - {row.circuit_short_name}/drivers.csv")
    stint_df = stint_df.merge(drivers, on="driver_number", how="inner")
    results = pd.read_csv(f"data/raw/2025/{row.country_code} - {row.circuit_short_name}/session_result.csv")
    race_laps = results["number_of_laps"].max()
    valid_drivers = results[results["number_of_laps"] >= 0.75 * race_laps]["driver_number"]
    stint_df = stint_df[stint_df["driver_number"].isin(valid_drivers)].copy()

    compound_colors = {
        "SOFT": "#C0392B",
        "MEDIUM": "#D4A017",
        "HARD": "#d9d9d9",
        "INTERMEDIATE": "#1E5631",
        "WET": "#2C5F7C"
    }



    def label_to_color(label, compound_colors):
        compound = label.split("-")[-1]
        return compound_colors.get(compound, "#999999")

    df = stint_df.copy()
    # 1. узлы
    df["node"] = "S" + df["stint_number"].astype(str) + "-" + df["compound"]

    labels = list(dict.fromkeys(df["node"]))
    label_to_id = {l: i for i, l in enumerate(labels)}
    # 2. переходы
    df = df.sort_values(["name_acronym", "stint_number"])
    node_colors = [
        label_to_color(label, compound_colors)
        for label in labels
    ]

    flows = (
        df.groupby("name_acronym")
        .apply(lambda x: list(zip(x["node"][:-1], x["node"][1:])))
        .explode()
        .dropna()
        .value_counts()
        .reset_index(name="value")
    )

    flows[["source_label", "target_label"]] = pd.DataFrame(
        flows["index"].tolist(), index=flows.index
    )

    # ---------- HOVER: полный путь шин ----------
    # 1. путь шин для каждого пилота
    driver_paths = (
        df.groupby("name_acronym")["compound"]
        .apply(lambda x: " → ".join(x))
        .to_dict()
    )

    # 2. какие пилоты проходили через ноду
    node_drivers = (
        df.groupby("node")["name_acronym"]
        .unique()
        .to_dict()
    )

    # 3. hover-текст для нод
    node_hover = {}
    for node, drivers in node_drivers.items():
        node_hover[node] = "<br>".join(
            [f"{d}: {driver_paths[d]}" for d in drivers]
        )

    customdata = [node_hover.get(label, "") for label in labels]

    # 3. индексы
    source = flows["source_label"].map(label_to_id).tolist()
    target = flows["target_label"].map(label_to_id).tolist()
    value  = flows["value"].tolist()
    def hex_to_rgba(hex_color, alpha=0.5):
        hex_color = hex_color.lstrip("#")
        r, g, b = [int(hex_color[i:i+2], 16) for i in (0, 2, 4)]
        return f"rgba({r},{g},{b},{alpha})"

    link_colors = [
        hex_to_rgba(
            compound_colors.get(labels[s].split('-')[-1], "#999999"),
            alpha=0.5
        )
        for s in source
    ]

    fig = go.Figure(go.Sankey(
        node=dict(
            label=labels,
            color=node_colors,
            pad=15,
            thickness=20,
            customdata=customdata,
            hovertemplate=(
                "<b>%{label}</b><br><br>"
                "%{customdata}"
                "<extra></extra>"
            )
        ),
        link=dict(
            source=source,
            target=target,
            value=value,
            color=link_colors
        )
    ))

    fig.update_layout(
        font=dict(
            size=14,
            family="Arial",
            color="#999999"
        ),
        paper_bgcolor="rgba(0,0,0,0)",   # фон всей фигуры
        plot_bgcolor="rgba(0,0,0,0)"

    )

    with st.container(border=True):
        st.plotly_chart(fig, width='stretch')



def colored_metric_by_quantile(
    df,
    metric_name: str,
    session_key,
    precision: int = 2
) -> str:


    if metric_name not in df.columns:
        raise ValueError(f"Metric '{metric_name}' not found in DataFrame")

    metric_series = df[metric_name].dropna()

    if metric_series.empty:
        return ":gray[NA]"

    q25 = metric_series.quantile(0.25)
    q75 = metric_series.quantile(0.75)

    row = df[df["session_key"] == session_key]

    if row.empty:
        return ":gray[NA]"

    value = row[metric_name].iloc[0]

    if pd.isna(value):
        return ":gray[NA]"

    value_str = f"{value:.{precision}f}"

    if value < q25:
        return f":red[{value_str}]"
    elif value <= q75:
        return f":orange[{value_str}]"
    else:
        return f":green[{value_str}]"


def plot_first_pit(row):

    pit_df = pd.read_csv(f"data/raw/2025/{row.country_code} - {row.circuit_short_name}/pit.csv")
    drivers = pd.read_csv(f"data/raw/2025/{row.country_code} - {row.circuit_short_name}/drivers.csv")
    pit_df = pit_df.merge(drivers, on="driver_number", how="inner")
    session_result = pd.read_csv(f"data/raw/2025/{row.country_code} - {row.circuit_short_name}/session_result.csv")
    race_laps = session_result["number_of_laps"].max()

    first_pits = (
        pit_df
        .sort_values(["name_acronym", "lap_number"])
        .groupby("name_acronym", as_index=False)
        .first()
    )

    fig = px.box(
        first_pits,
        x="lap_number",
        points="all",
        hover_data=["name_acronym"]
    )

    fig.update_layout(
        xaxis_title="First Pit Lap",
        boxmode="group",
    )
    fig.update_xaxes(
        range=[0, race_laps],
        showgrid=True
    )
    fig.update_traces(
        customdata=first_pits[["name_acronym"]],
        hovertemplate=(
            "%{customdata[0]}: Lap %{x}<br><extra></extra>"
        )
    )

    st.plotly_chart(fig, use_container_width=True)


def plot_strategy_outcome(row):

    stint_df = pd.read_csv(f"data/raw/2025/{row.country_code} - {row.circuit_short_name}/stints.csv")
    drivers = pd.read_csv(f"data/raw/2025/{row.country_code} - {row.circuit_short_name}/drivers.csv")
    stint_df = stint_df.merge(drivers, on="driver_number", how="inner")
    results = pd.read_csv(f"data/raw/2025/{row.country_code} - {row.circuit_short_name}/session_result.csv")
    race_laps = results["number_of_laps"].max()
    valid_drivers = results[results["number_of_laps"] >= 0.75 * race_laps]["driver_number"]
    stint_df = stint_df[stint_df["driver_number"].isin(valid_drivers)].copy()

    stint_df["node"] = "S" + stint_df["stint_number"].astype(str) + "-" + stint_df["compound"]

    flows = (
        stint_df.groupby(["driver_number", "name_acronym"])["node"]
        .apply(list)
        .reset_index()
        .rename(columns={"node": "Strategy"})
    )

    flows = flows.merge(results, on="driver_number", how="inner")
    N = len(flows)
    flows["position"] = flows["position"].fillna(N + 1)
    flows["Strategy"] = flows["Strategy"].apply(tuple)


    mu = flows["position"].mean()
    between_var = 0.0
    for strat, group in flows.groupby("Strategy"):
        n_s = len(group)
        mu_s = group["position"].mean()
        between_var += n_s * (mu_s - mu) ** 2

    between_var /= N

    # Общая дисперсия
    total_var = ((flows["position"] - mu) ** 2).sum() / N

    flows = (
        flows
        .groupby("Strategy")
        .agg(
            avg_position=("position", "mean"),
            Drivers=("name_acronym", list)
        )
        .reset_index()
        .sort_values("avg_position")
        .rename(columns={"avg_position": "Average Position"})
    )
    # st.text(f"{between_var}")
    # st.text(f"{total_var}")
    # st.text(f"{between_var / total_var}")
    
    st.dataframe(flows[["Average Position","Strategy","Drivers"]], hide_index=True)


def plot_race_timeline(row):

    results = pd.read_csv(f"data/raw/2025/{row.country_code} - {row.circuit_short_name}/session_result.csv")
    drivers = pd.read_csv(f"data/raw/2025/{row.country_code} - {row.circuit_short_name}/drivers.csv")
    results = results.merge(drivers, on="driver_number", how="inner")
    race_laps = results["number_of_laps"].max()
    dnf_df = results[results[["dnf", "dns", "dsq"]].any(axis=1)]
    dnf_df["message"] = np.select(
        [
            dnf_df["dnf"],
            dnf_df["dns"],
            dnf_df["dsq"]
        ],
        [
            "CAR " + dnf_df["driver_number"].astype(str)
            + " (" + dnf_df["name_acronym"] + ") - DNF",

            "CAR " + dnf_df["driver_number"].astype(str)
            + " (" + dnf_df["name_acronym"] + ") - DNS",

            "CAR " + dnf_df["driver_number"].astype(str)
            + " (" + dnf_df["name_acronym"] + ") - DSQ",
        ],
        default=None
    )
    dnf_df["lap_number"] = np.select(
        [
            dnf_df["dnf"],
            dnf_df["dns"],
            dnf_df["dsq"]
        ],
        [   
            dnf_df["number_of_laps"], 
            0,
            race_laps
        ]
    )
    jitter = 0.03
    dnf_df["lap_number_jitter"] = dnf_df["lap_number"] + np.random.uniform(-jitter, jitter, len(dnf_df))
    x_range = [-3, race_laps+5]
    race_control = pd.read_csv(f"data/raw/2025/{row.country_code} - {row.circuit_short_name}/race_control.csv")
    sc_mask = race_control["message"].str.contains(
            "SAFETY CAR DEPLOYED|VIRTUAL SAFETY CAR DEPLOYED|RACE WILL START BEHIND THE SAFETY CAR",
            case=False,
            na=False
        )
    df_sc = race_control[sc_mask].copy()

    investigation_mask = race_control["message"].str.contains(
            "UNDER INVESTIGATION|NOTED",
            case=False,
            na=False
        )
    i_df = race_control[investigation_mask].copy()


    # Создание графика
    fig = go.Figure()

    #Safety car
    fig.add_trace(go.Scatter(
        x=df_sc["lap_number"],
        y=[0]*len(df_sc["lap_number"]),
        mode="markers+text",
        marker=dict(size=10, color="#1E5631", opacity=0.7),
        text=['🚔']*len(df_sc["lap_number"]),
        textposition="top center",
        showlegend=False,
        customdata=df_sc['message'],
        hovertemplate=(
            "<b>Lap %{x}</b><br><br>"
            "%{customdata}"
            "<extra></extra>"
            )
    ))

    #Investigation
    fig.add_trace(go.Scatter(
        x=i_df["lap_number"],
        y=[0]*len(i_df["lap_number"]),
        mode="markers+text",
        marker=dict(size=10, 
                    symbol="diamond",
                    color="#D4A017", 
                    opacity=0.7),
        text=['🔎']*len(i_df["lap_number"]),
        textposition="bottom center",
        showlegend=False,
        customdata=i_df['message'],
        hovertemplate=(
            "<b>Lap %{x}</b><br><br>"
            "%{customdata}"
            "<extra></extra>"
            )
    ))

    #DNS DNF DSQ
    fig.add_trace(go.Scatter(
        x=dnf_df["lap_number_jitter"],
        y=[0]*len(dnf_df["lap_number"]),
        mode="markers+text",
        marker=dict(size=10, 
                    symbol="x",
                    color="#C0392B", 
                    opacity=0.7),
        text=['']*len(dnf_df["lap_number"]),
        textposition="bottom center",
        showlegend=False,
        customdata=dnf_df['message'],
        hovertemplate=(
            "<b>Lap %{x:.0f}</b><br><br>"
            "%{customdata}"
            "<extra></extra>"
            )
    ))

    # Линия-ось
    fig.add_shape(
        type="line",
        x0=x_range[0],
        x1=x_range[1],
        y0=0,
        y1=0,
        line=dict(color="#888", width=2)
    )

    fig.update_layout(
        height=150,
        margin=dict(l=20, r=20, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )

    # Расширяем ось X
    fig.update_xaxes(range=x_range, showgrid=True, zeroline=False, ticks="outside", tickformat=".")
    fig.update_yaxes(visible=False)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("❌ DNF/DNS/DSQ", len(dnf_df), border=True)
    with col2:
        st.metric("🚔 Safety Car", len(df_sc), border=True)
    with col3:
        st.metric("🔎 Investigations", len(i_df), border=True)
    st.plotly_chart(fig, use_container_width=True)
 

def colored_progress(value, label="", color="#007bff"):
    """
    value: float 0-1
    label: подпись над прогресс-баром
    color: HEX или rgb()
    """
    width_percent = int(value * 100)
    st.markdown(f"""
        <div style="margin:4px 0; font-size:12px; opacity:0.7;">{label}</div>
        <div style="
            background-color:#ddd;
            border-radius:6px;
            height:12px;
            width:100%;
        ">
            <div style="
                width:{width_percent}%;
                background-color:{color};
                height:100%;
                border-radius:6px;
                text-align:right;
            "></div>
        </div>
    """, unsafe_allow_html=True)




