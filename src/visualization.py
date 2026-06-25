"""Plotly and NetworkX visualizations for the Streamlit dashboard."""

from __future__ import annotations

from collections import Counter
from typing import Any

import networkx as nx
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


STATE_COLORS = {
    "Reconnaissance": "#38BDF8",
    "Initial Access": "#FB923C",
    "Execution": "#A78BFA",
    "Privilege Escalation": "#F43F5E",
    "Lateral Movement": "#EF4444",
    "Collection": "#34D399",
    "Exfiltration": "#FACC15",
    "Noise": "#94A3B8",
    "S1_Point_of_Entry": "#FB923C",
    "S2_C2_Communications": "#818CF8",
    "S3_Lateral_Movement": "#EF4444",
    "S4_Asset_Data_Discovery": "#34D399",
    "S5_Data_Exfiltration": "#FACC15",
    "S6_Non_Complete": "#94A3B8",
}


def _layout(fig: go.Figure, title: str) -> go.Figure:
    fig.update_layout(
        title=title,
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15,23,42,0.35)",
        font=dict(color="#E5E7EB"),
        margin=dict(l=30, r=30, t=70, b=35),
    )
    return fig


def alert_timeline(alerts: list[str], predicted_path: list[str], true_states: list[str] | None = None, confidences: list[float] | None = None, alert_techniques: dict[str, str] | None = None) -> go.Figure:
    df = pd.DataFrame(
        {
            "index": list(range(len(alerts))),
            "alert": alerts,
            "predicted_state": predicted_path,
            "true_state": true_states or ["N/A"] * len(alerts),
            "confidence": confidences or [0.0] * len(alerts),
            "technique": [alert_techniques.get(alert, "None") if alert_techniques else "None" for alert in alerts],
        }
    )
    fig = px.scatter(
        df,
        x="index",
        y="predicted_state",
        color="predicted_state",
        color_discrete_map=STATE_COLORS,
        size=[18] * len(df),
        hover_data=["alert", "true_state", "confidence", "technique"],
    )
    fig.update_traces(marker=dict(line=dict(width=1, color="#E5E7EB")))
    fig.update_xaxes(title="Alert index", dtick=1)
    fig.update_yaxes(title="Predicted phase")
    return _layout(fig, "Observed Alert Timeline with Inferred Hidden States")


def probability_evolution_chart(evolution: pd.DataFrame, states: list[str]) -> go.Figure:
    long = evolution.melt(id_vars=["t", "alert"], value_vars=states, var_name="state", value_name="probability")
    fig = px.line(long, x="t", y="probability", color="state", color_discrete_map=STATE_COLORS, hover_data=["alert"])
    fig.update_yaxes(tickformat=".0%", range=[0, 1])
    fig.update_xaxes(title="Alert index", dtick=1)
    return _layout(fig, "State Probability Evolution")


def next_state_bar_chart(next_distribution: dict[str, float], top_n: int = 5) -> go.Figure:
    items = sorted(next_distribution.items(), key=lambda item: item[1], reverse=True)[:top_n]
    df = pd.DataFrame(items, columns=["state", "probability"])
    fig = px.bar(df, x="probability", y="state", orientation="h", color="state", color_discrete_map=STATE_COLORS, text=df["probability"].map(lambda v: f"{v:.0%}"))
    fig.update_xaxes(tickformat=".0%", range=[0, max(0.05, df["probability"].max() * 1.15)])
    fig.update_layout(showlegend=False)
    return _layout(fig, "Predicted Next-State Distribution")


def transition_matrix_heatmap(model) -> go.Figure:
    fig = px.imshow(
        model.transition_matrix,
        x=model.states,
        y=model.states,
        color_continuous_scale="Blues",
        text_auto=".2f",
        labels=dict(x="Next state", y="Current state", color="P"),
    )
    return _layout(fig, "Transition Matrix P(next state | current state)")


def emission_matrix_heatmap(model, selected_state: str | None = None) -> go.Figure:
    states = [selected_state] if selected_state else model.states
    indexes = [model.state_to_index[state] for state in states]
    matrix = model.emission_matrix[indexes, :]
    fig = px.imshow(
        matrix,
        x=model.observations,
        y=states,
        color_continuous_scale="Viridis",
        labels=dict(x="Alert", y="State", color="P"),
    )
    fig.update_xaxes(tickangle=45)
    return _layout(fig, "Emission Matrix P(alert | state)")


def attack_path_graph(model, predicted_path: list[str], next_state: str | None = None) -> go.Figure:
    graph = nx.DiGraph()
    counts = Counter(predicted_path)
    for state, count in counts.items():
        graph.add_node(state, count=count)
    for left, right in zip(predicted_path, predicted_path[1:]):
        graph.add_edge(left, right, weight=model.get_transition_probability(left, right))
    if next_state:
        graph.add_node(next_state, count=counts.get(next_state, 0) + 1, predicted=True)
        graph.add_edge(predicted_path[-1], next_state, weight=model.get_transition_probability(predicted_path[-1], next_state), predicted=True)
    pos = nx.spring_layout(graph, seed=7, k=0.8)
    edge_x, edge_y = [], []
    annotations = []
    for u, v, data in graph.edges(data=True):
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
        annotations.append(
            dict(x=(x0 + x1) / 2, y=(y0 + y1) / 2, text=f"{data['weight']:.2f}", showarrow=False, font=dict(size=11, color="#CBD5E1"))
        )
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=edge_x, y=edge_y, mode="lines", line=dict(width=2, color="#64748B"), hoverinfo="skip"))
    fig.add_trace(
        go.Scatter(
            x=[pos[n][0] for n in graph.nodes],
            y=[pos[n][1] for n in graph.nodes],
            text=list(graph.nodes),
            mode="markers+text",
            textposition="top center",
            marker=dict(
                size=[28 + 6 * graph.nodes[n].get("count", 1) for n in graph.nodes],
                color=[STATE_COLORS.get(n, "#38BDF8") for n in graph.nodes],
                line=dict(width=2, color="#E5E7EB"),
            ),
            hovertext=[f"{n}<br>Visits: {graph.nodes[n].get('count', 1)}" for n in graph.nodes],
            hoverinfo="text",
        )
    )
    fig.update_layout(annotations=annotations, xaxis=dict(visible=False), yaxis=dict(visible=False), showlegend=False)
    return _layout(fig, "Inferred Attack Path Graph")


def mitre_ranking_chart(techniques: list[dict[str, Any]], top_n: int = 10) -> go.Figure:
    df = pd.DataFrame(techniques[:top_n])
    if df.empty:
        df = pd.DataFrame([{"technique_id": "None", "technique_name": "No direct mapping", "score": 0, "source_state": "N/A"}])
    df["label"] = df["technique_id"] + " " + df["technique_name"]
    fig = px.bar(df, x="score", y="label", orientation="h", color="source_state", color_discrete_map=STATE_COLORS, hover_data=["confidence", "comment"] if "confidence" in df else None)
    fig.update_xaxes(range=[0, max(10, df["score"].max() * 1.15)], title="Technique score")
    return _layout(fig, "Top Predicted MITRE ATT&CK Techniques")


def confidence_gauge(confidence: float, title: str = "Current State Confidence") -> go.Figure:
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=confidence * 100,
            number={"suffix": "%"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#38BDF8"},
                "steps": [
                    {"range": [0, 40], "color": "#7F1D1D"},
                    {"range": [40, 70], "color": "#78350F"},
                    {"range": [70, 100], "color": "#064E3B"},
                ],
            },
        )
    )
    return _layout(fig, title)


def robustness_accuracy_chart(results: pd.DataFrame, x: str) -> go.Figure:
    grouped = results.groupby(["profile", x], as_index=False)["phase_accuracy"].mean()
    fig = px.line(grouped, x=x, y="phase_accuracy", color="profile", markers=True)
    fig.update_yaxes(tickformat=".0%", range=[0, 1])
    return _layout(fig, f"Phase Accuracy vs {x.replace('_', ' ').title()}")


def robustness_heatmap(results: pd.DataFrame) -> go.Figure:
    pivot = results.pivot_table(index="noise_rate", columns="missing_rate", values="phase_accuracy", aggfunc="mean")
    fig = px.imshow(pivot, text_auto=".0%", color_continuous_scale="RdYlGn", zmin=0, zmax=1, labels=dict(color="Accuracy"))
    return _layout(fig, "Mean Phase Accuracy by Noise and Missing-Alert Rate")

