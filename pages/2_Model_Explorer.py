from __future__ import annotations

import pandas as pd
import streamlit as st

from src.model_profiles import list_available_profiles, load_profile
from src.utils import distribution_frame, inject_css, panel
from src.visualization import emission_matrix_heatmap, transition_matrix_heatmap


st.set_page_config(page_title="Model Explorer", layout="wide")
inject_css()
st.title("Model Explorer")

profile = st.sidebar.selectbox("Profile", list_available_profiles())
model = load_profile(profile)
state = st.sidebar.selectbox("State", model.states)
alert = st.sidebar.selectbox("Alert", model.observations)

st.caption(model.description)

a, b = st.columns([1, 2])
with a:
    st.subheader("Initial Distribution")
    st.dataframe(distribution_frame(dict(zip(model.states, model.initial_probs))), hide_index=True, use_container_width=True)
with b:
    st.plotly_chart(transition_matrix_heatmap(model), use_container_width=True)

st.subheader("Emission Explorer")
st.plotly_chart(emission_matrix_heatmap(model, state), use_container_width=True)

left, right, third = st.columns(3)
with left:
    outgoing = pd.DataFrame(
        {"next_state": model.states, "transition_probability": model.transition_matrix[model.state_to_index[state]]}
    ).sort_values("transition_probability", ascending=False)
    st.markdown(f"#### Top outgoing transitions from `{state}`")
    st.dataframe(outgoing, hide_index=True, use_container_width=True)
with right:
    emissions = pd.DataFrame(
        {"alert": model.observations, "emission_probability": model.emission_matrix[model.state_to_index[state]]}
    ).sort_values("emission_probability", ascending=False)
    st.markdown(f"#### Top emitted alerts from `{state}`")
    st.dataframe(emissions.head(12), hide_index=True, use_container_width=True)
with third:
    emitters = pd.DataFrame(
        {"state": model.states, "emission_probability": model.emission_matrix[:, model.observation_to_index[alert]]}
    ).sort_values("emission_probability", ascending=False)
    st.markdown(f"#### States most compatible with `{alert}`")
    st.dataframe(emitters, hide_index=True, use_container_width=True)

panel(
    "Forward inference updates the current state distribution after each alert by multiplying the previous posterior by "
    "the transition matrix and then by the alert emission likelihood. Viterbi uses the same transition and emission "
    "probabilities in log space to recover the most likely hidden path across the whole sequence."
)

