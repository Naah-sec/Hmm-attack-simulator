# ATT&CK-HMM: An Explainable Hidden Markov Simulator for Predicting Multi-Stage Attacker Progression from IDS Alerts

ATTACK_HMM_SIMULATOR is a defensive research prototype that simulates IDS alert labels and uses Hidden Markov Models to infer hidden attacker progression stages. It does not perform scanning, exploitation, credential attacks, packet attacks, malware behavior, or command execution against systems.

## Problem Statement

Security teams often receive alert streams as isolated observations, while intrusions progress through latent phases such as initial access, execution, lateral movement, collection, and exfiltration. This project models those phases explicitly and explains how alert labels support the inferred attack stage.

## Research Motivation

HMMs are a strong fit for multi-stage intrusion reasoning because they model a hidden process from noisy observations. IDS alerts are imperfect emissions of an underlying campaign state, and transition probabilities provide a principled way to reason about likely next stages.

## Main Contribution

- A manual NumPy implementation of forward inference, Viterbi decoding, sequence likelihood, and next-state prediction.
- Two transparent HMM profiles, with no baseline models:
  - Published APT-HMM profile using a six-state published-style transition matrix adapted to this alert vocabulary.
  - ATT&CK-Enriched HMM profile using an eight-state ATT&CK-inspired model.
- MITRE ATT&CK mapping for states and alerts.
- A polished Streamlit dashboard for simulation, model exploration, robustness evaluation, and exports.

## Architecture

```text
Synthetic IDS alert labels
  -> scenario simulator with noise and missing-alert controls
  -> selected HMM profile
  -> forward posterior, Viterbi path, next-state forecast
  -> MITRE ATT&CK enrichment
  -> dashboard charts, explanations, reports, Navigator layer
```

## HMM Mathematical Formulation

- Hidden states: attack phases defined by the selected profile.
- Observations: IDS alert labels from `data/alert_catalog.json`.
- Initial distribution: `pi_i = P(S_0 = i)`.
- Transition matrix: `A_ij = P(S_t = j | S_{t-1} = i)`.
- Emission matrix: `B_i(o) = P(O_t = o | S_t = i)`.
- Forward algorithm: updates `P(S_t | O_1:t)` after each alert and normalizes at every step.
- Viterbi algorithm: finds the maximum likelihood hidden-state path in log space to avoid underflow.

## MITRE ATT&CK Enrichment

`data/mitre_mapping.json` maps model states and key alert labels to ATT&CK techniques. Technique scores are derived from the current or next state distribution and exported as an ATT&CK Navigator-compatible layer.

## Dashboard Pages

- Live Attack Simulator: run scenarios or custom alert sequences and inspect predictions.
- Model Explorer: inspect initial, transition, and emission probabilities.
- MITRE ATT&CK View: browse state-to-technique and alert-to-technique mappings.
- Robustness Lab: evaluate the two HMM profiles under noise and missing alerts.
- Report Export: download JSON, Markdown, CSV, and Navigator outputs.

## Installation

Linux/macOS:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Windows:

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

## Running Robustness Experiments

Open the Robustness Lab page, select one profile or compare both HMM profiles, choose scenarios, noise rates, missing-alert rates, and seed count, then run the experiment. Results are saved to `outputs/experiment_results.csv`.

## Exported Outputs

- `outputs/prediction_report.json`
- `outputs/prediction_report.md`
- `outputs/experiment_results.csv`
- `outputs/attack_navigator_layer.json`
- `outputs/reports/session_state.json`

## Limitations

- Synthetic IDS alerts are used.
- Emission probabilities are adapted for the project alert vocabulary.
- The published transition matrix has different states than the enriched model.
- MITRE mapping is approximate and intended for defensive research explanation.
- No real IDS integration is performed.

## Future Work

- Train emissions from real alert datasets.
- Parse Suricata or Wazuh alerts.
- Mine ATT&CK Attack Flow sequences.
- Add supervised training from labeled campaigns.
- Add an analyst feedback loop.
- Add richer ATT&CK Navigator integration.

