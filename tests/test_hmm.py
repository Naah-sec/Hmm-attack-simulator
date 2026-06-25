import numpy as np

from src.model_profiles import load_attack_enriched_hmm


def test_forward_returns_probability_vectors():
    model = load_attack_enriched_hmm()
    frame = model.forward(["PORT_SCAN", "EXPLOIT_PUBLIC_APP", "DATA_STAGING"])
    assert len(frame) == 3
    assert np.allclose(frame[model.states].sum(axis=1), 1.0)


def test_viterbi_path_length():
    model = load_attack_enriched_hmm()
    result = model.viterbi(["PORT_SCAN", "EXPLOIT_PUBLIC_APP", "EXFIL_OVER_HTTP"])
    assert len(result["path"]) == 3
    assert len(result["steps"]) == 3


def test_predict_next_distribution_sums_to_one():
    model = load_attack_enriched_hmm()
    dist = model.predict_next_distribution(["PORT_SCAN"])
    assert np.isclose(sum(dist.values()), 1.0)


def test_unknown_alert_maps_to_unknown_bucket():
    model = load_attack_enriched_hmm()
    known = model.forward(["UNKNOWN_ALERT"])
    unknown = model.forward(["NOT_A_REAL_ALERT"])
    assert np.allclose(known[model.states].iloc[0], unknown[model.states].iloc[0])

