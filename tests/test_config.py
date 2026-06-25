import numpy as np

from src.config_loader import load_mitre_mapping
from src.model_profiles import load_attack_enriched_hmm, load_published_apt_hmm


def test_profiles_load_and_validate():
    for model in [load_attack_enriched_hmm(), load_published_apt_hmm()]:
        assert len(model.states) > 0
        assert len(model.observations) == 28
        assert model.transition_matrix.shape == (len(model.states), len(model.states))
        assert model.emission_matrix.shape == (len(model.states), len(model.observations))
        assert np.allclose(model.transition_matrix.sum(axis=1), 1.0)
        assert np.allclose(model.emission_matrix.sum(axis=1), 1.0)


def test_mitre_mapping_exists_for_non_noise_states():
    mapping = load_mitre_mapping()["profiles"]
    for profile, states in mapping.items():
        for state, techniques in states.items():
            if "Noise" in state or "Non_Complete" in state:
                continue
            assert techniques, f"{profile} {state} must have mapped techniques"

