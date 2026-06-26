from src.config_loader import alert_vocabulary
from src.model_profiles import ENRICHED_PROFILE, PUBLISHED_PROFILE
from src.simulator import get_scenario_names, inject_noise, load_scenarios, remove_missing_alerts, simulate_scenario


def test_scenario_loading_works():
    assert "Web Intrusion to Exfiltration" in get_scenario_names()


def test_all_scenarios_have_valid_aligned_alerts():
    valid_alerts = set(alert_vocabulary())
    valid_states = {
        "Reconnaissance",
        "Initial Access",
        "Execution",
        "Privilege Escalation",
        "Lateral Movement",
        "Collection",
        "Exfiltration",
        "Noise",
    }
    for name, scenario in load_scenarios().items():
        assert len(scenario["alerts"]) == len(scenario["true_states"]), name
        assert set(scenario["alerts"]).issubset(valid_alerts), name
        assert set(scenario["true_states"]).issubset(valid_states), name


def test_noise_injection_preserves_alignment():
    alerts = ["PORT_SCAN", "EXPLOIT_PUBLIC_APP"]
    states = ["Reconnaissance", "Initial Access"]
    new_alerts, new_states = inject_noise(alerts, states, 0.5, 1)
    assert len(new_alerts) == len(new_states)
    assert len(new_alerts) >= len(alerts)


def test_missing_alert_removal_preserves_alignment():
    alerts, states = remove_missing_alerts(["A", "B", "C"], ["X", "Y", "Z"], 0.5, 2)
    assert len(alerts) == len(states)
    assert len(alerts) >= 1


def test_simulate_scenario_maps_published_states():
    sim = simulate_scenario("Web Intrusion to Exfiltration", PUBLISHED_PROFILE)
    assert all(state.startswith("S") for state in sim["true_states"])
    sim2 = simulate_scenario("Web Intrusion to Exfiltration", ENRICHED_PROFILE)
    assert not all(state.startswith("S") for state in sim2["true_states"])
