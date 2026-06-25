import json

from src.export import export_attack_navigator_layer, export_prediction_report_json, export_prediction_report_markdown


def test_json_report_exports(tmp_path):
    out = export_prediction_report_json(
        "ATT&CK-Enriched HMM",
        "Scenario",
        0.0,
        0.0,
        ["PORT_SCAN"],
        ["Reconnaissance"],
        {"Reconnaissance": 1.0},
        {"Initial Access": 1.0},
        [],
        {"executive_summary": "x", "caveats": []},
        path=tmp_path / "report.json",
    )
    assert out.exists()
    assert json.loads(out.read_text())["selected_profile"] == "ATT&CK-Enriched HMM"


def test_markdown_report_exports(tmp_path):
    out = export_prediction_report_markdown(
        "ATT&CK-Enriched HMM",
        "Scenario",
        ["PORT_SCAN"],
        ["Reconnaissance"],
        {"executive_summary": "x", "analyst_explanation": "y", "caveats": []},
        [],
        path=tmp_path / "report.md",
    )
    assert out.exists()
    assert "ATT&CK-HMM Prediction Report" in out.read_text()


def test_navigator_layer_exports(tmp_path):
    out = export_attack_navigator_layer(
        [{"technique_id": "T1041", "score": 85, "comment": "Predicted from Exfiltration phase."}],
        path=tmp_path / "layer.json",
    )
    payload = json.loads(out.read_text())
    assert payload["techniques"][0]["techniqueID"] == "T1041"

