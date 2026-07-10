from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


def load_workflow() -> dict:
    # BaseLoader keeps GitHub's top-level `on` key as text instead of applying
    # YAML 1.1's legacy boolean coercion.
    return yaml.load(
        (ROOT / ".github" / "workflows" / "update.yml").read_text(encoding="utf-8"),
        Loader=yaml.BaseLoader,
    )


def test_pages_workflow_routes_push_without_network_update():
    workflow = load_workflow()
    assert {"push", "schedule", "workflow_dispatch"} <= set(workflow["on"])
    assert workflow["on"]["push"]["branches"] == ["main"]

    job = workflow["jobs"]["update-and-deploy"]
    steps = {step["name"]: step for step in job["steps"]}
    assert steps["Update index"]["if"] == "github.event_name != 'push'"
    assert "--sleep 0.8" in steps["Update index"]["run"]
    assert steps["Commit refreshed metadata"]["if"] == "github.event_name != 'push'"
    assert "scripts/build_site.py" in steps["Build site"]["run"]


def test_pages_workflow_has_required_deployment_contract():
    workflow = load_workflow()
    assert workflow["permissions"]["contents"] == "write"
    assert workflow["permissions"]["pages"] == "write"
    assert workflow["permissions"]["id-token"] == "write"

    job = workflow["jobs"]["update-and-deploy"]
    assert job["environment"]["name"] == "github-pages"
    steps = {step["name"]: step for step in job["steps"]}
    assert steps["Configure GitHub Pages"]["uses"] == "actions/configure-pages@v5"
    assert steps["Upload GitHub Pages artifact"]["uses"] == "actions/upload-pages-artifact@v4"
    assert steps["Upload GitHub Pages artifact"]["with"]["path"] == "_site"
    assert steps["Deploy GitHub Pages"]["uses"] == "actions/deploy-pages@v4"
    assert steps["Deploy GitHub Pages"]["id"] == "deployment"
