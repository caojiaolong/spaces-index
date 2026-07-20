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

    job = workflow["jobs"]["prepare"]
    steps = {step["name"]: step for step in job["steps"]}
    assert steps["Update index"]["if"] == "github.event_name != 'push'"
    assert "--sleep 0.8" in steps["Update index"]["run"]
    assert steps["Commit refreshed metadata"]["if"] == (
        "github.event_name != 'push' && "
        "steps.changes.outputs.should_deploy == 'true'"
    )
    assert "scripts/build_site.py" in steps["Build site"]["run"]


def test_pages_workflow_skips_build_and_deploy_without_changes():
    workflow = load_workflow()
    prepare = workflow["jobs"]["prepare"]
    steps = {step["name"]: step for step in prepare["steps"]}

    assert prepare["outputs"]["should_deploy"] == (
        "${{ steps.changes.outputs.should_deploy }}"
    )
    assert steps["Check for deployable changes"]["id"] == "changes"
    change_check = steps["Check for deployable changes"]["run"]
    assert '"$GITHUB_EVENT_NAME" == "push"' in change_check
    assert "git status --porcelain -- README.md docs data" in change_check
    assert "should_deploy=false" in change_check

    guarded_steps = (
        "Run tests",
        "Build site",
        "Configure GitHub Pages",
        "Upload GitHub Pages artifact",
    )
    for name in guarded_steps:
        assert steps[name]["if"] == "steps.changes.outputs.should_deploy == 'true'"

    deploy = workflow["jobs"]["deploy"]
    assert deploy["needs"] == "prepare"
    assert deploy["if"] == "needs.prepare.outputs.should_deploy == 'true'"


def test_pages_workflow_has_required_deployment_contract():
    workflow = load_workflow()
    prepare = workflow["jobs"]["prepare"]
    deploy = workflow["jobs"]["deploy"]

    assert prepare["permissions"]["contents"] == "write"
    assert prepare["permissions"]["pages"] == "write"
    assert "id-token" not in prepare["permissions"]
    assert deploy["permissions"]["pages"] == "write"
    assert deploy["permissions"]["id-token"] == "write"

    assert deploy["environment"]["name"] == "github-pages"
    prepare_steps = {step["name"]: step for step in prepare["steps"]}
    deploy_steps = {step["name"]: step for step in deploy["steps"]}
    assert prepare_steps["Checkout"]["uses"] == "actions/checkout@v6"
    assert prepare_steps["Set up Python"]["uses"] == "actions/setup-python@v6"
    assert prepare_steps["Set up uv"]["uses"] == (
        "astral-sh/setup-uv@08807647e7069bb48b6ef5acd8ec9567f424441b"
    )
    assert prepare_steps["Configure GitHub Pages"]["uses"] == "actions/configure-pages@v6"
    assert prepare_steps["Upload GitHub Pages artifact"]["uses"] == (
        "actions/upload-pages-artifact@v5"
    )
    assert prepare_steps["Upload GitHub Pages artifact"]["with"]["path"] == "_site"
    assert deploy_steps["Deploy GitHub Pages"]["uses"] == "actions/deploy-pages@v5"
    assert deploy_steps["Deploy GitHub Pages"]["id"] == "deployment"
