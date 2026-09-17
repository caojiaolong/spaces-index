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
    assert "--sleep 3" in steps["Update index"]["run"]
    assert "--audience public" in steps["Update index"]["run"]
    assert "--skip-build" in steps["Update index"]["run"]
    assert steps["Commit refreshed metadata"]["if"] == (
        "github.event_name != 'push' && "
        "steps.changes.outputs.should_commit == 'true'"
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
    assert change_check == "uv run python scripts/check_site_changes.py"

    guarded_steps = (
        "Run tests",
        "Configure GitHub Pages",
        "Upload GitHub Pages artifact",
    )
    for name in guarded_steps:
        assert steps[name]["if"] == "steps.changes.outputs.should_deploy == 'true'"
    assert steps["Build site"]["if"] == "steps.changes.outputs.should_deploy == 'true'"

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


def test_unified_update_builds_once_and_deploys_withdrawals_after_failure():
    steps = {step["name"]: step for step in load_workflow()["jobs"]["prepare"]["steps"]}
    mirror = steps["Update index"]
    assert mirror["if"] == "github.event_name != 'push'"
    assert mirror["continue-on-error"] == "true"
    assert "scripts/update_all.py --audience public --sleep 3 --skip-build" in mirror["run"]
    assert "steps.update.outcome" not in steps["Build site"]["if"]
    assert "Update pilot Markdown" not in steps
    assert "data/articles" in steps["Commit refreshed metadata"]["run"]


def test_only_small_network_state_is_restored_and_saved_across_runs():
    steps = {step["name"]: step for step in load_workflow()["jobs"]["prepare"]["steps"]}
    restore, save = steps["Restore source request state"], steps["Save source request state"]
    assert restore["if"] == "github.event_name != 'push'"
    assert "always()" in save["if"] and "github.event_name != 'push'" in save["if"]
    assert restore["with"]["key"] == save["with"]["key"]
    assert "github.run_id" in save["with"]["key"] and "github.run_attempt" in save["with"]["key"]
    assert restore["with"]["restore-keys"] in save["with"]["key"]
    assert set(restore["with"]["path"].splitlines()) == {".cache/mirror/robots*.json", ".cache/image-cache/hosts-*.json"}
    assert save["with"]["path"] == restore["with"]["path"]


def test_article_storage_is_included_in_full_site_updates():
    steps = {step["name"]: step for step in load_workflow()["jobs"]["prepare"]["steps"]}
    command = steps["Commit refreshed metadata"]["run"]
    assert "git add -- README.md docs config data/README.md data/posts_raw.json" in command
    for line in command.splitlines():
        if line.strip().startswith("git add "):
            assert "data" not in line.split()
    assert "data/overrides.yaml data/articles" in command
    assert "data/articles/9119" not in command and "data/articles/11882" not in command
