import json
from pathlib import Path

from scripts.build_site import build_catalog


ROOT = Path(__file__).resolve().parents[1]
POST_FIELDS = {
    "id",
    "title",
    "url",
    "date",
    "sourceCategory",
    "sourceTags",
    "sourceSummary",
    "topics",
    "series",
    "seriesId",
    "seriesIndex",
    "level",
    "seriesTopic",
    "notes",
}


def test_repository_catalog_respects_metadata_and_url_contract():
    raw_posts = json.loads(
        (ROOT / "data" / "posts_classified.json").read_text(encoding="utf-8")
    )
    catalog = build_catalog(raw_posts)

    assert catalog["stats"]["postCount"] == len(raw_posts)
    assert catalog["stats"]["topicCount"] == len(catalog["topics"])
    assert catalog["stats"]["seriesCount"] == len(catalog["series"])
    assert all(set(post) == POST_FIELDS for post in catalog["posts"])
    assert all(post["url"] is None or post["url"].startswith("https://") for post in catalog["posts"])
    assert all(
        post["sourceSummary"] is None or len(post["sourceSummary"]) <= 320
        for post in catalog["posts"]
    )
    assert all(series["count"] >= 2 for series in catalog["series"])
