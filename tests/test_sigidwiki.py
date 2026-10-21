from __future__ import annotations

from sigid.sigidwiki import SigidWikiCatalogImporter


def test_importer_paginates_and_keeps_wiki_claims_and_media_link_only():
    calls = []
    wikitext = """{{Signal
|Title=Example FSK
|Mode=USB
|Modulation=4FSK
|Bandwidth=600 Hz
|Signal file=example.mp3
|Picture=example.png
|Signal description=An example signal.
}}
"""

    def fetch(_endpoint, params):
        calls.append(params)
        if params.get("list") == "categorymembers":
            if params["cmtitle"] == "Category:Signal":
                if "cmcontinue" not in params:
                    return {
                        "query": {"categorymembers": [{"title": "Example FSK"}]},
                        "continue": {"cmcontinue": "next-page"},
                    }
                return {"query": {"categorymembers": [{"title": "Second Signal"}]}}
            return {"query": {"categorymembers": [{"title": "Unidentified burst"}]}}
        pages = {}
        for title in params["titles"].split("|"):
            page_id = {"Example FSK": 1, "Second Signal": 2, "Unidentified burst": 3}[title]
            pages[str(page_id)] = {
                "pageid": page_id,
                "title": title,
                "images": [{"title": "File:extra.png"}] if title == "Example FSK" else [],
                "revisions": [{"revid": 10 + page_id, "timestamp": "2026-10-20T12:00:00Z", "*": wikitext if title == "Example FSK" else "No Signal template."}],
            }
        return {"query": {"pages": pages}}

    catalog = SigidWikiCatalogImporter(fetch_json=fetch, sleep=lambda _duration: None).import_catalog(
        categories=("Signal", "Unidentified"), limit_per_category=2
    )
    assert len(catalog["records"]) == 3
    known = catalog["records"][0]
    assert known["source_reported_fields"]["Modulation"] == "4FSK"
    assert known["source_label_status"] == "unreviewed_external"
    assert known["asset_reuse_status"] == "not_assessed"
    assert known["asset_handling"] == "link_only"
    assert {asset["title"] for asset in known["assets"]} == {
        "File:example.mp3", "File:example.png", "File:extra.png"
    }
    assert all(asset["page_url"].startswith("https://www.sigidwiki.com/wiki/File:") for asset in known["assets"])
    assert catalog["records"][1]["source_reported_fields"] == {}
    assert any(call.get("cmcontinue") == "next-page" for call in calls)
    assert all(call.get("prop") != "imageinfo" for call in calls)
