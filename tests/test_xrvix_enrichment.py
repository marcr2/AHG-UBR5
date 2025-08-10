import os
import json
import sys

# Ensure project root is on sys.path for direct module imports
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from process_xrvix_dumps_json import (
    enrich_batch_metadata_from_citations,
    enrich_all_batches_metadata_from_citations,
    CITATION_CACHE,
    merge_citation_results_into_cache,
    save_citation_cache,
)


def make_batch_file(tmp_dir, source, batch_idx, metas):
    source_dir = os.path.join(tmp_dir, source)
    os.makedirs(source_dir, exist_ok=True)
    batch_path = os.path.join(source_dir, f"batch_{batch_idx:04d}.json")
    data = {
        "source": source,
        "batch_num": batch_idx,
        "embeddings": [[0.0, 1.0]],
        "chunks": ["dummy text"],
        "metadata": metas,
    }
    with open(batch_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    return batch_path


def test_enrich_batch_metadata_from_citations_updates_pending_only():
    current_batch = {
        "metadata": [
            {
                "source": "biorxiv",
                "paper_index": 1,
                "citation_count": "pending",
                "journal": "pending",
                "impact_factor": "pending",
            },
            {
                "source": "biorxiv",
                "paper_index": 2,
                "citation_count": "5",
                "journal": "Some Journal",
                "impact_factor": "10.2",
            },
        ]
    }
    citations = {
        "biorxiv_1": {
            "citation_count": "12",
            "journal": "Journal A",
            "impact_factor": "7.5",
        },
        "biorxiv_2": {
            "citation_count": "9",
            "journal": "Journal B",
            "impact_factor": "8.1",
        },
    }
    enrich_batch_metadata_from_citations(current_batch, citations)

    # First record should be filled
    m0 = current_batch["metadata"][0]
    assert m0["citation_count"] == "12"
    assert m0["journal"] == "Journal A"
    assert m0["impact_factor"] == "7.5"

    # Second record should remain unchanged because it wasn't pending
    m1 = current_batch["metadata"][1]
    assert m1["citation_count"] == "5"
    assert m1["journal"] == "Some Journal"
    assert m1["impact_factor"] == "10.2"


def test_enrich_all_batches_metadata_from_citations_sweeps_dir(tmp_path):
    embeddings_dir = tmp_path / "xrvix_embeddings"
    os.makedirs(embeddings_dir, exist_ok=True)

    # Create two batch files with pending fields
    make_batch_file(
        str(embeddings_dir),
        "biorxiv",
        0,
        [
            {
                "source": "biorxiv",
                "paper_index": 1,
                "citation_count": "pending",
                "journal": "pending",
                "impact_factor": "pending",
            },
        ],
    )
    make_batch_file(
        str(embeddings_dir),
        "medrxiv",
        1,
        [
            {
                "source": "medrxiv",
                "paper_index": 10,
                "citation_count": "pending",
                "journal": "pending",
                "impact_factor": "pending",
            },
        ],
    )

    # Build a cache and persist
    merge_citation_results_into_cache(
        {
            "biorxiv_1": {
                "citation_count": "3",
                "journal": "J A",
                "impact_factor": "1.1",
            },
            "medrxiv_10": {
                "citation_count": "7",
                "journal": "J B",
                "impact_factor": "2.2",
            },
        }
    )
    save_citation_cache(str(embeddings_dir))

    # Run sweep with implicit loading
    stats = enrich_all_batches_metadata_from_citations(str(embeddings_dir))
    assert stats["files_scanned"] >= 2
    assert stats["files_modified"] >= 2
    assert stats["records_updated"] >= 2

    # Validate batch contents updated
    for source, idx, key in [("biorxiv", 0, "biorxiv_1"), ("medrxiv", 1, "medrxiv_10")]:
        batch_path = os.path.join(str(embeddings_dir), source, f"batch_{idx:04d}.json")
        with open(batch_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        meta = data["metadata"][0]
        cache = CITATION_CACHE[key]
        assert meta["citation_count"] == cache["citation_count"]
        assert meta["journal"] == cache["journal"]
        assert meta["impact_factor"] == cache["impact_factor"]
