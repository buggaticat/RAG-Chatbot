from __future__ import annotations

from types import SimpleNamespace

from rag.validation_layers.deterministic_layer import validate_deterministic_output


def _chunks():
    """Build a small retrieval set used by the deterministic layer tests."""

    return [
        SimpleNamespace(metadata={"doc_id": "doc-1", "paper_id": "doc-1", "chunk_id": "c1", "section_id": "1"}, text="The model achieved 92.5% accuracy on 2024 data."),
        SimpleNamespace(metadata={"doc_id": "doc-1", "paper_id": "doc-1", "chunk_id": "c2", "section_id": "2"}, text="The loss was 0.12 after 30 epochs."),
        SimpleNamespace(metadata={"doc_id": "doc-1", "paper_id": "doc-1", "chunk_id": "t1", "section_id": "3"}, text="Table 1 | Metric | Value | Accuracy | 92.5% | Loss | 0.12 |"),
    ]


def test_accepts_valid_answer_with_matching_citations_and_numbers():
    output = {
        "answer": "The model achieved 92.5% accuracy.",
        "citations": [{"doc_id": "doc-1", "chunk_id": "c1"}],
        "confidence": "high",
    }

    result = validate_deterministic_output(output, _chunks())

    assert result.is_valid is True
    assert result.errors == []


def test_rejects_not_found_with_citations():
    output = {
        "answer": "Not found in context",
        "citations": [{"doc_id": "doc-1", "chunk_id": "c1"}],
        "confidence": "low",
    }

    result = validate_deterministic_output(output, _chunks())

    assert result.is_valid is False
    assert "If the answer is 'Not found in context', citations must be empty." in result.errors


def test_rejects_invalid_confidence():
    output = {
        "answer": "The model achieved 92.5% accuracy.",
        "citations": [{"doc_id": "doc-1", "chunk_id": "c1"}],
        "confidence": "certain",
    }

    result = validate_deterministic_output(output, _chunks())

    assert result.is_valid is False
    assert "'confidence' must be one of: low, medium, high." in result.errors


def test_rejects_numeric_values_not_present_in_citations():
    output = {
        "answer": "The model achieved 95% accuracy.",
        "citations": [{"doc_id": "doc-1", "chunk_id": "c1"}],
        "confidence": "medium",
    }

    result = validate_deterministic_output(output, _chunks())

    assert result.is_valid is False
    assert "Numeric value '95%' must appear exactly in cited context." in result.errors


def test_rejects_citations_not_in_retrieved_context():
    output = {
        "answer": "The model achieved 92.5% accuracy.",
        "citations": [{"doc_id": "doc-9", "chunk_id": "c9"}],
        "confidence": "medium",
    }

    result = validate_deterministic_output(output, _chunks())

    assert result.is_valid is False
    assert "Citations must reference retrieved chunks only: doc-9:c9" in result.errors


def test_rejects_duplicate_citations():
    output = {
        "answer": "The model achieved 92.5% accuracy.",
        "citations": [
            {"doc_id": "doc-1", "chunk_id": "c1"},
            {"doc_id": "doc-1", "chunk_id": "c1"},
        ],
        "confidence": "medium",
    }

    result = validate_deterministic_output(output, _chunks())

    assert result.is_valid is False
    assert "Citations must not contain duplicates." in result.errors


def test_rejects_not_found_when_answer_is_not_short():
    output = {
        "answer": "Not found in context because the answer is not explicitly stated anywhere.",
        "citations": [],
        "confidence": "low",
    }

    result = validate_deterministic_output(output, _chunks())

    assert result.is_valid is False
    assert "Abstention answers must stay short and exact." in result.errors


def test_rejects_date_literals_missing_from_context():
    output = {
        "answer": "The study was published on 2025-01-15.",
        "citations": [{"doc_id": "doc-1", "chunk_id": "c1"}],
        "confidence": "medium",
    }

    result = validate_deterministic_output(output, _chunks())

    assert result.is_valid is False
    assert "Date literal '2025-01-15' must appear exactly in cited context." in result.errors


def test_accepts_table_answers_with_table_like_chunks():
    output = {
        "answer": "Table 1 shows Accuracy at 92.5%.",
        "citations": [{"doc_id": "doc-1", "chunk_id": "t1"}],
        "confidence": "medium",
    }

    result = validate_deterministic_output(output, _chunks())

    assert result.is_valid is True


def test_rejects_citation_metadata_mismatch():
    output = {
        "answer": "The model achieved 92.5% accuracy.",
        "citations": [{"doc_id": "doc-2", "chunk_id": "c1"}],
        "confidence": "medium",
    }

    result = validate_deterministic_output(output, _chunks())

    assert result.is_valid is False
    assert "Citation doc-2:c1 does not match the cited paper metadata." in result.errors
