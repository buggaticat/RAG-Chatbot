from __future__ import annotations

from eval.retrieval_and_answer_quality import (
    RetrievalAndAnswerQualityEvaluator,
    RetrievalAndAnswerQualitySample,
)


class FakeMetric:
    def __init__(self, value: float) -> None:
        self.value = value
        self.calls: list[dict[str, object]] = []

    async def ascore(self, **kwargs):
        self.calls.append(kwargs)
        return self.value


def test_retrieval_and_answer_quality_evaluator_uses_ragas_metric_inputs():
    context_precision = FakeMetric(0.9)
    id_precision = FakeMetric(0.5)
    context_recall = FakeMetric(0.8)
    id_recall = FakeMetric(0.25)
    faithfulness = FakeMetric(0.75)
    response_groundedness = FakeMetric(1.0)

    evaluator = RetrievalAndAnswerQualityEvaluator(
        context_precision_metric=context_precision,
        id_based_context_precision_metric=id_precision,
        context_recall_metric=context_recall,
        id_based_context_recall_metric=id_recall,
        faithfulness_metric=faithfulness,
        response_groundedness_metric=response_groundedness,
    )

    sample = RetrievalAndAnswerQualitySample(
        user_input="Where is the Eiffel Tower located?",
        reference="The Eiffel Tower is located in Paris.",
        retrieved_contexts=["The Eiffel Tower is located in Paris."],
        response="The Eiffel Tower is located in Paris.",
        retrieved_context_ids=["doc_1", "doc_2", "doc_3"],
        reference_context_ids=["doc_1", "doc_4"],
    )

    report = evaluator.evaluate([sample])

    assert report.context_precision_with_reference is not None
    assert report.context_precision_with_reference.summary["mean"] == 0.9
    assert report.id_based_context_precision is not None
    assert report.id_based_context_precision.summary["mean"] == 0.5
    assert report.context_recall is not None
    assert report.context_recall.summary["mean"] == 0.8
    assert report.id_based_context_recall is not None
    assert report.id_based_context_recall.summary["mean"] == 0.25
    assert report.faithfulness is not None
    assert report.faithfulness.summary["mean"] == 0.75
    assert report.response_groundedness is not None
    assert report.response_groundedness.summary["mean"] == 1.0
    assert report.hallucination_rate == 0.25

    assert context_precision.calls[0]["user_input"] == sample.user_input
    assert context_precision.calls[0]["reference"] == sample.reference
    assert context_precision.calls[0]["retrieved_contexts"] == sample.retrieved_contexts
    assert id_precision.calls[0]["retrieved_context_ids"] == sample.retrieved_context_ids
    assert id_precision.calls[0]["reference_context_ids"] == sample.reference_context_ids
    assert context_recall.calls[0]["user_input"] == sample.user_input
    assert context_recall.calls[0]["reference"] == sample.reference
    assert faithfulness.calls[0]["response"] == sample.response
    assert faithfulness.calls[0]["retrieved_contexts"] == sample.retrieved_contexts
    assert response_groundedness.calls[0]["response"] == sample.response
    assert response_groundedness.calls[0]["retrieved_contexts"] == sample.retrieved_contexts


def test_retrieval_and_answer_quality_evaluator_sync_wrapper_returns_report():
    evaluator = RetrievalAndAnswerQualityEvaluator(
        context_precision_metric=FakeMetric(1.0),
        context_recall_metric=FakeMetric(1.0),
        faithfulness_metric=FakeMetric(1.0),
        response_groundedness_metric=FakeMetric(1.0),
    )

    report = evaluator.evaluate(
        [
            RetrievalAndAnswerQualitySample(
                user_input="Question",
                reference="Answer",
                retrieved_contexts=["Answer"],
                response="Answer",
            )
        ]
    )

    assert report.context_precision_with_reference is not None
    assert report.context_recall is not None
    assert report.faithfulness is not None
    assert report.response_groundedness is not None
