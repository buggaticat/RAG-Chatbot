from __future__ import annotations

from eval.retrieval_and_answer_quality import (
    RetrievalAndAnswerQualityEvaluator,
    RetrievalAndAnswerQualitySample,
)
from eval.runner import RAGEvaluationSuite


class FakeMetric:
    def __init__(self, value: float) -> None:
        self.value = value

    async def ascore(self, **kwargs):
        return self.value


def test_runner_includes_retrieval_and_answer_quality_report():
    retrieval_report = RetrievalAndAnswerQualityEvaluator(
        context_precision_metric=FakeMetric(1.0),
        context_recall_metric=FakeMetric(1.0),
        faithfulness_metric=FakeMetric(1.0),
        response_groundedness_metric=FakeMetric(1.0),
    ).evaluate(
        [
            RetrievalAndAnswerQualitySample(
                user_input="Where was Einstein born?",
                reference="Albert Einstein was born in Ulm, Germany.",
                retrieved_contexts=["Albert Einstein was born in Ulm, Germany."],
                response="Albert Einstein was born in Ulm, Germany.",
            )
        ]
    )

    suite = RAGEvaluationSuite(retrieval_and_answer_quality_report=retrieval_report)
    report = suite.build_report().to_dict()

    assert "retrieval_and_answer_quality" in report
    assert report["retrieval_and_answer_quality"]["hallucination_rate"] == 0.0
