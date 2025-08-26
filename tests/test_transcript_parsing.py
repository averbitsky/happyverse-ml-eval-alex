# Save as: tests/test_transcript_parsing.py

import textwrap

from intervieweval.evaluators.orchestrator import EvaluationOrchestrator


def _parse(transcript: str, questions):
    """
    Helper to call the private method.

    :param transcript: The full transcript text.
    :param questions: List of questions to look for.
    :return: List of (question, answer) pairs.
    """
    return EvaluationOrchestrator._parse_transcript(transcript, questions)


def test_qn_style_removes_question_prefix_basic() -> None:
    """
    Test that the QnA style with prefixes works and removes the question from the start of the answer.

    :return: None
    """
    question = (
        "Walk me through how you integrated a 5G core (e.g., Open5GS, Magma, or similar) into an enterprise network. "
        "What challenges did you face with routing, NAT, or DNS, and how did you resolve them?"
    )
    transcript = textwrap.dedent(
        f"""
        Q1: {question}
        A1: {question}
        "We took a very modular, service-driven approach to 5G core integration. The first step was ..."
    """
    ).strip()

    qs = [question]
    pairs = _parse(transcript, qs)
    assert len(pairs) == 1
    q, ans = pairs[0]
    assert q == question
    # The answer should NOT start with the question text
    assert not ans.strip().lower().startswith(question.lower())
    # It should start with the real content after removal
    assert ans.strip().startswith('"We took a very modular')


def test_qn_style_handles_quotes_and_whitespace() -> None:
    """
    Test that the QnA style with quotes and whitespace works correctly.

    :return: None.
    """
    q = "Explain your approach to migrating a legacy ETL to a streaming system (e.g., Kafka + Flink)."
    transcript = textwrap.dedent(
        f"""
        Q2:   {q}
        A2: "{q}"    We began by decoupling ingestion from transformation to isolate backpressure...
    """
    ).strip()
    qs = ["placeholder", q]
    pairs = _parse(transcript, qs)
    assert len(pairs) == 1
    _, ans = pairs[0]
    assert not ans.strip().lower().startswith(q.lower())
    assert ans.strip().startswith("We began by decoupling ingestion")


def test_qn_style_with_no_a_prefix() -> None:
    """
    Test that the QnA style works even if there is no 'A' prefix before the answer.

    :return: None.
    """
    q = "How do you handle schema evolution in a data lake without breaking downstream jobs?"
    transcript = textwrap.dedent(
        f"""
        Q1: {q}
        {q}
        We prefer schema-on-read for most use-cases; we add compatibility guards at ingestion...
    """
    ).strip()
    qs = [q]
    pairs = _parse(transcript, qs)
    assert len(pairs) == 1
    _, ans = pairs[0]
    assert not ans.strip().lower().startswith(q.lower())
    assert ans.strip().startswith("We prefer schema-on-read")


def test_plain_text_fallback_with_two_questions() -> None:
    """
    Test that the plain text style works with two questions and does not produce false positives.

    :return: None.
    """
    q1 = "Describe your approach to blue/green deployments for a high-traffic API."
    q2 = "How did you tune Postgres for a write-heavy workload?"
    transcript = textwrap.dedent(
        f"""
        Intro blah blah...

        {q1}
        We fronted traffic with a router that could pin users to a color; health checks + canaries controlled flips.

        {q2}
        We adjusted shared_buffers, checkpoint_timeout, wal_compression and used partitioning to reduce contention.
    """
    ).strip()

    qs = [q1, q2]
    pairs = _parse(transcript, qs)
    assert len(pairs) == 2

    (pq1, a1), (pq2, a2) = pairs
    assert pq1 == q1 and pq2 == q2
    assert not a1.strip().lower().startswith(q1.lower())
    assert not a2.strip().lower().startswith(q2.lower())
    assert a1.strip().startswith("We fronted traffic with a router")
    assert a2.strip().startswith("We adjusted shared_buffers")


def test_no_false_positive_when_question_not_repeated_in_answer() -> None:
    """
    Test that there are no false positives when the question is not repeated in the answer.

    :return: None.
    """
    q = "How do you ensure reproducibility in ML experiments?"
    transcript = textwrap.dedent(
        f"""
        Q1: {q}
        A1: We pin package versions, use containerized runners, record random seeds, and snapshot data manifests.
    """
    ).strip()
    qs = [q]
    (pq, ans) = _parse(transcript, qs)[0]
    assert pq == q
    # Nothing to strip; answer should be preserved
    assert ans.strip().startswith("We pin package versions")


def test_handles_smart_quotes_and_optional_prefixes() -> None:
    """
    Test that the parser handles smart quotes and optional 'Question:' prefixes.

    :return: None.
    """
    q = "What is your strategy for canary releases in Kubernetes?"
    # Includes smart quotes and 'Question:' prefix
    transcript = textwrap.dedent(
        f"""
        Q1: {q}
        A1: “Question: {q}”  Typically we leverage progressive delivery via weighted routing and SLO-based rollback.
    """
    ).strip()
    qs = [q]
    (pq, ans) = _parse(transcript, qs)[0]
    assert pq == q
    assert not ans.strip().lower().startswith(q.lower())
    assert ans.strip().startswith("Typically we leverage progressive delivery")


if __name__ == "__main__":
    import sys
    import pytest

    sys.exit(pytest.main([__file__, "-q"]))
