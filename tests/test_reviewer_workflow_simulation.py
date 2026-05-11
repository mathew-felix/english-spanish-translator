from scripts.reviewer_workflow_simulation import run_simulation


def test_reviewer_workflow_simulation_quantifies_traceability_gain():
    result = run_simulation()

    assert result["tasks"] == 5
    assert result["plain_translation_traceability"]["coverage_percent"] == 33.33
    assert result["review_workflow_traceability"]["coverage_percent"] == 100.0
    assert result["evidence_lookup"]["coverage_percent"] == 100.0
