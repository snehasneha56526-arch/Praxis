"""
tests/test_memory.py - Unit tests for PraxisMemory.
"""

from praxis_env.memory import PraxisMemory


def test_save_finding_returns_confirmation_and_persists():
    memory = PraxisMemory()
    response = memory.save_finding("db_pool", "exhausted")

    assert response == "Saved: db_pool"
    assert memory.saved_findings["db_pool"] == "exhausted"


def test_recall_specific_key():
    memory = PraxisMemory(saved_findings={"root_cause": "bad_config"})
    assert memory.recall_memory("root_cause") == "bad_config"


def test_recall_unknown_key_returns_helpful_message():
    memory = PraxisMemory(saved_findings={"root_cause": "bad_config"})
    assert (
        memory.recall_memory("missing_key") == "No saved finding for key: missing_key"
    )


def test_recall_all_returns_deterministic_sorted_lines():
    memory = PraxisMemory(saved_findings={"b": "value_b", "a": "value_a"})
    assert memory.recall_memory() == "a: value_a\nb: value_b"


def test_recall_all_when_empty():
    memory = PraxisMemory()
    assert memory.recall_memory() == "No findings saved."


def test_get_observation_context_pre_cutoff_returns_last_ten_entries():
    memory = PraxisMemory(CONTEXT_CUTOFF_STEP=30)
    full_log = [f"log_{i}" for i in range(15)]

    context = memory.get_observation_context(full_log=full_log, step=29)

    assert context == "\n".join(full_log[-10:])


def test_get_observation_context_post_cutoff_returns_banner_and_findings():
    memory = PraxisMemory(
        saved_findings={"db_pool": "exhausted"}, CONTEXT_CUTOFF_STEP=30
    )

    context = memory.get_observation_context(full_log=["ignored"], step=30)

    assert "[CONTEXT LIMIT REACHED — Step 30/30]" in context
    assert "Full investigation log is no longer available." in context
    assert "Your saved findings:" in context
    assert "db_pool: exhausted" in context
    assert "Use recall_memory to retrieve specific findings." in context
    assert "Use save_finding to persist new evidence." in context


def test_is_active_matches_cutoff_rule():
    memory = PraxisMemory(CONTEXT_CUTOFF_STEP=30)
    assert memory.is_active(29) is False
    assert memory.is_active(30) is True
    assert memory.is_active(31) is True


def test_reset_clears_saved_findings():
    memory = PraxisMemory(saved_findings={"k": "v"})
    memory.reset()
    assert memory.saved_findings == {}


def test_determinism_same_inputs_same_output_bytes():
    memory = PraxisMemory(
        saved_findings={"alpha": "1", "beta": "2"}, CONTEXT_CUTOFF_STEP=30
    )
    full_log = [f"log_{i}" for i in range(20)]

    output_1 = memory.get_observation_context(full_log=full_log, step=35)
    output_2 = memory.get_observation_context(full_log=full_log, step=35)

    assert output_1 == output_2
    assert output_1.encode("utf-8") == output_2.encode("utf-8")
