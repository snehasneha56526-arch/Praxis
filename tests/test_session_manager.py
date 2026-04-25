"""tests/test_session_manager.py - Session allocation behavior."""

from server.session_manager import SessionManager


def test_allocate_passes_session_id_into_environment_state():
    manager = SessionManager(max_sessions=4)
    allocation = manager.allocate(task_name="single-service-alert")
    state = allocation.session.env.state()
    assert state.session_id == allocation.session.session_id
