"""
server.session_manager - Session allocation and lookup for FastAPI routes.

Each session owns an isolated PraxisEnvironment instance so concurrent clients
do not share mutable scenario state.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from uuid import uuid4

from praxis_env.models import PraxisObservation
from server.praxis_environment import PraxisEnvironment

logger = logging.getLogger(__name__)


@dataclass
class Session:
    """In-memory session record."""

    session_id: str
    env: PraxisEnvironment
    task_name: str
    created_at: float
    last_activity_at: float
    lock: threading.Lock = field(default_factory=threading.Lock)


@dataclass(frozen=True)
class SessionAllocation:
    """Return type for allocate(): created session + initial observation."""

    session: Session
    observation: PraxisObservation
    metadata: dict[str, int | str | None] = field(default_factory=dict)


class SessionManager:
    """Thread-safe in-memory session registry with LRU eviction."""

    def __init__(self, max_sessions: int | None = None) -> None:
        resolved_max_sessions = max_sessions
        if resolved_max_sessions is None:
            resolved_max_sessions = int(os.getenv("PRAXIS_MAX_SESSIONS", "128"))
        self.max_sessions = max(1, resolved_max_sessions)
        self._sessions: OrderedDict[str, Session] = OrderedDict()
        self._lock = threading.Lock()

    def allocate(self, task_name: str, seed: int | None = None) -> SessionAllocation:
        """
        Create a fresh session and initialise its environment via reset().

        `seed` is accepted for API compatibility; deterministic scenarios may
        ignore it until procedural tasks are introduced.
        """
        session_id = str(uuid4())
        env = PraxisEnvironment()
        observation = env.reset(task_name=task_name, seed=seed, session_id=session_id)
        now = time.time()
        session = Session(
            session_id=session_id,
            env=env,
            task_name=task_name,
            created_at=now,
            last_activity_at=now,
        )

        with self._lock:
            if len(self._sessions) >= self.max_sessions:
                evicted_session_id, evicted_session = self._sessions.popitem(last=False)
                logger.warning(
                    "Evicted session=%s task=%s reason=lru",
                    evicted_session_id,
                    evicted_session.task_name,
                )
            self._sessions[session_id] = session
        return SessionAllocation(
            session=session,
            observation=observation,
            metadata=env.last_reset_metadata,
        )

    def get(self, session_id: str) -> Session | None:
        """Return session by id without mutating LRU order."""
        with self._lock:
            return self._sessions.get(session_id)

    def touch(self, session_id: str) -> bool:
        """Mark session as recently used for LRU ordering."""
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                return False
            session.last_activity_at = time.time()
            self._sessions.move_to_end(session_id)
            return True

    def close(self, session_id: str) -> bool:
        """Remove a session if present."""
        with self._lock:
            return self._sessions.pop(session_id, None) is not None
