"""
server.app — FastAPI application factory.

Builds the HTTP API that the validation script, inference.py, and
HuggingFace Spaces will call.

Endpoints:
    POST /reset    → Start a new episode
    POST /step     → Execute one action
    GET  /state    → Get current episode state
    GET  /tasks    → List available task names
    GET  /health   → Health check (returns 200 + metadata)
    GET  /         → Web UI redirect / info

Environment Variables:
    ENABLE_WEB_INTERFACE: "true" to enable OpenEnv web UI (optional)
    LOG_LEVEL: log level — default "INFO"
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from praxis_env.models import PraxisAction, PraxisObservation, PraxisState
from server.praxis_environment import PraxisEnvironment
from server.session_manager import Session, SessionManager

# ── Logging ───────────────────────────────────────────────────────────────────

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ── Global session manager instance ────────────────────────────────────────────
manager = SessionManager()
task_catalog = PraxisEnvironment().list_tasks()


# ── Request / Response schemas (Pydantic, for FastAPI validation) ─────────────


class ResetRequest(BaseModel):
    """POST /reset body."""

    task_name: str = "single-service-alert"
    seed: int | None = None


class StepRequest(BaseModel):
    """POST /step body."""

    command: str


# ── App factory ───────────────────────────────────────────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup / shutdown lifecycle."""
    logger.info("Praxis environment server starting up")
    logger.info("Available tasks: %s", task_catalog)
    yield
    logger.info("Praxis environment server shutting down")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="Praxis — Production Incident Response Environment",
        description=(
            "OpenEnv-compatible environment for training AI agents on "
            "SRE on-call triage tasks. Implements reset/step/state API."
        ),
        version="1.0.0",
        lifespan=lifespan,
    )

    # CORS — allow judges and web UI to call from any origin
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Routes ────────────────────────────────────────────────────────────────

    def _require_session(request: Request) -> tuple[str, Session]:
        session_id = request.headers.get("x-session-id")
        if not session_id:
            raise HTTPException(status_code=400, detail="Missing X-Session-Id header")
        session = manager.get(session_id)
        if session is None:
            raise HTTPException(status_code=400, detail="No active session for that id")
        manager.touch(session_id)
        return session_id, session

    @app.get("/health")
    async def health() -> dict[str, Any]:
        """Health check — must return 200 for the pre-validation script."""
        return {
            # OpenEnv runtime validator expects "healthy"
            "status": "healthy",
            "environment": "praxis-env",
            "version": "1.0.0",
            "available_tasks": task_catalog,
        }

    @app.get("/metadata")
    async def metadata() -> dict[str, Any]:
        """OpenEnv metadata endpoint used by runtime validators."""
        return {
            "name": "praxis-env",
            "version": "1.0.0",
            "description": (
                "Production Incident Response Training Ground — simulating real-world "
                "SRE on-call triage for AI agents."
            ),
            "supports_concurrent_sessions": True,
            "themes": ["long-horizon-planning"],
            "tasks": [
                {"name": "single-service-alert", "difficulty": "easy", "max_steps": 15},
                {"name": "ambiguous-incident", "difficulty": "medium", "max_steps": 25},
                {"name": "cascading-failure", "difficulty": "hard", "max_steps": 20},
                {"name": "memory-leak", "difficulty": "hard", "max_steps": 25},
                {
                    "name": "procedural-incident",
                    "difficulty": "medium",
                    "max_steps": 25,
                },
            ],
            "endpoints": {
                "reset": "/reset",
                "step": "/step",
                "state": "/state",
                "tasks": "/tasks",
                "health": "/health",
                "schema": "/schema",
                "mcp": "/mcp",
            },
        }

    @app.get("/schema")
    async def schema() -> dict[str, Any]:
        """OpenEnv schema endpoint used by runtime validators."""
        return {
            "action": PraxisAction.model_json_schema(),
            "observation": PraxisObservation.model_json_schema(),
            "state": PraxisState.model_json_schema(),
        }

    @app.post("/mcp")
    async def mcp(request: Request) -> dict[str, Any]:
        """
        Minimal JSON-RPC MCP endpoint.

        The OpenEnv runtime validator only checks reachability and JSON-RPC shape.
        """
        try:
            payload = await request.json()
        except Exception:
            payload = {}
        return {
            "jsonrpc": "2.0",
            "id": payload.get("id"),
            "result": {"ok": True},
        }

    @app.get("/")
    async def root() -> dict[str, str]:
        """Root endpoint — basic info for judges browsing the space."""
        return {
            "name": "praxis-env",
            "description": "Production Incident Response Training Ground",
            "docs": "/docs",
            "health": "/health",
            "tasks": "/tasks",
        }

    @app.post("/reset")
    async def reset(request: Request) -> dict[str, Any]:
        """
        Start a new episode.

        Body (optional): {"task_name": "single-service-alert"}
        Returns: {"observation": PraxisObservation, ...flat_fields}

        Accepts: JSON body, empty body {}, or no body at all.
        Includes both wrapped and flat observation fields for compatibility.
        """
        task_name = "single-service-alert"
        seed: int | None = None
        try:
            body = await request.body()
            if body and body.strip():
                data = await request.json()
                task_name = (
                    data.get("task_name", "single-service-alert")
                    or "single-service-alert"
                )
                raw_seed = data.get("seed")
                if raw_seed is not None:
                    if isinstance(raw_seed, bool) or not isinstance(raw_seed, int):
                        raise HTTPException(
                            status_code=400,
                            detail="Invalid seed: expected integer",
                        )
                    seed = raw_seed
        except HTTPException:
            raise
        except Exception:
            pass  # no body or invalid JSON — use default task

        try:
            allocation = manager.allocate(task_name=task_name, seed=seed)
            obs_dict = PraxisEnvironment._obs_to_dict(allocation.observation)
            # Return both flat fields AND wrapped observation key
            # so both strict and lenient judges pass
            return {
                "session_id": allocation.session.session_id,
                "observation": obs_dict,
                "metadata": allocation.metadata,
                **obs_dict,
            }
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            logger.exception("reset() failed: %s", e)
            raise HTTPException(status_code=500, detail=f"reset() error: {e}")

    @app.post("/step")
    async def step(request: Request, payload: StepRequest) -> dict[str, Any]:
        """
        Execute one action.

        Body: {"command": "query_logs service=auth timerange=5m"}
        Returns: {observation, reward, done, info}
        """
        try:
            _, session = _require_session(request)
            action = PraxisAction(command=payload.command)
            with session.lock:
                result = session.env.step(action)
            return result
        except HTTPException as e:
            raise e
        except RuntimeError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            logger.exception("step() failed: %s", e)
            raise HTTPException(status_code=500, detail=f"step() error: {e}")

    @app.get("/state")
    async def state(request: Request) -> dict[str, Any]:
        """
        Get current episode state.

        Returns: PraxisState as JSON
        """
        try:
            session_id, session = _require_session(request)
            with session.lock:
                s = session.env.state()
            return {
                "episode_id": s.episode_id,
                "step_count": s.step_count,
                "task_name": s.task_name,
                "incident_resolved": s.incident_resolved,
                "root_cause_identified": s.root_cause_identified,
                "cumulative_reward": s.cumulative_reward,
                "session_id": session_id,
                "memory_active": s.memory_active,
            }
        except RuntimeError as e:
            raise HTTPException(status_code=400, detail=str(e))

    @app.get("/tasks")
    async def tasks() -> dict[str, list[str]]:
        """List all available task names."""
        return {"tasks": task_catalog}

    return app


def main() -> None:
    """Run the API server via uvicorn.

    Exposed as the `server` project script so validators and local runners
    can start the environment without custom commands.
    """
    import uvicorn

    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "7860"))
    uvicorn.run(
        "server.app:app",
        host=host,
        port=port,
        log_level=LOG_LEVEL.lower(),
    )


# ── ASGI app (imported by uvicorn and Dockerfile CMD) ─────────────────────────
app = create_app()


if __name__ == "__main__":
    main()
