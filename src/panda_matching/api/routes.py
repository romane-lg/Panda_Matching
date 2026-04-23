from __future__ import annotations

import uuid
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from panda_matching.agent.router import route_chat_message
from panda_matching.agent.tools import (
    append_chat_message,
    blockers_data,
    ensure_chat_session,
    explain_match_data,
    json_safe,
    load_chat_state,
    save_chat_state,
    top_matches_data,
)
from panda_matching.api.chat_ui import render_chat_ui
from panda_matching.db.session import get_sessionmaker

app = FastAPI(title="Panda Matching API", version="0.1.0")


class ChatRequest(BaseModel):
    message: str
    session_id: str | None = None


class ChatResponse(BaseModel):
    session_id: str
    intent: str
    response: str
    data: dict[str, Any] | None = None


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/chat", response_class=HTMLResponse)
def chat_ui() -> str:
    return render_chat_ui()


@app.get("/matches/top")
def get_top_matches(
    panda_name: str = Query(..., min_length=1),
    k: int = Query(5, ge=1, le=50),
) -> dict[str, Any]:
    session_maker = get_sessionmaker()
    with session_maker() as session:
        return top_matches_data(session, panda_name=panda_name, k=k)


@app.get("/matches/explain")
def explain_match(
    focal_id: str = Query(..., min_length=1),
    candidate_id: str = Query(..., min_length=1),
) -> dict[str, Any]:
    session_maker = get_sessionmaker()
    with session_maker() as session:
        return explain_match_data(session, focal_id=focal_id, candidate_id=candidate_id)


@app.get("/matches/blockers")
def get_blockers(
    focal_id: str = Query(..., min_length=1),
) -> dict[str, Any]:
    session_maker = get_sessionmaker()
    with session_maker() as session:
        return blockers_data(session, focal_id=focal_id)


@app.post("/agent/chat", response_model=ChatResponse)
def chat_agent(payload: ChatRequest) -> ChatResponse:
    session_id = payload.session_id or str(uuid.uuid4())
    message = payload.message.strip()

    session_maker = get_sessionmaker()
    with session_maker() as session:
        ensure_chat_session(session, session_id)
        memory = load_chat_state(session, session_id)
        append_chat_message(
            session,
            session_id=session_id,
            role="user",
            message=message,
        )

        try:
            turn = route_chat_message(session, message, memory)
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

        safe_data = None if turn.data is None else json_safe(turn.data)
        append_chat_message(
            session,
            session_id=session_id,
            role="assistant",
            message=turn.response,
            intent=turn.intent,
            data=safe_data,
        )
        save_chat_state(session, session_id=session_id, memory=memory)
        session.commit()
        return ChatResponse(
            session_id=session_id,
            intent=turn.intent,
            response=turn.response,
            data=None,
        )


def run() -> None:
    import uvicorn

    uvicorn.run("panda_matching.api:app", host="0.0.0.0", port=8000, reload=False)
