from fastapi import APIRouter, HTTPException, Request

from app.models.schemas import AskRequest, AskResponse
from app.services.embedding_backends import EmbeddingModelMismatchError

router = APIRouter(tags=["rag"])


@router.post("/ask", response_model=AskResponse)
async def ask(body: AskRequest, request: Request) -> AskResponse:
    if not body.question.strip():
        raise HTTPException(status_code=422, detail="question must not be empty")

    rag_svc = request.app.state.rag_service

    try:
        result = await rag_svc.ask(
            session_id=body.session_id,
            question=body.question,
            document_id=body.document_id,
            strategy=body.strategy,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except EmbeddingModelMismatchError as exc:
        raise HTTPException(
            status_code=409,
            detail={
                "message": str(exc),
                "reindex_url": f"/documents/{body.document_id}/reindex",
            },
        ) from exc
    except PermissionError as exc:
        raise HTTPException(
            status_code=409, detail="Document is not ready", headers={"X-Status": str(exc)}
        ) from exc
    except NotImplementedError as exc:
        raise HTTPException(status_code=501, detail=str(exc)) from exc

    return AskResponse(**result)
