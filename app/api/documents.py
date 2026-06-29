from fastapi import APIRouter, BackgroundTasks, Form, HTTPException, Query, Request, UploadFile

from app.models.schemas import DocumentOut, DocumentUploadResponse

router = APIRouter(prefix="/documents", tags=["documents"])

MAX_FILE_SIZE = 20 * 1024 * 1024  # 20 MB


@router.post("", status_code=202, response_model=DocumentUploadResponse)
async def upload_document(
    file: UploadFile,
    background_tasks: BackgroundTasks,
    request: Request,
    session_id: str = Form(...),
) -> DocumentUploadResponse:
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=422, detail="Only PDF files are accepted")

    if file.size and file.size > MAX_FILE_SIZE:
        raise HTTPException(status_code=422, detail="File exceeds 20 MB limit")
    file_bytes = await file.read()
    if len(file_bytes) > MAX_FILE_SIZE:
        raise HTTPException(status_code=422, detail="File exceeds 20 MB limit")

    doc_svc = request.app.state.doc_service
    doc = await doc_svc.create_document(session_id, file.filename or "upload.pdf", file_bytes)
    background_tasks.add_task(doc_svc._index_document, doc.id)

    return DocumentUploadResponse(id=doc.id, filename=doc.filename, status=doc.status, file_size=doc.file_size)


@router.get("", response_model=list[DocumentOut])
async def list_documents(session_id: str, request: Request) -> list[DocumentOut]:
    doc_svc = request.app.state.doc_service
    docs = await doc_svc.list_documents(session_id)
    return [DocumentOut.model_validate(d) for d in docs]


@router.get("/{document_id}", response_model=DocumentOut)
async def get_document(document_id: int, session_id: str, request: Request) -> DocumentOut:
    doc_svc = request.app.state.doc_service
    doc = await doc_svc.get_document(document_id, session_id)
    if doc is None:
        raise HTTPException(status_code=404, detail="Document not found")
    return DocumentOut.model_validate(doc)


@router.delete("/{document_id}", status_code=204)
async def delete_document(document_id: int, session_id: str, request: Request) -> None:
    doc_svc = request.app.state.doc_service
    deleted = await doc_svc.delete_document(document_id, session_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Document not found")


@router.post("/{document_id}/reindex", status_code=202, response_model=DocumentUploadResponse)
async def reindex_document(
    document_id: int,
    background_tasks: BackgroundTasks,
    request: Request,
    session_id: str = Query(...),
) -> DocumentUploadResponse:
    doc_svc = request.app.state.doc_service
    doc = await doc_svc.start_reindex(document_id, session_id)
    if doc is None:
        raise HTTPException(status_code=404, detail="Document not found")
    background_tasks.add_task(doc_svc._index_document, doc.id)
    return DocumentUploadResponse(id=doc.id, filename=doc.filename, status=doc.status, file_size=doc.file_size)
