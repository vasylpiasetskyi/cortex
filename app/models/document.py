import enum
from datetime import UTC, datetime

import sqlalchemy as sa
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.db import Base


class DocumentStatus(enum.StrEnum):
    indexing = "indexing"
    ready = "ready"
    error = "error"


class Document(Base):
    __tablename__ = "documents"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    session_id: Mapped[str] = mapped_column(sa.String(255), index=True, nullable=False)
    filename: Mapped[str] = mapped_column(sa.String(255), nullable=False)
    file_path: Mapped[str] = mapped_column(sa.String(512), nullable=False)
    file_size: Mapped[int] = mapped_column(sa.Integer, nullable=False)
    status: Mapped[str] = mapped_column(sa.String(20), nullable=False, default=DocumentStatus.indexing)
    error_msg: Mapped[str | None] = mapped_column(sa.Text, nullable=True)
    embedding_model: Mapped[str | None] = mapped_column(sa.String(100), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        sa.DateTime(timezone=True),
        default=lambda: datetime.now(UTC),
        nullable=False,
    )

    chunks: Mapped[list["Chunk"]] = relationship(
        "Chunk", back_populates="document", cascade="all, delete-orphan"
    )


class Chunk(Base):
    __tablename__ = "chunks"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    document_id: Mapped[int] = mapped_column(
        sa.Integer, sa.ForeignKey("documents.id", ondelete="CASCADE"), nullable=False
    )
    text: Mapped[str] = mapped_column(sa.Text, nullable=False)
    page: Mapped[int] = mapped_column(sa.Integer, nullable=False)
    chunk_index: Mapped[int] = mapped_column(sa.Integer, nullable=False)
    token_count: Mapped[int] = mapped_column(sa.Integer, nullable=False)

    document: Mapped["Document"] = relationship("Document", back_populates="chunks")
