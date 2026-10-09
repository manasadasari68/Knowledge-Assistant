"""Track which embedding model produced each chunk vector.

Revision ID: 0002_chunk_embedding_model
Revises: 0001_document_storage
"""

from alembic import op
import sqlalchemy as sa

revision = "0002_chunk_embedding_model"
down_revision = "0001_document_storage"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "document_chunks",
        sa.Column("embedding_model", sa.String(length=255), nullable=True),
    )
    op.create_index(
        "ix_document_chunks_embedding_model",
        "document_chunks",
        ["embedding_model"],
    )


def downgrade() -> None:
    op.drop_index("ix_document_chunks_embedding_model", table_name="document_chunks")
    op.drop_column("document_chunks", "embedding_model")
