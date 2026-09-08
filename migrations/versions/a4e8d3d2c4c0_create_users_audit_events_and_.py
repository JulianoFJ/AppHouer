"""create users, audit_events and amostragem tables

Revision ID: a4e8d3d2c4c0
Revises: 
Create Date: 2026-09-08 16:30:12.977341

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB


# revision identifiers, used by Alembic.
revision: str = 'a4e8d3d2c4c0'
down_revision: Union[str, Sequence[str], None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "users",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("login", sa.Text, nullable=False, unique=True),
        sa.Column("nome", sa.Text, nullable=False),
        sa.Column("senha_hash", sa.Text, nullable=False),
        sa.Column("perfil", sa.Text, nullable=False, server_default="usuario"),
        sa.Column("ativo", sa.Boolean, nullable=False, server_default=sa.true()),
        sa.Column("criado_em", sa.DateTime(timezone=True), nullable=False,
                   server_default=sa.text("now()")),
    )

    # Sem FK para `users`: um evento de auditoria tem que sobreviver à exclusão do
    # usuário que o gerou, exatamente como a linha do CSV/planilha de hoje sobrevive.
    op.create_table(
        "audit_events",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("timestamp_utc", sa.DateTime(timezone=True), nullable=False),
        sa.Column("evento", sa.Text, nullable=False),
        sa.Column("usuario", sa.Text),
        sa.Column("nome", sa.Text),
        sa.Column("perfil", sa.Text),
        sa.Column("sessao_id", sa.Text),
        sa.Column("segundos_sessao", sa.Float),
        sa.Column("alvo", sa.Text),
        sa.Column("detalhe", sa.Text),
    )
    op.create_index("ix_audit_events_timestamp_utc", "audit_events", ["timestamp_utc"])
    op.create_index("ix_audit_events_sessao_id", "audit_events", ["sessao_id"])

    op.create_table(
        "amostragem_execucoes",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("criado_em", sa.DateTime(timezone=True), nullable=False,
                   server_default=sa.text("now()")),
        # ON DELETE SET NULL: a execução documenta um sorteio já realizado — apagar o
        # usuário não pode apagar (nem invalidar) o registro do que foi sorteado.
        sa.Column("usuario_id", sa.Integer, sa.ForeignKey("users.id", ondelete="SET NULL")),
        sa.Column("identificacao", sa.Text),
        # JSONB, não colunas fixas: os parâmetros de amostragem (NBR 5426, áreas
        # especiais) ainda estão evoluindo — engessar o schema forçaria uma migration
        # a cada campo novo.
        sa.Column("parametros", JSONB, nullable=False, server_default=sa.text("'{}'::jsonb")),
        sa.Column("status", sa.Text, nullable=False, server_default="concluida"),
    )

    op.create_table(
        "amostragem_pontos",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("execucao_id", sa.Integer,
                   sa.ForeignKey("amostragem_execucoes.id", ondelete="CASCADE"),
                   nullable=False),
        sa.Column("dados", JSONB, nullable=False),
        sa.Column("selecionado", sa.Boolean, nullable=False, server_default=sa.true()),
        sa.Column("revisao_manual", JSONB),
    )
    op.create_index("ix_amostragem_pontos_execucao_id", "amostragem_pontos", ["execucao_id"])


def downgrade() -> None:
    op.drop_table("amostragem_pontos")
    op.drop_table("amostragem_execucoes")
    op.drop_index("ix_audit_events_sessao_id", table_name="audit_events")
    op.drop_index("ix_audit_events_timestamp_utc", table_name="audit_events")
    op.drop_table("audit_events")
    op.drop_table("users")
