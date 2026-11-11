"""Real pgvector retrieval with deterministic embeddings and mocked generation."""

import asyncio
import json
from collections import Counter
from contextlib import asynccontextmanager
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock
from uuid import uuid4

import httpx
import pytest
from fastapi import FastAPI
from google.genai import errors, types
from PIL import Image
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import async_sessionmaker

from config import GeminiSettings
from routes.attachments import make_attachments_router
from routes.explain import make_explain_router
from services import attachments, chat_history, retrieval


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture
def configured(monkeypatch):
    monkeypatch.setenv("AI_SERVICE_API_KEY", "test-secret")
    settings = GeminiSettings(llm_provider="gemini_enterprise", google_cloud_project="test",
                              model_names=["test-model"])
    generate = AsyncMock(return_value=SimpleNamespace(candidates=[object()], text="A grounded answer."))
    api = SimpleNamespace(models=SimpleNamespace(generate_content=generate))
    context = AsyncMock()
    context.__aenter__.return_value = api
    client = Mock(return_value=SimpleNamespace(aio=context))
    embed = AsyncMock(return_value=[[1.0] + [0.0] * 767])
    monkeypatch.setattr(retrieval, "embed_chunks", embed)
    monkeypatch.setattr(retrieval, "create_client", client)
    app = FastAPI()
    app.include_router(make_explain_router(settings))
    app.state.ingest_database = None
    return SimpleNamespace(app=app, settings=settings, generate=generate, embed=embed, client=client)


@pytest.fixture
def seeded(storage_session, monkeypatch, configured):
    @asynccontextmanager
    async def seed(rows=None):
        async with storage_session() as (session, schema):
            monkeypatch.setattr(retrieval, "text", lambda sql: text(sql.replace("concept_explainer", schema)))
            monkeypatch.setattr(chat_history, "text", lambda sql: text(sql.replace("concept_explainer", schema)))

            @asynccontextmanager
            async def connection():
                yield session.bind

            database = SimpleNamespace(sessions=async_sessionmaker(
                session.bind, expire_on_commit=False, join_transaction_mode="create_savepoint"),
                engine=SimpleNamespace(connect=connection), dispose=AsyncMock())
            rows = rows if rows is not None else [
                ("exact", [1], "offering", [1, 3]),
                ("diagonal", [0.707, 0.707], "offering", [1]),
                ("half", [0.5] * 4, "offering", [1]),
                ("orthogonal", [0, 1], "offering", [1]),
                ("other-offering", [1], "other", [1]),
                ("other-week", [1], "offering", [2]),
            ]
            for document, vector, offering, weeks in rows:
                await session.execute(text(f"""
                    INSERT INTO {schema}.chunks (document_id, content, embedding, metadata)
                    VALUES (:document, :content, CAST(:vector AS vector), CAST(:metadata AS jsonb))
                """), {"document": document, "content": document + " course material " * 30,
                       "vector": json.dumps(vector + [0] * (768 - len(vector))),
                       "metadata": json.dumps({"course_offering_id": offering, "weeks": weeks,
                                               "start_week": 1, "end_week": 3,
                                               "title": document, "material_type": "NOTES"})})
            await session.commit()
            configured.app.state.ingest_database = database
            yield database
    return seed


async def post(app, *, question="Explain the topic", week=1, offering="offering", token="test-secret",
               attachment_ids=None, session_id=None):
    body = {"question": question, "week": week, "course_offering_id": offering}
    if attachment_ids is not None:
        body["attachment_ids"] = attachment_ids
    if session_id is not None:
        body["session_id"] = session_id
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        return await client.post("/explain", json=body,
                                 headers={"Authorization": f"Bearer {token}"} if token else {})


@pytest.mark.anyio
async def test_explain_valid_answer_with_sources(configured, seeded):
    async with seeded():
        response = await post(configured.app)
    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "A grounded answer."
    assert [s["document_id"] for s in body["sources"]] == ["exact", "diagonal", "half"]
    assert body["sources"][0] == {"document_id": "exact", "title": "exact", "material_type": "NOTES",
                                  "chunk_preview": ("exact" + " course material " * 30)[:200],
                                  "source_type": "course_material", "attachment_id": None, "filename": None}
    assert body["retrieval_info"] == {"chunks_available": 4, "chunks_above_threshold": 3,
                                      "similarity_threshold": 0.4, "attachment_chunks_available": 0,
                                      "attachment_chunks_above_threshold": 0, "attachment_images_count": 0}
    configured.embed.assert_awaited_once_with(["Explain the topic"], configured.settings)
    configured.client.assert_called_once_with(configured.settings)
    kwargs = configured.generate.await_args.kwargs
    assert kwargs["model"] == "test-model"
    assert kwargs["config"].temperature == 0.3
    context = "\n\n".join(f"--- Source: {name} ---\n{name}" + " course material " * 30
                            for name in ["exact", "diagonal", "half"])
    assert kwargs["contents"] == retrieval.EXPLAIN_PROMPT_TEMPLATE.format(context=context, question="Explain the topic")


@pytest.mark.anyio
async def test_explain_empty_week(configured, seeded):
    async with seeded():
        body = (await post(configured.app, week=9)).json()
    assert body == {"answer": "No course material has been ingested for Week 9 of this offering yet.",
                    "sources": [], "retrieval_info": {"chunks_available": 0, "chunks_above_threshold": 0,
                                                      "similarity_threshold": 0.4, "attachment_chunks_available": 0,
                                                      "attachment_chunks_above_threshold": 0, "attachment_images_count": 0}}
    configured.generate.assert_not_awaited()


@pytest.mark.anyio
async def test_explain_below_threshold(configured, seeded):
    async with seeded([("orthogonal", [0, 1], "offering", [1])]):
        body = (await post(configured.app)).json()
    assert body["answer"] == "The course material for Week 1 does not cover this question."
    assert body["sources"] == []
    assert body["retrieval_info"] == {"chunks_available": 1, "chunks_above_threshold": 0,
                                      "similarity_threshold": 0.4, "attachment_chunks_available": 0,
                                      "attachment_chunks_above_threshold": 0, "attachment_images_count": 0}
    configured.generate.assert_not_awaited()


@pytest.mark.anyio
async def test_explain_filters_offering_and_week(configured, seeded):
    async with seeded():
        body = (await post(configured.app, week=3)).json()
    assert [s["document_id"] for s in body["sources"]] == ["exact"]
    assert body["retrieval_info"]["chunks_available"] == 1
    assert body["retrieval_info"]["chunks_above_threshold"] == 1


@pytest.mark.anyio
async def test_explain_orders_and_limits_top_k(configured, seeded):
    rows = [(str(i), [1, i / 20], "offering", [1]) for i in range(10)]
    async with seeded(rows):
        body = (await post(configured.app)).json()
    assert [s["document_id"] for s in body["sources"]] == [str(i) for i in range(8)]
    assert body["retrieval_info"]["chunks_above_threshold"] == 10


@pytest.mark.anyio
@pytest.mark.parametrize("token", [None, "wrong"])
async def test_explain_auth_failure(configured, token):
    assert (await post(configured.app, token=token)).status_code == 401
    configured.embed.assert_not_awaited()


@pytest.mark.anyio
@pytest.mark.parametrize("failure", ["missing_url", "missing_state", "query_error"])
async def test_explain_database_unavailable(configured, monkeypatch, failure):
    if failure == "missing_url":
        monkeypatch.delenv("DATABASE_URL", raising=False)
    if failure == "query_error":
        configured.app.state.ingest_database = object()
        monkeypatch.setattr(retrieval, "_query_chunks", AsyncMock(side_effect=SQLAlchemyError("private")))
    response = await post(configured.app)
    assert response.status_code == 503
    assert "private" not in response.text


@pytest.mark.anyio
async def test_explain_developer_mode(configured):
    app = FastAPI()
    app.include_router(make_explain_router(GeminiSettings()))
    app.state.ingest_database = object()
    assert (await post(app)).status_code == 503


@pytest.mark.anyio
@pytest.mark.parametrize("kwargs", [{"question": ""}, {"question": "x" * 4001}, {"week": 0}, {"offering": ""}])
async def test_explain_validation_failure(configured, kwargs):
    assert (await post(configured.app, **kwargs)).status_code == 422


@pytest.mark.anyio
@pytest.mark.parametrize("failure", ["embedding", "generation", "empty_candidates", "empty_text"])
async def test_explain_provider_failure(configured, seeded, failure):
    if failure == "embedding":
        configured.embed.side_effect = RuntimeError("private")
    elif failure == "generation":
        configured.generate.side_effect = RuntimeError("private")
    else:
        configured.generate.return_value = SimpleNamespace(
            candidates=[] if failure == "empty_candidates" else [object()], text="")
    async with seeded():
        response = await post(configured.app)
    assert response.status_code == (503 if failure == "embedding" else 200)
    assert "private" not in response.text
    if failure != "embedding":
        assert response.json()["answer"] == retrieval.FALLBACK_ANSWER


@pytest.mark.anyio
async def test_explain_available_when_callback_config_missing(configured, seeded, monkeypatch):
    import main

    monkeypatch.delenv("AI_SERVICE_CALLBACK_SECRET", raising=False)
    monkeypatch.delenv("AI_SERVICE_CALLBACK_ALLOWED_HOSTS", raising=False)
    monkeypatch.setattr(main, "gemini_settings", configured.settings)
    recover = AsyncMock()
    monkeypatch.setattr(main, "sweep_expired_attachments", AsyncMock(return_value=0))
    monkeypatch.setattr(main, "recover_stuck_jobs", recover)
    async with seeded() as database:
        monkeypatch.setattr(main, "Database", Mock(return_value=database))
        app = configured.app
        app.router.lifespan_context = main.lifespan
        app.add_api_route("/health", main.health_check)
        monkeypatch.setattr(main, "app", app)
        async with app.router.lifespan_context(app):
            assert app.state.ingest_database is database
            response = await post(app)
            assert response.status_code == 200
            assert response.json()["answer"] == "A grounded answer."
            assert response.json()["sources"]
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
                health = (await client.get("/health")).json()
            assert health["ingest_status"] == "disabled (missing AI_SERVICE_CALLBACK_SECRET)"
            assert app.state.worker_tasks == []
    recover.assert_not_awaited()


@pytest.mark.anyio
async def test_failed_startup_disposes_database(configured, monkeypatch):
    import main

    database = SimpleNamespace(dispose=AsyncMock())
    monkeypatch.setenv("DATABASE_URL", "postgresql://test.invalid/test")
    monkeypatch.setattr(main, "Database", Mock(return_value=database))
    probe = AsyncMock(side_effect=SQLAlchemyError("private"))
    monkeypatch.setattr(main, "probe_ingestion_database", probe)
    monkeypatch.setattr(main.asyncio, "sleep", AsyncMock())
    await main.configure_ingestion(configured.app)
    assert probe.await_count == 3
    database.dispose.assert_awaited_once()
    assert configured.app.state.ingest_database is None
    assert configured.app.state.ingest_status == "disabled (db unreachable)"


@pytest.mark.anyio
@pytest.mark.parametrize("result", ["answer", "empty_candidates", "empty_text", "error"])
async def test_generate_answer_resilience(configured, result):
    if result == "error":
        configured.generate.side_effect = RuntimeError("private")
    else:
        configured.generate.return_value = SimpleNamespace(
            candidates=[] if result == "empty_candidates" else [object()],
            text="An answer." if result == "answer" else "")
    answer = await retrieval._generate_answer("Explain", [
        {"metadata": {"title": "Notes"}, "content": "Material"}
    ], configured.settings)
    assert answer == ("An answer." if result == "answer" else retrieval.FALLBACK_ANSWER)


@pytest.fixture
def attachment_api(configured, worker_database, monkeypatch):
    @asynccontextmanager
    async def setup(course_vector=None):
        async with worker_database() as (database, schema):
            def isolated_text(sql):
                return text(sql.replace("concept_explainer", schema))

            monkeypatch.setattr(attachments, "text", isolated_text)
            monkeypatch.setattr(retrieval, "text", isolated_text)
            app = configured.app
            app.include_router(make_attachments_router())
            app.state.ingest_database = database
            app.state.gemini_settings = configured.settings
            if course_vector is not None:
                async with database.sessions() as session, session.begin():
                    await session.execute(text(f"""
                        INSERT INTO {schema}.chunks (document_id, content, embedding, metadata)
                        VALUES ('lecture', 'Course lecture content', CAST(:vector AS vector),
                                CAST(:metadata AS jsonb))
                    """), {
                        "vector": json.dumps(course_vector + [0] * (768 - len(course_vector))),
                        "metadata": json.dumps({"course_offering_id": "offering", "weeks": [1],
                                                "title": "Lecture", "material_type": "NOTES"}),
                    })
            configured.embed.side_effect = lambda chunks, settings: [[1.0] + [0.0] * 767 for _ in chunks]
            yield database, schema
    return setup


@pytest.mark.anyio
async def test_explain_persists_user_and_assistant_messages(configured, attachment_api):
    marker = "[AI-GENERATED EXAMPLE]"
    configured.generate.return_value.text = f"{marker}\nWork through this example."
    async with attachment_api([1.0]) as (database, schema):
        response = await post(configured.app, session_id="test-session-1")
        assert response.status_code == 200
        async with database.sessions() as session:
            rows = (await session.execute(text(f"""
                SELECT role, content, sources_json, jsonb_typeof(sources_json) AS sources_type
                FROM {schema}.chat_messages
                WHERE session_id = 'test-session-1'
                ORDER BY created_at, id
            """))).mappings().all()
        messages = await chat_history.fetch_recent_messages(database, "test-session-1")
    assert len(rows) == 2
    assert [row["role"] for row in rows] == ["user", "assistant"]
    assert [row["content"] for row in rows] == ["Explain the topic", response.json()["answer"]]
    assert rows[0]["sources_json"] is None
    assert rows[0]["sources_type"] is None
    assert rows[1]["sources_type"] == "array"
    assert rows[1]["sources_json"] == [
        {"source_type": "course_material", "attachment_id": None, "filename": None,
         **response.json()["sources"][0]},
        response.json()["sources"][1],
    ]
    assert messages[1].sources_json == rows[1]["sources_json"]
    assert messages[0].created_at < messages[1].created_at


@pytest.mark.anyio
async def test_explain_fetches_history_for_followup(configured, attachment_api):
    async with attachment_api([1.0]) as (database, _):
        for role, content in [
            ("user", "What is recursion?"), ("assistant", "A function calls itself."),
            ("user", "What stops it?"), ("assistant", "A base case."),
        ]:
            await chat_history.persist_message(database, "followup", role, content)
        await chat_history.persist_message(database, "other", "user", "Unrelated session")
        response = await post(configured.app, question="Why does it matter?", session_id="followup")
    assert response.status_code == 200
    prompt = configured.generate.await_args.kwargs["contents"]
    assert (
        "Conversation so far:\nUser: What is recursion?\nAssistant: A function calls itself.\n\n"
        "User: What stops it?\nAssistant: A base case.\n\n"
    ) in prompt
    assert "Student's question: Why does it matter?" in prompt
    assert prompt.index("Conversation so far:") < prompt.index("Student's question:")
    assert "User: Why does it matter?" not in prompt
    assert "Unrelated session" not in prompt


@pytest.mark.anyio
async def test_explain_empty_session_id_skips_history(configured, attachment_api, caplog):
    async with attachment_api([1.0]) as (database, schema):
        response = await post(configured.app, session_id="")
        async with database.sessions() as session:
            assert await session.scalar(text(f"SELECT count(*) FROM {schema}.chat_messages")) == 0
    assert response.status_code == 200
    assert "Conversation so far:" not in configured.generate.await_args.kwargs["contents"]
    assert any(
        record.levelname == "WARNING"
        and record.getMessage() == "chat_history.skipped reason=missing_session_id"
        for record in caplog.records
    )


@pytest.mark.anyio
async def test_explain_history_fetches_last_10_only(configured, attachment_api):
    async with attachment_api([1.0]) as (database, _):
        async with database.sessions.begin() as session:
            for index in range(15):
                await chat_history.persist_message(
                    database, "long", "user", f"Prior turn [{index}]", session=session,
                )
        response = await post(configured.app, session_id="long")
    assert response.status_code == 200
    prompt = configured.generate.await_args.kwargs["contents"]
    assert all(f"Prior turn [{index}]" not in prompt for index in range(5))
    expected = "".join(f"User: Prior turn [{index}]\n" for index in range(5, 15))
    assert f"Conversation so far:\n{expected}" in prompt


@pytest.mark.anyio
async def test_explain_history_format(configured, attachment_api):
    long_message = "x" * 2000
    async with attachment_api([1.0]) as (database, _):
        await chat_history.persist_message(database, "format", "user", long_message + "Truncated")
        await chat_history.persist_message(database, "format", "assistant", "Two lines\nwith braces {here}.")
        response = await post(configured.app, session_id="format")
    assert response.status_code == 200
    prompt = configured.generate.await_args.kwargs["contents"]
    assert f"User: {long_message}\nAssistant: Two lines\nwith braces {{here}}.\n\n" in prompt
    assert "Truncated" not in prompt


@pytest.mark.anyio
async def test_explain_short_circuit_persists_messages(configured, attachment_api):
    async with attachment_api([0.0, 1.0]) as (database, _):
        for week, session_id, answer in [
            (2, "empty", "No course material has been ingested for Week 2 of this offering yet."),
            (1, "irrelevant", "The course material for Week 1 does not cover this question."),
        ]:
            response = await post(configured.app, week=week, session_id=session_id)
            assert response.status_code == 200
            assert response.json()["answer"] == answer
            messages = await chat_history.fetch_recent_messages(database, session_id)
            assert [(message.role, message.content, message.sources_json) for message in messages] == [
                ("user", "Explain the topic", None), ("assistant", answer, []),
            ]
    configured.generate.assert_not_awaited()


@pytest.mark.anyio
async def test_explain_no_greeting_in_prompt(configured, attachment_api):
    async with attachment_api([1.0]):
        response = await post(configured.app)
    assert response.status_code == 200
    prompt = configured.generate.await_args.kwargs["contents"]
    assert "Do not introduce yourself or greet the student." in prompt
    assert "greeting with no academic content" in prompt
    assert "Hello! I am Stewart" not in prompt


@pytest.mark.anyio
async def test_explain_prompt_includes_ai_generated_instruction(configured, attachment_api):
    async with attachment_api([1.0]):
        response = await post(configured.app)
    assert response.status_code == 200
    prompt = configured.generate.await_args.kwargs["contents"]
    assert "Worked problems and examples:" in prompt
    assert "for a suitable example" in prompt
    assert "literal marker [AI-GENERATED EXAMPLE] on its own line" in prompt
    assert "do not refuse - generate one with" in prompt


@pytest.mark.anyio
async def test_explain_prompt_contains_formatting_instructions(configured, attachment_api):
    async with attachment_api([1.0]):
        response = await post(configured.app)
    assert response.status_code == 200
    configured.client.assert_called_once_with(configured.settings)
    configured.generate.assert_awaited_once()
    prompt = configured.generate.await_args.kwargs["contents"]
    phase_4a_rules, heading, formatting_rules = prompt.partition("Formatting rules:")
    assert heading == "Formatting rules:"
    for instruction in ("##", "Key takeaway", "LaTeX", "$", "[AI-GENERATED EXAMPLE]",
                        "Do not append", "Do not repeat"):
        assert instruction in formatting_rules
    assert "Do not introduce yourself or greet the student." in phase_4a_rules
    assert "greeting with no academic content" in phase_4a_rules
    assert "Worked problems and examples:" in phase_4a_rules
    assert "literal marker [AI-GENERATED EXAMPLE] on its own line" in phase_4a_rules
    assert "do not refuse - generate one with" in phase_4a_rules


@pytest.mark.anyio
async def test_explain_detects_ai_generated_example(configured, attachment_api):
    marker = "[AI-GENERATED EXAMPLE]"
    configured.generate.return_value.text = f"{marker}\n  Add   two\n numbers. {'x' * 200}"
    async with attachment_api([1.0]):
        response = await post(configured.app)
    assert response.status_code == 200
    assert response.json()["sources"][-1] == {
        "source_type": "ai_generated", "document_id": None, "title": "AI-generated example",
        "material_type": "AI_GENERATED", "chunk_preview": f"{marker} Add two numbers. {'x' * 183}",
        "attachment_id": None, "filename": None,
    }


@pytest.mark.anyio
async def test_explain_no_ai_generated_source_when_marker_absent(configured, attachment_api):
    async with attachment_api([1.0]):
        response = await post(configured.app)
    assert response.status_code == 200
    assert len(response.json()["sources"]) == 1
    assert all(source.get("source_type") != "ai_generated" for source in response.json()["sources"])


@pytest.mark.anyio
async def test_explain_does_not_strip_marker_from_answer(configured, attachment_api):
    answer = "Here is a worked example.\n\n[AI-GENERATED EXAMPLE]\nAdd two numbers."
    configured.generate.return_value.text = answer
    async with attachment_api([1.0]):
        response = await post(configured.app)
    assert response.status_code == 200
    assert response.json()["answer"] == answer


@pytest.mark.anyio
async def test_explain_multiple_markers_only_one_source(configured, attachment_api):
    marker = "[AI-GENERATED EXAMPLE]"
    configured.generate.return_value.text = f"{marker}\nFirst example.\n{marker}\nSecond example."
    async with attachment_api([1.0]):
        response = await post(configured.app)
    assert response.status_code == 200
    sources = [source for source in response.json()["sources"] if source.get("source_type") == "ai_generated"]
    assert len(sources) == 1
    assert sources[0]["chunk_preview"].startswith(f"{marker} First example.")


async def insert_chat_message_with_age(database, schema, session_id, days):
    async with database.sessions.begin() as session:
        await session.execute(text(f"""
            INSERT INTO {schema}.chat_messages (session_id, role, content, created_at)
            VALUES (:session_id, 'user', 'Question', now() - :age)
        """), {"session_id": session_id, "age": timedelta(days=days)})


@pytest.mark.anyio
async def test_sweep_expired_chat_messages(worker_database):
    async with worker_database() as (database, schema):
        await insert_chat_message_with_age(database, schema, "expired", 106)
        await insert_chat_message_with_age(database, schema, "fresh", 30)
        assert await chat_history.sweep_expired_chat_messages(database) == 1
        assert await chat_history.fetch_recent_messages(database, "expired") == []
        assert len(await chat_history.fetch_recent_messages(database, "fresh")) == 1


@pytest.mark.anyio
async def test_startup_sweep_includes_chat_messages(configured, attachment_api, monkeypatch, caplog):
    import main

    async with attachment_api() as (database, schema):
        await insert_chat_message_with_age(database, schema, "expired", 106)
        database.dispose = AsyncMock()

        async def configure(application):
            application.state.ingest_database = database
            application.state.ingest_status = "disabled (missing AI_SERVICE_CALLBACK_SECRET)"

        monkeypatch.setattr(main, "gemini_settings", configured.settings)
        monkeypatch.setattr(main, "configure_ingestion", configure)
        caplog.set_level("INFO", logger=main.cleanup_logger.name)
        async with main.lifespan(configured.app):
            assert await chat_history.fetch_recent_messages(database, "expired") == []
            task = configured.app.state.cleanup_sweep_task
            assert not task.done()
        assert task.cancelled()
    assert "attachments.swept count=0" in caplog.text
    assert "chat_messages.swept count=1" in caplog.text


async def upload(app, filename, content, mime_type):
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post("/attachments", files={"file": (filename, content, mime_type)},
                                     headers={"Authorization": "Bearer test-secret"})
    assert response.status_code == 200
    return response.json()["attachment_id"]


@pytest.fixture
def image_file(tmp_path):
    async def make(filename):
        def save_and_read():
            path = tmp_path / filename
            with Image.new("RGB", (2, 2), "blue") as image:
                image.save(path)
            return path.read_bytes(), Image.MIME[Image.EXTENSION[path.suffix]]

        return await asyncio.to_thread(save_and_read)

    return make


@pytest.mark.anyio
async def test_explain_with_one_text_attachment(configured, attachment_api):
    pdf = Path(__file__).parent / "fixtures" / "sample.pdf"
    configured.generate.return_value.text = "Lecture PDF content is in notes.pdf."
    async with attachment_api() as (database, schema):
        identifier = await upload(configured.app, "notes.pdf", await asyncio.to_thread(pdf.read_bytes),
                                  "application/pdf")
        response = await post(configured.app, attachment_ids=[identifier])
        async with database.sessions() as session:
            assert await session.scalar(text(f"SELECT count(*) FROM {schema}.chunks")) == 0
    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "Lecture PDF content is in notes.pdf."
    assert body["sources"] == [{
        "document_id": None, "title": "notes.pdf", "material_type": "ATTACHMENT",
        "chunk_preview": "Lecture PDF content", "source_type": "attachment",
        "attachment_id": identifier, "filename": "notes.pdf",
    }]
    assert body["retrieval_info"]["chunks_available"] == 0
    assert body["retrieval_info"]["attachment_chunks_available"] == 1
    assert body["retrieval_info"]["attachment_chunks_above_threshold"] == 1
    assert [call.args[0] for call in configured.embed.await_args_list] == [
        ["Explain the topic"], ["Lecture PDF content"],
    ]
    assert "--- Attached files ---\n--- Attachment: notes.pdf ---\nLecture PDF content" in (
        configured.generate.await_args.kwargs["contents"]
    )


@pytest.mark.anyio
async def test_explain_with_attachment_answers_when_material_does_not(configured, attachment_api):
    configured.generate.return_value.text = "The answer comes from notes.txt."
    async with attachment_api(course_vector=[0, 1]):
        identifier = await upload(configured.app, "notes.txt", b"Useful attachment content", "text/plain")
        response = await post(configured.app, attachment_ids=[identifier])
    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "The answer comes from notes.txt."
    assert [source["source_type"] for source in body["sources"]] == ["attachment"]
    assert body["retrieval_info"]["chunks_available"] == 1
    assert body["retrieval_info"]["chunks_above_threshold"] == 0
    assert "Course lecture content" not in configured.generate.await_args.kwargs["contents"]


@pytest.mark.anyio
async def test_explain_with_image_attachment_succeeds(configured, attachment_api, image_file):
    async with attachment_api(course_vector=[1]):
        identifier = await upload(configured.app, "image.png", *(await image_file("image.png")))
        response = await post(configured.app, attachment_ids=[identifier])
    assert response.status_code == 200
    course_source, image_source = response.json()["sources"]
    assert course_source["source_type"] == "course_material"
    assert image_source == {
        "document_id": None, "title": "image.png", "material_type": "ATTACHMENT",
        "chunk_preview": "[image: image.png]", "source_type": "attachment",
        "attachment_id": identifier, "filename": "image.png",
    }


@pytest.mark.anyio
async def test_explain_image_only_request_succeeds(configured, attachment_api, image_file):
    async with attachment_api():
        identifier = await upload(configured.app, "image.png", *(await image_file("image.png")))
        response = await post(configured.app, attachment_ids=[identifier])
    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "A grounded answer."
    assert body["retrieval_info"]["chunks_available"] == 0
    assert body["retrieval_info"]["attachment_chunks_available"] == 0
    source, = body["sources"]
    assert source["attachment_id"] == identifier
    assert source["chunk_preview"] == "[image: image.png]"
    configured.embed.assert_awaited_once_with(["Explain the topic"], configured.settings)


@pytest.mark.anyio
async def test_explain_mixed_text_and_image_request(configured, attachment_api, image_file):
    async with attachment_api():
        text_id = await upload(configured.app, "z-notes.txt", b"Useful notes", "text/plain")
        image_id = await upload(configured.app, "diagram.webp", *(await image_file("diagram.webp")))
        second_text_id = await upload(configured.app, "a-notes.txt", b"More notes", "text/plain")
        response = await post(configured.app, attachment_ids=[text_id, image_id, second_text_id])
    assert response.status_code == 200
    body = response.json()
    assert [source["attachment_id"] for source in body["sources"]] == [text_id, second_text_id, image_id]
    assert [source["chunk_preview"] for source in body["sources"]] == [
        "Useful notes", "More notes", "[image: diagram.webp]",
    ]
    assert body["retrieval_info"]["attachment_chunks_available"] == 2
    assert body["retrieval_info"]["attachment_images_count"] == 1


@pytest.mark.anyio
async def test_explain_multiple_images(configured, attachment_api, image_file):
    filenames = ["z-figure.webp", "m-chart.jpg", "a-diagram.png"]
    async with attachment_api():
        identifiers = [await upload(configured.app, name, *(await image_file(name))) for name in filenames]
        response = await post(configured.app, attachment_ids=identifiers)
    assert response.status_code == 200
    body = response.json()
    assert [source["attachment_id"] for source in body["sources"]] == identifiers
    assert [source["chunk_preview"] for source in body["sources"]] == [f"[image: {name}]" for name in filenames]
    assert body["retrieval_info"]["attachment_images_count"] == 3


@pytest.mark.anyio
async def test_explain_image_passed_to_gemini_as_multimodal_part(configured, attachment_api, image_file, monkeypatch):
    content, mime_type = await image_file("diagram.jpg")
    from_bytes = Mock(wraps=types.Part.from_bytes)
    monkeypatch.setattr(types.Part, "from_bytes", from_bytes)
    async with attachment_api():
        identifier = await upload(configured.app, "diagram.jpg", content, mime_type)
        response = await post(configured.app, attachment_ids=[identifier])
    assert response.status_code == 200
    contents = configured.generate.await_args.kwargs["contents"]
    assert isinstance(contents, list)
    prompt, part = contents
    assert isinstance(prompt, str)
    assert "Explain the topic" in prompt
    assert "diagram.jpg" in prompt
    assert "Attached images" in prompt
    assert isinstance(part, types.Part)
    assert part.inline_data.mime_type == "image/jpeg"
    assert part.inline_data.data == content
    from_bytes.assert_called_once_with(data=content, mime_type=mime_type)


@pytest.mark.anyio
async def test_explain_image_prompt_includes_filenames(configured, attachment_api, image_file):
    filenames = ["diagram.png", "chart.webp"]
    async with attachment_api():
        identifiers = [await upload(configured.app, name, *(await image_file(name))) for name in filenames]
        response = await post(configured.app, attachment_ids=identifiers)
    assert response.status_code == 200
    prompt = configured.generate.await_args.kwargs["contents"][0]
    assert "Attached images (2): diagram.png, chart.webp" in prompt
    assert "cite the filename" in prompt


@pytest.mark.anyio
async def test_explain_gemini_vision_failure_returns_422(configured, attachment_api, image_file, caplog):
    private_message = "Gemini rejected the image: private image metadata"
    configured.generate.side_effect = errors.ClientError(400, {"error": {"message": private_message}})
    async with attachment_api():
        first = await upload(configured.app, "first.png", *(await image_file("first.png")))
        second = await upload(configured.app, "second.png", *(await image_file("second.png")))
        response = await post(configured.app, attachment_ids=[first, second])
    assert response.status_code == 422
    assert response.json() == {
        "detail": f"Image processing failed. Attachment {first} could not be read by the vision model.",
    }
    assert f"attachment_id={first} error_type=ClientError" in caplog.text
    assert private_message not in caplog.text
    configured.generate.assert_awaited_once()


@pytest.mark.anyio
async def test_explain_text_only_request_prompt_omits_image_block(configured, attachment_api):
    async with attachment_api():
        identifier = await upload(configured.app, "notes.txt", b"Useful notes", "text/plain")
        response = await post(configured.app, attachment_ids=[identifier])
    assert response.status_code == 200
    prompt = configured.generate.await_args.kwargs["contents"]
    assert isinstance(prompt, str)
    assert "Attached images" not in prompt
    assert "Useful notes\n\nStudent's question:" in prompt


@pytest.mark.anyio
@pytest.mark.parametrize("image_count", [0, 1, 3])
async def test_explain_image_count_in_retrieval_info(configured, attachment_api, image_file, image_count):
    async with attachment_api():
        identifiers = [await upload(configured.app, "notes.txt", b"Useful notes", "text/plain")]
        for index in range(image_count):
            filename = f"image-{index}.png"
            identifiers.append(await upload(configured.app, filename, *(await image_file(filename))))
        response = await post(configured.app, attachment_ids=identifiers)
    assert response.status_code == 200
    assert response.json()["retrieval_info"]["attachment_images_count"] == image_count


@pytest.mark.anyio
async def test_explain_rejects_expired_attachment(configured, attachment_api):
    async with attachment_api() as (database, schema):
        identifier = await upload(configured.app, "notes.txt", b"Expired content", "text/plain")
        async with database.sessions() as session, session.begin():
            await session.execute(text(f"""
                UPDATE {schema}.attachments SET expires_at = now() - interval '1 minute'
                WHERE id = :id
            """), {"id": identifier})
        response = await post(configured.app, attachment_ids=[identifier])
    assert response.status_code == 422
    assert response.json() == {"detail": f"Attachment {identifier} not found or expired"}


@pytest.mark.anyio
async def test_explain_rejects_unknown_attachment_id(configured, attachment_api):
    identifier = str(uuid4())
    async with attachment_api():
        response = await post(configured.app, attachment_ids=[identifier])
    assert response.status_code == 422
    assert response.json() == {"detail": f"Attachment {identifier} not found or expired"}


@pytest.mark.anyio
async def test_explain_rejects_more_than_eight_attachments(configured):
    response = await post(configured.app, attachment_ids=[str(uuid4()) for _ in range(9)])
    assert response.status_code == 422
    assert response.json() == {"detail": "Maximum 8 attachments per request"}


@pytest.mark.anyio
async def test_explain_rejects_extraction_failure(configured, attachment_api, image_file, monkeypatch):
    paths = []
    extract_text = retrieval.extract_text

    def record_path(path, mime_type):
        paths.append(Path(path))
        return extract_text(path, mime_type)

    monkeypatch.setattr(retrieval, "extract_text", record_path)
    async with attachment_api():
        identifier = await upload(configured.app, "broken.pdf", b"%PDF-1.7\ncorrupt", "application/pdf")
        image = await upload(configured.app, "image.png", *(await image_file("image.png")))
        response = await post(configured.app, attachment_ids=[identifier, image])
    assert response.status_code == 422
    assert response.json() == {"detail": f"Attachment {identifier}: Unable to extract text from PDF"}
    assert len(paths) == 1
    assert not await asyncio.to_thread(paths[0].exists)
    configured.embed.assert_not_awaited()


def test_explain_attachment_chunk_cap_distribution():
    files = [(str(i), f"{i}.txt", [f"{i}-{j}" for j in range(count)])
             for i, count in enumerate([100, 200, 300])]
    selected = retrieval._allocate_chunks(files)
    assert len(selected) == 100
    assert selected == [(identifier, filename, chunk)
                        for (identifier, filename, chunks), count in zip(files, [17, 33, 50])
                        for chunk in chunks[:count]]

    files = [(str(i), f"{i}.txt", ["chunk"] * count)
             for i, count in enumerate([490, 490, 1, 1, 1, 1, 1, 1])]
    selected = retrieval._allocate_chunks(files)
    assert Counter(identifier for identifier, _, _ in selected) == {
        "0": 49, "1": 45, "2": 1, "3": 1, "4": 1, "5": 1, "6": 1, "7": 1,
    }


def test_explain_attachment_chunk_cap_under_threshold():
    files = [("a", "a.txt", ["a"] * 10), ("b", "b.txt", ["b"] * 15), ("c", "c.txt", ["c"] * 25)]
    assert retrieval._allocate_chunks(files) == [
        (identifier, filename, chunk) for identifier, filename, chunks in files for chunk in chunks
    ]


@pytest.mark.anyio
@pytest.mark.parametrize(("filename", "mime_type", "label"), [
    ("notes.txt", "text/plain", "TXT"), ("notes.md", "text/markdown", "MD"),
])
async def test_explain_txt_and_md_attachments(configured, attachment_api, filename, mime_type, label):
    async with attachment_api():
        identifier = await upload(configured.app, filename, b"  # Useful notes\xff\n", mime_type)
        response = await post(configured.app, attachment_ids=[identifier])
        empty = await upload(configured.app, filename, b" \n\t", mime_type)
        rejected = await post(configured.app, attachment_ids=[empty])
    assert response.status_code == 200
    assert response.json()["sources"][0]["filename"] == filename
    assert response.json()["sources"][0]["chunk_preview"] == "# Useful notes\ufffd"
    assert rejected.status_code == 422
    assert rejected.json() == {"detail": f"Attachment {empty}: No extractable text in {label}"}


@pytest.mark.anyio
async def test_explain_mixed_material_and_attachment(configured, attachment_api):
    content = "\n".join(f"Attachment paragraph {i} " + "content " * 120 for i in range(105))
    async with attachment_api(course_vector=[1]):
        identifier = await upload(configured.app, "notes.md", content.encode(), "text/markdown")
        response = await post(configured.app, attachment_ids=[identifier])
    assert response.status_code == 200
    body = response.json()
    assert [source["source_type"] for source in body["sources"]] == ["course_material"] + ["attachment"] * 8
    assert body["sources"][0]["document_id"] == "lecture"
    assert all(source["attachment_id"] == identifier for source in body["sources"][1:])
    info = body["retrieval_info"]
    assert info["attachment_chunks_available"] == 105
    assert info["attachment_chunks_above_threshold"] == 100
    assert len(configured.embed.await_args_list[1].args[0]) == 100
    prompt = configured.generate.await_args.kwargs["contents"]
    assert "--- Course material ---\n--- Source: Lecture ---\nCourse lecture content" in prompt
    assert "--- Attached files ---\n--- Attachment: notes.md ---\nAttachment paragraph 0" in prompt
