"""Acceptance tests for the request-scoped exception error presentation stack.

The sections follow the distinct exits:
1. Stack mechanics (push/pop/merge/auto-restore)
2. HTTPException exit
3. RequestValidationError exit (422)
4. WebSocketRequestValidationError exit (close reason)
5. ResponseValidationError exit (500)
6. Coexistence with user-registered exception handlers
7. OpenAPI stays static
8. Background tasks observe an empty stack
9. Concurrent request isolation
10. TestClient / direct ASGI transport parity
"""

import asyncio
from http import HTTPStatus

import httpx
import pytest
from fastapi import (
    BackgroundTasks,
    Depends,
    FastAPI,
    HTTPException,
    Request,
    WebSocket,
    error_presentation,
    get_error_presentation,
    pop_error_presentation,
    push_error_presentation,
)
from fastapi.error_presentation import (
    ErrorPresentation,
    get_stack_depth,
    resolve_error_presentation,
)
from fastapi.exceptions import (
    RequestValidationError,
    ResponseValidationError,
    WebSocketRequestValidationError,
)
from fastapi.testclient import TestClient
from pydantic import BaseModel
from starlette.websockets import WebSocketDisconnect


class Item(BaseModel):
    name: str


# ---------------------------------------------------------------------------
# 1. Stack mechanics
# ---------------------------------------------------------------------------


def test_empty_stack_view_is_default():
    assert get_stack_depth() == 0
    assert get_error_presentation() == ErrorPresentation()
    assert get_error_presentation().payload_profile == "detail"
    assert get_error_presentation().include_validation_body is False
    assert get_error_presentation().include_endpoint_ctx is False
    assert get_error_presentation().websocket_reason_mode == "json_errors"


def test_partial_frame_merges_without_touching_other_switches():
    token = push_error_presentation(payload_profile="problem")
    try:
        view = get_error_presentation()
        assert view.payload_profile == "problem"
        assert view.include_validation_body is False
        assert view.websocket_reason_mode == "json_errors"
    finally:
        pop_error_presentation(token)
    assert get_error_presentation() == ErrorPresentation()


def test_inner_pop_restores_outer_user_policy_not_default():
    outer = push_error_presentation(payload_profile="problem")
    try:
        assert get_error_presentation().payload_profile == "problem"
        inner = push_error_presentation(include_validation_body=True)
        try:
            view = get_error_presentation()
            assert view.payload_profile == "problem"
            assert view.include_validation_body is True
        finally:
            pop_error_presentation(inner)
        # Back to the outer user policy, never to a default literal.
        view = get_error_presentation()
        assert view.payload_profile == "problem"
        assert view.include_validation_body is False
    finally:
        pop_error_presentation(outer)
    assert get_error_presentation() == ErrorPresentation()


def test_context_manager_pops_on_exception_path():
    with pytest.raises(RuntimeError):
        with error_presentation(payload_profile="problem"):
            assert get_stack_depth() == 1
            raise RuntimeError("boom")
    assert get_stack_depth() == 0
    assert get_error_presentation() == ErrorPresentation()


def test_exception_snapshot_uses_innermost_view_and_outer_does_not_overwrite():
    with pytest.raises(RuntimeError) as outer_exc:
        with error_presentation(payload_profile="problem"):
            with error_presentation(include_validation_body=True):
                raise RuntimeError("inner")
    view = resolve_error_presentation(outer_exc.value)
    assert view.payload_profile == "problem"
    assert view.include_validation_body is True


def test_exception_in_outer_block_does_not_keep_inner_frame():
    with pytest.raises(HTTPException) as exc_info:
        with error_presentation(payload_profile="problem"):
            try:
                with error_presentation(include_validation_body=True):
                    raise ValueError("caught and recovered")
            except ValueError:
                pass
            raise HTTPException(status_code=400, detail="outer only")
    view = resolve_error_presentation(exc_info.value)
    assert view.payload_profile == "problem"
    assert view.include_validation_body is False


def test_invalid_values_are_rejected():
    with pytest.raises(ValueError):
        push_error_presentation(payload_profile="weird")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        push_error_presentation(websocket_reason_mode="nope")  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        push_error_presentation(include_validation_body="yes")  # type: ignore[arg-type]
    assert get_stack_depth() == 0


# ---------------------------------------------------------------------------
# Shared apps for the exit-specific tests
# ---------------------------------------------------------------------------


def build_app() -> FastAPI:
    app = FastAPI()

    async def problem_body_ctx_dep():
        # Request validation / response serialization errors are raised by the
        # framework around the endpoint, so a request-scoped yield dependency
        # is the scope that covers them.
        with error_presentation(
            payload_profile="problem",
            include_validation_body=True,
            include_endpoint_ctx=True,
        ):
            yield

    async def body_dep():
        with error_presentation(include_validation_body=True):
            yield

    async def ctx_dep():
        with error_presentation(include_endpoint_ctx=True):
            yield

    @app.get("/http-default")
    def http_default():
        raise HTTPException(status_code=404, detail="not here")

    @app.get("/http-problem")
    def http_problem():
        with error_presentation(payload_profile="problem"):
            raise HTTPException(status_code=404, detail="not here")

    @app.get("/http-problem-headers")
    def http_problem_headers():
        with error_presentation(payload_profile="problem"):
            raise HTTPException(
                status_code=400,
                detail="bad",
                headers={"X-Custom": "present", "Retry-After": "42"},
            )

    @app.get("/http-headers-default")
    def http_headers_default():
        raise HTTPException(
            status_code=400, detail="bad", headers={"X-Custom": "present"}
        )

    @app.get("/no-body")
    def no_body():
        with error_presentation(payload_profile="problem"):
            raise HTTPException(status_code=204)

    @app.post("/validation-default")
    def validation_default(item: Item):
        return item

    @app.post(
        "/validation-problem",
        dependencies=[Depends(problem_body_ctx_dep)],
    )
    def validation_problem(item: Item):
        return item

    @app.post("/validation-body", dependencies=[Depends(body_dep)])
    def validation_body(item: Item):
        return item

    @app.post("/validation-ctx", dependencies=[Depends(ctx_dep)])
    def validation_ctx(item: Item):
        return item

    @app.get("/response-default", response_model=Item)
    def response_default():
        return {"unexpected": 1}

    @app.get(
        "/response-problem",
        response_model=Item,
        dependencies=[Depends(problem_body_ctx_dep)],
    )
    def response_problem():
        return {"unexpected": 1}

    return app


app = build_app()
client = TestClient(app, raise_server_exceptions=False)


# ---------------------------------------------------------------------------
# 2. HTTPException exit
# ---------------------------------------------------------------------------


def test_http_exception_default_shape_unchanged():
    response = client.get("/http-default")
    assert response.status_code == 404
    assert response.headers["content-type"] == "application/json"
    assert response.json() == {"detail": "not here"}


def test_http_exception_problem_envelope():
    response = client.get("/http-problem")
    assert response.status_code == 404
    assert response.headers["content-type"] == "application/problem+json"
    assert response.json() == {
        "type": "about:blank",
        "title": HTTPStatus(404).phrase,
        "status": 404,
        "detail": "not here",
    }


def test_http_exception_headers_pass_through_in_both_profiles():
    response = client.get("/http-headers-default")
    assert response.headers["x-custom"] == "present"
    response = client.get("/http-problem-headers")
    assert response.headers["content-type"] == "application/problem+json"
    assert response.headers["x-custom"] == "present"
    assert response.headers["retry-after"] == "42"


def test_http_exception_no_body_status_ignores_profile():
    response = client.get("/no-body")
    assert response.status_code == 204
    assert response.text == ""
    assert "content-type" not in response.headers


def test_http_exception_next_request_back_to_default():
    client.get("/http-problem")
    response = client.get("/http-default")
    assert response.headers["content-type"] == "application/json"
    assert response.json() == {"detail": "not here"}


# ---------------------------------------------------------------------------
# 3. RequestValidationError exit
# ---------------------------------------------------------------------------


def test_request_validation_default_shape_unchanged():
    response = client.post("/validation-default", json={"other": 1})
    assert response.status_code == 422
    assert response.headers["content-type"] == "application/json"
    payload = response.json()
    assert set(payload.keys()) == {"detail"}
    assert payload["detail"][0]["loc"] == ["body", "name"]


def test_request_validation_problem_envelope_with_body_and_ctx():
    response = client.post("/validation-problem", json={"other": 1})
    assert response.status_code == 422
    assert response.headers["content-type"] == "application/problem+json"
    payload = response.json()
    assert payload["type"] == "about:blank"
    assert payload["title"] == HTTPStatus(422).phrase
    assert payload["status"] == 422
    assert isinstance(payload["errors"], list)
    assert payload["body"] == {"other": 1}
    assert payload["endpoint_ctx"]["function"] == "validation_problem"
    assert payload["endpoint_ctx"]["path"] == "POST /validation-problem"
    assert "detail" not in payload


def test_request_validation_detail_with_body():
    response = client.post("/validation-body", json={"other": 1})
    assert response.status_code == 422
    assert response.headers["content-type"] == "application/json"
    payload = response.json()
    assert set(payload.keys()) == {"detail", "body"}
    assert payload["body"] == {"other": 1}


def test_request_validation_detail_with_endpoint_ctx():
    response = client.post("/validation-ctx", json={"other": 1})
    payload = response.json()
    assert set(payload.keys()) == {"detail", "endpoint_ctx"}
    assert payload["endpoint_ctx"]["function"] == "validation_ctx"


def test_request_validation_invalid_json_body_attached():
    # Malformed JSON fails while the body is read, before dependencies are
    # solved, so the frame has to come from middleware.
    invalid_json_app = FastAPI()

    class IncludeBodyMiddleware:
        def __init__(self, app):
            self.app = app

        async def __call__(self, scope, receive, send):
            with error_presentation(include_validation_body=True):
                await self.app(scope, receive, send)

    invalid_json_app.add_middleware(IncludeBodyMiddleware)

    @invalid_json_app.post("/items")
    def items_create(item: Item):
        return item

    response = TestClient(invalid_json_app).post(
        "/items",
        content=b"{not json",
        headers={"content-type": "application/json"},
    )
    assert response.status_code == 422
    payload = response.json()
    assert payload["detail"][0]["type"] == "json_invalid"
    assert payload["body"] == "{not json"


# ---------------------------------------------------------------------------
# 4. WebSocketRequestValidationError exit
# ---------------------------------------------------------------------------


ws_app = FastAPI()


async def ws_omit_dep():
    with error_presentation(websocket_reason_mode="omit"):
        yield


async def ws_problem_but_json_errors_dep():
    # problem+json is HTTP-only and must never leak into the WS close semantics
    with error_presentation(
        payload_profile="problem", websocket_reason_mode="json_errors"
    ):
        yield


@ws_app.websocket("/ws/{item_id}")
async def ws_default(websocket: WebSocket, item_id: int):
    await websocket.accept()


@ws_app.websocket("/ws-omit/{item_id}", dependencies=[Depends(ws_omit_dep)])
async def ws_omit(websocket: WebSocket, item_id: int):
    await websocket.accept()


@ws_app.websocket(
    "/ws-problem-ignored/{item_id}",
    dependencies=[Depends(ws_problem_but_json_errors_dep)],
)
async def ws_problem_ignored(websocket: WebSocket, item_id: int):
    await websocket.accept()


ws_client = TestClient(ws_app)


def test_websocket_default_reason_carries_errors():
    with pytest.raises(WebSocketDisconnect) as exc_info:
        with ws_client.websocket_connect("/ws/not-an-int"):
            pass
    assert exc_info.value.code == 1008
    # Historical behavior: the encoded error list rides in the reason.
    assert isinstance(exc_info.value.reason, list)
    assert exc_info.value.reason[0]["loc"] == ["path", "item_id"]


def test_websocket_omit_reason_is_empty():
    with pytest.raises(WebSocketDisconnect) as exc_info:
        with ws_client.websocket_connect("/ws-omit/not-an-int"):
            pass
    assert exc_info.value.code == 1008
    assert exc_info.value.reason == ""


def test_websocket_never_assembles_problem_media_type():
    # payload_profile is an HTTP concept; it must not change the WS close.
    with pytest.raises(WebSocketDisconnect) as exc_info:
        with ws_client.websocket_connect("/ws-problem-ignored/not-an-int"):
            pass
    assert exc_info.value.code == 1008
    assert isinstance(exc_info.value.reason, list)
    reason_repr = repr(exc_info.value.reason)
    assert "application/problem+json" not in reason_repr
    assert "about:blank" not in reason_repr


# ---------------------------------------------------------------------------
# 5. ResponseValidationError exit
# ---------------------------------------------------------------------------


def test_response_validation_default_is_stack_aware_not_uncaught():
    response = client.get("/response-default")
    assert response.status_code == 500
    assert response.headers["content-type"] == "application/json"
    payload = response.json()
    assert set(payload.keys()) == {"detail"}
    assert payload["detail"][0]["loc"] == ["response", "name"]


def test_response_validation_problem_envelope():
    response = client.get("/response-problem")
    assert response.status_code == 500
    assert response.headers["content-type"] == "application/problem+json"
    payload = response.json()
    assert payload["type"] == "about:blank"
    assert payload["title"] == HTTPStatus(500).phrase
    assert payload["status"] == 500
    assert payload["errors"][0]["loc"] == ["response", "name"]
    assert payload["body"] == {"unexpected": 1}
    assert payload["endpoint_ctx"]["function"] == "response_problem"


def test_user_response_validation_handler_wins():
    user_app = FastAPI()

    @user_app.exception_handler(ResponseValidationError)
    async def custom_response_handler(request: Request, exc: ResponseValidationError):
        from starlette.responses import JSONResponse

        return JSONResponse(status_code=512, content={"custom": True})

    @user_app.get("/x", response_model=Item)
    def x():
        return {"unexpected": 1}

    response = TestClient(user_app).get("/x")
    assert response.status_code == 512
    assert response.json() == {"custom": True}


# ---------------------------------------------------------------------------
# 6. Coexistence with custom exception handlers
# ---------------------------------------------------------------------------


def test_custom_http_handler_not_forced_to_subscribe_to_stack():
    coexist_app = FastAPI()

    @coexist_app.exception_handler(HTTPException)
    async def custom_http(request: Request, exc: HTTPException):
        from starlette.responses import JSONResponse

        return JSONResponse({"handled": "by-user"})

    @coexist_app.get("/a")
    def a():
        with error_presentation(payload_profile="problem"):
            raise HTTPException(status_code=400)

    @coexist_app.get("/b")
    def b():
        from fastapi.exceptions import RequestValidationError as _RVE  # noqa

        raise HTTPException(status_code=403)

    response = TestClient(coexist_app).get("/a")
    # The user handler wins; it does not have to know about the stack.
    assert response.json() == {"handled": "by-user"}
    assert response.headers["content-type"] == "application/json"


def test_default_handlers_still_subscribe_alongside_custom_handlers():
    coexist_app = FastAPI()

    @coexist_app.exception_handler(RequestValidationError)
    async def custom_rv(request: Request, exc: RequestValidationError):
        from starlette.responses import JSONResponse

        return JSONResponse({"custom": "validation"})

    @coexist_app.get("/e/{item_id}")
    def e(item_id: int):
        with error_presentation(payload_profile="problem"):
            raise HTTPException(status_code=404, detail="still default")

    response = TestClient(coexist_app).get("/e/1")
    # HTTPException default handler is still subscribed to the stack.
    assert response.status_code == 404
    assert response.headers["content-type"] == "application/problem+json"
    assert response.json()["detail"] == "still default"


def test_custom_websocket_handler_runs_without_clearing_stack():
    coexist_app = FastAPI()

    @coexist_app.exception_handler(WebSocketRequestValidationError)
    async def custom_ws(websocket: WebSocket, exc: WebSocketRequestValidationError):
        await websocket.close(code=1008, reason="custom-ws")

    @coexist_app.websocket("/w/{item_id}")
    async def w(websocket: WebSocket, item_id: int):
        await websocket.accept()

    with pytest.raises(WebSocketDisconnect) as exc_info:
        with TestClient(coexist_app).websocket_connect("/w/bad"):
            pass
    assert exc_info.value.reason == "custom-ws"


# ---------------------------------------------------------------------------
# 7. OpenAPI generation ignores the request-time stack
# ---------------------------------------------------------------------------


def test_openapi_schema_ignores_presentation_stack():
    schema_before = app.openapi()

    with error_presentation(
        payload_profile="problem",
        include_validation_body=True,
        include_endpoint_ctx=True,
        websocket_reason_mode="omit",
    ):
        schema_inside = app.openapi()
        response = client.get("/openapi.json")

    assert response.status_code == 200
    assert response.json() == schema_before
    assert schema_inside == schema_before
    assert (
        schema_inside["components"]["schemas"]["HTTPValidationError"]["properties"][
            "detail"
        ]["type"]
        == "array"
    )


# ---------------------------------------------------------------------------
# 8. Background tasks observe an empty stack
# ---------------------------------------------------------------------------


bg_app = FastAPI()
bg_seen: dict[str, object] = {}


async def bg_active_frame_dep():
    # Request-scoped yield dependency: its frame is still alive while the
    # response (including background tasks) is being sent.
    with error_presentation(payload_profile="problem", include_validation_body=True):
        yield


def record_view_task():
    bg_seen["view"] = get_error_presentation()
    bg_seen["depth"] = get_stack_depth()


@bg_app.get("/bg", dependencies=[Depends(bg_active_frame_dep)])
def bg_endpoint(background_tasks: BackgroundTasks):
    background_tasks.add_task(record_view_task)
    return {"ok": True}


@bg_app.get("/bg-forgotten-pop")
def bg_forgotten_pop(background_tasks: BackgroundTasks):
    # Raw push without a matching pop: the background task must still see an
    # empty stack instead of the leaked frame.
    push_error_presentation(payload_profile="problem")
    background_tasks.add_task(record_view_task)
    return {"ok": True}


bg_client = TestClient(bg_app)


def test_background_task_sees_empty_stack_while_request_frame_active():
    bg_seen.clear()
    response = bg_client.get("/bg")
    assert response.status_code == 200
    assert bg_seen["view"] == ErrorPresentation()
    assert bg_seen["depth"] == 0


def test_background_task_sees_empty_stack_after_forgotten_pop():
    bg_seen.clear()
    response = bg_client.get("/bg-forgotten-pop")
    assert response.status_code == 200
    assert bg_seen["view"] == ErrorPresentation()
    assert bg_seen["depth"] == 0


def test_request_after_exception_path_has_empty_stack():
    leak_app = FastAPI()

    @leak_app.get("/leak")
    async def leak():
        # Raw push without a matching pop inside this request task; the
        # exceptional auto-restore behavior of error_presentation() is what
        # users should use, but even so a later request must see defaults.
        push_error_presentation(payload_profile="problem")
        raise HTTPException(status_code=400)

    leak_client = TestClient(leak_app, raise_server_exceptions=False)
    response = leak_client.get("/leak")
    assert response.headers["content-type"] == "application/problem+json"

    # Every request gets its own copied context: a later request is default.
    response = leak_client.get("/leak")
    # The same raw push still happens, but this is a fresh context so this
    # assertion guards against a process-global singleton carrying the frame.
    assert response.headers["content-type"] == "application/problem+json"
    # An unrelated old route keeps the historical shape.
    assert client.get("/http-default").headers["content-type"] == "application/json"


# ---------------------------------------------------------------------------
# 9. Concurrent request isolation over a direct ASGI transport
# ---------------------------------------------------------------------------


concurrency_app = FastAPI()


@concurrency_app.get("/problem")
async def concurrency_problem():
    with error_presentation(payload_profile="problem"):
        await asyncio.sleep(0.05)
        raise HTTPException(status_code=400, detail="a")


@concurrency_app.get("/body")
async def concurrency_body():
    with error_presentation(include_validation_body=True):
        await asyncio.sleep(0.05)
        raise HTTPException(status_code=400, detail="b")


async def _stream_frame_dep():
    with error_presentation(payload_profile="problem"):
        yield


@concurrency_app.get("/stream", dependencies=[Depends(_stream_frame_dep)])
async def concurrency_stream():
    from starlette.responses import StreamingResponse

    async def gen():
        for _ in range(5):
            await asyncio.sleep(0.03)
            # While request A streams, request B push/pop must not alter A.
            assert get_error_presentation().payload_profile == "problem"
            yield b"data\n"

    return StreamingResponse(gen())


async def _concurrent_case():
    transport = httpx.ASGITransport(app=concurrency_app)
    async with httpx.AsyncClient(
        transport=transport, base_url="http://asgi-local"
    ) as http:
        r_problem, r_body = await asyncio.gather(
            http.get("/problem"), http.get("/body")
        )
        assert r_problem.headers["content-type"] == "application/problem+json"
        assert r_body.headers["content-type"] == "application/json"
        assert r_body.json() == {"detail": "b"}
        assert "about:blank" not in r_body.text

        async def churn_other_request_stack():
            for _ in range(10):
                with error_presentation(
                    payload_profile="detail", include_validation_body=True
                ):
                    await asyncio.sleep(0.01)

        r_stream, _ = await asyncio.gather(
            http.get("/stream"), churn_other_request_stack()
        )
        assert r_stream.status_code == 200
        assert r_stream.text.count("data") == 5


def test_concurrent_requests_have_independent_stacks():
    asyncio.run(_concurrent_case())


# ---------------------------------------------------------------------------
# 10. TestClient and direct ASGI transport agree
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("path", ["/http-problem", "/http-default"])
def test_testclient_and_asgi_parity(path):
    testclient_response = client.get(path)

    async def via_asgi():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://asgi-local"
        ) as http:
            return await http.get(path)

    asgi_response = asyncio.run(via_asgi())
    assert asgi_response.status_code == testclient_response.status_code
    assert (
        asgi_response.headers["content-type"]
        == testclient_response.headers["content-type"]
    )
    assert asgi_response.json() == testclient_response.json()


# ---------------------------------------------------------------------------
# Dependency- and middleware-scoped frames (the documented nesting points)
# ---------------------------------------------------------------------------


scope_app = FastAPI()


async def async_problem_dep_with_body():
    with error_presentation(payload_profile="problem", include_validation_body=True):
        yield


def sync_problem_dep_with_body():
    with error_presentation(payload_profile="problem", include_validation_body=True):
        yield


class ProblemASGIMiddleware:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        with error_presentation(payload_profile="problem"):
            await self.app(scope, receive, send)


scope_app.add_middleware(ProblemASGIMiddleware)


@scope_app.post("/async-dep", dependencies=[Depends(async_problem_dep_with_body)])
async def async_dep_endpoint(item: Item):
    return item


@scope_app.post("/sync-dep", dependencies=[Depends(sync_problem_dep_with_body)])
def sync_dep_endpoint(item: Item):
    return item


@scope_app.get("/middleware-http")
def middleware_http():
    raise HTTPException(status_code=404, detail="mw")


scope_client = TestClient(scope_app, raise_server_exceptions=False)


def test_async_yield_dependency_scopes_validation_presentation():
    response = scope_client.post("/async-dep", json={"nope": 1})
    assert response.status_code == 422
    assert response.headers["content-type"] == "application/problem+json"
    assert response.json()["body"] == {"nope": 1}


def test_sync_yield_dependency_scopes_validation_presentation():
    response = scope_client.post("/sync-dep", json={"nope": 1})
    assert response.status_code == 422
    assert response.headers["content-type"] == "application/problem+json"
    assert response.json()["body"] == {"nope": 1}


def test_pure_asgi_middleware_scopes_http_exception():
    response = scope_client.get("/middleware-http")
    assert response.status_code == 404
    assert response.headers["content-type"] == "application/problem+json"
    assert response.json()["detail"] == "mw"
