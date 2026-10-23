from http import HTTPStatus

from fastapi.encoders import jsonable_encoder
from fastapi.error_presentation import ErrorPresentation, resolve_error_presentation
from fastapi.exceptions import (
    RequestValidationError,
    ResponseValidationError,
    ValidationException,
    WebSocketRequestValidationError,
)
from fastapi.utils import is_body_allowed_for_status_code
from fastapi.websockets import WebSocket
from starlette.exceptions import HTTPException
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.status import WS_1008_POLICY_VIOLATION

HTTP_422_UNPROCESSABLE_ENTITY = 422
HTTP_500_INTERNAL_SERVER_ERROR = 500

PROBLEM_CONTENT_TYPE = "application/problem+json"
PROBLEM_DEFAULT_TYPE = "about:blank"


def _status_phrase(status_code: int) -> str:
    try:
        return HTTPStatus(status_code).phrase
    except ValueError:
        return ""


def _problem_response(
    *,
    status_code: int,
    content: dict[str, object],
    headers: dict[str, str] | None = None,
) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content=content,
        headers=headers,
        media_type=PROBLEM_CONTENT_TYPE,
    )


def _validation_problem_content(
    *,
    status_code: int,
    title: str,
    errors: object,
) -> dict[str, object]:
    return {
        "type": PROBLEM_DEFAULT_TYPE,
        "title": title,
        "status": status_code,
        "errors": errors,
    }


def _validation_extra_content(
    exc: ValidationException,
    presentation: ErrorPresentation,
) -> dict[str, object]:
    extra: dict[str, object] = {}
    if presentation.include_validation_body:
        extra["body"] = jsonable_encoder(getattr(exc, "body", None))
    if presentation.include_endpoint_ctx:
        extra["endpoint_ctx"] = jsonable_encoder(exc.endpoint_ctx)
    return extra


# ----------------------------------------------------------------------------
# HTTP exit: HTTPException
# ----------------------------------------------------------------------------


async def http_exception_handler(request: Request, exc: HTTPException) -> Response:
    headers = getattr(exc, "headers", None)
    if not is_body_allowed_for_status_code(exc.status_code):
        # The no-body rule is independent of the presentation stack.
        return Response(status_code=exc.status_code, headers=headers)
    presentation = resolve_error_presentation(exc)
    if presentation.payload_profile == "problem":
        content: dict[str, object] = {
            "type": PROBLEM_DEFAULT_TYPE,
            "title": _status_phrase(exc.status_code),
            "status": exc.status_code,
            "detail": exc.detail,
        }
        return _problem_response(
            status_code=exc.status_code, content=content, headers=headers
        )
    return JSONResponse(
        {"detail": exc.detail}, status_code=exc.status_code, headers=headers
    )


# ----------------------------------------------------------------------------
# HTTP exit: RequestValidationError (fixed 422)
# ----------------------------------------------------------------------------


async def request_validation_exception_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    presentation = resolve_error_presentation(exc)
    errors = jsonable_encoder(exc.errors())
    extra = _validation_extra_content(exc, presentation)
    if presentation.payload_profile == "problem":
        content = _validation_problem_content(
            status_code=HTTP_422_UNPROCESSABLE_ENTITY,
            title=_status_phrase(HTTP_422_UNPROCESSABLE_ENTITY),
            errors=errors,
        )
        content.update(extra)
        return _problem_response(
            status_code=HTTP_422_UNPROCESSABLE_ENTITY, content=content
        )
    content = {"detail": errors}
    content.update(extra)
    return JSONResponse(status_code=HTTP_422_UNPROCESSABLE_ENTITY, content=content)


# ----------------------------------------------------------------------------
# HTTP exit: ResponseValidationError (500, stack-aware default path)
# ----------------------------------------------------------------------------


async def response_validation_exception_handler(
    request: Request, exc: ResponseValidationError
) -> JSONResponse:
    # Called from routing before the response head is sent.  Response
    # validation failures raised while a streaming body is produced happen
    # too late to be rendered and are not dispatched here.
    presentation = resolve_error_presentation(exc)
    errors = jsonable_encoder(exc.errors())
    extra = _validation_extra_content(exc, presentation)
    if presentation.payload_profile == "problem":
        content = _validation_problem_content(
            status_code=HTTP_500_INTERNAL_SERVER_ERROR,
            title=_status_phrase(HTTP_500_INTERNAL_SERVER_ERROR),
            errors=errors,
        )
        content.update(extra)
        return _problem_response(
            status_code=HTTP_500_INTERNAL_SERVER_ERROR, content=content
        )
    content = {"detail": errors}
    content.update(extra)
    return JSONResponse(status_code=HTTP_500_INTERNAL_SERVER_ERROR, content=content)


# ----------------------------------------------------------------------------
# WebSocket exit: WebSocketRequestValidationError
#
# Only the WebSocket close semantics (and websocket_reason_mode) apply here.
# There is no HTTP response, so no Content-Type and no problem+json media type
# is ever assembled for this exit.
# ----------------------------------------------------------------------------


async def websocket_request_validation_exception_handler(
    websocket: WebSocket, exc: WebSocketRequestValidationError
) -> None:
    presentation = resolve_error_presentation(exc)
    if presentation.websocket_reason_mode == "json_errors":
        # Historical behavior: the encoded error list rides in the close reason.
        reason = jsonable_encoder(exc.errors())
    else:  # "omit": do not stuff errors into the close reason.
        reason = None
    await websocket.close(code=WS_1008_POLICY_VIOLATION, reason=reason)
