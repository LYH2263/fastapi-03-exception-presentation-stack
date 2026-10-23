"""Request-scoped stack of exception error presentation strategies.

The stack lives in a :class:`contextvars.ContextVar`, so concurrent HTTP
requests (and concurrent WebSocket connections) each see an independent stack
even when they enter the framework through the same shared application and
middlewares.

A frame only overrides the switches it sets explicitly; the read interface
(:func:`get_error_presentation`) merges defaults, outer frames and the stack
top into a complete :class:`ErrorPresentation` view.  When the stack is empty
the view is exactly the default presentation, i.e. what the built-in default
exception handlers produced before this module existed.

The :func:`error_presentation` context manager pushes on entry and pops on
exit, including exceptional exits.  Exceptions raised while a frame is active
carry a frozen snapshot of the merged view with them, because by the time an
exception handler runs the ``with`` block has already unwound and popped its
frame.  The snapshot is only taken by the innermost frame an exception crosses,
so it always contains the fullest view that was active at the raise site.
"""

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar, Token
from dataclasses import dataclass
from typing import Literal

from starlette.background import BackgroundTask

PayloadProfile = Literal["detail", "problem"]
"""
- ``"detail"``: keep the historical ``{"detail": ...}`` JSON shape.
- ``"problem"``: use an ``application/problem+json`` envelope (RFC 9457-ish)
  with at least ``type``, ``title``, ``status`` and ``detail``/``errors``.
"""

WebSocketReasonMode = Literal["json_errors", "omit"]
"""
- ``"json_errors"``: the WebSocket close reason still carries the encoded
  validation error list (the historical behavior).
- ``"omit"``: the close reason does not carry the error list, an empty reason
  is sent instead.
"""

_PAYLOAD_PROFILES = ("detail", "problem")
_WEBSOCKET_REASON_MODES = ("json_errors", "omit")


@dataclass(frozen=True)
class ErrorPresentation:
    """Merged, read-only view of the error presentation stack top."""

    include_validation_body: bool = False
    include_endpoint_ctx: bool = False
    payload_profile: PayloadProfile = "detail"
    websocket_reason_mode: WebSocketReasonMode = "json_errors"


_DEFAULT_PRESENTATION = ErrorPresentation()

# Each frame is a mapping of only the keys explicitly overridden by that push.
_Frame = Mapping[str, object]

# ``None`` means "no frame was ever pushed in this context".  The value is a
# tuple of immutable frame mappings, rebuilt on every push/pop so that one
# request can never mutate another request's stack.
_error_presentation_stack: ContextVar[tuple[_Frame, ...] | None] = ContextVar(
    "fastapi_error_presentation_stack", default=None
)

# Attribute used to carry the merged view on exceptions that propagate through
# an active ``error_presentation()`` block.
_PRESENTATION_SNAPSHOT_ATTR = "__fastapi_error_presentation__"


def _build_frame(
    *,
    include_validation_body: bool | None,
    include_endpoint_ctx: bool | None,
    payload_profile: PayloadProfile | None,
    websocket_reason_mode: WebSocketReasonMode | None,
) -> _Frame:
    frame: dict[str, object] = {}
    if include_validation_body is not None:
        if not isinstance(include_validation_body, bool):
            raise TypeError("include_validation_body must be a bool")
        frame["include_validation_body"] = include_validation_body
    if include_endpoint_ctx is not None:
        if not isinstance(include_endpoint_ctx, bool):
            raise TypeError("include_endpoint_ctx must be a bool")
        frame["include_endpoint_ctx"] = include_endpoint_ctx
    if payload_profile is not None:
        if payload_profile not in _PAYLOAD_PROFILES:
            raise ValueError(
                "payload_profile must be one of: " + ", ".join(_PAYLOAD_PROFILES)
            )
        frame["payload_profile"] = payload_profile
    if websocket_reason_mode is not None:
        if websocket_reason_mode not in _WEBSOCKET_REASON_MODES:
            raise ValueError(
                "websocket_reason_mode must be one of: "
                + ", ".join(_WEBSOCKET_REASON_MODES)
            )
        frame["websocket_reason_mode"] = websocket_reason_mode
    return frame


def _push_frame(frame: _Frame) -> Token[tuple[_Frame, ...] | None]:
    current = _error_presentation_stack.get()
    return _error_presentation_stack.set((current or ()) + (frame,))


def push_error_presentation(
    *,
    include_validation_body: bool | None = None,
    include_endpoint_ctx: bool | None = None,
    payload_profile: PayloadProfile | None = None,
    websocket_reason_mode: WebSocketReasonMode | None = None,
) -> Token[tuple[_Frame, ...] | None]:
    """Push a presentation frame onto the current request's stack.

    Every argument is optional; only the provided switches override the outer
    view.  Pair with :func:`pop_error_presentation` (or use
    :func:`error_presentation`) so the frame is always removed.
    """
    return _push_frame(
        _build_frame(
            include_validation_body=include_validation_body,
            include_endpoint_ctx=include_endpoint_ctx,
            payload_profile=payload_profile,
            websocket_reason_mode=websocket_reason_mode,
        )
    )


def pop_error_presentation(
    token: Token[tuple[_Frame, ...] | None],
) -> None:
    """Pop the frame pushed with ``token``.

    Resetting via the original token restores exactly the outer frames left by
    an outer push; it never falls back to a process-wide singleton or to the
    default literal.
    """
    try:
        _error_presentation_stack.reset(token)
    except ValueError as exc:
        # Sync (non-async) dependencies/endpoints run in worker threads whose
        # contextvars context is a throwaway copy: the push and the pop may
        # live in different copied contexts.  The frame never reached this
        # context, so there is nothing to reset (and nothing leaks, the copied
        # context is discarded).  Genuine same-context token reuse stays loud.
        if "different Context" not in str(exc):
            raise


@contextmanager
def error_presentation(
    *,
    include_validation_body: bool | None = None,
    include_endpoint_ctx: bool | None = None,
    payload_profile: PayloadProfile | None = None,
    websocket_reason_mode: WebSocketReasonMode | None = None,
) -> Iterator[None]:
    """Context manager that pushes a frame and always pops it again.

    Usable from endpoints, dependencies (including ``yield`` dependencies) and
    pure ASGI middlewares.  On the exceptional path the frame is still removed
    from the context variable, but the exception leaves the block carrying a
    snapshot of the merged view so the default exception handlers can render
    the error with the presentation that was active where it was raised.
    """
    frame = _build_frame(
        include_validation_body=include_validation_body,
        include_endpoint_ctx=include_endpoint_ctx,
        payload_profile=payload_profile,
        websocket_reason_mode=websocket_reason_mode,
    )
    token = _push_frame(frame)
    try:
        yield
    except BaseException as exc:
        _snapshot_on_exception(exc, frame)
        raise
    finally:
        pop_error_presentation(token)


def _merged_view(frames: tuple[_Frame, ...]) -> ErrorPresentation:
    values: dict[str, object] = {
        "include_validation_body": False,
        "include_endpoint_ctx": False,
        "payload_profile": "detail",
        "websocket_reason_mode": "json_errors",
    }
    for frame in frames:
        values.update(frame)
    return ErrorPresentation(**values)  # type: ignore[arg-type]


def get_error_presentation() -> ErrorPresentation:
    """Return the merged view of the current stack top.

    An empty stack returns the default view, byte-for-byte equivalent to the
    historical built-in handler output.
    """
    frames = _error_presentation_stack.get()
    if not frames:
        return _DEFAULT_PRESENTATION
    return _merged_view(frames)


def get_stack_depth() -> int:
    """Number of frames currently pushed in this context."""
    frames = _error_presentation_stack.get()
    return 0 if frames is None else len(frames)


def _snapshot_on_exception(exc: BaseException, own_frame: _Frame) -> None:
    # The innermost frame an exception crosses unwinds first and sees every
    # outer frame still active, so its merged view is the fullest one.  Outer
    # frames must not overwrite it.  ``own_frame`` is merged explicitly because
    # sync endpoints/dependencies run in copied worker-thread contexts where
    # the push may not be visible in this context.
    existing = None
    try:
        existing = getattr(exc, _PRESENTATION_SNAPSHOT_ATTR, None)
    except Exception:
        existing = None
    if isinstance(existing, ErrorPresentation):
        return
    visible = _error_presentation_stack.get()
    snapshot = _merged_view((visible or ()) + (own_frame,))
    try:
        setattr(exc, _PRESENTATION_SNAPSHOT_ATTR, snapshot)
    except Exception:
        # Some exception objects do not allow extra attributes; the context
        # variable itself still provides the view for handlers reached before
        # an explicit pop.
        pass


def resolve_error_presentation(exc: BaseException) -> ErrorPresentation:
    """Resolve the presentation for an exception about to be rendered.

    Prefers the snapshot captured when the exception crossed an active frame,
    otherwise reads the current stack top.
    """
    snapshot = None
    try:
        snapshot = getattr(exc, _PRESENTATION_SNAPSHOT_ATTR, None)
    except Exception:
        snapshot = None
    if isinstance(snapshot, ErrorPresentation):
        return snapshot
    return get_error_presentation()


class _EmptyErrorPresentationBackground(BackgroundTask):
    """Wrap response background tasks so they observe an empty stack.

    Background tasks run after the response has been handed to the client;
    they must not read frames left by the request, including frames that were
    pushed without a matching pop.
    """

    def __init__(self, background: BackgroundTask) -> None:
        # Intentionally do not call super().__init__: the wrapped task is
        # invoked directly and BackgroundTask's func/args are irrelevant.
        self._background = background

    async def __call__(self) -> None:
        token = _error_presentation_stack.set(None)
        try:
            await self._background()
        finally:
            _error_presentation_stack.reset(token)


def background_with_empty_error_presentation(
    background: BackgroundTask,
) -> BackgroundTask:
    """Wrap a response background task to run with an empty stack."""
    return _EmptyErrorPresentationBackground(background)
