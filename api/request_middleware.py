
import logging
import time
import uuid

from fastapi import Request

from rag.observability import current_request_id

logger = logging.getLogger("finsight.api")


async def request_tracking_middleware(request: Request, call_next):
    request_id = str(uuid.uuid4())
    start_time = time.perf_counter()

    request.state.request_id = request_id

    # Make the ID visible to LLM logging for this request.
    token = current_request_id.set(request_id)

    try:
        response = await call_next(request)
    except Exception:
        logger.exception(
            "Request failed | id=%s | method=%s | path=%s",
            request_id,
            request.method,
            request.url.path,
        )
        raise
    finally:
        current_request_id.reset(token)

    duration_ms = round(
        (time.perf_counter() - start_time) * 1000,
        2,
    )

    response.headers["X-Request-ID"] = request_id

    logger.info(
        "Request completed | id=%s | method=%s | path=%s | status=%s | duration_ms=%s",
        request_id,
        request.method,
        request.url.path,
        response.status_code,
        duration_ms,
    )

    return response