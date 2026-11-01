"""Launch on the reserved service port with a psycopg-compatible loop."""

import asyncio
import os
import sys

import uvicorn


def _bind() -> tuple[str, int]:
    port_env = os.getenv("PORT")
    if port_env:
        return "0.0.0.0", int(port_env)
    return "127.0.0.1", 8010


if __name__ == "__main__":
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    host, port = _bind()
    uvicorn.run("main:app", host=host, port=port, loop="asyncio", reload=False)
