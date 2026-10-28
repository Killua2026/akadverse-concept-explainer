"""Launch on the reserved service port with a psycopg-compatible loop."""

import asyncio
import sys

import uvicorn


if __name__ == "__main__":
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    uvicorn.run("main:app", host="127.0.0.1", port=8010, loop="asyncio")
