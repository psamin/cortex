"""Local runner for a hosted Cortex MCP server.

A Manufact-hosted MCP server cannot reach ``localhost:8000``. Instead of exposing
the backend through a tunnel, this runner dials OUT to the hosted server and
long-polls it for tool calls (``mcp/src/relay.ts``), forwards each one to the
local backend's ``POST /api/tools/{name}`` and posts the response back. Nothing
listens on a public port, and the hosted server's URL never changes.

    uv run python -m cortex.mcp_runner --server https://<slug>.run.mcp-use.com

The link is authenticated with CORTEX_MCP_KEY, which must be set to the same
value here (``.env`` or env) and on the hosted server.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from typing import Any
from urllib.parse import quote

import httpx
from pydantic_settings import BaseSettings, SettingsConfigDict

from .config import ROOT

log = logging.getLogger("cortex.mcp_runner")

DEFAULT_API = "http://127.0.0.1:8000/api"
POLL_TIMEOUT_S = 40.0  # the server holds a poll for ~25 s
TOOL_TIMEOUT_S = 960.0  # backend wait_s tops out at 900 s
# only these headers from the MCP server reach the backend; never Authorization or Host
FORWARD_PREFIXES = ("x-cortex-",)
FORWARD_EXACT = {"content-type", "accept"}
# read-only GETs the hosted server may relay (MCP resources, stage hints); nothing that writes
READ_PREFIXES = ("/projects", "/evaluations", "/jobs")


class RunnerSettings(BaseSettings):
    """Read from env or the repo-root .env, like the backend's own settings."""

    model_config = SettingsConfigDict(env_file=(ROOT / ".env",), extra="ignore")
    cortex_mcp_key: str | None = None
    cortex_mcp_server_url: str | None = None
    cortex_runner_api_url: str = DEFAULT_API


class RunnerError(RuntimeError):
    """The link cannot work (wrong key, relay disabled); retrying would not help."""


def server_base(url: str) -> str:
    """Accept the server's root or its /mcp endpoint URL."""
    url = url.rstrip("/")
    return url[: -len("/mcp")] if url.endswith("/mcp") else url


def read_path_ok(path: Any) -> bool:
    """A relayed GET may only read: a plain path under READ_PREFIXES, no traversal, no other host."""
    if not isinstance(path, str) or not path.startswith("/") or path.startswith("//"):
        return False
    route = path.split("?", 1)[0]
    if ".." in route or "\\" in route or "%" in route:
        return False
    return any(route == p or route.startswith(p + "/") for p in READ_PREFIXES)


def forward_headers(headers: Any) -> dict[str, str]:
    if not isinstance(headers, dict):
        return {}
    out = {}
    for k, v in headers.items():
        lk = str(k).lower()
        if isinstance(v, str) and (lk in FORWARD_EXACT or lk.startswith(FORWARD_PREFIXES)):
            out[lk] = v
    return out


class Runner:
    def __init__(self, server: str, key: str, api: str = DEFAULT_API, client: httpx.AsyncClient | None = None):
        if not key:
            raise RunnerError("CORTEX_MCP_KEY is not set: put the hosted server's key in .env or pass --key")
        self.server = server_base(server)
        self.api = api.rstrip("/")
        self.auth = {"Authorization": f"Bearer {key}"}
        self.client = client or httpx.AsyncClient()
        self.connected = False
        self._tasks: set[asyncio.Task] = set()

    async def poll_once(self) -> list[dict[str, Any]]:
        """One long-poll. Returns the calls the server handed over (possibly none)."""
        r = await self.client.post(f"{self.server}/relay/poll", headers=self.auth, timeout=POLL_TIMEOUT_S)
        if r.status_code == 401:
            raise RunnerError(f"{self.server} rejected the runner key (401): CORTEX_MCP_KEY differs from the server's")
        if r.status_code == 503:
            raise RunnerError(f"{self.server} has the runner relay disabled: set CORTEX_MCP_KEY on the MCP server")
        if r.status_code == 404:
            raise RunnerError(f"{self.server}/relay/poll not found: is this a Cortex MCP server with the relay?")
        r.raise_for_status()
        if not self.connected:
            self.connected = True
            log.info("connected to %s, forwarding tool calls to %s", self.server, self.api)
        if r.status_code == 204 or not r.content:
            return []
        calls = r.json().get("calls", [])
        return [c for c in calls if isinstance(c, dict) and isinstance(c.get("id"), str) and isinstance(c.get("tool"), str)]

    async def handle(self, call: dict[str, Any]) -> None:
        """Forward one call to the local backend and post the result back."""
        tool, args, path = call["tool"], call.get("args") or {}, call.get("path")
        try:
            if path is not None:
                if not read_path_ok(path):
                    result = {"id": call["id"], "status": 403,
                              "body": json.dumps({"detail": f"the runner only relays reads under {READ_PREFIXES}"})}
                else:
                    r = await self.client.get(f"{self.api}{path}", headers=forward_headers(call.get("headers")),
                                              timeout=60.0)
                    result = {"id": call["id"], "status": r.status_code, "body": r.text}
                tool = f"GET {path}"
            else:
                r = await self.client.post(f"{self.api}/tools/{quote(tool, safe='')}", json=args,
                                           headers=forward_headers(call.get("headers")), timeout=TOOL_TIMEOUT_S)
                result = {"id": call["id"], "status": r.status_code, "body": r.text}
        except httpx.HTTPError as e:
            detail = f"the Cortex runner could not reach the backend at {self.api} ({type(e).__name__}: {e})"
            result = {"id": call["id"], "status": 502, "body": json.dumps({"detail": detail})}
        log.info("%s -> %s", tool if path is not None else f"cortex.{tool}", result["status"])
        try:
            await self.client.post(f"{self.server}/relay/result", json=result, headers=self.auth, timeout=30.0)
        except httpx.HTTPError as e:
            log.warning("could not return the result of cortex.%s: %s", tool, e)

    def _spawn(self, call: dict[str, Any]) -> None:
        t = asyncio.create_task(self.handle(call))
        self._tasks.add(t)
        t.add_done_callback(self._tasks.discard)

    async def run(self) -> None:
        """Poll forever; calls run concurrently. Network errors back off, auth errors raise."""
        delay = 1.0
        while True:
            try:
                for call in await self.poll_once():
                    self._spawn(call)
                delay = 1.0
            except RunnerError:
                raise
            except (httpx.HTTPError, ValueError) as e:
                if self.connected:
                    log.warning("lost the link to %s (%s); retrying", self.server, e)
                    self.connected = False
                else:
                    log.warning("cannot reach %s (%s); retrying in %.0f s", self.server, e, delay)
                await asyncio.sleep(delay)
                delay = min(delay * 2, 30.0)


def main(argv: list[str] | None = None) -> int:
    cfg = RunnerSettings()
    p = argparse.ArgumentParser(prog="python -m cortex.mcp_runner", description=__doc__.split("\n\n")[0])
    p.add_argument("--server", default=cfg.cortex_mcp_server_url,
                   help="hosted MCP server URL, e.g. https://<slug>.run.mcp-use.com (env CORTEX_MCP_SERVER_URL)")
    p.add_argument("--api", default=cfg.cortex_runner_api_url,
                   help=f"local backend API base (default {DEFAULT_API})")
    p.add_argument("--key", default=None, help="shared secret (default: CORTEX_MCP_KEY from env or .env)")
    a = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s runner %(message)s", datefmt="%H:%M:%S")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    if not a.server:
        p.error("--server (or CORTEX_MCP_SERVER_URL) is required")
    try:
        asyncio.run(Runner(a.server, a.key or cfg.cortex_mcp_key or "", a.api).run())
    except RunnerError as e:
        log.error("%s", e)
        return 2
    except KeyboardInterrupt:
        return 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
