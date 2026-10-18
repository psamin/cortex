"""Closing a robot session is best effort: a worker that exits on its own mid-close must not fail the collection."""

import asyncio
from types import SimpleNamespace

from cortex.services import robot_session_service as rs


class _Stdin:
    def write(self, b):
        pass

    async def drain(self):
        pass


class _Proc:
    """Doesn't exit within the wait, then is already gone when killed (the Windows ProcessLookupError case)."""

    returncode = None
    stdin = _Stdin()

    async def wait(self):
        await asyncio.sleep(10)

    def kill(self):
        raise ProcessLookupError


def test_close_survives_a_worker_that_exited_before_the_kill(monkeypatch):
    monkeypatch.setattr(rs, "CLOSE_TIMEOUT_S", 0.05)
    rs.SESSIONS["rs_test"] = SimpleNamespace(proc=_Proc())
    asyncio.run(rs.close("rs_test"))  # must not raise
    assert "rs_test" not in rs.SESSIONS
