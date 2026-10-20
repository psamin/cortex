"""#365: work cut off by a backend restart doesn't stay active forever."""

from cortex.config import Settings
from cortex.context import build_ctx
from cortex.models import Condition
from cortex.rl.proposals import propose
from cortex.rl.types import RLSandbox, Weakness
from cortex.services import rl_service


def _sandbox(status: str) -> RLSandbox:
    w = Weakness(key="shifted_object", branch_id="br_1", evaluation_id="ev_1", checkpoint_id="ck_1",
                 condition=Condition(), cube_offset=(0.02, 0.0), n=10, successes=3, success_rate=0.3,
                 ci95=(0.11, 0.6), target=0.8, gap=0.5, dominant_failure="grasp_failed", score=0.3)
    sb = propose("prj_1", [w])[0]
    sb.status = status  # type: ignore[assignment]
    return sb


async def test_rl_sandboxes_cut_off_by_a_restart_are_failed(tmp_path):
    ctx = await build_ctx(Settings(data_dir=tmp_path, _env_file=None))
    rows = {st: _sandbox(st) for st in ("piloting", "accepted", "running", "ready", "succeeded")}
    for sb in rows.values():
        await ctx.repo.save("rl_sandboxes", sb)
    ctx2 = await build_ctx(Settings(data_dir=tmp_path, _env_file=None))
    assert await rl_service.sweep_interrupted(ctx2) == 3
    after = {st: (await ctx2.repo.get("rl_sandboxes", sb.id, RLSandbox)) for st, sb in rows.items()}
    for st in ("piloting", "accepted", "running"):
        assert after[st].status == "failed" and "backend restart" in after[st].reason
    assert after["ready"].status == "ready" and after["succeeded"].status == "succeeded"
