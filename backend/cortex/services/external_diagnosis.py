"""Failure diagnosis by the connected MCP client's own model ("bring your own model").

When Cortex has no model route of its own (no OpenRouter key, the InsForge gateway refuses), the
MCP client can lend its model instead. Two backend tools carry it:

  get_diagnosis_inputs       runs the measured part of an analysis (quality checks, simulator
                             failures), picks the same keyframe targets the built-in VLM route
                             would review and stores the analysis as `running`, one pending
                             diagnosis per target. Returns each target's contact sheet, measured
                             telemetry and the exact vlm_diagnostics prompt.
  submit_failure_hypotheses  the client's answers for that analysis: raw model replies (parsed
                             here) or structured hypotheses. Every answer is validated against
                             the fixed taxonomy; one invalid answer rejects the whole submission
                             and nothing is stored. Accepted answers become unverified VLM
                             hypotheses that record the route and the model the client reported,
                             and the failure clusters are rebuilt as for any other VLM route.

Routes: "mcp_sampling" (the client's model answered a sampling/createMessage request from the
Cortex MCP server) or "mcp_agent" (the connected agent looked at the images itself). The model
name is whatever the client reported; Cortex cannot verify it.
"""

from __future__ import annotations

import base64
import json
import re
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from ..context import Ctx
from ..errors import UserError
from ..models import Analysis, EpisodeDiagnosis, Hypothesis, Project
from . import analysis_service
from . import recording_service as rec
from . import vlm_diagnostics as vlm

N_KEYFRAMES = 6  # keyframes_from_video / keyframes_from_array default
PENDING = "awaiting an answer from the MCP client's model"
UNANSWERED = "the MCP client submitted no answer for this target"
ROUTE_LABEL = {"mcp_sampling": "MCP sampling", "mcp_agent": "MCP agent"}


def taxonomy_text() -> str:
    return "\n".join(f"- {k}: {v}" for k, v in vlm.TAXONOMY.items())


def prompt_for(task: str, telemetry: dict[str, Any], n: int = N_KEYFRAMES) -> str:
    """The same prompt vlm_diagnostics sends to a model route."""
    return vlm.PROMPT.format(task=task, n=n, telemetry=json.dumps(telemetry), taxonomy=taxonomy_text())


# ------------------------------------------------------------------ validation

class SubmittedHypothesis(BaseModel):
    category: str = Field(description="one of the fixed failure taxonomy categories")
    confidence: float = Field(ge=0, le=1, description="the model's own confidence, 0-1 (not a probability)")
    rationale: str = Field(min_length=1, max_length=300)
    frame_indices: list[int] = Field(default_factory=list, max_length=N_KEYFRAMES,
                                     description=f"keyframes (0-{N_KEYFRAMES - 1}) that show it")

    @field_validator("category")
    @classmethod
    def _in_taxonomy(cls, v: str) -> str:
        if v not in vlm.TAXONOMY:
            raise ValueError(f"'{v}' is not in the failure taxonomy ({', '.join(vlm.TAXONOMY)})")
        return v

    @field_validator("frame_indices")
    @classmethod
    def _in_range(cls, v: list[int]) -> list[int]:
        bad = [i for i in v if not 0 <= i < N_KEYFRAMES]
        if bad:
            raise ValueError(f"frame indices {bad} are outside the {N_KEYFRAMES} keyframes (0-{N_KEYFRAMES - 1})")
        return v


class ModelReply(BaseModel):
    """What the prompt asks the model to return."""

    hypotheses: list[SubmittedHypothesis] = Field(min_length=1, max_length=3)


class TargetAnswer(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ref_id: str = Field(description="target ref_id from get_diagnosis_inputs")
    reply_text: str | None = Field(None, max_length=8000, description="the model's raw reply; parsed and validated "
                                                                       "by Cortex")
    hypotheses: list[SubmittedHypothesis] | None = Field(None, min_length=1, max_length=3,
                                                         description="structured hypotheses (instead of reply_text)")
    error: str | None = Field(None, max_length=300, description="the client could not get an answer (declined, "
                                                                "failed); recorded, never replaced by a guess")

    @model_validator(mode="after")
    def _exactly_one(self) -> TargetAnswer:
        if sum(x is not None for x in (self.reply_text, self.hypotheses, self.error)) != 1:
            raise ValueError("give exactly one of reply_text, hypotheses or error")
        return self


class DiagnosisInputsArgs(BaseModel):
    project_id: str
    max_targets: int = Field(4, ge=1, le=6, description="episodes to review (failed evaluation rollouts first)")


class SubmitArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    project_id: str
    analysis_id: str = Field(description="analysis_id from get_diagnosis_inputs")
    route: Literal["mcp_sampling", "mcp_agent"] = Field(
        description="'mcp_sampling' when the answers came from a sampling/createMessage request, 'mcp_agent' when "
                    "you (the agent) looked at the contact sheets yourself")
    model: str = Field(min_length=1, max_length=120, description="the model that produced the answers, as reported "
                                                                 "by the MCP client (sampling result `model`)")
    client: str | None = Field(None, max_length=120, description="MCP client name/version, if known")
    answers: list[TargetAnswer] = Field(min_length=1, max_length=6)


def parse_reply(text: str) -> list[SubmittedHypothesis]:
    """Strict counterpart of vlm.parse_hypotheses: anything off-taxonomy or malformed raises."""
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        raise ValueError("no JSON object in the model reply")
    try:
        data = json.loads(m.group(0))
    except json.JSONDecodeError as e:
        raise ValueError(f"the model reply is not valid JSON ({e.msg})") from None
    return ModelReply.model_validate(data).hypotheses


def _errors(e: ValidationError) -> str:
    return "; ".join(f"{'.'.join(map(str, x['loc'])) or '(reply)'}: {x['msg']}" for x in e.errors())


# ------------------------------------------------------------------ the two steps

async def get_inputs(ctx: Ctx, project: Project, max_targets: int, actor: str) -> dict[str, Any]:
    an = Analysis(project_id=project.id, vlm_provider="MCP client model (awaiting answers)")
    await ctx.repo.save("analyses", an)
    eps, failures = await analysis_service.measure(ctx, an)
    targets = await analysis_service.vlm_targets(ctx, eps, failures, max_targets)
    if not targets:
        an.vlm_provider = None
        an.summary = "No failed or flagged episodes to review. "
        await analysis_service.finish(ctx, an, eps, failures, actor, "get_diagnosis_inputs",
                                      {"max_targets": max_targets})
        return {"analysis_id": an.id, "targets": [], "summary": an.summary + "Analysis completed without VLM review."}

    an.diagnoses = [EpisodeDiagnosis(episode_id=ref, model="(pending)", hypotheses=[], error=PENDING)
                    for ref, _, _ in targets]
    await ctx.repo.save("analyses", an)
    ctx.changed(project.id, "analyses", an.id)
    await ctx.audit(actor, "get_diagnosis_inputs", project.id, {"max_targets": max_targets}, "ok",
                    f"{an.id}: {len(targets)} targets awaiting the client's model")
    return {
        "analysis_id": an.id,
        "task": project.spec.goal,
        "taxonomy": vlm.TAXONOMY,
        "targets": [{"ref_id": ref, "telemetry": tel, "prompt": prompt_for(project.spec.goal, tel, len(frames)),
                     "image_b64": base64.b64encode(vlm.contact_sheet(frames)).decode(), "mime_type": "image/jpeg"}
                    for ref, frames, tel in targets],
        "summary": f"Analysis {an.id}: {len(targets)} episodes to review ({len(failures)} failed evaluation "
                   f"rollouts measured). Answer each target's prompt, then call submit_failure_hypotheses.",
    }


async def submit(ctx: Ctx, a: SubmitArgs, actor: str) -> dict[str, Any]:
    an = await ctx.repo.get("analyses", a.analysis_id, Analysis)
    if an is None or an.project_id != a.project_id:
        raise KeyError(a.analysis_id)
    pending = {d.episode_id for d in an.diagnoses if d.error == PENDING}
    if an.status != "running" or not pending:
        raise UserError(f"analysis {an.id} is not waiting for hypotheses (status {an.status}); "
                        "call get_diagnosis_inputs for a new one")

    problems: list[str] = []
    accepted: dict[str, EpisodeDiagnosis] = {}
    for ans in a.answers:
        if ans.ref_id not in pending:
            problems.append(f"{ans.ref_id}: not a target of {an.id} (targets: {', '.join(sorted(pending))})")
            continue
        if ans.ref_id in accepted:
            problems.append(f"{ans.ref_id}: answered twice")
            continue
        try:
            hyps = parse_reply(ans.reply_text) if ans.reply_text is not None else ans.hypotheses or []
        except ValidationError as e:
            problems.append(f"{ans.ref_id}: {_errors(e)}")
            continue
        except ValueError as e:
            problems.append(f"{ans.ref_id}: {e}")
            continue
        accepted[ans.ref_id] = EpisodeDiagnosis(
            episode_id=ans.ref_id, model=a.model, route=a.route, client=a.client, error=ans.error,
            hypotheses=[Hypothesis(**h.model_dump()) for h in hyps])
    if problems:
        await ctx.audit(actor, "submit_failure_hypotheses", a.project_id,
                        {"analysis_id": an.id, "route": a.route, "model": a.model}, "refused", "; ".join(problems)[:300])
        raise UserError("Hypotheses rejected, nothing was stored: " + " | ".join(problems))

    an.diagnoses = [accepted.get(d.episode_id) or EpisodeDiagnosis(
        episode_id=d.episode_id, model=a.model, route=a.route, client=a.client, hypotheses=[], error=UNANSWERED)
        if d.error == PENDING else d for d in an.diagnoses]
    an.vlm_provider = f"{ROUTE_LABEL[a.route]}: {a.model} (client-reported)"
    eps = await rec.list_episodes(ctx, a.project_id)
    failures = await analysis_service.measured_failures(ctx, a.project_id)
    await analysis_service.finish(ctx, an, eps, failures, actor, "submit_failure_hypotheses",
                                  {"analysis_id": an.id, "route": a.route, "model": a.model, "client": a.client})
    clusters = [c for c in await analysis_service.current_clusters(ctx, a.project_id) if c.source == "vlm"]
    return {"summary": f"Stored {sum(1 for d in accepted.values() if d.hypotheses)} unverified hypotheses from "
                       f"{a.model} via {a.route}. {an.summary}",
            "ref_ids": [an.id] + [c.id for c in clusters], "analysis_id": an.id,
            "diagnoses": [d.model_dump(mode="json") for d in an.diagnoses],
            "vlm_clusters": [c.model_dump(mode="json") for c in clusters],
            "note": "VLM hypotheses are unverified until a human confirms them in the studio"}


# ------------------------------------------------------------------ backend tools (registered in agent_orchestrator)

async def t_get_inputs(ctx, a: DiagnosisInputsArgs, actor):
    from .agent_orchestrator import get_project

    return await get_inputs(ctx, await get_project(ctx, a.project_id), a.max_targets, actor)


async def t_submit(ctx, a: SubmitArgs, actor):
    return await submit(ctx, a, actor)


TOOLS = [
    ("get_diagnosis_inputs",
     "Bring your own model: run quality checks and measured failure analysis, then return the keyframe contact "
     "sheets, measured telemetry and fixed-taxonomy prompt for the episodes a VLM should review. Answer them with "
     "your model and call submit_failure_hypotheses with the analysis_id.",
     DiagnosisInputsArgs, t_get_inputs),
    ("submit_failure_hypotheses",
     "Store your model's failure hypotheses for an analysis from get_diagnosis_inputs. Validated against the "
     "fixed taxonomy (any invalid answer rejects the submission); stored as unverified VLM hypotheses with the "
     "route and the model you report, then failure clusters are rebuilt.",
     SubmitArgs, t_submit),
]
