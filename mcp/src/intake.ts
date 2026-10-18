import { randomBytes } from "node:crypto";
import { readFile, stat } from "node:fs/promises";
import { basename, extname } from "node:path";
import { createRequestStateCodec, inputRequired, inputResponse, type CallToolResult, type InputRequiredResult, type RequestContext } from "mcp-use";
import { callTool, identityOf } from "./backend.js";
import { z } from "zod";

// A new scene starts with the human's choice of where it comes from: a photo of their setup (read by Cortex's
// Claude Code), their own description, or the default tabletop. Asked as a form when the client can show one;
// otherwise the tool description tells the agent to ask. A photo path is read here, next to the human.
//
// Intake questions go to the human, not the model. start_intake / update_intake return typed questions
// ({path, type, options, patch, ...}); when the client can show a form (2026-07-28 `input_required` with the
// elicitation capability) the tool answers with an elicitation form built from those questions, turns the
// human's answers into update_intake patches, applies them, and repeats while required questions remain
// (later questions, e.g. "which zone is the target?", only get options once earlier answers exist).
// Clients that cannot show a form get the old result plus the questions spelled out for the model to ask.

type Json = Record<string, unknown>;
type Question = { path: string; type: string; prompt: string; why: string; options?: string[]; default?: unknown; required?: boolean; patch?: Json | null; option_patches?: Record<string, Json> };
/** How to turn one form field back into a patch; kept in the signed requestState between rounds. */
type Field = { key: string; path: string; kind: "enum" | "multi" | "confirm" | "accept" | "string" | "number" | "describe"; patch?: Json | null; option_patches?: Record<string, Json> };
/** Between rounds. `start_args`: the scene-source form is open and no project exists yet. */
type State = { project_id: string; round: number; fields: Field[]; start_args?: Json };

const MAX_ROUNDS = 4;
// One process serves every round on a single host; set CORTEX_MCP_STATE_KEY (>= 32 chars) when several
// replicas share the endpoint so any of them can verify the state.
const codec = createRequestStateCodec<State>({ key: process.env.CORTEX_MCP_STATE_KEY ?? randomBytes(32), ttlSeconds: 1800 });
/** For ServerConfig.requestState: verifies (HMAC + expiry) every echoed requestState before a handler runs. */
export const intakeRequestState = { verify: codec.verify };

const getPath = (o: unknown, path: string): unknown =>
  path.split(".").reduce<unknown>((v, k) => (v && typeof v === "object" ? (v as Json)[k] : undefined), o);
const nest = (path: string, value: unknown): Json =>
  path.split(".").reverse().reduce<unknown>((v, k) => ({ [k]: v }), value) as Json;

const LIST_KEYS = new Set(["objects", "zones", "cameras", "obstacles"]);
/** Merge patches the way the backend would apply them one after another (name-keyed lists concatenate). */
function merge(a: Json, b: Json): Json {
  const out: Json = { ...a };
  for (const [k, v] of Object.entries(b)) {
    const prev = out[k];
    if (LIST_KEYS.has(k) && Array.isArray(prev) && Array.isArray(v)) out[k] = [...prev, ...v];
    else if (prev && v && typeof prev === "object" && typeof v === "object" && !Array.isArray(prev) && !Array.isArray(v)) out[k] = merge(prev as Json, v as Json);
    else out[k] = v;
  }
  return out;
}

const fieldKey = (i: number, path: string) => `q${i}_${path.replace(/[^a-zA-Z0-9]+/g, "_")}`.slice(0, 60);

/** Map backend questions onto flat elicitation fields (the form schema only allows primitives and enum arrays). */
function buildForm(questions: Question[], round: number) {
  const properties: Record<string, Json> = {};
  const required: string[] = [];
  const fields: Field[] = [];
  questions.forEach((q, i) => {
    if (round > 1 && q.required === false) return; // optional questions are offered once
    const key = fieldKey(i, q.path);
    const title = q.path;
    const description = `${q.prompt} (${q.why})`;
    const suggested = q.patch ? getPath(q.patch, q.path) : undefined;
    const opts = q.options ?? [];
    let kind: Field["kind"];
    let schema: Json;
    if (q.type === "enum" && opts.length) {
      kind = "enum";
      const perOption = q.option_patches && Object.keys(q.option_patches).length > 0;
      const guess = perOption && q.patch ? opts.find((o) => JSON.stringify(q.option_patches?.[o]) === JSON.stringify(q.patch)) : (suggested ?? q.default);
      // Defaults must match the field type, or the client rejects the whole form.
      const dflt = typeof guess === "string" && opts.includes(guess) ? guess : undefined;
      schema = { type: "string", title, description, enum: opts, ...(dflt ? { default: dflt } : {}) };
    } else if (q.type === "list" && opts.length) {
      kind = "multi";
      // Suggested lists hold names (task.place) or {name} objects (objects): keep the names that are options.
      const dflt = Array.isArray(suggested)
        ? suggested.map((x) => (typeof x === "string" ? x : (x as Json | null)?.name)).filter((x): x is string => typeof x === "string" && opts.includes(x))
        : [];
      schema = { type: "array", title, description, items: { type: "string", enum: opts }, ...(dflt.length ? { default: dflt } : {}) };
    } else if (q.type === "confirm") {
      kind = "confirm";
      schema = { type: "boolean", title, description: `${description} Uncheck if any value is wrong, then tell the agent the correct one.`, default: true };
    } else if ((q.type === "string" || q.type === "number") && (q.patch || q.required)) {
      kind = q.type;
      schema = { type: q.type, title, description, ...(typeof suggested === q.type ? { default: suggested } : {}) };
    } else if (q.patch) {
      kind = "accept";
      schema = { type: "boolean", title: `Use suggested ${q.path}`, description: `${description} Suggested: ${JSON.stringify(q.patch)}`, default: true };
    } else {
      return; // no options and no suggestion yet (it depends on another answer): ask in a later round
    }
    properties[key] = schema;
    if (q.required !== false && kind !== "confirm" && kind !== "accept") required.push(key);
    fields.push({ key, path: q.path, kind, patch: q.patch, option_patches: q.option_patches });
    if (q.path === "environment") {
      const dkey = `${key}_description`;
      properties[dkey] = { type: "string", title: "Surroundings (free text)", description: "e.g. 'white desk by a window, afternoon light, shelf on the left'. Leave empty to skip." };
      fields.push({ key: dkey, path: "environment.description", kind: "describe" });
    }
  });
  return { schema: { type: "object", properties, ...(required.length ? { required } : {}) }, fields };
}

/** Turn accepted form content into one update_intake patch plus notes for the model. */
function toPatch(fields: Field[], content: Json): { patch: Json; notes: string[] } {
  let patch: Json = {};
  const notes: string[] = [];
  for (const f of fields) {
    const v = content[f.key];
    if (v === undefined || v === null || v === "") continue;
    let p: Json | null | undefined;
    switch (f.kind) {
      case "enum":
        p = f.option_patches?.[String(v)] ?? (f.path.startsWith("objects.") ? undefined : nest(f.path, v));
        if (!p) notes.push(`${f.path}: the human chose '${v}' but Cortex had no values for it; set them with cortex_update_intake.`);
        break;
      case "multi":
        p = Array.isArray(v) ? (f.path === "objects" ? { objects: v.map((name) => ({ name })) } : nest(f.path, v)) : undefined;
        break;
      case "confirm":
      case "accept":
        if (v === true) p = f.patch;
        else notes.push(`${f.path}: the human did not accept the suggestion; ask them for the correct values.`);
        break;
      case "string":
      case "number":
      case "describe":
        p = nest(f.path, f.kind === "number" ? Number(v) : v);
        break;
    }
    if (p) patch = merge(patch, p);
  }
  return { patch, notes };
}

function canAsk(ctx: RequestContext): boolean {
  // ctx.client reads the 2026-07-28 per-request envelope. mcp-use serves 2025-era clients statelessly, so the
  // SDK's legacy shim cannot send them a server->client elicitation: they get the text fallback instead.
  return ctx.client.can("elicitation");
}

function withQuestionsText(r: CallToolResult, data: Json, extra: string[] = []): CallToolResult {
  const qs = (Array.isArray(data.questions) ? data.questions : []) as Question[];
  const open = qs.filter((q) => q.required !== false);
  const lines = qs.map((q) => `- ${q.path}${q.required === false ? " (optional)" : ""}: ${q.prompt}${q.options?.length ? ` Options: ${q.options.join(" | ")}.` : ""}${q.patch ? ` Suggested patch: ${JSON.stringify(q.patch)}` : ""}`);
  const ask = open.length
    ? `ASK THE HUMAN these intake questions (do not guess the answers), then send their answers with cortex_update_intake:\n${lines.join("\n")}`
    : undefined;
  const text = (r.content ?? []).map((c) => (c.type === "text" ? c.text : "")).join("\n");
  return {
    ...r,
    content: [{ type: "text", text: [text, ...extra, ask].filter(Boolean).join("\n") }],
    structuredContent: { ...data, ...(extra.length ? { human_input_notes: extra } : {}), ...(ask ? { next_step: ask } : {}) },
  };
}

// ---------------------------------------------------------------- where the scene comes from (start_intake)

const PHOTO_EXT = new Set([".jpg", ".jpeg", ".png", ".webp"]);
const MAX_PHOTO_BYTES = 15 * 1024 * 1024;
const SOURCES = ["photo", "description", "default"] as const;

const hasSource = (a: Json) => Boolean(a.scene_source || a.image_path || a.image_b64 || a.description);

/** This server runs next to the human, so a photo path is read here and sent to the backend as base64. */
async function withPhoto(args: Json): Promise<Json | string> {
  const { image_path, ...rest } = args;
  const path = typeof image_path === "string" ? image_path.trim().replace(/^["']|["']$/g, "") : "";
  if (!path) return rest;
  if (!PHOTO_EXT.has(extname(path).toLowerCase())) return `image_path must be a JPEG, PNG or WebP photo (got ${basename(path)}).`;
  try {
    const st = await stat(path);
    if (!st.isFile()) return `image_path ${path} is not a file.`;
    if (st.size > MAX_PHOTO_BYTES) return `The photo at ${path} is larger than 15 MB; send a smaller one.`;
    return { ...rest, image_b64: (await readFile(path)).toString("base64") };
  } catch {
    return `Cortex could not read a photo at ${path}. Check the path with the human.`;
  }
}

/** The first form for a new scene: photo, description or the default. */
const SOURCE_FORM = {
  type: "object",
  properties: {
    scene_source: {
      type: "string",
      title: "How should Cortex build the scene?",
      description: "photo: from a picture of your setup (give its file path below). description: from your own words below. " +
        "default: the built-in tabletop, a red cube and a target zone.",
      enum: [...SOURCES],
      enumNames: ["From a photo of my setup", "From my description", "Use the default scene"],
      default: "photo",
    },
    image_path: {
      type: "string",
      title: "Photo of your setup (file path)",
      description: "e.g. C:\\Users\\me\\Pictures\\desk.jpg. A shot from above and in front of the arm, with the whole table in view, works best.",
    },
    description: {
      type: "string",
      title: "Or describe your setup",
      description: "The objects and their sizes, where they sit relative to the arm, the target, and anything in the way.",
    },
  },
  required: ["scene_source"],
};

const SOURCE_HINT =
  "This scene came from the task prompt alone. For a sim that matches the human's real setup, offer them the three ways in: " +
  "a photo (cortex_start_intake with project_id and image_path), a detailed description (description), or the default scene " +
  "(scene_source='default').";

/** Turn the scene form's answer into start_intake args, or say what is still missing. */
function sourceArgs(base: Json, content: Json): Json | string {
  const src = String(content.scene_source ?? "");
  const path = typeof content.image_path === "string" ? content.image_path.trim() : "";
  const text = typeof content.description === "string" ? content.description.trim() : "";
  if (src === "photo" && !path) return "The human chose a photo but gave no file path. Ask them for the path, then call cortex_start_intake with image_path.";
  if (src === "description" && !text) return "The human chose to describe the scene but wrote nothing. Ask them to describe it, then call cortex_start_intake with description.";
  return { ...base, scene_source: src, ...(src === "photo" ? { image_path: path } : {}), ...(text && src !== "default" ? { description: text } : {}) };
}

const textResult = (text: string, isError = false): CallToolResult => ({ isError, content: [{ type: "text", text }] });

/**
 * start_intake / update_intake with human input. A new scene first asks where it comes from (photo, description,
 * default) when the client can show forms and the call did not say. On re-entry (`requestState` verified by the
 * codec) it applies the human's answers to the project from round one and never starts a second intake.
 */
export async function runIntake(tool: "start_intake" | "update_intake", input: Json, ctx: RequestContext): Promise<CallToolResult | InputRequiredResult> {
  const { ask_human, ...given } = input;
  let args: Json = given;
  const state = ctx.requestState<State>();
  let result: CallToolResult;
  let round = 1;
  const notes: string[] = [];
  if (tool === "start_intake" && !state && ask_human !== false && !hasSource(args) && canAsk(ctx)) {
    return inputRequired({
      inputRequests: {
        scene: inputRequired.elicit({
          message: "Cortex will rebuild your robot's workspace in simulation. Start from a photo, your description, or the default scene?",
          requestedSchema: SOURCE_FORM as never,
        }),
      },
      requestState: await codec.mint({ project_id: "", round: 0, fields: [], start_args: args }),
    });
  }
  if (state?.start_args) {
    const answer = inputResponse(ctx.inputResponses, "scene");
    if (answer.kind !== "elicit" || answer.action !== "accept" || !answer.content) {
      const why = answer.kind === "elicit" ? `The human chose '${answer.action}' on the scene form.` : "No form answer came back.";
      return textResult(`${why} Nothing was created. Ask them in chat: a photo of their setup, a description, or the default scene.`);
    }
    const next = sourceArgs(state.start_args, answer.content as Json);
    if (typeof next === "string") return textResult(next);
    args = next;
  }
  if (tool === "start_intake") {
    const withImage = await withPhoto(args);
    if (typeof withImage === "string") return textResult(withImage + " Nothing was created.", true);
    if (!hasSource(args)) notes.push(SOURCE_HINT);
    args = withImage;
  }
  if (state && typeof state === "object" && state.project_id) {
    round = state.round + 1;
    const answer = inputResponse(ctx.inputResponses, "intake");
    if (answer.kind !== "elicit" || answer.action !== "accept" || !answer.content) {
      const r = await callTool("get_intake_status", { project_id: state.project_id }, ctx.signal, identityOf(ctx));
      const why = answer.kind === "elicit" ? `The human chose '${answer.action}' on the intake form.` : "No form answer came back.";
      return r.isError ? r : withQuestionsText(r, (r.structuredContent ?? {}) as Json, [`${why} Ask in chat instead, or stop if they do not want to continue.`]);
    }
    const { patch, notes: n } = toPatch(state.fields, answer.content);
    notes.push(...n);
    result = Object.keys(patch).length
      ? await callTool("update_intake", { project_id: state.project_id, patch }, ctx.signal, identityOf(ctx))
      : await callTool("get_intake_status", { project_id: state.project_id }, ctx.signal, identityOf(ctx));
    if (result.isError) return result;
    notes.unshift(`Applied the human's form answers (round ${state.round}): ${JSON.stringify(patch)}`);
  } else {
    result = await callTool(tool, args, ctx.signal, identityOf(ctx));
    if (result.isError) return result;
  }

  const data = (result.structuredContent ?? {}) as Json;
  const projectId = String(data.project_id ?? args.project_id ?? state?.project_id ?? "");
  const questions = (Array.isArray(data.questions) ? data.questions : []) as Question[];
  const openRequired = questions.some((q) => q.required !== false);
  const askNow = ask_human !== false && projectId && canAsk(ctx) && round <= MAX_ROUNDS && (openRequired || (round === 1 && questions.length > 0));
  if (askNow) {
    const { schema, fields } = buildForm(questions, round);
    if (fields.length) {
      return inputRequired({
        inputRequests: {
          intake: inputRequired.elicit({
            message:
              `Cortex needs a few details to build the simulation for project ${projectId}` +
              `${round > 1 ? ` (round ${round})` : ""}. Sizes in metres, positions from the arm base (+x forward, +y left).`,
            requestedSchema: schema as never,
          }),
        },
        requestState: await codec.mint({ project_id: projectId, round, fields }),
      });
    }
  }
  return withQuestionsText(result, { ...data, project_id: projectId || undefined }, notes);
}

/** Adds the `ask_human` flag to the intake tools' input schemas. */
export function withAskHuman(schema: z.ZodObject): z.ZodObject {
  return schema.extend({
    ask_human: z.boolean().optional().describe(
      "Default true: if the client supports forms (elicitation), Cortex asks the human the open intake questions directly " +
        "and applies their answers. false = just return the questions.",
    ),
  });
}
