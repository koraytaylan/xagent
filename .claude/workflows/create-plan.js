export const meta = {
  name: 'create-plan',
  description: 'Author the next implementation plan(s) under docs/plans/ in the exact house style the implement-plan workflow consumes. A read-only SURVEY step scans docs/plans/ (the roll-up board + every per-plan STATUS.md), the docs/plans/README.md authoring guide, CONTRIBUTING.md, and recent git history to learn the next NNNN sequence number, the project quality-gate commands, a recent complete plan to mirror, and a ranked list of candidate next plans — derived from the user brief if one was given, otherwise auto-derived from the deferred / gated / follow-on / debt work the STATUS docs explicitly record. SELECT picks which (and how many) plans to author and fixes each plan a stable NNNN-Title-Case-Kebab folder. For each selected plan a research-backed BLUEPRINT agent reads the authoring guide, a reference plan, and the actual codebase, then returns a structured plan blueprint with real file:line anchors, workstreams, and junior-executable tasks (kebab ids, Depends-on edges, falsifiable Done-when gates). RENDER writes the full triad (SCOPE.md, ARCHITECTURE.md, TASKS.md) plus the per-plan STATUS.md from that single shared blueprint so the workstream ids stay 1:1 across all three docs and the TASKS.md parses under implement-plan exactly. VERIFY runs an adversarial critic against the README checklist and the implement-plan parse contract (grep-confirming a sample of cited anchors), looping a bounded fixer until clean. VALIDATE then runs implement-plan itself in dryRun on the freshly authored plan to prove its TASKS.md parses into an acyclic dependency DAG, folding any safety edges it surfaces back into the file. Finally a single ROLL-UP step adds a 📋 Planned row per new plan to docs/plans/STATUS.md. All files are written UNCOMMITTED for human review (or committed when commit:true). The plans are then ready to execute with implement-plan.',
  phases: [
    { title: 'Survey', detail: 'read docs/plans/ (roll-up + per-plan STATUS), the README authoring guide, CONTRIBUTING, recent git; learn next NNNN, gate commands, a reference plan, and candidate next plans (from the brief or auto-derived from deferred/gated/follow-on work)' },
    { title: 'Select', detail: 'choose which and how many plans to author; assign each a stable NNNN-Title-Case-Kebab folder and fixed file paths' },
    { title: 'Blueprint', detail: 'per plan: a research agent reads the authoring guide, a reference plan, and the codebase, then returns a structured plan blueprint with real file:line anchors, workstreams, and junior-executable tasks' },
    { title: 'Render', detail: 'per plan: write SCOPE.md, ARCHITECTURE.md, TASKS.md, and the per-plan STATUS.md from the one shared blueprint so the triad stays internally consistent and TASKS.md parses under implement-plan' },
    { title: 'Verify', detail: 'per plan: an adversarial critic checks the files against the README checklist and the implement-plan parse contract (grep-confirming cited anchors); a bounded fixer addresses blockers' },
    { title: 'Validate', detail: 'per plan: run implement-plan in dryRun on the authored plan to prove TASKS.md parses into an acyclic DAG; fold any surfaced safety edges back into the file' },
    { title: 'Roll-up', detail: 'add a 📋 Planned row per new plan to the docs/plans/ roll-up board and bump its Last updated line (uncommitted, or committed when commit:true)' },
  ],
}

// ── Inputs ──────────────────────────────────────────────────────────────────
// `args` is a free-text brief string ("plan the credit-path learning fix") OR an object:
//   { brief: "...", count: 1, plansDir: "docs/plans", baseNumber: "0012",
//     dryRun: false, verify: true, validate: true, commit: false,
//     authorModel, criticModel, choreModel, maxVerifyIters: 2 }
// `brief`         what the next plan(s) should address. If omitted, the next plan(s) are auto-derived
//                 from the deferred / gated / follow-on / debt work the STATUS docs explicitly record.
// `count`         how many plans to author (default 1; "auto" = author every distinct high-priority
//                 candidate the survey found, capped at MAX_PLANS). Ignored topics are logged.
// `plansDir`      where plans live (default "docs/plans"); the survey confirms/overrides it.
// `baseNumber`    force the first plan's NNNN (default: one past the highest existing plan number).
// `dryRun`        read-only: run Survey + Select and print the plans that WOULD be authored
//                 (numbers, slugs, rationale, provisional workstreams), then STOP. Writes nothing.
// `verify`        run the adversarial critic + bounded fixer per plan (default true).
// `validate`      run implement-plan dryRun per authored plan as the machine parse/DAG gate (default true).
// `commit`        git-add + commit the authored plan files and the roll-up edit (default false —
//                 files are left uncommitted for human review, matching implement-plan's convention).
// `maxVerifyIters` critic→fix rounds before a plan is left as-authored for inspection (default 2).
//
// What it produces, per plan, under <plansDir>/NNNN-Title-Case-Kebab/:
//   SCOPE.md, ARCHITECTURE.md, TASKS.md, STATUS.md  — the authored triad + the 📋 Planned tracker,
// plus one 📋 Planned row appended to <plansDir>/STATUS.md. These are exactly what implement-plan
// consumes: `implement-plan NNNN` (dryRun first) executes the result.

let _args = args
if (typeof _args === 'string') {
  const s = _args.trim()
  if (s.startsWith('{') || s.startsWith('[')) { try { _args = JSON.parse(s) } catch { /* keep as string */ } }
}
// A bare non-JSON string is the plan brief (the most natural single argument for "plan X").
const cfg = (_args && typeof _args === 'object') ? _args : (_args ? { brief: String(_args) } : {})
const BRIEF = (cfg.brief != null ? String(cfg.brief) : '').trim()
const PLANS_DIR_HINT = (cfg.plansDir || 'docs/plans').replace(/\/+$/, '')
const BASE_NUMBER = cfg.baseNumber != null ? String(cfg.baseNumber) : ''  // != null so baseNumber:0 is honored
const DRY_RUN = !!cfg.dryRun
const DO_VERIFY = cfg.verify !== false
const DO_VALIDATE = cfg.validate !== false
const DO_COMMIT = !!cfg.commit
const MAX_VERIFY_ITERS = Math.max(1, cfg.maxVerifyIters || 2)
const MAX_PLANS = 5                              // hard cap so "auto" / a vague brief can never fan out unbounded
const AUTHOR_MODEL = cfg.authorModel || undefined  // undefined ⇒ inherit the session model (best for authoring)
const CRITIC_MODEL = cfg.criticModel || undefined
const CHORE_MODEL = cfg.choreModel || 'haiku'    // cheap, mechanical git/edit steps

// How many plans to author. A brief defaults to a single plan; auto-derivation defaults to one too,
// unless count says otherwise. "auto" lets the survey's candidate count decide (capped).
const COUNT_RAW = cfg.count
const WANT_AUTO = COUNT_RAW === 'auto' || COUNT_RAW === 'all'
const WANT_COUNT = WANT_AUTO ? MAX_PLANS : Math.max(1, Math.min(MAX_PLANS, Number(COUNT_RAW) || 1))

// ── small deterministic helpers ───────────────────────────────────────────────
const pad4 = (n) => String(n).padStart(4, '0')
const isKebab = (s) => /^[a-z0-9]+(?:-[a-z0-9]+)*$/.test(s)
// Normalize a human title to the Title-Case-Kebab folder convention (acronyms kept upper).
function toTitleKebab(s) {
  return String(s || '')
    .trim()
    .replace(/[^A-Za-z0-9]+/g, ' ')
    .split(/\s+/)
    .filter(Boolean)
    .map(w => (w === w.toUpperCase() && w.length > 1) ? w : (w.charAt(0).toUpperCase() + w.slice(1).toLowerCase()))
    .join('-')
}

// ── Schemas ───────────────────────────────────────────────────────────────────
const SURVEY_SCHEMA = {
  type: 'object',
  additionalProperties: false,
  required: ['repoRoot', 'startRef', 'plansDir', 'rollupPath', 'readmePath', 'highestNumber', 'nextNumber', 'gateCommands', 'candidates'],
  properties: {
    repoRoot: { type: 'string', description: 'git rev-parse --show-toplevel' },
    startRef: { type: 'string', description: 'git rev-parse --abbrev-ref HEAD' },
    plansDir: { type: 'string', description: 'the plans root directory relative to repoRoot (e.g. docs/plans), confirmed to exist' },
    rollupPath: { type: 'string', description: 'the roll-up board path relative to repoRoot (e.g. docs/plans/STATUS.md); "" if none exists' },
    readmePath: { type: 'string', description: 'the authoring-guide README relative to repoRoot (e.g. docs/plans/README.md); "" if none' },
    contributingPath: { type: 'string', description: 'CONTRIBUTING.md path relative to repoRoot if it exists, else ""' },
    referencePlanDir: { type: 'string', description: 'a recent COMPLETE plan folder (relative to repoRoot) whose triad best exemplifies the house style, for authors to mirror' },
    highestNumber: { type: 'string', description: 'the highest existing 4-digit plan number, e.g. "0011"' },
    nextNumber: { type: 'string', description: 'highestNumber + 1, zero-padded to 4 digits, e.g. "0012"' },
    gateCommands: { type: 'array', items: { type: 'string' }, description: 'the project standing quality-gate commands every task must keep green, verbatim (e.g. ["cargo fmt --all -- --check", "cargo clippy --workspace --all-targets -- -D warnings", "cargo test -p xagent-sandbox"]) — read from the README/CONTRIBUTING/CLAUDE.md' },
    existingPlans: {
      type: 'array',
      description: 'one entry per existing plan folder',
      items: {
        type: 'object', additionalProperties: false, required: ['number', 'slug', 'status'],
        properties: {
          number: { type: 'string' }, slug: { type: 'string' },
          status: { type: 'string', description: 'Planned / In progress / Complete / Blocked / Superseded (from the roll-up)' },
          outcome: { type: 'string' },
        },
      },
    },
    candidates: {
      type: 'array',
      description: 'ranked candidate next plans. If a user brief was given, the FIRST candidate must realize that brief; otherwise candidates are auto-derived ONLY from work the STATUS docs / CONTRIBUTING / code explicitly mark as deferred, gated, follow-on, or unpaid debt — each MUST cite where that intent is recorded. Most-actionable first.',
      items: {
        type: 'object',
        additionalProperties: false,
        required: ['title', 'slug', 'rationale', 'evidence', 'kind', 'provisionalWorkstreams'],
        properties: {
          title: { type: 'string', description: 'human plan title' },
          slug: { type: 'string', description: 'Title-Case-Kebab slug for the folder (acronyms upper), NNNN excluded' },
          rationale: { type: 'string', description: 'why this is the right next plan' },
          evidence: { type: 'array', items: { type: 'string' }, description: 'where the need is recorded: STATUS line / decision doc / CONTRIBUTING rule / code anchor (file:line)' },
          kind: { type: 'string', enum: ['follow-on', 'deferred', 'gated', 'debt', 'brief', 'new'] },
          sourcePlan: { type: 'string', description: 'the plan number this follows from, if any; "" otherwise' },
          provisionalWorkstreams: {
            type: 'array',
            items: { type: 'object', additionalProperties: false, required: ['name', 'summary'], properties: { name: { type: 'string' }, summary: { type: 'string' } } },
            description: 'a first-pass split into 1-N workstreams (the blueprint step refines this)',
          },
        },
      },
    },
  },
}

const BLUEPRINT_SCHEMA = {
  type: 'object',
  additionalProperties: false,
  required: ['mission', 'summaryParagraph', 'typography', 'whyThisPlan', 'findings', 'workstreams', 'tasks', 'originMapping', 'lockedDecisions', 'outOfScope', 'gateCommands', 'fileManifest', 'plannedOutcome', 'rootCause', 'approach'],
  properties: {
    mission: { type: 'string', description: 'one-sentence mission statement (rendered as the SCOPE blockquote; no leading ">")' },
    summaryParagraph: { type: 'string', description: 'the 3-8 line TASKS.md intro paragraph: what the whole plan accomplishes as a run-on list of concrete deltas in execution order' },
    typography: { type: 'string', enum: ['em-dash', 'ascii'], description: 'punctuation convention to keep consistent across all three docs ("—"/"→" vs "-"/"->")' },
    whyThisPlan: { type: 'string', description: 'markdown for SCOPE "## Why this plan": a numbered list, each item a **bold lead-in.** then the problem, every load-bearing claim citing file.ext:line with the symbol in backticks. Findings here MUST match the findings[] array below 1:1 by number.' },
    findings: {
      type: 'array',
      description: 'the numbered findings in "Why this plan", machine-mirrored for the origin→workstream coverage check',
      items: {
        type: 'object', additionalProperties: false, required: ['num', 'leadIn', 'workstream'],
        properties: {
          num: { type: 'integer' }, leadIn: { type: 'string' },
          anchors: { type: 'array', items: { type: 'string' }, description: 'file.ext:line(-range) anchors cited for this finding' },
          workstream: { type: 'string', description: 'the workstream id (e.g. "0001") that addresses this finding' },
        },
      },
    },
    workstreams: {
      type: 'array',
      description: 'the plan workstreams, ids "0001".."000N", 1:1 with the SCOPE in-scope bullets, the ARCHITECTURE sections, and the TASKS phase headers',
      items: {
        type: 'object', additionalProperties: false, required: ['id', 'name', 'inScope', 'architecture'],
        properties: {
          id: { type: 'string', description: '4-digit workstream id, "0001".."000N"' },
          name: { type: 'string', description: 'workstream name (used in all three docs verbatim)' },
          inScope: { type: 'string', description: 'the one SCOPE "In scope" bullet body for this workstream' },
          architecture: { type: 'string', description: 'markdown body for ARCHITECTURE "## {id} — {name}": opens with "Today ..." (present behavior, file:line), then "Edits:" with bold-labeled deltas (minimal fenced rust/wgsl snippet + doc-comment), then a "Properties that make this safe:" argument' },
        },
      },
    },
    tasks: {
      type: 'array',
      description: 'every executable task, in execution order; probe/baseline tasks first and listed as Depends-on of the changes they gate',
      items: {
        type: 'object', additionalProperties: false, required: ['id', 'title', 'workstream', 'kind', 'context', 'steps', 'dependsOn', 'doneWhen', 'touches', 'gated'],
        properties: {
          id: { type: 'string', description: 'stable lower-kebab-case id (reused as branch task/{id})' },
          title: { type: 'string', description: 'Title Case description; for a GATED task it MUST end with " (GATED)"' },
          workstream: { type: 'string', description: 'the owning workstream id' },
          kind: { type: 'string', enum: ['probe', 'change', 'spike', 'doc'], description: 'probe/baseline (authored first, gated on by changes), code change, spike/decision (resolves into a NNNN-NAME-DECISION.md), or documentation-only' },
          context: { type: 'string', description: '1-2 prose paragraphs of current/broken behavior with precise symbol (file.ext:START-END) anchors and why the change is needed; for a GATED task open with a bold **Gate:** paragraph naming the upstream tasks that must land first' },
          steps: { type: 'array', items: { type: 'string' }, description: 'ordered imperative steps; prescriptive and copy-paste-ready — name exact file+symbol, embed verbatim fenced rust/wgsl/bash including full test functions; GPU tests embed the GpuKernel::is_available() self-skip guard' },
          dependsOn: { type: 'array', items: { type: 'string' }, description: 'direct prerequisite task ids only ([] = none)' },
          doneWhen: { type: 'string', description: 'the falsifiable acceptance criterion; behavioral tasks phrase it red-green; almost always closes with the gate-commands shorthand. A documentation-only task may exempt the green gate but must say it is documentation-only.' },
          touches: { type: 'array', items: { type: 'string' }, description: 'repo-relative files the task will create or modify (bias toward MORE)' },
          gated: { type: 'boolean', description: 'true iff a conditional Phase-2 follow-up; its title ends with " (GATED)" and Done-when is binary land-or-revert-and-record' },
        },
      },
    },
    originMapping: {
      type: 'array',
      description: 'the SCOPE origin→workstream table: every finding number must appear',
      items: { type: 'object', additionalProperties: false, required: ['finding', 'workstream'], properties: { finding: { type: 'string' }, workstream: { type: 'string' } } },
    },
    lockedDecisions: { type: 'array', items: { type: 'object', additionalProperties: false, required: ['title', 'body'], properties: { title: { type: 'string' }, body: { type: 'string', description: 'the commitment, its rationale, and the explicit gate/condition under which it holds or is revisited' } } } },
    outOfScope: { type: 'array', items: { type: 'object', additionalProperties: false, required: ['item', 'why'], properties: { item: { type: 'string' }, why: { type: 'string' } } } },
    testStrategy: { type: 'string', description: 'optional ARCHITECTURE "## Test strategy": named tests with file, setup, exact assertions, ending in the CI gate commands' },
    interactions: { type: 'string', description: 'optional ARCHITECTURE "## Interaction with prior work": bold-led bullets tying the plan to prior plans/specs/reviews' },
    decisionDocs: {
      type: 'array',
      description: 'optional NNNN-NAME-DECISION.md stubs for spike/decision tasks (numbered to the workstream they resolve)',
      items: { type: 'object', additionalProperties: false, required: ['filename', 'body'], properties: { filename: { type: 'string', description: 'e.g. 0003-SHARED-DEVICE-DECISION.md' }, body: { type: 'string' } } },
    },
    gateCommands: { type: 'array', items: { type: 'string' }, description: 'the project standing quality-gate commands, verbatim' },
    fileManifest: { type: 'array', items: { type: 'string' }, description: 'every source file the plan touches (backtick repo-relative paths) for the ARCHITECTURE opening blockquote' },
    plannedOutcome: { type: 'string', description: 'one-line goal/expected outcome for the STATUS docs and the roll-up row' },
    rootCause: { type: 'string', description: 'the per-plan STATUS "Root cause" line' },
    approach: { type: 'string', description: 'the per-plan STATUS "Approach" line' },
  },
}

const VERIFY_SCHEMA = {
  type: 'object',
  additionalProperties: false,
  required: ['approved', 'parseable', 'findings', 'summary'],
  properties: {
    approved: { type: 'boolean', description: 'true ONLY if there are no blocker findings and the TASKS.md satisfies the implement-plan parse contract' },
    parseable: { type: 'boolean', description: 'true if TASKS.md parses under the implement-plan contract (## NNNN — Name phases; ### {kebab-id} — {Title}; - **Depends on:**; - **Done when:**; unique kebab ids; deps reference existing ids; acyclic)' },
    findings: {
      type: 'array',
      items: {
        type: 'object', additionalProperties: false, required: ['severity', 'note'],
        properties: { severity: { type: 'string', enum: ['blocker', 'nit'] }, file: { type: 'string' }, note: { type: 'string' } },
      },
    },
    summary: { type: 'string' },
  },
}

const OK_SCHEMA = { type: 'object', additionalProperties: false, required: ['ok'], properties: { ok: { type: 'boolean' }, error: { type: 'string' } } }

// ── Prompt builders (function declarations hoist) ─────────────────────────────
function surveyPrompt(plansDirHint, brief) {
  let p = `You are the SURVEY step of a create-plan run: you gather the facts needed to author the project's next implementation plan(s). Do everything READ-ONLY — no edits, no commits, no branch changes.\n\n`
  p += `1. GIT ROOTS: repoRoot (\`git rev-parse --show-toplevel\`), startRef (\`git rev-parse --abbrev-ref HEAD\`).\n`
  p += `2. LOCATE THE PLANS ROOT. Confirm "${plansDirHint}" exists under repoRoot (it is the conventional location). Set plansDir to it. Find its authoring-guide README (readmePath, e.g. ${plansDirHint}/README.md) and its roll-up board (rollupPath, e.g. ${plansDirHint}/STATUS.md). READ the README in full — it is the authoritative authoring guide (folder/naming rule, the SCOPE/ARCHITECTURE/TASKS triad, the per-task format, the lifecycle, the new-plan checklist). Find CONTRIBUTING.md (contributingPath) if present.\n`
  p += `3. ENUMERATE EXISTING PLANS. List every NNNN-* folder under plansDir; for each, record number, slug, and its status + one-line outcome from the roll-up board. Determine highestNumber (the largest 4-digit prefix) and nextNumber (highestNumber + 1, zero-padded). Pick referencePlanDir: a recent COMPLETE plan whose triad best exemplifies the current house style (a small, clean one is ideal for mirroring).\n`
  p += `4. GATE COMMANDS. Extract the project's standing quality-gate commands every task must keep green, verbatim, from the README "Conventions"/lifecycle, CONTRIBUTING, and/or CLAUDE.md (e.g. the fmt/clippy/test trio). Return them in gateCommands.\n`
  p += `5. CANDIDATE NEXT PLAN(S). Produce a ranked candidates[] list, most-actionable first:\n`
  if (brief) {
    p += `   - A USER BRIEF was given: "${brief}". The FIRST candidate MUST realize this brief — give it a house-style title + Title-Case-Kebab slug, a rationale, and a first-pass split into 1-N provisional workstreams. Set its kind to "brief". You MAY add further auto-derived candidates after it.\n`
  } else {
    p += `   - NO brief was given: AUTO-DERIVE candidates ONLY from work the project explicitly records as unfinished-but-intended. Mine: the roll-up board and every per-plan STATUS.md for phrases like "follow-on", "deferred", "the follow-on is", "GATED", "revisit", "next plan", "next suspect", decision docs that say "When to revisit"; CONTRIBUTING rules with standing debt; and clear TODO-shaped intent in code. Each candidate MUST cite in evidence[] exactly where that intent is recorded (STATUS line, decision-doc path, CONTRIBUTING rule, or file:line). Do NOT invent net-new scope that nothing in the repo asks for — if genuinely nothing is recorded, return a single best-judgment candidate with kind "new" and say so in its rationale.\n`
  }
  p += `   For every candidate also give a first-pass split into 1-N provisional workstreams (name + one-line summary). The blueprint step refines these.\n\n`
  p += `Return the full structured result. Use exact 4-digit numbers. Do not modify anything.`
  return p
}

// The house-style contract the renderers and the blueprint author must honor. Kept compact here; the
// README (readmePath) is the full authoritative guide and every author is told to read it.
const TASKS_CONTRACT = [
  'TASKS.md MUST parse under the implement-plan contract, EXACTLY:',
  '  • Workstream phase headers: "## NNNN — Name" (one per workstream, ids 0001..000N).',
  '  • Each task: a heading "### {kebab-id} — {Title}" followed by 1-2 context paragraphs, a "**Steps:**" ordered list, then two bullets:',
  '        - **Depends on:** {comma-separated prerequisite kebab-ids, or — if none}',
  '        - **Done when:** {falsifiable acceptance criterion}',
  '  • Task ids are unique, lower-kebab-case; every Depends-on id references another task id in the file; the dependency graph is acyclic.',
  '  • A GATED task ends its title with " (GATED)", opens context with a bold **Gate:** paragraph, and has a binary land-or-revert-and-record Done-when.',
  '  • Done-when almost always closes by citing the project gate commands (state the full forms once in a "**Conventions**" block, abbreviate thereafter).',
].join('\n')

function blueprintPrompt(facts, plan) {
  const f = JSON.stringify({
    repoRoot: facts.repoRoot, plansDir: facts.plansDir, readmePath: facts.readmePath,
    contributingPath: facts.contributingPath, referencePlanDir: facts.referencePlanDir,
    gateCommands: facts.gateCommands,
  }, null, 2)
  const c = JSON.stringify({
    number: plan.number, slug: plan.slug, title: plan.title, planDir: plan.planDir,
    rationale: plan.rationale, evidence: plan.evidence, kind: plan.kind, sourcePlan: plan.sourcePlan,
    provisionalWorkstreams: plan.provisionalWorkstreams,
  }, null, 2)
  let p = `You are the BLUEPRINT step of a create-plan run. Produce a precise, research-backed STRUCTURED blueprint for ONE new implementation plan. This blueprint is the single source of truth the renderers turn into SCOPE.md / ARCHITECTURE.md / TASKS.md / STATUS.md, so it must be internally consistent and grounded in the ACTUAL codebase. This step is READ-ONLY: research by reading and grepping; make NO edits.\n\n`
  p += `Plan to design (its folder identity is FIXED — use these exact values; do NOT renumber or rename):\n${c}\n\n`
  p += `Survey facts:\n${f}\n\n`
  p += `Do this:\n`
  p += `1. READ the authoring guide at ${facts.repoRoot}/${facts.readmePath} in full — it defines the SCOPE/ARCHITECTURE/TASKS triad, the per-task format, and the new-plan checklist. READ the reference plan's triad under ${facts.repoRoot}/${facts.referencePlanDir} (SCOPE.md, ARCHITECTURE.md, TASKS.md, STATUS.md) to mirror the exact house style, depth, and tone. READ ${facts.contributingPath ? facts.repoRoot + '/' + facts.contributingPath : 'CONTRIBUTING.md if present'} for the rules source must obey.\n`
  p += `2. RESEARCH the codebase for THIS plan's subject. Read the evidence cited in the candidate, then grep/read the real source: find the symbols, current behavior, and exact file:line anchors you will cite. Every load-bearing claim in the blueprint MUST point at a real symbol at a real file:line you actually opened — fabricated or guessed anchors are the primary failure mode; do not produce one. Line numbers are hints, but the symbol must exist where you say.\n`
  p += `3. DESIGN the plan as workstreams (ids 0001..000N) and dependency-ordered, junior-executable tasks. Test-first / measurement-first: author any probe or baseline task BEFORE the change tasks that measure against it, and list it in those tasks' dependsOn. Tasks that touch disjoint files with no cross-reference get an empty dependsOn so implement-plan runs them in parallel; only add a dependsOn edge when a task truly needs another's code/types/artifacts or would write the same file.\n`
  p += `4. Write each task for a junior engineer who has never seen the codebase: name the exact file + symbol, give exact values with their rationale, embed the verbatim test (the test is the acceptance gate), and make Done-when something a reviewer can mark pass/fail mechanically. GPU-touching tests embed the standard self-skip guard verbatim. Leave ZERO design decisions to the executor. For every task, re-read its steps and Done-when and ask whether a junior would still have to ask "which file?", "what value?", or "how should this behave?" — if so the task is underspecified; answer it in the task before returning.\n`
  p += `5. Make the workstream ids 1:1 across findings → in-scope bullets → ARCHITECTURE sections → TASKS phases, and ensure every numbered finding appears in originMapping. Pick ONE typography and keep it consistent. Carry the gate commands through verbatim: ${JSON.stringify(facts.gateCommands)}.\n\n`
  p += TASKS_CONTRACT + `\n\n`
  p += `Return the structured blueprint. Be thorough and concrete — this is the contract the implementers will follow.`
  return p
}

function renderScopePrompt(repoRoot, plan, readmePath, refDir, bpJson) {
  return `Write the SCOPE.md for plan ${plan.number} from the structured blueprint below, in the project's exact house style.\n`
    + `First READ ${repoRoot}/${readmePath} (the "SCOPE.md" section defines the fixed section order) and ${repoRoot}/${refDir}/SCOPE.md (mirror its structure/tone).\n`
    + `Write the file to ${repoRoot}/${plan.planDir}/SCOPE.md (create the directory if needed). Required structure, in order: a top "# Scope — Plan ${plan.number}" title; the one-sentence mission as a blockquote directly under the title; "## Why this plan" (the blueprint's whyThisPlan, numbered with bold lead-ins and file:line citations); "## In scope" (one bullet per workstream, "- **{workstream.id} — {workstream.name}.** {inScope}" where workstream.id is the 4-digit id); "## Origin -> workstream mapping" (a table covering EVERY finding from originMapping); "## Locked decisions" (each with its gate/condition); "## Out of scope" (mirrors In scope to draw the boundary); and the two closing See-references to ARCHITECTURE.md and TASKS.md. If the blueprint's whyThisPlan adjudicates prior review claims (cites verified-against provenance or rejects claims), also add the optional provenance paragraph and the "Review claims rejected during verification" table from the README's SCOPE.md spec. Keep the blueprint's typography consistent.\n`
    + `Blueprint:\n${bpJson}\n\n`
    + `Write ONLY this file. Do NOT commit or touch any other file. Return ok=true on success, ok=false + error otherwise.`
}

function renderArchPrompt(repoRoot, plan, readmePath, refDir, bpJson) {
  return `Write the ARCHITECTURE.md for plan ${plan.number} from the structured blueprint below, in the project's exact house style.\n`
    + `First READ ${repoRoot}/${readmePath} (the "ARCHITECTURE.md" section) and ${repoRoot}/${refDir}/ARCHITECTURE.md (mirror it).\n`
    + `Write the file to ${repoRoot}/${plan.planDir}/ARCHITECTURE.md. Required: a "# Architecture — Plan ${plan.number} (deltas)" title (the literal "(deltas)" suffix is mandatory); an opening blockquote manifest listing every touched file (fileManifest) as backtick repo-relative paths, ending with the line "Line numbers are hints; locate by symbol."; then one "## {id} — {name}" section PER WORKSTREAM (use the blueprint's per-workstream architecture body — each opens with "Today ..." stating present behavior at file:line, then "Edits:" with bold-labeled deltas shown as minimal fenced rust/wgsl snippets carrying justifying doc-comments (never full function bodies), then a "Properties that make this safe:" invariant argument). Append "## Test strategy" and "## Interaction with prior work" only if the blueprint provides them. Defer boundary/decision policy to SCOPE. Keep typography consistent.\n`
    + `Blueprint:\n${bpJson}\n\n`
    + `Write ONLY this file. Do NOT commit or touch any other file. Return ok=true on success, ok=false + error otherwise.`
}

function renderTasksPrompt(repoRoot, plan, readmePath, refDir, bpJson, gateCommands) {
  return `Write the TASKS.md for plan ${plan.number} from the structured blueprint below. This file is MACHINE-CONSUMED by the implement-plan workflow, so it MUST follow the parse contract exactly — a deviation breaks scheduling.\n`
    + `First READ ${repoRoot}/${readmePath} (the "TASKS.md" + "Task format" sections) and ${repoRoot}/${refDir}/TASKS.md (mirror its structure precisely).\n\n`
    + TASKS_CONTRACT + `\n\n`
    + `Write the file to ${repoRoot}/${plan.planDir}/TASKS.md with: a "# XAgent Plan ${plan.number} — ${plan.title}" title; the blueprint's summaryParagraph; a "See [SCOPE.md](SCOPE.md) ... [ARCHITECTURE.md](ARCHITECTURE.md) ..." references line; a "**Conventions**" block (each task has a stable kebab id = branch task/{id}; Depends-on lists direct prerequisites only, "—" means none; Done-when is the verifiable criterion and every task keeps the gate commands green — state the full forms here once: ${JSON.stringify(gateCommands)}; GPU tests self-skip without an adapter; line numbers are hints, locate by symbol); a "---" rule; then one PHASE HEADER per workstream written EXACTLY as "## {workstream.id} — {workstream.name}" (workstream.id is the 4-digit id like "0001", e.g. "## 0001 — Foo"; this is NOT a task id), each phase holding its tasks written EXACTLY as "### {task.id} — {Title}" (task.id is the lower-kebab task id like "add-foo") + context + "**Steps:**" ordered list (embed the verbatim fenced code/tests from the blueprint) + "- **Depends on:** ..." + "- **Done when:** ..."; separate tasks/phases with "---"; and a closing "**End of plan ${plan.number} TASKS.**" line. Render the blueprint's tasks and dependsOn edges VERBATIM — do not add, drop, or reorder dependency edges. Keep typography consistent.\n`
    + `Blueprint:\n${bpJson}\n\n`
    + `Also, if the blueprint has decisionDocs, write each one to ${repoRoot}/${plan.planDir}/{filename}.\n`
    + `Write ONLY files inside ${plan.planDir}. Do NOT commit or touch any other file. Return ok=true on success, ok=false + error otherwise.`
}

function renderStatusPrompt(repoRoot, plan, readmePath, refDir, bpJson) {
  return `Write the per-plan STATUS.md for plan ${plan.number} (the living task-level tracker), status 📋 Planned since it is newly authored and not yet started.\n`
    + `First READ ${repoRoot}/${readmePath} (the "STATUS.md" two-tier section) and ${repoRoot}/${refDir}/STATUS.md (mirror its shape).\n`
    + `Run \`date +%Y-%m-%d\` for today's date. Write the file to ${repoRoot}/${plan.planDir}/STATUS.md with: a "# Plan ${plan.number} — ${plan.title} — status" title; a one-line intro that task-level status lives here and the roll-up row in ../STATUS.md must stay in sync; a "**Status:** 📋 Planned." line; a "_Last updated: {today}, against {branch}._" line (use the current branch); a "- **Goal:**" line (plannedOutcome); a "- **Root cause:**" line (rootCause); an "- **Approach:**" line (approach); and a workstream table "| WS | Workstream | Tasks | State |" with one row per workstream — the Tasks column a comma-separated list of that workstream's task ids each wrapped in backticks (e.g. \`add-foo\`, \`add-bar\`), and State a single 📋 Planned. Do NOT fabricate measured results — this plan has not run.\n`
    + `Blueprint:\n${bpJson}\n\n`
    + `Write ONLY this file. Do NOT commit or touch any other file. Return ok=true on success, ok=false + error otherwise.`
}

function verifyPlanPrompt(repoRoot, plan, readmePath) {
  return `You are an adversarial REVIEWER of a freshly authored implementation plan. Be skeptical: assume the author cut corners. This is READ-ONLY — make NO edits; only report findings.\n\n`
    + `The plan is at ${repoRoot}/${plan.planDir}/ (SCOPE.md, ARCHITECTURE.md, TASKS.md, STATUS.md). The authoring guide + its "Checklist for a new plan" is at ${repoRoot}/${readmePath}.\n\n`
    + `Check, and raise a blocker for any failure:\n`
    + `1. PARSE CONTRACT (set parseable=false on any miss): TASKS.md has "## NNNN — Name" workstream phases; each task is "### {kebab-id} — {Title}" with a "- **Depends on:**" bullet and a "- **Done when:**" bullet; task ids are unique + lower-kebab; every Depends-on id references a task id present in the file; the dependency graph is acyclic; GATED task titles end with " (GATED)".\n`
    + `2. TRIAD CONSISTENCY: workstream ids are 1:1 across SCOPE "In scope", the ARCHITECTURE "## {id}" sections, and the TASKS phases; every numbered finding in SCOPE "Why this plan" appears in the origin→workstream table; ARCHITECTURE title ends with "(deltas)" and opens with the file-manifest blockquote ending "Line numbers are hints; locate by symbol."\n`
    + `3. ANCHOR REALITY: pick several file:line / symbol anchors cited across SCOPE and ARCHITECTURE and grep the repo to confirm the named symbol actually exists where claimed. Raise a blocker for any anchor that does not resolve.\n`
    + `4. JUNIOR-EXECUTABILITY: each task names exact files/symbols/values, pins the test verbatim, leaves no "which?/what?/how?" open, and has a Done-when a reviewer can mark pass/fail mechanically. Done-when cites the project gate commands except where a task is explicitly documentation-only. GPU-touching tests embed the GpuKernel::is_available() self-skip guard.\n`
    + `5. STATUS hygiene: per-plan STATUS.md exists with status 📋 Planned and a workstream table; it claims NO measured results.\n`
    + `6. MEASUREMENT-FIRST ORDERING (README checklist item 8): any probe/baseline task is authored BEFORE the change tasks that measure against it, and those changes list it in Depends-on.\n`
    + `7. ARCHITECTURE SHAPE: every "## {id} — {name}" workstream section opens with the word "Today" stating present behavior at a file:line, and ends with a substantive "Properties that make this safe:" invariant argument (not placeholder text). Raise a blocker for any section missing either.\n`
    + `8. DECISION DOCS (README checklist item 12): every spike/decision task (one whose Done-when is itself a decision, typically framed "Prototype only if …") has a matching NNNN-NAME-DECISION.md in the plan folder, numbered to that task's workstream. GATED tasks carry " (GATED)", a bold **Gate:** precondition, and a binary land-or-revert-and-record Done-when.\n`
    + `9. TYPOGRAPHY (README checklist item 13): one punctuation convention (em-dash/arrow vs ASCII) is used consistently across SCOPE, ARCHITECTURE, and TASKS — flag a doc that mixes them.\n\n`
    + `Then run the README's own "Checklist for a new plan" end to end and raise a blocker for any unmet item not already covered above.\n`
    + `Set approved=true ONLY if there are zero blocker findings AND parseable=true. List every blocker and nit with the file and a concrete, actionable note.`
}

function fixPlanPrompt(repoRoot, plan, findings) {
  return `A reviewer found issues in the authored plan at ${repoRoot}/${plan.planDir}/. Fix EACH finding by editing the affected file(s) in place; preserve everything that is already correct, and keep the implement-plan TASKS.md parse contract intact. If a finding is about a wrong file:line anchor, grep the repo for the real location and correct it. Do NOT commit.\n\n`
    + TASKS_CONTRACT + `\n\nFindings to address:\n`
    + findings.map(x => `- [${x.severity}] ${x.file ? x.file + ': ' : ''}${x.note}`).join('\n')
    + `\n\nEdit only files inside ${plan.planDir}. Return ok=true once every blocker is addressed, ok=false + error otherwise.`
}

function foldEdgesPrompt(repoRoot, tasksPath, addedEdges) {
  return `The implement-plan dependency verifier found safety edges MISSING from the authored ${repoRoot}/${tasksPath} — pairs of tasks that would run in parallel but share files or have an unwritten prerequisite. Make the authored graph complete by adding each as a real "Depends on" edge so future runs need no inference.\n`
    + `For each edge "to ← from" below, add "from" to the "- **Depends on:**" bullet of task "to" (replace a "—"/"-" with the id; otherwise append ", {id}"). Change NOTHING else. Keep the parse contract intact. Do NOT commit.\n`
    + addedEdges.map(e => `- ${e.to} ← ${e.from}${e.reason ? ` (${e.reason})` : ''}`).join('\n')
    + `\n\nEdit ONLY ${tasksPath}. Return ok=true on success, ok=false + error otherwise.`
}

function rollupPrompt(repoRoot, rollupPath, rows) {
  return `Add a 📋 Planned row to the plans roll-up board at ${repoRoot}/${rollupPath} for each newly authored plan below, then refresh the board's "Last updated" line (run \`date +%Y-%m-%d\` for today's date, and use the current branch). The board is "one row per plan, no per-task detail". Insert each new row in plan-number order; leave every existing row byte-for-byte untouched. Each row's columns: number | title | 📋 Planned | tasks count "0/N" (N = the plan's task count) | the one-line outcome/goal | a link to the plan's own STATUS.md (e.g. [status](NNNN-Slug/STATUS.md)).\n`
    + `New plans:\n${JSON.stringify(rows, null, 2)}\n\n`
    + `Edit ONLY ${rollupPath}. Do NOT commit. Return ok=true on success, ok=false + error otherwise.`
}

function commitPrompt(repoRoot, paths, planLabels) {
  return `Stage and commit the newly authored plan files in ${repoRoot}. Stage exactly these paths and nothing else: ${paths.map(p => `"${p}"`).join(', ')}. Then commit with EXACTLY this message (write it to a temp file and use \`git commit -F <file>\`):\n`
    + `-----\ndocs(plans): author plan(s) ${planLabels.join(', ')}\n\nAuthored via create-plan in the docs/plans house style; ready for implement-plan.\n-----\n`
    + `Do NOT push. Return ok=true + the new commit sha in error-or-note, ok=false + error otherwise.`
}

// Deterministic graph checks over a blueprint's tasks (mirrors implement-plan's parse expectations).
function checkBlueprintGraph(bp) {
  const issues = []
  const tasks = Array.isArray(bp.tasks) ? bp.tasks : []
  if (!tasks.length) { issues.push('blueprint has no tasks'); return issues }
  const ids = tasks.map(t => t.id)
  const idSet = new Set(ids)
  if (idSet.size !== ids.length) issues.push(`duplicate task ids: ${ids.filter((x, i) => ids.indexOf(x) !== i).join(', ')}`)
  for (const t of tasks) {
    if (!isKebab(String(t.id || ''))) issues.push(`task id not kebab-case: "${t.id}"`)
    for (const dep of (t.dependsOn || [])) if (!idSet.has(dep)) issues.push(`task ${t.id} depends on unknown id "${dep}"`)
    if (t.gated && !/\(GATED\)\s*$/.test(String(t.title || ''))) issues.push(`gated task ${t.id} title must end with "(GATED)"`)
    const ws = String(t.workstream || '')
    if (!(bp.workstreams || []).some(w => w.id === ws)) issues.push(`task ${t.id} references unknown workstream "${ws}"`)
  }
  // Cycle detection (Kahn).
  const indeg = new Map(ids.map(id => [id, 0]))
  const adj = new Map(ids.map(id => [id, []]))
  for (const t of tasks) for (const dep of (t.dependsOn || [])) if (idSet.has(dep)) { adj.get(dep).push(t.id); indeg.set(t.id, indeg.get(t.id) + 1) }
  const q = ids.filter(id => indeg.get(id) === 0)
  let seen = 0
  while (q.length) { const id = q.shift(); seen++; for (const n of adj.get(id)) { indeg.set(n, indeg.get(n) - 1); if (indeg.get(n) === 0) q.push(n) } }
  if (seen !== ids.length) issues.push('dependency cycle among tasks')
  // Findings coverage. Match the finding NUMBER as a bounded token so "(1)" / "1" count but "10" does
  // NOT satisfy finding 1 (a plain includes() would false-positive), while still accepting the house
  // convention of descriptive origin cells like "First problem (1)".
  const mapped = [...new Set((bp.originMapping || []).map(m => String(m.finding)))]
  for (const fnd of (bp.findings || [])) {
    const key = String(fnd.num)
    const tokenRe = new RegExp(`(^|[^0-9])${key}([^0-9]|$)`)   // key is an integer ⇒ safe to interpolate
    if (!mapped.some(m => tokenRe.test(m))) issues.push(`finding ${key} missing from originMapping`)
  }
  // Measurement-first ordering (README checklist item 8): a probe/baseline task a change depends on
  // must be authored EARLIER in the file than the change that gates on it.
  const indexOf = new Map(ids.map((id, i) => [id, i]))
  const kindOf = new Map(tasks.map(t => [t.id, t.kind]))
  for (let i = 0; i < tasks.length; i++) {
    for (const dep of (tasks[i].dependsOn || [])) {
      if (kindOf.get(dep) === 'probe' && (indexOf.get(dep) ?? 0) > i) {
        issues.push(`probe/baseline task ${dep} is authored after ${tasks[i].id} which depends on it (probes must come first)`)
      }
    }
  }
  return issues
}

// ── 1. Survey ──────────────────────────────────────────────────────────────────
phase('Survey')
const survey = await agent(
  surveyPrompt(PLANS_DIR_HINT, BRIEF),
  { label: 'survey', phase: 'Survey', model: CRITIC_MODEL, effort: 'high', agentType: 'Explore', schema: SURVEY_SCHEMA }
)
if (!survey || !survey.repoRoot || !survey.plansDir) {
  log('Survey failed (no repo root / plans dir).')
  return { error: 'survey failed' }
}
if (!Array.isArray(survey.candidates) || survey.candidates.length === 0) {
  log('Survey found no candidate next plans (and no brief). Nothing to author.')
  return { error: 'no candidates', survey }
}
const REPO = survey.repoRoot
const PLANS_DIR = survey.plansDir.replace(/\/+$/, '')
const README_PATH = survey.readmePath || `${PLANS_DIR}/README.md`
const REF_DIR = survey.referencePlanDir || ''
const ROLLUP_PATH = survey.rollupPath || `${PLANS_DIR}/STATUS.md`
const GATES = Array.isArray(survey.gateCommands) && survey.gateCommands.length ? survey.gateCommands : ['<project quality-gate commands>']
log(`Survey: highest plan ${survey.highestNumber || '?'}; next ${survey.nextNumber}; ${survey.candidates.length} candidate(s); reference ${REF_DIR || '(none)'}; gates ${JSON.stringify(GATES)}.`)

// ── 2. Select: choose plans + fix each a stable NNNN-Title-Case-Kebab folder ────
phase('Select')
const firstNum = BASE_NUMBER ? parseInt(BASE_NUMBER, 10) : (parseInt(survey.nextNumber, 10) || (parseInt(survey.highestNumber, 10) || 0) + 1)
const chosen = survey.candidates.slice(0, WANT_COUNT)
const skipped = survey.candidates.slice(WANT_COUNT)
const plans = chosen.map((c, i) => {
  const number = pad4(firstNum + i)
  let slug = toTitleKebab(c.slug || c.title)
  if (!slug) slug = `Plan-${number}`
  const planDir = `${PLANS_DIR}/${number}-${slug}`
  return {
    number, slug, title: c.title || slug.replace(/-/g, ' '), planDir,
    tasksPath: `${planDir}/TASKS.md`, statusPath: `${planDir}/STATUS.md`,
    rationale: c.rationale || '', evidence: c.evidence || [], kind: c.kind || 'new',
    sourcePlan: c.sourcePlan || '', provisionalWorkstreams: c.provisionalWorkstreams || [],
  }
})
const facts = { repoRoot: REPO, plansDir: PLANS_DIR, readmePath: README_PATH, contributingPath: survey.contributingPath || '', referencePlanDir: REF_DIR, gateCommands: GATES }
log(`Selected ${plans.length} plan(s) to author: ${plans.map(p => `${p.number}-${p.slug}`).join(', ')}${skipped.length ? ` (skipping ${skipped.length} lower-priority candidate(s): ${skipped.map(c => c.title).join('; ')})` : ''}.`)
for (const p of plans) log(`  ${p.number}-${p.slug} [${p.kind}] — ${p.rationale}${p.evidence.length ? ` (evidence: ${p.evidence.join('; ')})` : ''}`)

// ── Dry run: report what WOULD be authored and STOP (Survey + Select were read-only). ─
if (DRY_RUN) {
  phase('Dry run')
  log(`DRY RUN — nothing is written. Re-run without dryRun to author the plan(s).`)
  for (const p of plans) log(`  would author ${p.planDir}/ with provisional workstreams: ${(p.provisionalWorkstreams || []).map(w => w.name).join(' | ') || '(blueprint decides)'}`)
  return {
    dryRun: true, plansDir: PLANS_DIR, nextNumber: survey.nextNumber,
    wouldAuthor: plans.map(p => ({ number: p.number, slug: p.slug, planDir: p.planDir, kind: p.kind, rationale: p.rationale, evidence: p.evidence })),
    skipped: skipped.map(c => ({ title: c.title, kind: c.kind })),
  }
}

// ── 3-6. Per-plan pipeline: Blueprint → Render → Verify(+fix) → Validate ────────
// pipeline() has NO barrier between stages, so plan B can be researching while plan A is rendering.
// Each plan writes only files inside its own folder, so concurrent plans never collide; the single
// shared file (the roll-up board) is written once, after, in the Roll-up barrier.
async function verifyAndFix(plan) {
  if (!DO_VERIFY) return { approved: true, parseable: true, findings: [], skipped: true }
  let verdict = null
  for (let i = 1; i <= MAX_VERIFY_ITERS; i++) {
    verdict = await agent(
      verifyPlanPrompt(REPO, plan, README_PATH),
      { label: `verify:${plan.number}#${i}`, phase: 'Verify', model: CRITIC_MODEL, effort: 'high', agentType: 'Explore', schema: VERIFY_SCHEMA }
    )
    if (verdict && verdict.approved && verdict.parseable) { log(`verify ${plan.number}: clean on round ${i}.`); break }
    const blockers = (verdict && verdict.findings || []).filter(f => f.severity === 'blocker')
    if (!blockers.length && verdict && verdict.parseable) { log(`verify ${plan.number}: only nits on round ${i}; accepting.`); break }
    if (i === MAX_VERIFY_ITERS) { log(`verify ${plan.number}: still ${blockers.length} blocker(s) after ${i} round(s); left as-authored for inspection.`); break }
    log(`verify ${plan.number}: round ${i} found ${blockers.length} blocker(s); fixing.`)
    const fixFindings = blockers.length ? blockers : [{ severity: 'blocker', note: 'reviewer rejected but listed no findings; re-check the parse contract and triad consistency' }]
    await agent(fixPlanPrompt(REPO, plan, fixFindings), { label: `fix:${plan.number}#${i}`, phase: 'Verify', model: AUTHOR_MODEL, schema: OK_SCHEMA })
  }
  return verdict || { approved: false, parseable: false, findings: [{ severity: 'blocker', note: 'no verdict' }] }
}

async function validatePlan(plan) {
  if (!DO_VALIDATE) return { skipped: true }
  // Run the real consumer in read-only dryRun: it parses the authored TASKS.md and builds the wave DAG.
  let res = null
  try {
    res = await workflow('implement-plan', { plan: plan.number, dryRun: true })
  } catch (e) {
    log(`validate ${plan.number}: implement-plan dryRun unavailable (${String((e && e.message) || e)}); skipping the machine parse check.`)
    return { skipped: true, error: String((e && e.message) || e) }
  }
  if (!res || res.error) { log(`validate ${plan.number}: implement-plan could not parse the plan (${(res && res.error) || 'no result'}).`); return { ok: false, error: (res && res.error) || 'no result' } }
  log(`validate ${plan.number}: parsed OK — ${res.taskCount} task(s), ${res.waveCount} wave(s), widest ${res.widestWave} concurrent.`)
  // Any safety edge the verifier had to add means our authored graph was incomplete — fold it in so
  // the file is self-sufficient for future runs.
  const added = Array.isArray(res.addedEdges) ? res.addedEdges : []
  let edgesFolded = true
  if (added.length) {
    log(`validate ${plan.number}: implement-plan added ${added.length} safety edge(s); folding them into TASKS.md.`)
    const fold = await agent(foldEdgesPrompt(REPO, plan.tasksPath, added), { label: `fold-edges:${plan.number}`, phase: 'Validate', model: CHORE_MODEL, schema: OK_SCHEMA })
    edgesFolded = !!(fold && fold.ok)
    if (!edgesFolded) log(`validate ${plan.number}: could not fold edges automatically (${(fold && fold.error) || 'unknown'}); add them by hand: ${added.map(e => `${e.to}←${e.from}`).join(', ')}.`)
  }
  return { ok: true, taskCount: res.taskCount, waveCount: res.waveCount, widestWave: res.widestWave, addedEdges: added, edgesFolded }
}

const authored = await pipeline(
  plans,
  // Stage 1 — Blueprint (research-backed structured plan).
  async (plan) => {
    const bp = await agent(
      blueprintPrompt(facts, plan),
      { label: `blueprint:${plan.number}`, phase: 'Blueprint', model: AUTHOR_MODEL, effort: 'high', agentType: 'Explore', schema: BLUEPRINT_SCHEMA }
    )
    if (!bp) throw new Error(`blueprint failed for ${plan.number}`)
    // Pin the folder identity (the agent must not renumber/rename); log any structural issues.
    const merged = { ...bp, number: plan.number, slug: plan.slug, title: plan.title, planDir: plan.planDir, tasksPath: plan.tasksPath, statusPath: plan.statusPath }
    const issues = checkBlueprintGraph(merged)
    if (issues.length) log(`blueprint ${plan.number}: ${issues.length} structural issue(s) — ${issues.join('; ')} (Verify will re-check after render).`)
    log(`blueprint ${plan.number}: ${(merged.workstreams || []).length} workstream(s), ${(merged.tasks || []).length} task(s).`)
    return merged
  },
  // Stage 2 — Render the triad + per-plan STATUS from the one blueprint (4 files, parallel).
  async (bp, plan) => {
    const bpJson = JSON.stringify(bp, null, 2)
    const writes = await parallel([
      () => agent(renderScopePrompt(REPO, plan, README_PATH, REF_DIR, bpJson), { label: `render-scope:${plan.number}`, phase: 'Render', model: AUTHOR_MODEL, schema: OK_SCHEMA }),
      () => agent(renderArchPrompt(REPO, plan, README_PATH, REF_DIR, bpJson), { label: `render-arch:${plan.number}`, phase: 'Render', model: AUTHOR_MODEL, schema: OK_SCHEMA }),
      () => agent(renderTasksPrompt(REPO, plan, README_PATH, REF_DIR, bpJson, GATES), { label: `render-tasks:${plan.number}`, phase: 'Render', model: AUTHOR_MODEL, schema: OK_SCHEMA }),
      () => agent(renderStatusPrompt(REPO, plan, README_PATH, REF_DIR, bpJson), { label: `render-status:${plan.number}`, phase: 'Render', model: AUTHOR_MODEL, schema: OK_SCHEMA }),
    ])
    const ok = writes.filter(w => w && w.ok).length
    if (ok < 4) log(`render ${plan.number}: ${ok}/4 files written cleanly (${writes.map((w, k) => (w && w.ok) ? null : ['scope', 'arch', 'tasks', 'status'][k]).filter(Boolean).join(', ')} reported a problem).`)
    else log(`render ${plan.number}: SCOPE/ARCHITECTURE/TASKS/STATUS written to ${plan.planDir}.`)
    return { bp, writeOk: ok }
  },
  // Stage 3 — Verify (adversarial critic) + bounded fixer.
  async (r, plan) => ({ ...r, verify: await verifyAndFix(plan) }),
  // Stage 4 — Validate via implement-plan dryRun (real consumer), fold any safety edges.
  async (r, plan) => {
    const validate = await validatePlan(plan)
    return {
      number: plan.number, slug: plan.slug, title: plan.title, planDir: plan.planDir, statusPath: plan.statusPath,
      kind: plan.kind,
      taskCount: (r.bp && r.bp.tasks && r.bp.tasks.length) || (validate && validate.taskCount) || 0,
      plannedOutcome: (r.bp && r.bp.plannedOutcome) || plan.rationale,
      writeOk: r.writeOk, verify: r.verify, validate,
    }
  },
)

const ok = authored.filter(Boolean)
if (!ok.length) { log('No plans were authored successfully.'); return { error: 'authoring failed', plansDir: PLANS_DIR } }

// ── 7. Roll-up: add a 📋 Planned row per new plan to the board (one shared-file write) ─
phase('Roll-up')
let rollupDone = false
if (ROLLUP_PATH) {
  const rows = ok.map(p => ({ number: p.number, title: p.title, taskCount: p.taskCount, outcome: p.plannedOutcome, statusLink: `${p.number}-${p.slug}/STATUS.md` }))
  const rr = await agent(rollupPrompt(REPO, ROLLUP_PATH, rows), { label: 'rollup', phase: 'Roll-up', model: CRITIC_MODEL, schema: OK_SCHEMA })
  rollupDone = !!(rr && rr.ok)
  log(rollupDone ? `Roll-up board updated with ${rows.length} 📋 Planned row(s) (UNCOMMITTED).` : `Roll-up update failed (${(rr && rr.error) || 'unknown'}); add the row(s) by hand.`)
} else {
  log('No roll-up board found; skipping the board row(s).')
}

// Optional commit (handy for ephemeral cloud runners that lose the working tree at session end).
let committed = false
if (DO_COMMIT) {
  const paths = []
  for (const p of ok) paths.push(`${p.planDir}`)
  if (rollupDone && ROLLUP_PATH) paths.push(ROLLUP_PATH)
  const cm = await agent(commitPrompt(REPO, paths, ok.map(p => p.number)), { label: 'commit', phase: 'Roll-up', model: CHORE_MODEL, schema: OK_SCHEMA })
  committed = !!(cm && cm.ok)
  log(committed ? `Committed the authored plan(s)${ROLLUP_PATH ? ' + roll-up row(s)' : ''} (NOT pushed).` : `Commit failed (${(cm && cm.error) || 'unknown'}).`)
}

for (const p of ok) {
  if (p.validate && p.validate.edgesFolded === false) log(`WARNING ${p.number}: implement-plan surfaced safety edges that could NOT be folded into TASKS.md automatically — add them by hand before implementing.`)
}
const summary = ok.map(p => `${p.number}-${p.slug} (${p.taskCount} tasks${p.verify && p.verify.approved ? ', verified' : p.verify && p.verify.skipped ? '' : ', verify had findings'}${p.validate && p.validate.ok ? ', parse-OK' : ''})`).join('; ')
log(`Done: authored ${ok.length} plan(s) — ${summary}. Review the files, then run implement-plan (dryRun first), e.g. implement-plan ${ok[0].number}.`)
return {
  plansDir: PLANS_DIR,
  authored: ok.map(p => ({
    number: p.number, slug: p.slug, title: p.title, planDir: p.planDir, kind: p.kind,
    taskCount: p.taskCount, plannedOutcome: p.plannedOutcome,
    filesOk: p.writeOk, verified: !!(p.verify && p.verify.approved), parseOk: !!(p.validate && p.validate.ok),
    edgesFolded: !(p.validate && p.validate.edgesFolded === false),
  })),
  rollupUpdated: rollupDone, committed,
  skipped: skipped.map(c => ({ title: c.title, kind: c.kind })),
}
