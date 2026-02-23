/**
 * LangCore API — TypeScript client example.
 *
 * Demonstrates:
 *   1. Submit an extraction from raw text.
 *   2. Submit an extraction from a URL.
 *   3. Submit a batch.
 *   4. Poll a task until it completes.
 *
 * Uses the built-in `fetch` API (Node.js 18+). No extra dependencies needed.
 *
 * Usage:
 *   npx ts-node examples/typescript/client.ts
 *   # or compile first:
 *   tsc examples/typescript/client.ts --target es2022 --moduleResolution node
 *   node examples/typescript/client.js
 */

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

const API_BASE = process.env.API_BASE ?? "http://localhost:8000/api/v1";
const DEFAULT_PROVIDER = "gpt-4o";
const POLL_INTERVAL_MS = 2_000;
const POLL_TIMEOUT_MS = 120_000;

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

interface SubmitResponse {
  task_id: string;
  status: string;
  message?: string;
}

interface Entity {
  extraction_class: string;
  extraction_text: string;
  attributes?: Record<string, string>;
  char_start?: number;
  char_end?: number;
}

interface TaskResult {
  entities: Entity[];
  metadata?: Record<string, unknown>;
}

interface TaskResponse {
  task_id: string;
  state: string;
  result?: TaskResult;
  error?: string;
}

interface BatchSubmitResponse {
  batch_id: string;
  task_ids?: string[];
  batch_task_id?: string;
}

interface ExtractionRequest {
  raw_text?: string;
  document_url?: string;
  provider?: string;
  passes?: number;
  idempotency_key?: string;
  extraction_config?: {
    prompt_description?: string;
    temperature?: number;
    examples?: unknown[];
  };
}

interface BatchRequest {
  batch_id: string;
  documents: ExtractionRequest[];
  provider?: string;
  callback_url?: string;
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/** POST /extract — submit a single extraction task. */
async function submitExtraction(
  payload: ExtractionRequest,
): Promise<SubmitResponse> {
  const res = await fetch(`${API_BASE}/extract`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!res.ok) {
    const detail = await res.text();
    throw new Error(`HTTP ${res.status}: ${detail}`);
  }
  return res.json() as Promise<SubmitResponse>;
}

/** POST /extract/batch — submit multiple documents. */
async function submitBatch(
  payload: BatchRequest,
): Promise<BatchSubmitResponse> {
  const res = await fetch(`${API_BASE}/extract/batch`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!res.ok) {
    const detail = await res.text();
    throw new Error(`HTTP ${res.status}: ${detail}`);
  }
  return res.json() as Promise<BatchSubmitResponse>;
}

/** GET /tasks/{taskId} — return current task state. */
async function getTaskStatus(taskId: string): Promise<TaskResponse> {
  const res = await fetch(`${API_BASE}/tasks/${taskId}`);
  if (!res.ok) {
    const detail = await res.text();
    throw new Error(`HTTP ${res.status}: ${detail}`);
  }
  return res.json() as Promise<TaskResponse>;
}

/** Poll /tasks/{taskId} until SUCCESS or FAILURE. */
async function pollTask(taskId: string): Promise<TaskResponse> {
  const deadline = Date.now() + POLL_TIMEOUT_MS;
  while (Date.now() < deadline) {
    const data = await getTaskStatus(taskId);
    console.log(`  [${taskId.slice(0, 8)}…] state=${data.state}`);
    if (data.state === "SUCCESS" || data.state === "FAILURE") {
      return data;
    }
    await new Promise((r) => setTimeout(r, POLL_INTERVAL_MS));
  }
  throw new Error(`Task ${taskId} did not finish within ${POLL_TIMEOUT_MS}ms`);
}

// ---------------------------------------------------------------------------
// Examples
// ---------------------------------------------------------------------------

async function exampleRawText(): Promise<void> {
  console.log("\n── Raw text extraction ──────────────────────────");
  const submit = await submitExtraction({
    raw_text:
      "AGREEMENT dated January 15, 2025 between Acme Corporation (Seller) " +
      "and Beta LLC (Buyer). Purchase price: $12,500 for 500 widgets at $25 " +
      "each. Payment: net 30 days. Governed by Delaware law.",
    provider: DEFAULT_PROVIDER,
    passes: 1,
    idempotency_key: "demo-raw-text-001",
    extraction_config: { temperature: 0.2 },
  });
  console.log(`Submitted: task_id=${submit.task_id}`);

  const final = await pollTask(submit.task_id);
  const entities = final.result?.entities ?? [];
  console.log(`Done — ${entities.length} entities extracted:`);
  for (const ent of entities) {
    console.log(
      `  [${ent.extraction_class}] ${JSON.stringify(ent.extraction_text)}`,
    );
  }
}

async function exampleUrl(): Promise<void> {
  console.log("\n── URL extraction ───────────────────────────────");
  const submit = await submitExtraction({
    document_url: "https://storage.example.com/contracts/agreement-2025.txt",
    provider: DEFAULT_PROVIDER,
    extraction_config: {
      prompt_description: "Extract any organisations, dates, and legal terms.",
      temperature: 0.1,
    },
  });
  console.log(`Submitted: task_id=${submit.task_id}`);

  const final = await pollTask(submit.task_id);
  const entities = final.result?.entities ?? [];
  console.log(`Done — ${entities.length} entities extracted.`);
}

async function exampleBatch(): Promise<void> {
  console.log("\n── Batch extraction ─────────────────────────────");
  const submit = await submitBatch({
    batch_id: "demo-batch-001",
    documents: [
      {
        raw_text:
          "Contract A: Acme Corp sells 500 units to Beta LLC for $12,500. " +
          "Delivery Q2 2025.",
      },
      {
        raw_text:
          "Contract B: Charlie Enterprises leases warehouse space from " +
          "Delta Holdings at $3,200/month for 24 months.",
      },
      {
        raw_text:
          "Contract C: Echo Inc purchases software licences from " +
          "Foxtrot SaaS Ltd at $9,000/year, auto-renewing annually.",
      },
    ],
    provider: DEFAULT_PROVIDER,
  });

  const taskIds = submit.task_ids ?? [];
  console.log(`Batch submitted — ${taskIds.length} task(s):`);
  for (const tid of taskIds) {
    console.log(`  task_id=${tid}`);
  }

  for (const tid of taskIds) {
    const final = await pollTask(tid);
    const entities = final.result?.entities ?? [];
    console.log(
      `  [${tid.slice(0, 8)}…] finished — ${entities.length} entities`,
    );
  }
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

(async () => {
  try {
    await exampleRawText();
    await exampleUrl();
    await exampleBatch();
  } catch (err) {
    console.error("Error:", err);
    process.exit(1);
  }
})();
