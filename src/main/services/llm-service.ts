/**
 * LlmService — centralized wrapper for all model API calls.
 *
 * Responsibilities:
 * 1. WRAP — thin wrapper around the OpenAI chat completions API
 * 2. RETRY — exponential backoff on transient failures
 * 3. RECORD — every call logged to llm_calls for cost tracking
 *
 * REDACTION: Never records email body/subject. Only IDs and metadata.
 */
import OpenAI from "openai";
import { randomUUID } from "crypto";
import { createLogger } from "./logger";

const log = createLogger("llm");

type TextBlock = {
  type: "text";
  text: string;
  cache_control?: { type: string };
};

type LlmRequestMessage = {
  role: "user" | "assistant";
  content: string;
};

export type MessageCreateParamsNonStreaming = {
  model: string;
  max_tokens?: number;
  system?: TextBlock[];
  messages: LlmRequestMessage[];
  thinking?: {
    type?: string;
    budget_tokens?: number;
  };
};

export type Message = {
  id: string;
  type: "message";
  role: "assistant";
  content: Array<{ type: "text"; text: string }>;
  model: string;
  stop_reason: "end_turn" | "max_tokens" | "stop_sequence";
  stop_sequence: string | null;
  usage: {
    input_tokens: number;
    output_tokens: number;
    cache_read_input_tokens?: number;
    cache_creation_input_tokens?: number;
  };
};

// Approximate pricing per million tokens. Last updated: 2026-04-04.
const PRICING: Record<
  string,
  { input: number; output: number; cacheRead: number; cacheWrite: number }
> = {
  "gpt-5.4": { input: 4.0, output: 32.0, cacheRead: 0.4, cacheWrite: 0 },
  "gpt-5.4-mini": { input: 0.75, output: 3.0, cacheRead: 0.075, cacheWrite: 0 },
  "gpt-5-mini": { input: 0.25, output: 2.0, cacheRead: 0.025, cacheWrite: 0 },
  "gpt-4.1": { input: 2.0, output: 8.0, cacheRead: 0.5, cacheWrite: 0 },
  "gpt-4.1-mini": { input: 0.4, output: 1.6, cacheRead: 0.1, cacheWrite: 0 },
};

const DEFAULT_PRICING = PRICING["gpt-5.4-mini"];

interface RetryConfig {
  maxRetries: number;
  initialDelayMs: number;
  maxDelayMs: number;
}

const RETRY_CONFIGS: Record<string, RetryConfig> = {
  rate_limit: { maxRetries: 5, initialDelayMs: 1000, maxDelayMs: 30000 },
  server_error: { maxRetries: 3, initialDelayMs: 2000, maxDelayMs: 30000 },
  connection: { maxRetries: 3, initialDelayMs: 1000, maxDelayMs: 10000 },
};

export interface LlmCallRecord {
  id: string;
  created_at: string;
  model: string;
  caller: string;
  email_id: string | null;
  account_id: string | null;
  input_tokens: number;
  output_tokens: number;
  cache_read_tokens: number;
  cache_create_tokens: number;
  cost_cents: number;
  duration_ms: number;
  success: number;
  error_message: string | null;
}

export interface UsageStats {
  today: { totalCostCents: number; totalCalls: number };
  thisWeek: { totalCostCents: number; totalCalls: number };
  thisMonth: { totalCostCents: number; totalCalls: number };
  byModel: Array<{ model: string; costCents: number; calls: number }>;
  byCaller: Array<{ caller: string; costCents: number; calls: number }>;
}

interface CreateOptions {
  caller: string;
  emailId?: string;
  accountId?: string;
  timeoutMs?: number;
}

type DatabaseInstance = {
  prepare: (sql: string) => {
    run: (...args: unknown[]) => void;
    get: (...args: unknown[]) => unknown;
    all: (...args: unknown[]) => unknown[];
  };
  exec: (sql: string) => void;
  transaction: <T>(fn: () => T) => () => T;
};

let _testClient: unknown = null;
let _defaultClient: OpenAI | null = null;
let _db: DatabaseInstance | null = null;
let _insertStmt: ReturnType<DatabaseInstance["prepare"]> | null = null;

function clearDbState(): void {
  _db = null;
  _insertStmt = null;
}

export function _setClientForTesting(client: unknown): void {
  _testClient = client;
}

export function resetClient(): void {
  _defaultClient = null;
}

export function getClient(): OpenAI {
  if (_testClient) return _testClient as OpenAI;
  if (!_defaultClient) _defaultClient = new OpenAI();
  return _defaultClient;
}

export function setLlmServiceDb(db: DatabaseInstance): void {
  clearDbState();
  _db = db;
  db.exec(`
    CREATE TABLE IF NOT EXISTS llm_calls (
      id TEXT PRIMARY KEY,
      created_at TEXT NOT NULL DEFAULT (datetime('now')),
      model TEXT NOT NULL,
      caller TEXT NOT NULL,
      email_id TEXT,
      account_id TEXT,
      input_tokens INTEGER NOT NULL,
      output_tokens INTEGER NOT NULL,
      cache_read_tokens INTEGER DEFAULT 0,
      cache_create_tokens INTEGER DEFAULT 0,
      cost_cents REAL NOT NULL,
      duration_ms INTEGER NOT NULL,
      success INTEGER NOT NULL DEFAULT 1,
      error_message TEXT
    );
    CREATE INDEX IF NOT EXISTS idx_llm_calls_created ON llm_calls(created_at);
    CREATE INDEX IF NOT EXISTS idx_llm_calls_caller ON llm_calls(caller);
  `);
  _insertStmt = db.prepare(`
    INSERT INTO llm_calls (id, model, caller, email_id, account_id,
      input_tokens, output_tokens, cache_read_tokens, cache_create_tokens,
      cost_cents, duration_ms, success, error_message)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
  `);
}

// Compatibility alias to avoid a wide rename in one patch.
export const setAnthropicServiceDb = setLlmServiceDb;

function calculateCostCents(
  model: string,
  inputTokens: number,
  outputTokens: number,
  cacheReadTokens: number,
  cacheCreateTokens: number,
): number {
  const pricing = PRICING[model] || DEFAULT_PRICING;
  const inputCost = (inputTokens * pricing.input) / 1_000_000;
  const outputCost = (outputTokens * pricing.output) / 1_000_000;
  const cacheReadCost = (cacheReadTokens * pricing.cacheRead) / 1_000_000;
  const cacheWriteCost = (cacheCreateTokens * pricing.cacheWrite) / 1_000_000;
  return (inputCost + outputCost + cacheReadCost + cacheWriteCost) * 100;
}

function recordCall(
  model: string,
  caller: string,
  emailId: string | null,
  accountId: string | null,
  inputTokens: number,
  outputTokens: number,
  cacheReadTokens: number,
  cacheCreateTokens: number,
  durationMs: number,
  success: boolean,
  errorMessage: string | null,
): void {
  if (!_insertStmt) {
    log.warn("LlmService: database not initialized, skipping call recording");
    return;
  }

  const costCents = calculateCostCents(
    model,
    inputTokens,
    outputTokens,
    cacheReadTokens,
    cacheCreateTokens,
  );

  try {
    _insertStmt.run(
      randomUUID(),
      model,
      caller,
      emailId,
      accountId,
      inputTokens,
      outputTokens,
      cacheReadTokens,
      cacheCreateTokens,
      costCents,
      durationMs,
      success ? 1 : 0,
      errorMessage,
    );
  } catch (err) {
    const message = err instanceof Error ? err.message : String(err);
    if (/database connection is not open/i.test(message)) {
      clearDbState();
      log.warn("LlmService: database connection closed, disabling call recording");
      return;
    }
    log.error({ err }, "Failed to record LLM call to database");
  }
}

function extractUsage(usage: Record<string, unknown> | null | undefined): {
  inputTokens: number;
  outputTokens: number;
  cacheReadTokens: number;
  cacheCreateTokens: number;
} {
  const promptTokenDetails = (usage?.prompt_tokens_details as Record<string, unknown> | undefined) || {};
  const cacheReadTokens =
    Number(usage?.cache_read_input_tokens) || Number(promptTokenDetails.cached_tokens) || 0;
  const rawInputTokens = Number(usage?.input_tokens) || 0;
  const rawPromptTokens = Number(usage?.prompt_tokens) || 0;
  const promptTokens =
    rawInputTokens ||
    Math.max(rawPromptTokens - cacheReadTokens, 0) ||
    Number(promptTokenDetails.audio_tokens) ||
    0;
  const completionTokens =
    Number(usage?.output_tokens) || Number(usage?.completion_tokens) || 0;
  const cacheCreateTokens = Number(usage?.cache_creation_input_tokens) || 0;

  return {
    inputTokens: promptTokens,
    outputTokens: completionTokens,
    cacheReadTokens,
    cacheCreateTokens,
  };
}

export function recordStreamingCall(
  model: string,
  caller: string,
  usage: Record<string, number>,
  durationMs: number,
  options?: { emailId?: string; accountId?: string },
): void {
  const extracted = extractUsage(usage);
  recordCall(
    model,
    caller,
    options?.emailId || null,
    options?.accountId || null,
    extracted.inputTokens,
    extracted.outputTokens,
    extracted.cacheReadTokens,
    extracted.cacheCreateTokens,
    durationMs,
    true,
    null,
  );
}

function asyncSleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function getRetryCategory(error: unknown): string | null {
  const status = Number((error as { status?: number } | undefined)?.status);
  const name = (error as { name?: string } | undefined)?.name;

  if (status === 429 || name === "RateLimitError") return "rate_limit";
  if ((status >= 500 && status < 600) || name === "InternalServerError") return "server_error";
  if (name === "APIConnectionError" || name === "TimeoutError") return "connection";
  return null;
}

function flattenSystemPrompt(blocks: TextBlock[] | undefined): string | undefined {
  if (!blocks || blocks.length === 0) return undefined;
  const text = blocks
    .filter((block) => block.type === "text")
    .map((block) => block.text)
    .join("\n\n")
    .trim();
  return text || undefined;
}

function mapFinishReason(
  finishReason: string | null | undefined,
): "end_turn" | "max_tokens" | "stop_sequence" {
  if (finishReason === "length") return "max_tokens";
  if (finishReason === "stop") return "end_turn";
  return "stop_sequence";
}

function extractCompletionText(content: unknown): string {
  if (typeof content === "string") return content;
  if (Array.isArray(content)) {
    return content
      .map((part) => {
        if (typeof part === "string") return part;
        if (part && typeof part === "object" && "text" in part) {
          return String((part as { text?: unknown }).text ?? "");
        }
        return "";
      })
      .join("");
  }
  return "";
}

async function callViaOpenAI(
  client: OpenAI,
  params: MessageCreateParamsNonStreaming,
  signal?: AbortSignal,
): Promise<Message> {
  const systemPrompt = flattenSystemPrompt(params.system);
  const messages = [
    ...(systemPrompt ? [{ role: "developer", content: systemPrompt }] : []),
    ...params.messages.map((message) => ({
      role: message.role,
      content: message.content,
    })),
  ];

  const completion = await client.chat.completions.create(
    {
      model: params.model,
      messages,
      max_completion_tokens: params.max_tokens,
    },
    { signal },
  );

  const choice = completion.choices[0];
  const text = extractCompletionText(choice?.message?.content);
  const usage = extractUsage(completion.usage as Record<string, unknown> | undefined);

  return {
    id: completion.id,
    type: "message",
    role: "assistant",
    content: [{ type: "text", text }],
    model: completion.model,
    stop_reason: mapFinishReason(choice?.finish_reason),
    stop_sequence: null,
    usage: {
      input_tokens: usage.inputTokens,
      output_tokens: usage.outputTokens,
      cache_read_input_tokens: usage.cacheReadTokens,
      cache_creation_input_tokens: usage.cacheCreateTokens,
    },
  };
}

async function callViaTestDouble(
  client: unknown,
  params: MessageCreateParamsNonStreaming,
  signal?: AbortSignal,
): Promise<Message> {
  const legacyClient = client as {
    messages?: {
      create?: (
        params: MessageCreateParamsNonStreaming,
        options?: { signal?: AbortSignal },
      ) => Promise<Message>;
    };
  };

  if (!legacyClient.messages?.create) {
    throw new Error("Testing client must expose chat.completions.create() or messages.create()");
  }

  return legacyClient.messages.create(params, { signal });
}

export async function createMessage(
  params: MessageCreateParamsNonStreaming,
  options: CreateOptions,
): Promise<Message> {
  const { caller, emailId, accountId, timeoutMs } = options;
  const model = params.model;
  const startTime = Date.now();
  const client = getClient();
  let lastError: unknown = null;
  let totalAttempts = 0;
  const maxPossibleRetries = Math.max(...Object.values(RETRY_CONFIGS).map((c) => c.maxRetries));

  for (let attempt = 0; attempt <= maxPossibleRetries; attempt++) {
    totalAttempts = attempt + 1;

    let abortController: AbortController | undefined;
    let timeoutHandle: ReturnType<typeof setTimeout> | undefined;
    if (timeoutMs) {
      abortController = new AbortController();
      timeoutHandle = setTimeout(() => abortController?.abort(), timeoutMs);
    }

    try {
      const response =
        (client as { chat?: { completions?: { create?: unknown } } }).chat?.completions?.create
          ? await callViaOpenAI(client, params, abortController?.signal)
          : await callViaTestDouble(client, params, abortController?.signal);

      const usage = extractUsage(response.usage as Record<string, unknown>);
      recordCall(
        model,
        caller,
        emailId || null,
        accountId || null,
        usage.inputTokens,
        usage.outputTokens,
        usage.cacheReadTokens,
        usage.cacheCreateTokens,
        Date.now() - startTime,
        true,
        null,
      );

      if (totalAttempts > 1) {
        log.info({ caller, model, attempts: totalAttempts }, "LLM call succeeded after retries");
      }

      return response;
    } catch (error) {
      lastError = error;
      const category = getRetryCategory(error);

      if (!category) break;

      const config = RETRY_CONFIGS[category];
      if (attempt >= config.maxRetries) break;

      const baseDelay = Math.min(config.initialDelayMs * Math.pow(2, attempt), config.maxDelayMs);
      const jitter = baseDelay * 0.1 * Math.random();
      const delay = baseDelay + jitter;

      log.warn(
        {
          caller,
          model,
          attempt: attempt + 1,
          maxRetries: config.maxRetries,
          category,
          delayMs: Math.round(delay),
        },
        "LLM call failed, retrying",
      );

      await asyncSleep(delay);
    } finally {
      if (timeoutHandle) clearTimeout(timeoutHandle);
    }
  }

  const errMsg = lastError instanceof Error ? lastError.message : String(lastError);
  recordCall(
    model,
    caller,
    emailId || null,
    accountId || null,
    0,
    0,
    0,
    0,
    Date.now() - startTime,
    false,
    errMsg,
  );

  throw lastError;
}

export function getUsageStats(): UsageStats {
  if (!_db) {
    return {
      today: { totalCostCents: 0, totalCalls: 0 },
      thisWeek: { totalCostCents: 0, totalCalls: 0 },
      thisMonth: { totalCostCents: 0, totalCalls: 0 },
      byModel: [],
      byCaller: [],
    };
  }

  try {
    const today = _db
      .prepare(
        "SELECT COALESCE(SUM(cost_cents), 0) as cost, COUNT(*) as calls FROM llm_calls WHERE date(created_at) = date('now')",
      )
      .get() as { cost: number; calls: number };

    const thisWeek = _db
      .prepare(
        "SELECT COALESCE(SUM(cost_cents), 0) as cost, COUNT(*) as calls FROM llm_calls WHERE created_at >= datetime('now', '-7 days')",
      )
      .get() as { cost: number; calls: number };

    const thisMonth = _db
      .prepare(
        "SELECT COALESCE(SUM(cost_cents), 0) as cost, COUNT(*) as calls FROM llm_calls WHERE created_at >= datetime('now', '-30 days')",
      )
      .get() as { cost: number; calls: number };

    const byModel = _db
      .prepare(
        "SELECT model, COALESCE(SUM(cost_cents), 0) as costCents, COUNT(*) as calls FROM llm_calls WHERE created_at >= datetime('now', '-30 days') GROUP BY model ORDER BY costCents DESC",
      )
      .all() as Array<{ model: string; costCents: number; calls: number }>;

    const byCaller = _db
      .prepare(
        "SELECT caller, COALESCE(SUM(cost_cents), 0) as costCents, COUNT(*) as calls FROM llm_calls WHERE created_at >= datetime('now', '-30 days') GROUP BY caller ORDER BY costCents DESC",
      )
      .all() as Array<{ caller: string; costCents: number; calls: number }>;

    return {
      today: { totalCostCents: today.cost, totalCalls: today.calls },
      thisWeek: { totalCostCents: thisWeek.cost, totalCalls: thisWeek.calls },
      thisMonth: { totalCostCents: thisMonth.cost, totalCalls: thisMonth.calls },
      byModel,
      byCaller,
    };
  } catch (err) {
    const message = err instanceof Error ? err.message : String(err);
    if (/database connection is not open/i.test(message)) {
      clearDbState();
      return {
        today: { totalCostCents: 0, totalCalls: 0 },
        thisWeek: { totalCostCents: 0, totalCalls: 0 },
        thisMonth: { totalCostCents: 0, totalCalls: 0 },
        byModel: [],
        byCaller: [],
      };
    }
    throw err;
  }
}

export function getCallHistory(limit: number = 50): LlmCallRecord[] {
  if (!_db) return [];

  try {
    return _db
      .prepare("SELECT * FROM llm_calls ORDER BY created_at DESC LIMIT ?")
      .all(limit) as LlmCallRecord[];
  } catch (err) {
    const message = err instanceof Error ? err.message : String(err);
    if (/database connection is not open/i.test(message)) {
      clearDbState();
      return [];
    }
    throw err;
  }
}
