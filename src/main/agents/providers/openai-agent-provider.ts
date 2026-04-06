import { z } from "zod";
import OpenAI from "openai";
import type {
  AgentEvent,
  AgentFrameworkConfig,
  AgentProvider,
  AgentProviderConfig,
  AgentRunParams,
  AgentRunResult,
  AgentToolSpec,
} from "../types";
import { createLogger } from "../../services/logger";
import { buildSystemPrompt } from "./claude-agent-provider";

const log = createLogger("openai-agent");
const MAX_TURNS = 25;

export class OpenAIAgentProvider implements AgentProvider {
  readonly config: AgentProviderConfig = {
    id: "openai",
    name: "OpenAI Agent",
    description: "OpenAI with built-in mail tools",
    auth: { type: "api_key", configKey: "OPENAI_API_KEY" },
  };

  private frameworkConfig: AgentFrameworkConfig;
  private activeAbortControllers = new Map<string, AbortController>();

  constructor(frameworkConfig: AgentFrameworkConfig) {
    this.frameworkConfig = frameworkConfig;
  }

  async *run(params: AgentRunParams): AsyncGenerator<AgentEvent, AgentRunResult, void> {
    const { taskId, prompt, context, tools, toolExecutor, signal, modelOverride } = params;
    const abortController = new AbortController();
    const onAbort = () => abortController.abort();
    signal.addEventListener("abort", onAbort, { once: true });
    this.activeAbortControllers.set(taskId, abortController);

    yield { type: "state", state: "running" };

    try {
      const apiKey = this.frameworkConfig.openaiApiKey || process.env.OPENAI_API_KEY;
      if (!apiKey) {
        throw new Error("OpenAI API key not configured");
      }

      const client = new OpenAI({ apiKey });
      const systemPrompt = buildSystemPrompt(context, tools, context.memoryContext);
      const messages: Array<Record<string, unknown>> = [
        { role: "developer", content: systemPrompt },
        { role: "user", content: prompt },
      ];

      for (let turn = 0; turn < MAX_TURNS; turn++) {
        if (abortController.signal.aborted) {
          yield { type: "state", state: "cancelled" };
          return { state: "cancelled" };
        }

        const completion = await client.chat.completions.create(
          {
            model: modelOverride ?? this.frameworkConfig.model,
            messages: messages as never,
            tools: tools.map(toOpenAIChatTool),
            tool_choice: "auto",
            parallel_tool_calls: false,
            max_completion_tokens: 4096,
          },
          { signal: abortController.signal },
        );

        const choice = completion.choices[0];
        const assistant = choice?.message;
        const assistantText = extractAssistantText(assistant?.content);
        const toolCalls = assistant?.tool_calls ?? [];

        messages.push({
          role: "assistant",
          content: assistant?.content ?? "",
          ...(toolCalls.length > 0 ? { tool_calls: toolCalls } : {}),
        });

        if (assistantText) {
          yield { type: "text_delta", text: assistantText };
        }

        if (toolCalls.length === 0) {
          yield { type: "done", summary: assistantText || "Completed" };
          return { state: "completed" };
        }

        for (const toolCall of toolCalls) {
          if (toolCall.type !== "function") continue;

          let input: Record<string, unknown> = {};
          try {
            input = JSON.parse(toolCall.function.arguments || "{}") as Record<string, unknown>;
          } catch {
            input = {};
          }

          yield {
            type: "tool_call_start",
            toolName: toolCall.function.name,
            toolCallId: toolCall.id,
            input,
          };

          let result: unknown;
          try {
            result = await toolExecutor(toolCall.function.name, input);
          } catch (err) {
            result = { error: err instanceof Error ? err.message : String(err) };
          }

          yield {
            type: "tool_call_end",
            toolCallId: toolCall.id,
            result,
          };

          messages.push({
            role: "tool",
            tool_call_id: toolCall.id,
            content: JSON.stringify(result),
          });
        }
      }

      yield { type: "error", message: `Agent reached the max turn limit (${MAX_TURNS})` };
      return { state: "failed" };
    } catch (err) {
      if (abortController.signal.aborted) {
        yield { type: "state", state: "cancelled" };
        return { state: "cancelled" };
      }

      const message = err instanceof Error ? err.message : String(err);
      log.error({ err }, "OpenAI agent run failed");
      yield { type: "error", message };
      return { state: "failed" };
    } finally {
      this.activeAbortControllers.delete(taskId);
      signal.removeEventListener("abort", onAbort);
    }
  }

  cancel(taskId: string): void {
    const controller = this.activeAbortControllers.get(taskId);
    controller?.abort();
    this.activeAbortControllers.delete(taskId);
  }

  async isAvailable(): Promise<boolean> {
    return Boolean(this.frameworkConfig.openaiApiKey || process.env.OPENAI_API_KEY);
  }

  updateConfig(config: Partial<AgentFrameworkConfig>): void {
    this.frameworkConfig = { ...this.frameworkConfig, ...config };
  }
}

function toOpenAIChatTool(spec: AgentToolSpec) {
  const jsonSchema = z.toJSONSchema(spec.inputSchema);
  const { $schema: _, ...parameters } = jsonSchema;
  return {
    type: "function" as const,
    function: {
      name: spec.name,
      description: spec.description,
      parameters,
    },
  };
}

function extractAssistantText(content: unknown): string {
  if (typeof content === "string") return content;
  if (!Array.isArray(content)) return "";

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
