import fs from "node:fs";
import path from "node:path";
import type { OpenClawConfig } from "../config/config.js";
import { createSubsystemLogger } from "../logging/subsystem.js";
import type {
  Options as SDKOptions,
  SDKMessage,
  ThinkingConfig,
} from "./agent-sdk-runner.runtime.js";
/**
 * Agent SDK runner — executes prompts via the Claude Agent SDK
 * (`@anthropic-ai/claude-agent-sdk`).  The SDK spawns the real Claude Code
 * binary which uses its own `~/.claude/` auth (Max subscription / API key),
 * completely separate from OpenClaw's setup-token auth.
 *
 * This is registered as the `agent-sdk` provider and wired into cron dispatch
 * alongside the existing CLI runner and embedded Pi runner.
 */
import { FailoverError } from "./failover-error.js";
import type { ThinkLevel } from "./model-selection.js";
import type { EmbeddedPiRunResult } from "./pi-embedded-runner/types.js";

const log = createSubsystemLogger("agent-sdk-runner");

// Keys to strip from the spawned process env so the SDK uses Claude CLI's
// native auth rather than a stray API key from the host environment.
const STRIP_ENV_KEYS = ["ANTHROPIC_API_KEY", "ANTHROPIC_API_KEY_OLD"];

export type AgentSdkRunnerParams = {
  sessionId: string;
  sessionKey?: string;
  agentId?: string;
  workspaceDir: string;
  config?: OpenClawConfig;
  prompt: string;
  /** Model portion after the provider slash, e.g. "opus" from "agent-sdk/opus". */
  model?: string;
  thinkLevel?: ThinkLevel;
  timeoutMs: number;
  runId: string;
  maxTurns?: number;
  allowedTools?: string[];
  disallowedTools?: string[];
  abortSignal?: AbortSignal;
};

/** Map OpenClaw ThinkLevel to SDK ThinkingConfig. */
function resolveThinkingConfig(level: ThinkLevel | undefined): ThinkingConfig | undefined {
  if (!level || level === "off") {
    return { type: "disabled" };
  }
  if (level === "adaptive") {
    return { type: "adaptive" };
  }
  // For explicit budget levels, map to token budgets.
  const budgetMap: Record<string, number> = {
    minimal: 1024,
    low: 4096,
    medium: 10240,
    high: 32768,
    xhigh: 65536,
  };
  const budget = budgetMap[level];
  if (budget !== undefined) {
    return { type: "enabled", budgetTokens: budget };
  }
  return { type: "adaptive" };
}

/** Map SDK model shorthand to full model ID. */
function resolveSDKModel(model: string | undefined): string | undefined {
  if (!model) {
    return undefined;
  }
  const aliases: Record<string, string> = {
    opus: "claude-opus-4-6",
    sonnet: "claude-sonnet-4-6",
    haiku: "claude-haiku-4-5-20251001",
  };
  return aliases[model.toLowerCase()] ?? model;
}

/** Read CLAUDE.md from the workspace directory if it exists. */
function readWorkspaceSystemPrompt(workspaceDir: string): string | undefined {
  for (const name of ["CLAUDE.md", "AGENTS.md"]) {
    const filePath = path.join(workspaceDir, name);
    try {
      return fs.readFileSync(filePath, "utf-8").trim() || undefined;
    } catch {
      // File doesn't exist, try next.
    }
  }
  return undefined;
}

/** Build a clean env with ANTHROPIC_API_KEY stripped. */
function buildCleanEnv(): Record<string, string | undefined> {
  const env: Record<string, string | undefined> = { ...process.env };
  for (const key of STRIP_ENV_KEYS) {
    delete env[key];
  }
  return env;
}

export async function runAgentSdkAgent(params: AgentSdkRunnerParams): Promise<EmbeddedPiRunResult> {
  const startMs = Date.now();
  const abortController = new AbortController();

  // Wire external abort signal.
  if (params.abortSignal) {
    if (params.abortSignal.aborted) {
      return errorResult(startMs, "Aborted before start", true);
    }
    params.abortSignal.addEventListener("abort", () => abortController.abort(), { once: true });
  }

  // Timeout via AbortController.
  const timeoutId = setTimeout(() => abortController.abort(), params.timeoutMs);

  try {
    // Dynamic import — SDK binary only loaded when this provider is used.
    const { query } = await import("./agent-sdk-runner.runtime.js");

    const sdkModel = resolveSDKModel(params.model);
    const thinkingConfig = resolveThinkingConfig(params.thinkLevel);
    const env = buildCleanEnv();
    const workspacePrompt = readWorkspaceSystemPrompt(params.workspaceDir);

    // Read MCP servers from agent defaults config.
    const mcpServers = params.config?.agents?.defaults?.mcpServers;

    const options: SDKOptions = {
      abortController,
      cwd: params.workspaceDir,
      model: sdkModel,
      maxTurns: params.maxTurns ?? 25,
      thinking: thinkingConfig,
      permissionMode: "bypassPermissions",
      allowDangerouslySkipPermissions: true,
      allowedTools:
        params.allowedTools ??
        (mcpServers && Object.keys(mcpServers).length > 0
          ? undefined // When MCP servers are configured, don't restrict tools so MCP tools are accessible.
          : ["Read", "Bash", "Glob", "Grep", "Write", "Edit"]),
      disallowedTools: params.disallowedTools,
      systemPrompt: workspacePrompt,
      persistSession: false,
      env,
      ...(mcpServers && Object.keys(mcpServers).length > 0
        ? { mcpServers: mcpServers as SDKOptions["mcpServers"] }
        : {}),
    };

    const stream = query({ prompt: params.prompt, options });

    const textParts: string[] = [];
    let lastRateLimitResetsAt: number | undefined;

    for await (const message of stream) {
      handleMessage(message, textParts, params, (resetsAt) => {
        lastRateLimitResetsAt = resetsAt;
      });
    }

    // The SDK generator may end cleanly on abort instead of throwing.
    // Detect timeout (our AbortController fired, not the caller's signal).
    if (abortController.signal.aborted && !params.abortSignal?.aborted) {
      return errorResult(startMs, `Agent SDK run timed out after ${params.timeoutMs}ms`, true);
    }
    if (params.abortSignal?.aborted) {
      return errorResult(startMs, "Aborted by caller", true);
    }

    // If we collected no text but had a rate limit rejection, throw failover.
    if (textParts.length === 0 && lastRateLimitResetsAt !== undefined) {
      const resetsIn = Math.max(0, lastRateLimitResetsAt - Date.now());
      throw new FailoverError(
        `Rate limited by Claude Max subscription (resets in ${Math.ceil(resetsIn / 60_000)}m)`,
        { reason: "rate_limit" },
      );
    }

    const text = textParts.join("\n\n").trim();
    const durationMs = Date.now() - startMs;

    return {
      payloads: text ? [{ text }] : undefined,
      meta: {
        durationMs,
        agentMeta: {
          sessionId: params.sessionId,
          provider: "agent-sdk",
          model: sdkModel ?? "claude-sonnet-4-6",
        },
        stopReason: "completed",
      },
    };
  } catch (err) {
    if (abortController.signal.aborted && !params.abortSignal?.aborted) {
      // Our timeout fired.
      return errorResult(startMs, `Agent SDK run timed out after ${params.timeoutMs}ms`, true);
    }
    if (params.abortSignal?.aborted) {
      return errorResult(startMs, "Aborted by caller", true);
    }

    // Rethrow FailoverErrors for the fallback wrapper.
    if (err instanceof FailoverError) {
      throw err;
    }

    // Classify common SDK errors.
    const message = err instanceof Error ? err.message : String(err);
    const failover = classifySdkError(message);
    if (failover) {
      throw failover;
    }

    log.error(`Agent SDK run failed: ${err instanceof Error ? err.message : String(err)}`);
    return errorResult(startMs, message, false);
  } finally {
    clearTimeout(timeoutId);
  }
}

/**
 * Process a single SDK message, extracting text and detecting errors.
 */
function handleMessage(
  message: SDKMessage,
  textParts: string[],
  params: AgentSdkRunnerParams,
  onRateLimit: (resetsAt: number) => void,
): void {
  switch (message.type) {
    case "assistant": {
      // Check for assistant-level errors (auth, billing, rate limit).
      if (message.error) {
        handleAssistantError(message.error, params);
      }
      // Extract text content from the assistant message.
      if (message.message?.content) {
        for (const block of message.message.content) {
          if ("text" in block && typeof block.text === "string" && block.text.trim()) {
            textParts.push(block.text);
          }
        }
      }
      break;
    }
    case "result": {
      handleResultMessage(message, textParts);
      break;
    }
    case "rate_limit_event": {
      const info = message.rate_limit_info;
      if (info.status === "rejected") {
        log.warn(`Rate limit rejected (resets at ${info.resetsAt}, type: ${info.rateLimitType})`);
        if (info.resetsAt) {
          onRateLimit(info.resetsAt);
        }
        throw new FailoverError(`Rate limited (${info.rateLimitType ?? "unknown"})`, {
          reason: "rate_limit",
        });
      }
      if (info.status === "allowed_warning") {
        log.warn(
          `Rate limit warning: ${(info.utilization ?? 0) * 100}% utilization` +
            (info.resetsAt ? `, resets at ${new Date(info.resetsAt).toISOString()}` : ""),
        );
      }
      break;
    }
    case "auth_status": {
      if (message.error) {
        throw new FailoverError(`Auth failed: ${message.error}`, { reason: "auth" });
      }
      break;
    }
    default:
      // Ignore other message types (system, streaming, tool progress, etc.)
      break;
  }
}

function handleAssistantError(error: string, params: AgentSdkRunnerParams): void {
  switch (error) {
    case "authentication_failed":
      throw new FailoverError("Claude CLI authentication failed", { reason: "auth" });
    case "billing_error":
      throw new FailoverError("Billing error from Claude CLI", { reason: "billing" });
    case "rate_limit":
      throw new FailoverError("Rate limited by Claude API", { reason: "rate_limit" });
    case "server_error":
      throw new FailoverError("Server error from Claude API", { reason: "overloaded" });
    case "invalid_request":
      throw new FailoverError("Invalid request to Claude API", { reason: "format" });
    default:
      log.warn(`Assistant error: ${error} (session ${params.sessionId})`);
      break;
  }
}

function handleResultMessage(message: SDKMessage & { type: "result" }, textParts: string[]): void {
  if (message.subtype === "success") {
    const success = message as SDKMessage & { type: "result"; subtype: "success"; result: string };
    if (success.result?.trim() && textParts.length === 0) {
      textParts.push(success.result);
    }
  } else if (message.subtype === "error_max_turns") {
    log.warn("Agent SDK run hit max turns limit");
  } else if (message.subtype === "error_during_execution") {
    const errResult = message as SDKMessage & {
      type: "result";
      subtype: "error_during_execution";
      errors: string[];
    };
    const errMsg = errResult.errors?.join("; ") ?? "Unknown execution error";
    log.error(`Agent SDK execution error: ${errMsg}`);
  }
}

/** Classify error messages into FailoverError. */
function classifySdkError(message: string): FailoverError | null {
  const lower = message.toLowerCase();
  if (
    lower.includes("authentication") ||
    lower.includes("not logged in") ||
    lower.includes("auth")
  ) {
    return new FailoverError(message, { reason: "auth" });
  }
  if (lower.includes("rate limit") || lower.includes("429")) {
    return new FailoverError(message, { reason: "rate_limit" });
  }
  if (lower.includes("overloaded") || lower.includes("503")) {
    return new FailoverError(message, { reason: "overloaded" });
  }
  if (lower.includes("billing") || lower.includes("402")) {
    return new FailoverError(message, { reason: "billing" });
  }
  return null;
}

function errorResult(startMs: number, message: string, aborted: boolean): EmbeddedPiRunResult {
  return {
    payloads: [{ text: message, isError: true }],
    meta: {
      durationMs: Date.now() - startMs,
      aborted,
    },
  };
}
