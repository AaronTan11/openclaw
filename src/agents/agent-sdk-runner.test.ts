import { beforeEach, describe, expect, it, vi } from "vitest";
import type { AgentSdkRunnerParams } from "./agent-sdk-runner.js";
import { FailoverError } from "./failover-error.js";
import { isAgentSdkProvider } from "./model-selection.js";

// Mock the dynamic import boundary — we never actually spawn the SDK binary.
const mockQuery = vi.fn();
vi.mock("./agent-sdk-runner.runtime.js", () => ({
  query: (...args: unknown[]) => mockQuery(...args),
}));

// Import after mocks are registered.
const { runAgentSdkAgent } = await import("./agent-sdk-runner.js");

function baseParams(overrides?: Partial<AgentSdkRunnerParams>): AgentSdkRunnerParams {
  return {
    sessionId: "test-session-1",
    workspaceDir: "/tmp/workspace",
    prompt: "Say hello",
    timeoutMs: 30_000,
    runId: "run-1",
    ...overrides,
  };
}

/** Helper: create an async generator from an array of SDK messages. */
async function* messagesGenerator(messages: Array<Record<string, unknown>>) {
  for (const msg of messages) {
    yield msg;
  }
}

describe("runAgentSdkAgent", () => {
  beforeEach(() => {
    mockQuery.mockReset();
  });

  it("returns text from a successful single-turn query", async () => {
    mockQuery.mockReturnValue(
      messagesGenerator([
        {
          type: "assistant",
          message: {
            content: [{ type: "text", text: "Hello!" }],
          },
          uuid: "uuid-1",
          session_id: "s1",
        },
        {
          type: "result",
          subtype: "success",
          result: "Hello!",
          duration_ms: 500,
          duration_api_ms: 400,
          is_error: false,
          num_turns: 1,
          total_cost_usd: 0,
          usage: { input_tokens: 10, output_tokens: 5 },
          modelUsage: {},
          permission_denials: [],
          uuid: "uuid-2",
          session_id: "s1",
        },
      ]),
    );

    const result = await runAgentSdkAgent(baseParams());

    expect(result.payloads).toEqual([{ text: "Hello!" }]);
    expect(result.meta.durationMs).toBeGreaterThan(0);
    expect(result.meta.agentMeta?.provider).toBe("agent-sdk");
    expect(result.meta.stopReason).toBe("completed");
  });

  it("returns error result on timeout", async () => {
    // The real SDK Query generator terminates when its AbortController fires.
    // Simulate that by resolving the pending promise on abort.
    mockQuery.mockImplementation(
      ({ options }: { options: { abortController: AbortController } }) => {
        const ac = options.abortController;
        // Return an async iterable that blocks until abort (simulating a long SDK run).
        return {
          [Symbol.asyncIterator]() {
            let done = false;
            return {
              async next() {
                if (done) {
                  return { done: true as const, value: undefined };
                }
                await new Promise<void>((resolve) => {
                  if (ac.signal.aborted) {
                    resolve();
                    return;
                  }
                  ac.signal.addEventListener("abort", () => resolve(), { once: true });
                });
                done = true;
                return { done: true as const, value: undefined };
              },
            };
          },
        };
      },
    );

    const result = await runAgentSdkAgent(baseParams({ timeoutMs: 50 }));

    expect(result.payloads).toHaveLength(1);
    expect(result.payloads![0].isError).toBe(true);
    expect(result.payloads![0].text).toContain("timed out");
    expect(result.meta.aborted).toBe(true);
  });

  it("throws FailoverError with 'rate_limit' on rate limit rejection", async () => {
    mockQuery.mockReturnValue(
      messagesGenerator([
        {
          type: "rate_limit_event",
          rate_limit_info: {
            status: "rejected",
            rateLimitType: "five_hour",
            resetsAt: Date.now() + 300_000,
          },
          uuid: "uuid-3",
          session_id: "s1",
        },
      ]),
    );

    await expect(runAgentSdkAgent(baseParams())).rejects.toThrow(FailoverError);
    try {
      await runAgentSdkAgent(baseParams());
    } catch (err) {
      expect(err).toBeInstanceOf(FailoverError);
      expect((err as FailoverError).reason).toBe("rate_limit");
    }
  });

  it("throws FailoverError with 'auth' on authentication error", async () => {
    mockQuery.mockReturnValue(
      messagesGenerator([
        {
          type: "assistant",
          error: "authentication_failed",
          message: { content: [] },
          uuid: "uuid-4",
          session_id: "s1",
        },
      ]),
    );

    await expect(runAgentSdkAgent(baseParams())).rejects.toThrow(FailoverError);
    try {
      await runAgentSdkAgent(baseParams());
    } catch (err) {
      expect((err as FailoverError).reason).toBe("auth");
    }
  });

  it("throws FailoverError with 'billing' on billing error", async () => {
    mockQuery.mockReturnValue(
      messagesGenerator([
        {
          type: "assistant",
          error: "billing_error",
          message: { content: [] },
          uuid: "uuid-5",
          session_id: "s1",
        },
      ]),
    );

    await expect(runAgentSdkAgent(baseParams())).rejects.toThrow(FailoverError);
    try {
      await runAgentSdkAgent(baseParams());
    } catch (err) {
      expect((err as FailoverError).reason).toBe("billing");
    }
  });

  it("handles auth_status error messages", async () => {
    mockQuery.mockReturnValue(
      messagesGenerator([
        {
          type: "auth_status",
          isAuthenticating: false,
          output: [],
          error: "Token expired",
          uuid: "uuid-6",
          session_id: "s1",
        },
      ]),
    );

    await expect(runAgentSdkAgent(baseParams())).rejects.toThrow(FailoverError);
    try {
      await runAgentSdkAgent(baseParams());
    } catch (err) {
      expect((err as FailoverError).reason).toBe("auth");
    }
  });

  it("passes model and thinking config correctly", async () => {
    mockQuery.mockReturnValue(
      messagesGenerator([
        {
          type: "result",
          subtype: "success",
          result: "Done",
          duration_ms: 100,
          duration_api_ms: 80,
          is_error: false,
          num_turns: 1,
          total_cost_usd: 0,
          usage: {},
          modelUsage: {},
          permission_denials: [],
          uuid: "uuid-7",
          session_id: "s1",
        },
      ]),
    );

    await runAgentSdkAgent(baseParams({ model: "opus", thinkLevel: "high" }));

    expect(mockQuery).toHaveBeenCalledOnce();
    const callArgs = mockQuery.mock.calls[0][0];
    expect(callArgs.options.model).toBe("claude-opus-4-6");
    expect(callArgs.options.thinking).toEqual({ type: "enabled", budgetTokens: 32768 });
    expect(callArgs.options.permissionMode).toBe("bypassPermissions");
    expect(callArgs.options.persistSession).toBe(false);
  });

  it("strips ANTHROPIC_API_KEY from env", async () => {
    const origKey = process.env.ANTHROPIC_API_KEY;
    process.env.ANTHROPIC_API_KEY = "sk-test-key";

    mockQuery.mockReturnValue(
      messagesGenerator([
        {
          type: "result",
          subtype: "success",
          result: "ok",
          duration_ms: 50,
          duration_api_ms: 40,
          is_error: false,
          num_turns: 1,
          total_cost_usd: 0,
          usage: {},
          modelUsage: {},
          permission_denials: [],
          uuid: "uuid-8",
          session_id: "s1",
        },
      ]),
    );

    await runAgentSdkAgent(baseParams());

    const passedEnv = mockQuery.mock.calls[0][0].options.env;
    expect(passedEnv.ANTHROPIC_API_KEY).toBeUndefined();

    // Restore.
    if (origKey !== undefined) {
      process.env.ANTHROPIC_API_KEY = origKey;
    } else {
      delete process.env.ANTHROPIC_API_KEY;
    }
  });

  it("passes maxTurns and allowedTools from params", async () => {
    mockQuery.mockReturnValue(
      messagesGenerator([
        {
          type: "result",
          subtype: "success",
          result: "ok",
          duration_ms: 50,
          duration_api_ms: 40,
          is_error: false,
          num_turns: 1,
          total_cost_usd: 0,
          usage: {},
          modelUsage: {},
          permission_denials: [],
          uuid: "uuid-9",
          session_id: "s1",
        },
      ]),
    );

    await runAgentSdkAgent(
      baseParams({
        maxTurns: 10,
        allowedTools: ["Read", "Bash"],
        disallowedTools: ["Write"],
      }),
    );

    const opts = mockQuery.mock.calls[0][0].options;
    expect(opts.maxTurns).toBe(10);
    expect(opts.allowedTools).toEqual(["Read", "Bash"]);
    expect(opts.disallowedTools).toEqual(["Write"]);
  });

  it("returns error result on abort signal", async () => {
    const controller = new AbortController();
    controller.abort();

    const result = await runAgentSdkAgent(baseParams({ abortSignal: controller.signal }));

    expect(result.payloads![0].isError).toBe(true);
    expect(result.payloads![0].text).toContain("Aborted");
    expect(result.meta.aborted).toBe(true);
  });

  it("resolves adaptive thinking for adaptive level", async () => {
    mockQuery.mockReturnValue(
      messagesGenerator([
        {
          type: "result",
          subtype: "success",
          result: "ok",
          duration_ms: 50,
          duration_api_ms: 40,
          is_error: false,
          num_turns: 1,
          total_cost_usd: 0,
          usage: {},
          modelUsage: {},
          permission_denials: [],
          uuid: "uuid-10",
          session_id: "s1",
        },
      ]),
    );

    await runAgentSdkAgent(baseParams({ thinkLevel: "adaptive" }));
    expect(mockQuery.mock.calls[0][0].options.thinking).toEqual({ type: "adaptive" });
  });

  it("disables thinking for 'off' level", async () => {
    mockQuery.mockReturnValue(
      messagesGenerator([
        {
          type: "result",
          subtype: "success",
          result: "ok",
          duration_ms: 50,
          duration_api_ms: 40,
          is_error: false,
          num_turns: 1,
          total_cost_usd: 0,
          usage: {},
          modelUsage: {},
          permission_denials: [],
          uuid: "uuid-11",
          session_id: "s1",
        },
      ]),
    );

    await runAgentSdkAgent(baseParams({ thinkLevel: "off" }));
    expect(mockQuery.mock.calls[0][0].options.thinking).toEqual({ type: "disabled" });
  });
});

describe("isAgentSdkProvider", () => {
  it("returns true for 'agent-sdk'", () => {
    expect(isAgentSdkProvider("agent-sdk")).toBe(true);
  });

  it("returns true for 'Agent-SDK' (case-insensitive)", () => {
    expect(isAgentSdkProvider("Agent-SDK")).toBe(true);
  });

  it("returns false for other providers", () => {
    expect(isAgentSdkProvider("claude-cli")).toBe(false);
    expect(isAgentSdkProvider("anthropic")).toBe(false);
    expect(isAgentSdkProvider("openrouter")).toBe(false);
  });
});
