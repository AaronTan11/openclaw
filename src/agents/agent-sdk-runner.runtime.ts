/**
 * Dynamic import boundary for @anthropic-ai/claude-agent-sdk.
 *
 * Per CLAUDE.md convention, lazy-loading boundaries prevent mixing static +
 * dynamic imports for the same module in production code paths.  Only the
 * agent-sdk runner imports this file (dynamically), so the ~56 MB SDK binary
 * is never loaded unless the `agent-sdk` provider is actually used.
 */
export { query } from "@anthropic-ai/claude-agent-sdk";
export type {
  Options,
  Query,
  SDKAssistantMessage,
  SDKMessage,
  SDKRateLimitEvent,
  SDKResultError,
  SDKResultMessage,
  SDKResultSuccess,
  ThinkingConfig,
} from "@anthropic-ai/claude-agent-sdk";
