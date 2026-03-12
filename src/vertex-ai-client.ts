/**
 * Google Vertex AI Client — Native provider for embedding & LLM.
 * Zero external dependencies: uses Node.js crypto for JWT signing
 * and native fetch() for REST API calls.
 *
 * Supports:
 *  - Embedding: text-embedding-005, text-embedding-004, gemini-embedding-001
 *  - LLM: gemini-2.0-flash, gemini-1.5-pro, etc.
 *  - Auth modes:
 *    1. GCE metadata server (credentials: "gce" or auto-detect on GCE VMs)
 *    2. Direct access token (credentials: "${GOOGLE_VERTEX_TOKEN}" or raw token)
 *    3. Service Account JSON key file (credentials: "/path/to/key.json" or inline JSON)
 */

import { createSign } from "node:crypto";
import { readFileSync } from "node:fs";
import { createHash } from "node:crypto";
import type { LlmClient } from "./llm-client.js";

// ============================================================================
// Types
// ============================================================================

export interface VertexAIConfig {
  projectId: string;
  location: string;
  model: string;
  /**
   * Authentication credentials. Supports multiple formats:
   * - "gce" or "metadata" — use GCE metadata server (for VMs with attached service account)
   * - "${ENV_VAR}" — resolve env var; if it's a raw access token, use directly
   * - "/path/to/key.json" — Service Account JSON key file
   * - '{"type":"service_account",...}' — inline Service Account JSON
   */
  credentials: string;
}

export interface VertexEmbeddingConfig extends VertexAIConfig {
  dimensions?: number;
  /** Task type for query embeddings (e.g. "RETRIEVAL_QUERY") */
  taskQuery?: string;
  /** Task type for passage/document embeddings (e.g. "RETRIEVAL_DOCUMENT") */
  taskPassage?: string;
}

interface ServiceAccountKey {
  type: string;
  project_id: string;
  private_key_id: string;
  private_key: string;
  client_email: string;
  client_id: string;
  auth_uri: string;
  token_uri: string;
}

interface CachedToken {
  accessToken: string;
  expiresAt: number; // epoch ms
}

// ============================================================================
// Embedding Cache (LRU with TTL) — mirrors the OpenAI embedder cache
// ============================================================================

interface CacheEntry {
  vector: number[];
  createdAt: number;
}

class EmbeddingCache {
  private cache = new Map<string, CacheEntry>();
  private readonly maxSize: number;
  private readonly ttlMs: number;
  public hits = 0;
  public misses = 0;

  constructor(maxSize = 256, ttlMinutes = 30) {
    this.maxSize = maxSize;
    this.ttlMs = ttlMinutes * 60_000;
  }

  private key(text: string, task?: string): string {
    const hash = createHash("sha256").update(`${task || ""}:${text}`).digest("hex").slice(0, 24);
    return hash;
  }

  get(text: string, task?: string): number[] | undefined {
    const k = this.key(text, task);
    const entry = this.cache.get(k);
    if (!entry) {
      this.misses++;
      return undefined;
    }
    if (Date.now() - entry.createdAt > this.ttlMs) {
      this.cache.delete(k);
      this.misses++;
      return undefined;
    }
    this.cache.delete(k);
    this.cache.set(k, entry);
    this.hits++;
    return entry.vector;
  }

  set(text: string, task: string | undefined, vector: number[]): void {
    const k = this.key(text, task);
    if (this.cache.size >= this.maxSize) {
      const firstKey = this.cache.keys().next().value;
      if (firstKey !== undefined) this.cache.delete(firstKey);
    }
    this.cache.set(k, { vector, createdAt: Date.now() });
  }

  get size(): number { return this.cache.size; }
  get stats(): { size: number; hits: number; misses: number; hitRate: string } {
    const total = this.hits + this.misses;
    return {
      size: this.cache.size,
      hits: this.hits,
      misses: this.misses,
      hitRate: total > 0 ? `${((this.hits / total) * 100).toFixed(1)}%` : "N/A",
    };
  }
}

// ============================================================================
// Retry with Exponential Backoff
// ============================================================================

/** HTTP status codes that should trigger a retry. */
const RETRYABLE_STATUS_CODES = new Set([429, 500, 502, 503, 504]);

interface RetryConfig {
  /** Max number of retry attempts (default: 3) */
  maxRetries?: number;
  /** Base delay in ms before first retry (default: 1000) */
  baseDelayMs?: number;
  /** Max delay in ms (default: 10000) */
  maxDelayMs?: number;
  /** Optional logger */
  log?: (msg: string) => void;
}

/**
 * Fetch with automatic retry on transient errors.
 * Uses exponential backoff with jitter. Respects Retry-After header.
 */
async function fetchWithRetry(
  url: string,
  options: RequestInit,
  config: RetryConfig = {},
): Promise<Response> {
  const maxRetries = config.maxRetries ?? 3;
  const baseDelayMs = config.baseDelayMs ?? 1000;
  const maxDelayMs = config.maxDelayMs ?? 10000;
  const log = config.log ?? (() => {});

  let lastError: Error | null = null;

  for (let attempt = 0; attempt <= maxRetries; attempt++) {
    try {
      const response = await fetch(url, options);

      // Success or non-retryable error — return immediately
      if (response.ok || !RETRYABLE_STATUS_CODES.has(response.status)) {
        return response;
      }

      // Retryable status code — check if we have retries left
      if (attempt >= maxRetries) {
        return response; // No more retries, let caller handle the error
      }

      // Calculate delay: exponential backoff with jitter
      let delay = Math.min(baseDelayMs * Math.pow(2, attempt), maxDelayMs);

      // Respect Retry-After header if present
      const retryAfter = response.headers.get("Retry-After");
      if (retryAfter) {
        const retryAfterMs = parseInt(retryAfter, 10) * 1000;
        if (!isNaN(retryAfterMs) && retryAfterMs > 0) {
          delay = Math.min(retryAfterMs, maxDelayMs);
        }
      }

      // Add jitter (±25%)
      const jitter = delay * 0.25 * (Math.random() * 2 - 1);
      delay = Math.round(delay + jitter);

      log(
        `[memory-lancedb-pro] Vertex AI retry ${attempt + 1}/${maxRetries} after ${response.status} (waiting ${delay}ms)`,
      );

      await new Promise((resolve) => setTimeout(resolve, delay));
    } catch (err) {
      // Network error (DNS, timeout, connection refused, etc.)
      lastError = err instanceof Error ? err : new Error(String(err));

      if (attempt >= maxRetries) {
        throw lastError;
      }

      const delay = Math.min(baseDelayMs * Math.pow(2, attempt), maxDelayMs);
      const jitter = delay * 0.25 * (Math.random() * 2 - 1);
      const finalDelay = Math.round(delay + jitter);

      log(
        `[memory-lancedb-pro] Vertex AI retry ${attempt + 1}/${maxRetries} after network error: ${lastError.message} (waiting ${finalDelay}ms)`,
      );

      await new Promise((resolve) => setTimeout(resolve, finalDelay));
    }
  }

  // Should never reach here, but just in case
  throw lastError ?? new Error("fetchWithRetry: unexpected state");
}

// ============================================================================

type AuthMode =
  | { type: "gce" }
  | { type: "access-token"; token: string }
  | { type: "service-account"; key: ServiceAccountKey };

/**
 * Resolve ${ENV_VAR} patterns in a string.
 */
function resolveEnvVarsInCredentials(input: string): string {
  return input.trim().replace(/\$\{([^}]+)\}/g, (_, envVar) => {
    const envValue = process.env[envVar];
    if (!envValue) {
      throw new Error(`Environment variable ${envVar} is not set (used in credentials)`);
    }
    return envValue;
  });
}

/**
 * Detect the auth mode from the credentials string.
 */
function detectAuthMode(credentials: string): AuthMode {
  const resolved = resolveEnvVarsInCredentials(credentials);

  // Mode 1: Explicit GCE metadata server
  if (resolved.toLowerCase() === "gce" || resolved.toLowerCase() === "metadata") {
    return { type: "gce" };
  }

  // Mode 3: Inline JSON or JSON file — try parsing as Service Account key
  if (resolved.startsWith("{")) {
    try {
      const key = JSON.parse(resolved) as ServiceAccountKey;
      if (key.private_key && key.client_email) {
        return { type: "service-account", key };
      }
    } catch {
      // Not valid JSON, treat as access token
    }
    // JSON but not a Service Account key → might be malformed
    return { type: "access-token", token: resolved };
  }

  // Try reading as JSON file path
  try {
    const fileContent = readFileSync(resolved, "utf-8");
    const key = JSON.parse(fileContent) as ServiceAccountKey;
    if (key.private_key && key.client_email) {
      return { type: "service-account", key };
    }
  } catch {
    // Not a readable JSON file
  }

  // Mode 2: It's a raw access token (e.g. from metadata server via env var)
  // Access tokens from Google typically start with "ya29." but we don't require that
  return { type: "access-token", token: resolved };
}

// ============================================================================
// Google Auth — Multi-mode Authentication
// ============================================================================

/**
 * Load and parse the Service Account JSON key.
 */
function loadServiceAccountKey(credentials: string): ServiceAccountKey {
  let json: string;
  const trimmed = credentials.trim();
  const resolved = resolveEnvVarsInCredentials(trimmed);

  if (resolved.startsWith("{")) {
    json = resolved;
  } else {
    try {
      json = readFileSync(resolved, "utf-8");
    } catch (err) {
      throw new Error(
        `Failed to read Service Account key file at "${resolved}": ${err instanceof Error ? err.message : String(err)}`
      );
    }
  }

  try {
    const key = JSON.parse(json) as ServiceAccountKey;
    if (!key.client_email || !key.private_key) {
      throw new Error("Service Account key is missing client_email or private_key");
    }
    return key;
  } catch (err) {
    if (err instanceof SyntaxError) {
      throw new Error(`Invalid JSON in Service Account key: ${err.message}`);
    }
    throw err;
  }
}

/**
 * Create a signed JWT for Google OAuth2 token exchange.
 */
function createJWT(serviceAccount: ServiceAccountKey, scopes: string[]): string {
  const now = Math.floor(Date.now() / 1000);
  const exp = now + 3600;

  const header = { alg: "RS256", typ: "JWT" };
  const payload = {
    iss: serviceAccount.client_email,
    scope: scopes.join(" "),
    aud: serviceAccount.token_uri || "https://oauth2.googleapis.com/token",
    iat: now,
    exp,
  };

  const headerB64 = Buffer.from(JSON.stringify(header)).toString("base64url");
  const payloadB64 = Buffer.from(JSON.stringify(payload)).toString("base64url");
  const signatureInput = `${headerB64}.${payloadB64}`;

  const sign = createSign("RSA-SHA256");
  sign.update(signatureInput);
  const signature = sign.sign(serviceAccount.private_key, "base64url");

  return `${signatureInput}.${signature}`;
}

class GoogleAuth {
  private readonly authMode: AuthMode;
  private cachedToken: CachedToken | null = null;
  private pendingRefresh: Promise<string> | null = null;

  constructor(credentials: string) {
    this.authMode = detectAuthMode(credentials);

    // For direct access tokens, cache immediately with 50 min TTL
    // (tokens from metadata typically last 1 hour)
    if (this.authMode.type === "access-token") {
      this.cachedToken = {
        accessToken: this.authMode.token,
        expiresAt: Date.now() + 50 * 60 * 1000,
      };
    }

    const modeLabel = this.authMode.type === "service-account"
      ? `service-account (${this.authMode.key.client_email})`
      : this.authMode.type;
    console.log(`[memory-lancedb-pro] Vertex AI auth mode: ${modeLabel}`);
  }

  /**
   * Get a valid access token, refreshing if necessary.
   * Thread-safe: coalesces concurrent refresh requests.
   */
  async getAccessToken(): Promise<string> {
    // Return cached token if still valid (with 5 min buffer)
    if (this.cachedToken && this.cachedToken.expiresAt > Date.now() + 5 * 60 * 1000) {
      return this.cachedToken.accessToken;
    }

    // For direct access tokens, if expired, there's no auto-refresh
    // (user must restart with a new token)
    if (this.authMode.type === "access-token") {
      // Still return the token even if "expired" — let the API reject it
      // with a clear error rather than failing silently here
      if (this.cachedToken) return this.cachedToken.accessToken;
      return this.authMode.token;
    }

    // Coalesce concurrent refresh requests
    if (this.pendingRefresh) {
      return this.pendingRefresh;
    }

    this.pendingRefresh = this.refreshToken();
    try {
      return await this.pendingRefresh;
    } finally {
      this.pendingRefresh = null;
    }
  }

  private async refreshToken(): Promise<string> {
    if (this.authMode.type === "gce") {
      return this.refreshFromMetadataServer();
    }
    if (this.authMode.type === "service-account") {
      return this.refreshFromServiceAccount(this.authMode.key);
    }
    // access-token mode — no refresh possible
    throw new Error("Access token expired. Restart with a fresh token.");
  }

  /**
   * Fetch access token from GCE metadata server.
   * Only works on Google Cloud VMs with attached service account.
   */
  private async refreshFromMetadataServer(): Promise<string> {
    const metadataUrl =
      "http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/token";

    try {
      const response = await fetch(metadataUrl, {
        headers: { "Metadata-Flavor": "Google" },
        signal: AbortSignal.timeout(5000),
      });

      if (!response.ok) {
        throw new Error(`Metadata server returned ${response.status}`);
      }

      const data = await response.json() as {
        access_token: string;
        expires_in: number;
        token_type: string;
      };

      if (!data.access_token) {
        throw new Error("Metadata server response missing access_token");
      }

      this.cachedToken = {
        accessToken: data.access_token,
        expiresAt: Date.now() + (data.expires_in || 3600) * 1000,
      };

      return data.access_token;
    } catch (err) {
      throw new Error(
        `GCE metadata server auth failed: ${err instanceof Error ? err.message : String(err)}. ` +
        `Make sure this VM has a service account with Vertex AI permissions.`
      );
    }
  }

  /**
   * Exchange Service Account JWT for access token.
   */
  private async refreshFromServiceAccount(key: ServiceAccountKey): Promise<string> {
    const jwt = createJWT(key, [
      "https://www.googleapis.com/auth/cloud-platform",
    ]);

    const tokenUrl = key.token_uri || "https://oauth2.googleapis.com/token";

    const response = await fetch(tokenUrl, {
      method: "POST",
      headers: { "Content-Type": "application/x-www-form-urlencoded" },
      body: new URLSearchParams({
        grant_type: "urn:ietf:params:oauth:grant-type:jwt-bearer",
        assertion: jwt,
      }).toString(),
    });

    if (!response.ok) {
      const errorText = await response.text().catch(() => "unknown");
      throw new Error(
        `Google OAuth2 token exchange failed (${response.status}): ${errorText}. ` +
        `Check Service Account credentials for ${key.client_email}.`
      );
    }

    const data = await response.json() as { access_token: string; expires_in: number };
    if (!data.access_token) {
      throw new Error("Google OAuth2 response missing access_token");
    }

    this.cachedToken = {
      accessToken: data.access_token,
      expiresAt: Date.now() + (data.expires_in || 3600) * 1000,
    };

    return data.access_token;
  }

  get mode(): string {
    return this.authMode.type;
  }
}

// ============================================================================
// Known Vertex AI Embedding Dimensions
// ============================================================================

const VERTEX_EMBEDDING_DIMENSIONS: Record<string, number> = {
  "text-embedding-005": 768,
  "text-embedding-004": 768,
  "text-multilingual-embedding-002": 768,
  "gemini-embedding-001": 3072,
};

// ============================================================================
// Vertex AI Embedder
// ============================================================================

export class VertexEmbedder {
  private readonly auth: GoogleAuth;
  private readonly _model: string;
  private readonly _projectId: string;
  private readonly _location: string;
  private readonly _taskQuery?: string;
  private readonly _taskPassage?: string;
  private readonly _requestDimensions?: number;
  private readonly _cache: EmbeddingCache;
  public readonly dimensions: number;

  constructor(config: VertexEmbeddingConfig) {
    this.auth = new GoogleAuth(config.credentials);
    this._model = config.model;
    this._projectId = config.projectId;
    this._location = config.location;
    this._taskQuery = config.taskQuery;
    this._taskPassage = config.taskPassage;
    this._requestDimensions = config.dimensions;

    // Resolve dimensions
    if (config.dimensions && config.dimensions > 0) {
      this.dimensions = config.dimensions;
    } else {
      const knownDims = VERTEX_EMBEDDING_DIMENSIONS[config.model];
      if (!knownDims) {
        throw new Error(
          `Unknown Vertex AI embedding model: ${config.model}. ` +
          `Set embedding.dimensions explicitly. Known models: ${Object.keys(VERTEX_EMBEDDING_DIMENSIONS).join(", ")}`
        );
      }
      this.dimensions = knownDims;
    }

    this._cache = new EmbeddingCache(256, 30);
    console.log(
      `[memory-lancedb-pro] Vertex AI embedder initialized: model=${config.model}, ` +
      `project=${config.projectId}, location=${config.location}, dims=${this.dimensions}`
    );
  }

  // --------------------------------------------------------------------------
  // Vertex AI REST API
  // --------------------------------------------------------------------------

  private get endpoint(): string {
    return `https://${this._location}-aiplatform.googleapis.com/v1/projects/${this._projectId}/locations/${this._location}/publishers/google/models/${this._model}:predict`;
  }

  private async callPredictAPI(instances: Array<{ content: string; task_type?: string }>): Promise<number[][]> {
    const accessToken = await this.auth.getAccessToken();

    const body: Record<string, unknown> = { instances };
    if (this._requestDimensions && this._requestDimensions > 0) {
      body.parameters = { outputDimensionality: this._requestDimensions };
    }

    const response = await fetchWithRetry(
      this.endpoint,
      {
        method: "POST",
        headers: {
          "Authorization": `Bearer ${accessToken}`,
          "Content-Type": "application/json",
        },
        body: JSON.stringify(body),
      },
      { maxRetries: 3, log: (msg) => console.log(msg) },
    );

    if (!response.ok) {
      const errorText = await response.text().catch(() => "unknown");
      throw new Error(
        `Vertex AI embedding API error (${response.status}): ${errorText}. ` +
        `Model: ${this._model}, Project: ${this._projectId}`
      );
    }

    const data = await response.json() as {
      predictions: Array<{
        embeddings: { values: number[]; statistics?: Record<string, unknown> };
      }>;
    };

    if (!data.predictions || !Array.isArray(data.predictions)) {
      throw new Error("Vertex AI embedding response missing predictions array");
    }

    return data.predictions.map((p) => {
      const values = p.embeddings?.values;
      if (!Array.isArray(values)) {
        throw new Error("Vertex AI embedding response missing embeddings.values");
      }
      return values;
    });
  }

  // --------------------------------------------------------------------------
  // Public API — compatible with Embedder interface
  // --------------------------------------------------------------------------

  get model(): string {
    return this._model;
  }

  get keyCount(): number {
    return 1; // Single service account
  }

  async embed(text: string): Promise<number[]> {
    return this.embedPassage(text);
  }

  async embedBatch(texts: string[]): Promise<number[][]> {
    return this.embedBatchPassage(texts);
  }

  async embedQuery(text: string): Promise<number[]> {
    return this.embedSingle(text, this._taskQuery);
  }

  async embedPassage(text: string): Promise<number[]> {
    return this.embedSingle(text, this._taskPassage);
  }

  async embedBatchQuery(texts: string[]): Promise<number[][]> {
    return this.embedMany(texts, this._taskQuery);
  }

  async embedBatchPassage(texts: string[]): Promise<number[][]> {
    return this.embedMany(texts, this._taskPassage);
  }

  async test(): Promise<{ success: boolean; error?: string; dimensions?: number }> {
    try {
      const testEmbedding = await this.embedPassage("test");
      return { success: true, dimensions: testEmbedding.length };
    } catch (error) {
      return {
        success: false,
        error: error instanceof Error ? error.message : String(error),
      };
    }
  }

  get cacheStats() {
    return {
      ...this._cache.stats,
      keyCount: 1,
    };
  }

  // --------------------------------------------------------------------------
  // Internals
  // --------------------------------------------------------------------------

  private validateEmbedding(embedding: number[]): void {
    if (!Array.isArray(embedding)) {
      throw new Error(`Embedding is not an array (got ${typeof embedding})`);
    }
    if (embedding.length !== this.dimensions) {
      throw new Error(
        `Embedding dimension mismatch: expected ${this.dimensions}, got ${embedding.length}`
      );
    }
  }

  private async embedSingle(text: string, task?: string): Promise<number[]> {
    if (!text || text.trim().length === 0) {
      throw new Error("Cannot embed empty text");
    }

    const cached = this._cache.get(text, task);
    if (cached) return cached;

    try {
      const instance: { content: string; task_type?: string } = { content: text };
      if (task) instance.task_type = task;

      const results = await this.callPredictAPI([instance]);
      const embedding = results[0];
      if (!embedding) {
        throw new Error("No embedding returned from Vertex AI");
      }

      this.validateEmbedding(embedding);
      this._cache.set(text, task, embedding);
      return embedding;
    } catch (error) {
      const msg = error instanceof Error ? error.message : String(error);
      throw new Error(
        `Failed to generate embedding from Vertex AI (${this._model}): ${msg}`,
        { cause: error instanceof Error ? error : undefined }
      );
    }
  }

  private async embedMany(texts: string[], task?: string): Promise<number[][]> {
    if (!texts || texts.length === 0) return [];

    const validTexts: string[] = [];
    const validIndices: number[] = [];

    texts.forEach((text, index) => {
      if (text && text.trim().length > 0) {
        validTexts.push(text);
        validIndices.push(index);
      }
    });

    if (validTexts.length === 0) {
      return texts.map(() => []);
    }

    try {
      // Vertex AI supports batch up to 250 instances
      const BATCH_SIZE = 250;
      const allEmbeddings: number[][] = [];

      for (let i = 0; i < validTexts.length; i += BATCH_SIZE) {
        const batch = validTexts.slice(i, i + BATCH_SIZE);
        const instances = batch.map((text) => {
          const inst: { content: string; task_type?: string } = { content: text };
          if (task) inst.task_type = task;
          return inst;
        });

        const batchResults = await this.callPredictAPI(instances);
        allEmbeddings.push(...batchResults);
      }

      const results: number[][] = new Array(texts.length);

      allEmbeddings.forEach((embedding, idx) => {
        const originalIndex = validIndices[idx];
        this.validateEmbedding(embedding);
        results[originalIndex] = embedding;
      });

      for (let i = 0; i < texts.length; i++) {
        if (!results[i]) results[i] = [];
      }

      return results;
    } catch (error) {
      const msg = error instanceof Error ? error.message : String(error);
      throw new Error(
        `Failed to generate batch embeddings from Vertex AI (${this._model}): ${msg}`,
        { cause: error instanceof Error ? error : undefined }
      );
    }
  }
}

// ============================================================================
// Vertex AI LLM Client
// ============================================================================

export interface VertexLlmConfig {
  projectId: string;
  location: string;
  model: string;
  credentials: string;
  timeoutMs?: number;
  log?: (msg: string) => void;
}

/**
 * Extract JSON from an LLM response that may be wrapped in markdown fences
 * or contain surrounding text.
 */
function extractJsonFromResponse(text: string): string | null {
  // Try markdown code fence first
  const fenceMatch = text.match(/```(?:json)?\s*\n?([\s\S]*?)```/);
  if (fenceMatch) {
    return fenceMatch[1].trim();
  }

  // Try balanced brace extraction
  const firstBrace = text.indexOf("{");
  if (firstBrace === -1) return null;

  let depth = 0;
  let lastBrace = -1;
  for (let i = firstBrace; i < text.length; i++) {
    if (text[i] === "{") depth++;
    else if (text[i] === "}") {
      depth--;
      if (depth === 0) {
        lastBrace = i;
        break;
      }
    }
  }

  if (lastBrace === -1) return null;
  return text.substring(firstBrace, lastBrace + 1);
}

function previewText(value: string, maxLen = 200): string {
  const normalized = value.replace(/\s+/g, " ").trim();
  if (normalized.length <= maxLen) return normalized;
  return `${normalized.slice(0, maxLen - 3)}...`;
}

export function createVertexLlmClient(config: VertexLlmConfig): LlmClient {
  const auth = new GoogleAuth(config.credentials);
  const log = config.log ?? (() => {});
  const timeoutMs = config.timeoutMs ?? 30000;

  const endpoint = `https://${config.location}-aiplatform.googleapis.com/v1/projects/${config.projectId}/locations/${config.location}/publishers/google/models/${config.model}:generateContent`;

  return {
    async completeJson<T>(prompt: string, label = "generic"): Promise<T | null> {
      try {
        const accessToken = await auth.getAccessToken();

        const controller = new AbortController();
        const timer = setTimeout(() => controller.abort(), timeoutMs);

        try {
          const response = await fetchWithRetry(
            endpoint,
            {
              method: "POST",
              headers: {
                "Authorization": `Bearer ${accessToken}`,
                "Content-Type": "application/json",
              },
              body: JSON.stringify({
                contents: [
                  {
                    role: "user",
                    parts: [{
                      text: "You are a memory extraction assistant. Always respond with valid JSON only.\n\n" + prompt,
                    }],
                  },
                ],
                generationConfig: {
                  temperature: 0.1,
                  responseMimeType: "application/json",
                },
              }),
              signal: controller.signal,
            },
            { maxRetries: 2, log },
          );

          clearTimeout(timer);

          if (!response.ok) {
            const errorText = await response.text().catch(() => "unknown");
            log(
              `memory-lancedb-pro: vertex-llm [${label}] API error (${response.status}): ${previewText(errorText)}`,
            );
            return null;
          }

          const data = await response.json() as {
            candidates?: Array<{
              content?: { parts?: Array<{ text?: string }> };
            }>;
          };

          const raw = data.candidates?.[0]?.content?.parts?.[0]?.text;
          if (!raw) {
            log(
              `memory-lancedb-pro: vertex-llm [${label}] empty response content from model ${config.model}`,
            );
            return null;
          }

          // With responseMimeType: "application/json", Gemini should return clean JSON
          // but we still use the extractor for robustness
          const jsonStr = extractJsonFromResponse(raw) || raw.trim();

          try {
            return JSON.parse(jsonStr) as T;
          } catch (err) {
            log(
              `memory-lancedb-pro: vertex-llm [${label}] JSON.parse failed: ${err instanceof Error ? err.message : String(err)} (jsonChars=${jsonStr.length}, jsonPreview=${JSON.stringify(previewText(jsonStr))})`,
            );
            return null;
          }
        } catch (err) {
          clearTimeout(timer);
          throw err;
        }
      } catch (err) {
        log(
          `memory-lancedb-pro: vertex-llm [${label}] request failed for model ${config.model}: ${err instanceof Error ? err.message : String(err)}`,
        );
        return null;
      }
    },
  };
}

// ============================================================================
// Exports
// ============================================================================

export { GoogleAuth, VERTEX_EMBEDDING_DIMENSIONS };
