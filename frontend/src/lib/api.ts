export type UiDashboardResponse = {
  kpis: {
    predictedAdmissions: number;
    icuOccupancyPct: number;
    availableIcuBeds: number;
    totalIcuBeds: number;
    staffRiskLevel: string;
    staffLoadIndex?: number;
  };
  forecast7d: Array<{ day: string; predicted: number | null; actual: number | null }>;
  icuProjection24h: Array<{ time: string; demand: number; capacity: number }>;
  icuDepartments?: Array<{
    department: string;
    total: number;
    occupied: number | null;
    available: number | null;
    predicted: number;
  }>;
  staff?: {
    riskLevel: string;
    currentWorkloadPerStaff?: number;
    nextDayPredWorkloadPerStaff?: number;
    nextDayRecommendation?: string;
    burnoutTrend7d?: Array<{ day: string; index: number | null }>;
  };
  emergencyForecast24h?: Array<{ time: string; admissions: number }>;
  alerts: Array<{
    id: string;
    severity: 'critical' | 'warning' | 'info';
    title: string;
    description: string;
    timestamp: string;
    action: string;
  }>;
  explainability: {
    factors: Array<{ id: string; label: string; impact: 'high' | 'medium' | 'low' }>;
    modelConfidence?: number;
  };
  timestamp?: string;
};

export type HealthStatus = {
  status: string;
  environment?: string;
};

export type ReadyStatus = {
  status: string;
  pipeline_cached?: boolean;
  error?: string;
};

export type WhatIfRequest = {
  admission_surge_pct: number;
  temperature_c: number;
  staff_availability_pct: number;
};

export type WhatIfResponse = {
  baseline: {
    admissions: number;
    icuOccupancyPct: number;
    staffLoadIndex: number;
  };
  projections: {
    admissions: number;
    icuOccupancyPct: number;
    staffLoadIndex: number;
  };
};

const API_BASE: string = (import.meta as any).env?.VITE_API_BASE_URL ?? '';
const DEFAULT_TIMEOUT_MS = 8000;
const DEFAULT_RETRIES = 2;

export class ApiError extends Error {
  status: number;
  code?: string;

  constructor(message: string, status: number, code?: string) {
    super(message);
    this.status = status;
    this.code = code;
  }
}

async function fetchWithTimeout(
  input: RequestInfo,
  init: RequestInit = {},
  timeoutMs: number = DEFAULT_TIMEOUT_MS
): Promise<Response> {
  const controller = new AbortController();
  const id = setTimeout(() => controller.abort(), timeoutMs);
  try {
    const res = await fetch(input, { ...init, signal: controller.signal });
    return res;
  } finally {
    clearTimeout(id);
  }
}

async function fetchJson<T>(
  path: string,
  options?: { timeoutMs?: number; retries?: number }
): Promise<T> {
  const url = `${API_BASE}${path}`;
  const timeoutMs = options?.timeoutMs ?? DEFAULT_TIMEOUT_MS;
  const retries = options?.retries ?? DEFAULT_RETRIES;

  let attempt = 0;
  // Basic retry with backoff for GET requests
  // (only for network/5xx errors, not 4xx client errors).
  // eslint-disable-next-line no-constant-condition
  while (true) {
    try {
      const res = await fetchWithTimeout(url, undefined, timeoutMs);
      if (!res.ok) {
        if (res.status >= 500 && attempt < retries) {
          attempt += 1;
          const backoffMs = 300 * attempt;
          await new Promise((resolve) => setTimeout(resolve, backoffMs));
          continue;
        }
        throw new ApiError(`Request failed: ${res.status} ${res.statusText}`, res.status);
      }
      return (await res.json()) as T;
    } catch (err) {
      if (attempt < retries && (err instanceof TypeError || (err as any)?.name === 'AbortError')) {
        attempt += 1;
        const backoffMs = 300 * attempt;
        await new Promise((resolve) => setTimeout(resolve, backoffMs));
        continue;
      }
      throw err;
    }
  }
}

async function postJson<T>(path: string, body: unknown): Promise<T> {
  const url = `${API_BASE}${path}`;
  const res = await fetchWithTimeout(url, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    let message = `Request failed: ${res.status} ${res.statusText}`;
    let code: string | undefined;
    try {
      const body: any = await res.json();
      if (body && typeof body === 'object') {
        if (typeof body.detail === 'string') {
          message = body.detail;
        }
        if (typeof body.code === 'string') {
          code = body.code;
        }
      }
    } catch {
      // ignore JSON parse errors and fall back to default message
    }
    throw new ApiError(message, res.status, code);
  }
  return (await res.json()) as T;
}

export function getUiDashboard(): Promise<UiDashboardResponse> {
  return fetchJson<UiDashboardResponse>('/api/ui/dashboard');
}

export function runWhatIf(body: WhatIfRequest): Promise<WhatIfResponse> {
  return postJson<WhatIfResponse>('/api/ui/whatif', body);
}

export function getHealth(): Promise<HealthStatus> {
  return fetchJson<HealthStatus>('/health');
}

export function getReady(): Promise<ReadyStatus> {
  return fetchJson<ReadyStatus>('/health/ready');
}

export function getMonitoringLastRun(): Promise<any> {
  return fetchJson<any>('/api/monitoring/last-run');
}

export function getMonitoringHistory(limit = 20): Promise<any> {
  return fetchJson<any>(`/api/monitoring/history?limit=${limit}`);
}
