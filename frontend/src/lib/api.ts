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

const API_BASE = (import.meta as any).env?.VITE_API_BASE_URL ?? '';

async function fetchJson<T>(path: string): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`);
  if (!res.ok) {
    throw new Error(`Request failed: ${res.status} ${res.statusText}`);
  }
  return (await res.json()) as T;
}

async function postJson<T>(path: string, body: unknown): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    throw new Error(`Request failed: ${res.status} ${res.statusText}`);
  }
  return (await res.json()) as T;
}

export function getUiDashboard(): Promise<UiDashboardResponse> {
  return fetchJson<UiDashboardResponse>('/api/ui/dashboard');
}

export function runWhatIf(body: WhatIfRequest): Promise<WhatIfResponse> {
  return postJson<WhatIfResponse>('/api/ui/whatif', body);
}
