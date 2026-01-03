import React, { createContext, useCallback, useContext, useEffect, useMemo, useState } from 'react';
import type { UiDashboardResponse } from './api';
import { getUiDashboard } from './api';

type DashboardDataState = {
  data: UiDashboardResponse | null;
  loading: boolean;
  error: string | null;
  refresh: () => void;
};

const DashboardDataContext = createContext<DashboardDataState | null>(null);

export function DashboardDataProvider({ children }: { children: React.ReactNode }) {
  const [data, setData] = useState<UiDashboardResponse | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  const load = useCallback(() => {
    setLoading(true);
    setError(null);
    getUiDashboard()
      .then((d) => {
        setData(d);
      })
      .catch((e: unknown) => {
        setError(e instanceof Error ? e.message : String(e));
      })
      .finally(() => setLoading(false));
  }, []);

  useEffect(() => {
    load();
  }, [load]);

  const value = useMemo(
    () => ({
      data,
      loading,
      error,
      refresh: load,
    }),
    [data, loading, error, load]
  );

  return <DashboardDataContext.Provider value={value}>{children}</DashboardDataContext.Provider>;
}

export function useDashboardData(): DashboardDataState {
  const ctx = useContext(DashboardDataContext);
  if (!ctx) {
    throw new Error('useDashboardData must be used within DashboardDataProvider');
  }
  return ctx;
}
