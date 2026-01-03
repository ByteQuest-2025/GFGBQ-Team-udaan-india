import { Activity, Database, Wifi, CheckCircle2, AlertCircle } from 'lucide-react';
import { useDashboardData } from '../lib/dashboardData';

export function SystemStatus() {
  const { data } = useDashboardData();

  const lastUpdate = (() => {
    const raw = data?.timestamp;
    if (!raw) return null;
    const d = new Date(raw);
    return Number.isNaN(d.getTime()) ? null : d;
  })();

  const formatTimeAgo = (date: Date) => {
    const now = new Date();
    const diff = now.getTime() - date.getTime();
    const minutes = Math.floor(diff / 60000);
    
    if (minutes < 1) return 'Just now';
    if (minutes === 1) return '1 min ago';
    if (minutes < 60) return `${minutes} min ago`;
    const hours = Math.floor(minutes / 60);
    if (hours === 1) return '1 hour ago';
    return `${hours} hours ago`;
  };

  return (
    <div className="flex items-center gap-6 px-4 py-2 bg-slate-50 border-b border-slate-200">
      <div className="flex items-center gap-2">
        <div className="flex items-center gap-1.5">
          <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse"></div>
          <span className="text-xs text-slate-600">ML Models Active</span>
        </div>
      </div>

      <div className="h-4 w-px bg-slate-300"></div>

      <div className="flex items-center gap-1.5">
        <Database className="w-3.5 h-3.5 text-slate-500" />
        <span className="text-xs text-slate-600">
          Data updated {lastUpdate ? formatTimeAgo(lastUpdate) : '—'}
        </span>
      </div>

      <div className="h-4 w-px bg-slate-300"></div>

      <div className="flex items-center gap-1.5">
        <CheckCircle2 className="w-3.5 h-3.5 text-green-600" />
        <span className="text-xs text-slate-600">All systems operational</span>
      </div>

      <div className="h-4 w-px bg-slate-300"></div>

      <div className="flex items-center gap-1.5">
        <Activity className="w-3.5 h-3.5 text-cyan-600" />
        <span className="text-xs text-slate-600">
          Model confidence:{' '}
          {typeof data?.explainability?.modelConfidence === 'number'
            ? `${(data.explainability.modelConfidence * 100).toFixed(1)}%`
            : '—'}
        </span>
      </div>
    </div>
  );
}
