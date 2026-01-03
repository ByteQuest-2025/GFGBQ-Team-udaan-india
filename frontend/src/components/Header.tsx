import { Bell, User } from 'lucide-react';
import { useEffect, useState } from 'react';
import { useDashboardData } from '../lib/dashboardData';

interface HeaderProps {
  alertCount?: number;
}

export function Header({ alertCount }: HeaderProps) {
  const { data } = useDashboardData();
  const [currentTime, setCurrentTime] = useState(new Date());
  const [showNotifications, setShowNotifications] = useState(false);

  useEffect(() => {
    const timer = setInterval(() => {
      setCurrentTime(new Date());
    }, 1000);

    return () => clearInterval(timer);
  }, []);

  const formatDate = (date: Date) => {
    return date.toLocaleDateString('en-US', { 
      weekday: 'long', 
      year: 'numeric', 
      month: 'long', 
      day: 'numeric' 
    });
  };

  const formatTime = (date: Date) => {
    return date.toLocaleTimeString('en-US', { 
      hour: '2-digit', 
      minute: '2-digit',
      second: '2-digit'
    });
  };

  const alerts = data?.alerts ?? [];
  const totalAlerts = alertCount ?? alerts.length ?? 0;

  const navigateToAlerts = () => {
    // Navigate back to dashboard and focus the alert panel section.
    window.dispatchEvent(new CustomEvent('phrel:navigate', { detail: 'dashboard' }));
    setShowNotifications(false);
    // Best-effort scroll to alerts area if present.
    setTimeout(() => {
      const el = document.querySelector('[data-section="alerts-panel"]');
      if (el && 'scrollIntoView' in el) {
        (el as HTMLElement).scrollIntoView({ behavior: 'smooth', block: 'start' });
      }
    }, 50);
  };

  return (
    <header className="bg-white border-b border-slate-200 px-8 py-4 flex items-center justify-between">
      <div className="flex items-center gap-4">
        <div className="w-10 h-10 bg-gradient-to-br from-cyan-600 to-blue-700 rounded-lg flex items-center justify-center">
          <span className="text-white text-xl">⚕</span>
        </div>
        <div>
          <h1 className="text-slate-900">Predictive Hospital Resource Intelligence</h1>
          <p className="text-sm text-slate-500">Real-time Emergency Load Forecasting</p>
        </div>
      </div>

      <div className="flex items-center gap-6">
        <div className="text-right">
          <div className="text-sm text-slate-600">{formatDate(currentTime)}</div>
          <div className="text-slate-900">{formatTime(currentTime)}</div>
        </div>

        <div className="relative">
          <button
            type="button"
            onClick={() => setShowNotifications((prev) => !prev)}
            className="relative p-2 text-slate-600 hover:bg-slate-100 rounded-lg transition-colors"
          >
            <Bell className="w-5 h-5" />
            {totalAlerts > 0 && (
              <span className="absolute top-1 right-1 w-5 h-5 bg-amber-500 text-white text-xs rounded-full flex items-center justify-center">
                {totalAlerts}
              </span>
            )}
          </button>
          {showNotifications && alerts.length > 0 && (
            <div className="absolute right-0 mt-2 w-80 bg-white border border-slate-200 rounded-lg shadow-lg z-20">
              <div className="px-4 py-2 border-b border-slate-200 flex items-center justify-between">
                <span className="text-sm font-medium text-slate-900">Active Alerts</span>
                <span className="text-xs text-slate-500">{alerts.length} open</span>
              </div>
              <div className="max-h-80 overflow-y-auto">
                {alerts.map((alert, idx) => (
                  <div
                    key={alert.id ?? idx}
                    className="px-4 py-3 border-b border-slate-100 last:border-b-0 hover:bg-slate-50 cursor-pointer"
                    onClick={navigateToAlerts}
                  >
                    <div className="flex items-center justify-between mb-1">
                      <span className="text-sm text-slate-900">{alert.title}</span>
                      <span className="text-xs px-2 py-0.5 rounded-full bg-amber-100 text-amber-700">
                        {alert.severity?.toUpperCase?.() ?? 'INFO'}
                      </span>
                    </div>
                    <p className="text-xs text-slate-600 mb-1">{alert.description}</p>
                    <div className="flex items-center justify-between text-[11px] text-slate-500">
                      <span>{alert.timestamp}</span>
                      <span>{alert.action}</span>
                    </div>
                  </div>
                ))}
              </div>
              <button
                type="button"
                onClick={navigateToAlerts}
                className="w-full px-4 py-2 text-xs text-cyan-700 hover:bg-slate-50 border-t border-slate-200"
              >
                View all alerts
              </button>
            </div>
          )}
        </div>

        <div className="flex items-center gap-3 pl-6 border-l border-slate-200">
          <div className="text-right">
            <div className="text-sm text-slate-900">Dr. Sarah Mitchell</div>
            <div className="text-xs text-slate-500">Hospital Administrator</div>
          </div>
          <div className="w-10 h-10 bg-cyan-600 rounded-full flex items-center justify-center">
            <User className="w-5 h-5 text-white" />
          </div>
        </div>
      </div>
    </header>
  );
}
