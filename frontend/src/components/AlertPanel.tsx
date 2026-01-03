import { AlertTriangle, Info, AlertCircle, ChevronRight } from 'lucide-react';
import { useDashboardData } from '../lib/dashboardData';

interface Alert {
  id: string;
  severity: 'critical' | 'warning' | 'info';
  title: string;
  description: string;
  timestamp: string;
  action: string;
}

const fallbackAlerts: Alert[] = [
  {
    id: '1',
    severity: 'warning',
    title: 'ICU Capacity Warning',
    description: 'ICU occupancy projected to reach 94% within 12 hours',
    timestamp: '2 min ago',
    action: 'Review staffing',
  },
  {
    id: '2',
    severity: 'info',
    title: 'Admission Surge Detected',
    description: 'Respiratory cases up 23% from weekly average',
    timestamp: '15 min ago',
    action: 'View details',
  },
  {
    id: '3',
    severity: 'critical',
    title: 'Staff Load High',
    description: 'Night shift understaffed by 2 nurses in ICU',
    timestamp: '32 min ago',
    action: 'Contact HR',
  },
];

export function AlertPanel() {
  const { data } = useDashboardData();
  const alerts: Alert[] = (data?.alerts?.length ? data.alerts : fallbackAlerts) as Alert[];
  const getSeverityStyles = (severity: Alert['severity']) => {
    switch (severity) {
      case 'critical':
        return {
          container: 'border-l-4 border-rose-500 bg-rose-50',
          icon: 'text-rose-600',
          iconBg: 'bg-rose-100',
          Icon: AlertCircle,
        };
      case 'warning':
        return {
          container: 'border-l-4 border-amber-500 bg-amber-50',
          icon: 'text-amber-600',
          iconBg: 'bg-amber-100',
          Icon: AlertTriangle,
        };
      case 'info':
        return {
          container: 'border-l-4 border-cyan-500 bg-cyan-50',
          icon: 'text-cyan-600',
          iconBg: 'bg-cyan-100',
          Icon: Info,
        };
    }
  };

  return (
    <div className="bg-white rounded-lg border border-slate-200 p-6">
      <div className="mb-4">
        <h2 className="text-slate-900 mb-1">Active Alerts</h2>
        <p className="text-sm text-slate-500">Real-time system notifications</p>
      </div>

      <div className="space-y-3">
        {alerts.map((alert) => {
          const styles = getSeverityStyles(alert.severity);
          const Icon = styles.Icon;

          return (
            <div
              key={alert.id}
              className={`${styles.container} rounded-lg p-4 transition-all hover:shadow-md`}
            >
              <div className="flex gap-3">
                <div className={`${styles.iconBg} p-2 rounded-lg h-fit`}>
                  <Icon className={`w-4 h-4 ${styles.icon}`} />
                </div>
                <div className="flex-1 min-w-0">
                  <div className="flex items-start justify-between gap-2 mb-1">
                    <h3 className="text-sm text-slate-900">{alert.title}</h3>
                    <span className="text-xs text-slate-500 whitespace-nowrap">
                      {alert.timestamp}
                    </span>
                  </div>
                  <p className="text-xs text-slate-600 mb-3">{alert.description}</p>
                  <button className="flex items-center gap-1 text-xs text-cyan-700 hover:text-cyan-800 transition-colors">
                    <span>{alert.action}</span>
                    <ChevronRight className="w-3 h-3" />
                  </button>
                </div>
              </div>
            </div>
          );
        })}
      </div>

      <button className="w-full mt-4 py-2 text-sm text-cyan-700 hover:bg-slate-50 rounded-lg transition-colors">
        View all alerts
      </button>
    </div>
  );
}
