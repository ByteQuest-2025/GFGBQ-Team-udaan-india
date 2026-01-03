import { Bell, User } from 'lucide-react';
import { useEffect, useState } from 'react';
import { useDashboardData } from '../lib/dashboardData';

interface HeaderProps {
  alertCount?: number;
}

export function Header({ alertCount }: HeaderProps) {
  const { data } = useDashboardData();
  const [currentTime, setCurrentTime] = useState(new Date());

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
          <button className="relative p-2 text-slate-600 hover:bg-slate-100 rounded-lg transition-colors">
            <Bell className="w-5 h-5" />
            {(alertCount ?? data?.alerts?.length ?? 0) > 0 && (
              <span className="absolute top-1 right-1 w-5 h-5 bg-amber-500 text-white text-xs rounded-full flex items-center justify-center">
                {alertCount ?? data?.alerts?.length ?? 0}
              </span>
            )}
          </button>
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
