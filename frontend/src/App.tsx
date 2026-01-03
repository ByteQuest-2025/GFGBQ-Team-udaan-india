import { useEffect, useState } from 'react';
import { Header } from './components/Header';
import { Sidebar } from './components/Sidebar';
import { SystemStatus } from './components/SystemStatus';
import { DashboardView } from './components/DashboardView';
import { EmergencyForecastView } from './components/EmergencyForecastView';
import { ICUView } from './components/ICUView';
import { StaffWorkloadView } from './components/StaffWorkloadView';
import { WhatIfSimulator } from './components/WhatIfSimulator';
import { ReportsView } from './components/ReportsView';
import { SettingsView } from './components/SettingsView';
import { SystemHealthView } from './components/SystemHealthView';
import { DashboardDataProvider } from './lib/dashboardData';
import { Toaster } from './components/ui/sonner';

export type ViewType =
  | 'dashboard'
  | 'emergency'
  | 'icu'
  | 'staff'
  | 'simulator'
  | 'reports'
  | 'settings'
  | 'system-health';

export default function App() {
  const [currentView, setCurrentView] = useState<ViewType>('dashboard');

  useEffect(() => {
    const handler = (event: Event) => {
      const detail = (event as any).detail as ViewType | undefined;
      if (detail) {
        setCurrentView(detail);
      }
    };

    window.addEventListener('phrel:navigate', handler as EventListener);
    return () => {
      window.removeEventListener('phrel:navigate', handler as EventListener);
    };
  }, []);

  const renderView = () => {
    switch (currentView) {
      case 'dashboard':
        return <DashboardView />;
      case 'emergency':
        return <EmergencyForecastView />;
      case 'icu':
        return <ICUView />;
      case 'staff':
        return <StaffWorkloadView />;
      case 'simulator':
        return <WhatIfSimulator />;
      case 'reports':
        return <ReportsView />;
      case 'settings':
        return <SettingsView />;
      case 'system-health':
        return <SystemHealthView />;
      default:
        return <DashboardView />;
    }
  };

  return (
    <DashboardDataProvider>
      <div className="flex h-screen bg-slate-50 overflow-hidden">
        <Sidebar currentView={currentView} onViewChange={setCurrentView} />
        <div className="flex-1 flex flex-col overflow-hidden">
          <Header />
          <SystemStatus />
          <main className="flex-1 overflow-y-auto">{renderView()}</main>
        </div>
      </div>
      <Toaster />
    </DashboardDataProvider>
  );
}