import { useState } from 'react';
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
import { DashboardDataProvider } from './lib/dashboardData';

export type ViewType = 'dashboard' | 'emergency' | 'icu' | 'staff' | 'simulator' | 'reports' | 'settings';

export default function App() {
  const [currentView, setCurrentView] = useState<ViewType>('dashboard');

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
    </DashboardDataProvider>
  );
}