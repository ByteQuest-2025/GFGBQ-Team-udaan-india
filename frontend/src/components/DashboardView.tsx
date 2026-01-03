import { KPICard } from './KPICard';
import { ForecastChart } from './ForecastChart';
import { ICUDemandChart } from './ICUDemandChart';
import { AlertPanel } from './AlertPanel';
import { ExplainabilityCard } from './ExplainabilityCard';
import { QuickActionButton } from './QuickActionButton';
import { TrendingUp, TrendingDown, Activity, Bed, Users, Gauge, AlertTriangle, Phone, FlaskConical } from 'lucide-react';
import { useDashboardData } from '../lib/dashboardData';

export function DashboardView() {
  const { data } = useDashboardData();
  const lastModelUpdate = new Date();
  lastModelUpdate.setMinutes(lastModelUpdate.getMinutes() - 8);

  const predictedAdmissions = data?.kpis?.predictedAdmissions;
  const icuPct = data?.kpis?.icuOccupancyPct;
  const availableBeds = data?.kpis?.availableIcuBeds;
  const totalBeds = data?.kpis?.totalIcuBeds;
  const staffRisk = data?.kpis?.staffRiskLevel;

  const formatTimeAgo = (date: Date) => {
    const now = new Date();
    const diff = now.getTime() - date.getTime();
    const minutes = Math.floor(diff / 60000);
    
    if (minutes < 1) return 'Just now';
    if (minutes === 1) return '1 min ago';
    return `${minutes} min ago`;
  };

  return (
    <div className="p-8 space-y-8">
      {/* Quick Actions Bar */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-slate-900 mb-1">Dashboard Overview</h1>
          <p className="text-sm text-slate-600">
            Last model update: {formatTimeAgo(lastModelUpdate)} • Confidence: 94.2%
          </p>
        </div>
        <div className="flex gap-3">
          <QuickActionButton 
            icon={AlertTriangle} 
            label="Prepare ICU" 
            variant="danger"
          />
          <QuickActionButton 
            icon={Phone} 
            label="Call On-Call Staff" 
            variant="secondary"
          />
          <QuickActionButton 
            icon={FlaskConical} 
            label="Run What-If Scenario" 
            variant="primary"
          />
        </div>
      </div>

      {/* KPI Summary Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        <KPICard
          title="Predicted Admissions"
          value={predictedAdmissions != null && Number.isFinite(predictedAdmissions) ? `${Math.round(predictedAdmissions)}` : '—'}
          unit="Next 24h"
          trend={{ direction: 'up', value: '+18%', label: 'vs yesterday' }}
          icon={Activity}
          iconBgColor="bg-cyan-100"
          iconColor="text-cyan-700"
        />
        <KPICard
          title="ICU Occupancy"
          value={icuPct != null && Number.isFinite(icuPct) ? `${icuPct.toFixed(1)}` : '—'}
          unit="%"
          trend={{ direction: 'up', value: '+5.2%', label: 'vs yesterday' }}
          icon={Gauge}
          iconBgColor="bg-amber-100"
          iconColor="text-amber-700"
          warning
        />
        <KPICard
          title="Available ICU Beds"
          value={availableBeds != null && Number.isFinite(availableBeds) ? `${Math.round(availableBeds)}` : '—'}
          unit={totalBeds != null && Number.isFinite(totalBeds) ? `of ${Math.round(totalBeds)} total` : 'of — total'}
          trend={{ direction: 'down', value: '-4', label: 'vs yesterday' }}
          icon={Bed}
          iconBgColor="bg-rose-100"
          iconColor="text-rose-700"
        />
        <KPICard
          title="Staff Load Index"
          value={staffRisk ? (staffRisk === 'HIGH' ? '8.5' : staffRisk === 'MEDIUM' ? '6.5' : staffRisk === 'LOW' ? '4.5' : '—') : '—'}
          unit="/ 10"
          trend={{ direction: 'down', value: '-0.3', label: 'vs yesterday' }}
          icon={Users}
          iconBgColor="bg-green-100"
          iconColor="text-green-700"
        />
      </div>

      {/* Main Content Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
        {/* Left Column - Charts */}
        <div className="lg:col-span-2 space-y-8">
          <ForecastChart />
          <ICUDemandChart />
        </div>

        {/* Right Column - Alerts & Explainability */}
        <div className="space-y-8">
          <AlertPanel />
          <ExplainabilityCard />
        </div>
      </div>
    </div>
  );
}