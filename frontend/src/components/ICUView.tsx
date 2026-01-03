import { AlertTriangle, TrendingUp, Clock } from 'lucide-react';
import { useDashboardData } from '../lib/dashboardData';
import { triggerAction } from '../lib/api';
import { toast } from 'sonner';

interface BedStatus {
  department: string;
  total: number;
  occupied: number;
  available: number;
  predicted: number;
  timeToCapacity?: string;
}

export function ICUView() {
  const { data } = useDashboardData();

  const getOccupancyColor = (percentage: number) => {
    if (percentage >= 90) return 'bg-rose-500';
    if (percentage >= 75) return 'bg-amber-500';
    return 'bg-green-500';
  };

  const getOccupancyPercentage = (occupied: number, total: number) => {
    if (!Number.isFinite(total) || total <= 0) return 0;
    if (!Number.isFinite(occupied) || occupied < 0) return 0;
    return Math.round((occupied / total) * 100);
  };

  const bedData: BedStatus[] = (() => {
    const depts = data?.icuDepartments;
    if (depts && depts.length > 0) {
      const mapped = depts
        .filter((d) => typeof d.total === 'number' && d.total > 0)
        .map((d) => {
          const occupied = typeof d.occupied === 'number' ? d.occupied : 0;
          const available = typeof d.available === 'number' ? d.available : Math.max(0, d.total - occupied);
          const predicted = typeof d.predicted === 'number' ? d.predicted : occupied;
          const occupancyPct = getOccupancyPercentage(occupied, d.total);
          const timeToCapacity = occupancyPct >= 90 ? '~6 hours' : occupancyPct >= 75 ? '~18 hours' : '~24 hours';
          return {
            department: d.department,
            total: d.total,
            occupied,
            available,
            predicted,
            timeToCapacity,
          };
        });
      mapped.sort((a, b) => (b.occupied / b.total) * 100 - (a.occupied / a.total) * 100);
      return mapped;
    }

    // Fallback uses KPI totals if department breakdown isn't available
    const total = data?.kpis?.totalIcuBeds ?? 0;
    const available = data?.kpis?.availableIcuBeds ?? 0;
    const occupied = Math.max(0, total - available);
    const predicted = Math.round(occupied * (1 + 0.05));
    return [
      {
        department: 'ICU',
        total,
        occupied,
        available,
        predicted,
        timeToCapacity: '—',
      },
    ];
  })();

  const totalBeds = bedData.reduce((sum, dept) => sum + dept.total, 0);
  const totalOccupied = bedData.reduce((sum, dept) => sum + dept.occupied, 0);
  const totalAvailable = bedData.reduce((sum, dept) => sum + dept.available, 0);

  const monitoringMode = totalBeds > 0 && totalOccupied === 0;

  return (
    <div className="p-8 space-y-6">
      {/* Header Stats */}
      <div className="bg-white rounded-lg border border-slate-200 p-6">
        <h1 className="text-slate-900 mb-6">ICU & Bed Management</h1>
        
        <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
          <div>
            <p className="text-sm text-slate-600 mb-1">Total ICU Beds</p>
            <p className="text-3xl text-slate-900">{totalBeds}</p>
          </div>
          <div>
            <p className="text-sm text-slate-600 mb-1">Currently Occupied</p>
            <p className="text-3xl text-slate-900">{totalOccupied}</p>
          </div>
          <div>
            <p className="text-sm text-slate-600 mb-1">Available Now</p>
            <p className="text-3xl text-green-600">{totalAvailable}</p>
          </div>
          <div>
            <p className="text-sm text-slate-600 mb-1">Overall Occupancy</p>
            <p className="text-3xl text-amber-600">{getOccupancyPercentage(totalOccupied, totalBeds)}%</p>
          </div>
        </div>
      </div>

      {/* Monitoring / Overflow Notice */}
      {monitoringMode ? (
        <div className="bg-slate-50 border-l-4 border-slate-400 rounded-lg p-6">
          <div className="flex items-start gap-3">
            <div className="bg-slate-200 p-2 rounded-lg">
              <AlertTriangle className="w-5 h-5 text-slate-700" />
            </div>
            <div className="flex-1">
              <h3 className="text-slate-900 mb-1">Monitoring mode</h3>
              <p className="text-sm text-slate-700">
                ICU occupancy currently reports as 0. Capacity risk is simulated under forecast assumptions.
              </p>
            </div>
          </div>
        </div>
      ) : (
        <div className="bg-amber-50 border-l-4 border-amber-500 rounded-lg p-6">
          <div className="flex items-start gap-3">
            <div className="bg-amber-100 p-2 rounded-lg">
              <AlertTriangle className="w-5 h-5 text-amber-600" />
            </div>
            <div className="flex-1">
              <h3 className="text-slate-900 mb-1">Predicted capacity risk</h3>
              <p className="text-sm text-slate-700 mb-3">
                Capacity risk simulated under forecast assumptions. Validate against live census before escalating.
              </p>
              <button
                type="button"
                onClick={async () => {
                  try {
                    await triggerAction({
                      action_type: 'activate_overflow_protocol',
                      source: 'icu_view',
                      payload: {
                        message: 'Convert 4 general beds to ICU and notify on-call teams',
                      },
                    });
                    toast.success('Action logged and escalation initiated');
                  } catch (e: any) {
                    toast.error(e?.message ?? 'Failed to log action');
                  }
                }}
                className="px-4 py-2 bg-amber-600 text-white rounded-lg text-sm hover:bg-amber-700 transition-colors"
              >
                Activate Overflow Protocol
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Bed Availability Table */}
      <div className="bg-white rounded-lg border border-slate-200 overflow-hidden">
        <div className="p-6 border-b border-slate-200">
          <h2 className="text-slate-900 mb-1">Bed Availability by Department</h2>
          <p className="text-sm text-slate-500">Sorted by highest risk first</p>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full">
            <thead className="bg-slate-50 border-b border-slate-200">
              <tr>
                <th className="text-left py-3 px-6 text-sm text-slate-600">Department</th>
                <th className="text-left py-3 px-6 text-sm text-slate-600">Total Beds</th>
                <th className="text-left py-3 px-6 text-sm text-slate-600">Occupied</th>
                <th className="text-left py-3 px-6 text-sm text-slate-600">Available</th>
                <th className="text-left py-3 px-6 text-sm text-slate-600">Predicted (24h)</th>
                <th className="text-left py-3 px-6 text-sm text-slate-600">Time to Capacity</th>
                <th className="text-left py-3 px-6 text-sm text-slate-600">Occupancy</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-200">
              {bedData.map((dept) => {
                const occupancyPercentage = getOccupancyPercentage(dept.occupied, dept.total);
                const predictedPercentage = getOccupancyPercentage(dept.predicted, dept.total);

                return (
                  <tr key={dept.department} className="hover:bg-slate-50 transition-colors cursor-pointer">
                    <td className="py-4 px-6 text-sm text-slate-900">{dept.department}</td>
                    <td className="py-4 px-6 text-sm text-slate-700">{dept.total}</td>
                    <td className="py-4 px-6 text-sm text-slate-700">{dept.occupied}</td>
                    <td className="py-4 px-6 text-sm">
                      <span className={dept.available <= 3 ? 'text-amber-600' : 'text-green-600'}>
                        {dept.available}
                      </span>
                    </td>
                    <td className="py-4 px-6 text-sm">
                      <div className="flex items-center gap-2">
                        <span className="text-slate-700">{dept.predicted}</span>
                        {dept.predicted > dept.occupied && (
                          <TrendingUp className="w-4 h-4 text-amber-500" />
                        )}
                      </div>
                    </td>
                    <td className="py-4 px-6 text-sm">
                      {dept.timeToCapacity && (
                        <div className="flex items-center gap-1.5">
                          <Clock className="w-3.5 h-3.5 text-amber-600" />
                          <span className={occupancyPercentage >= 85 ? 'text-amber-600' : 'text-slate-600'}>
                            {dept.timeToCapacity}
                          </span>
                        </div>
                      )}
                    </td>
                    <td className="py-4 px-6">
                      <div className="space-y-2">
                        <div className="flex items-center gap-3">
                          <div className="flex-1 bg-slate-200 rounded-full h-2 overflow-hidden">
                            <div
                              className={`h-full ${getOccupancyColor(occupancyPercentage)} transition-all`}
                              style={{ width: `${occupancyPercentage}%` }}
                            />
                          </div>
                          <span className="text-sm text-slate-700 w-12 text-right">
                            {occupancyPercentage}%
                          </span>
                        </div>
                        {predictedPercentage >= 95 && (
                          <p className="text-xs text-amber-600">⚠ Near capacity</p>
                        )}
                      </div>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}