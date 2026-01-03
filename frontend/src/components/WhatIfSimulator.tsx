import { useMemo, useState } from 'react';
import { Play, RotateCcw, Zap } from 'lucide-react';
import { runWhatIf, type WhatIfResponse } from '../lib/api';

interface Preset {
  name: string;
  icon: string;
  admissionSurge: number;
  temperature: number;
  staffAvailability: number;
}

const presets: Preset[] = [
  {
    name: 'Flu Outbreak',
    icon: '🦠',
    admissionSurge: 45,
    temperature: 8,
    staffAvailability: 85,
  },
  {
    name: 'Heatwave',
    icon: '🌡️',
    admissionSurge: 30,
    temperature: 32,
    staffAvailability: 90,
  },
  {
    name: 'Festival Surge',
    icon: '🎉',
    admissionSurge: 60,
    temperature: 18,
    staffAvailability: 80,
  },
];

export function WhatIfSimulator() {
  const [admissionSurge, setAdmissionSurge] = useState(0);
  const [temperature, setTemperature] = useState(15);
  const [staffAvailability, setStaffAvailability] = useState(100);

  const [result, setResult] = useState<WhatIfResponse | null>(null);
  const [running, setRunning] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const fallbackBaselineAdmissions = 248;
  const fallbackBaselineICU = 87.5;
  const fallbackBaselineStaffLoad = 6.8;

  const applyPreset = (preset: Preset) => {
    setAdmissionSurge(preset.admissionSurge);
    setTemperature(preset.temperature);
    setStaffAvailability(preset.staffAvailability);
  };

  const projections = useMemo(() => {
    if (result) {
      return {
        admissions: result.projections.admissions,
        icu: result.projections.icuOccupancyPct.toFixed(1),
        staffLoad: result.projections.staffLoadIndex.toFixed(1),
      };
    }

    // Fallback (local-only) if user hasn't run the backend simulation yet.
    const admissionImpact = (admissionSurge / 100) * fallbackBaselineAdmissions;
    const tempImpact = (15 - temperature) * 3; // Cold weather increases admissions
    const projectedAdmissions = Math.round(fallbackBaselineAdmissions + admissionImpact + tempImpact);

    const icuImpact = (admissionSurge / 100) * 10 + (15 - temperature) * 0.5;
    const projectedICU = Math.min(100, fallbackBaselineICU + icuImpact);

    const staffImpact = (100 - staffAvailability) / 100;
    const staffLoadImpact = fallbackBaselineStaffLoad + staffImpact * 3;

    return {
      admissions: projectedAdmissions,
      icu: projectedICU.toFixed(1),
      staffLoad: staffLoadImpact.toFixed(1),
    };
  }, [admissionSurge, temperature, staffAvailability, result]);

  const baselineAdmissions = result?.baseline.admissions ?? fallbackBaselineAdmissions;
  const baselineICU = result?.baseline.icuOccupancyPct ?? fallbackBaselineICU;
  const baselineStaffLoad = result?.baseline.staffLoadIndex ?? fallbackBaselineStaffLoad;

  const handleReset = () => {
    setAdmissionSurge(0);
    setTemperature(15);
    setStaffAvailability(100);
    setResult(null);
    setError(null);
  };

  const handleRun = async () => {
    setRunning(true);
    setError(null);
    try {
      const res = await runWhatIf({
        admission_surge_pct: admissionSurge,
        temperature_c: temperature,
        staff_availability_pct: staffAvailability,
      });
      setResult(res);
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : String(e));
      setResult(null);
    } finally {
      setRunning(false);
    }
  };

  return (
    <div className="p-8 space-y-6">
      {/* Header */}
      <div className="bg-white rounded-lg border border-slate-200 p-6">
        <h1 className="text-slate-900 mb-2">What-If Scenario Simulator</h1>
        <p className="text-sm text-slate-600">
          Adjust parameters to simulate different scenarios and see projected impact on hospital resources
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Control Panel */}
        <div className="bg-white rounded-lg border border-slate-200 p-6 space-y-6">
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-slate-900">Scenario Parameters</h2>
            <button
              onClick={handleReset}
              className="flex items-center gap-2 px-3 py-2 text-sm text-slate-600 hover:bg-slate-100 rounded-lg transition-colors"
            >
              <RotateCcw className="w-4 h-4" />
              Reset
            </button>
          </div>

          {/* Scenario Presets */}
          <div className="pb-6 border-b border-slate-200">
            <div className="flex items-center gap-2 mb-3">
              <Zap className="w-4 h-4 text-cyan-600" />
              <h3 className="text-sm text-slate-900">Quick Scenarios</h3>
            </div>
            <div className="grid grid-cols-3 gap-2">
              {presets.map((preset) => (
                <button
                  key={preset.name}
                  onClick={() => applyPreset(preset)}
                  className="p-3 bg-slate-50 hover:bg-cyan-50 border border-slate-200 hover:border-cyan-300 rounded-lg transition-colors text-center"
                >
                  <span className="text-2xl block mb-1">{preset.icon}</span>
                  <span className="text-xs text-slate-700">{preset.name}</span>
                </button>
              ))}
            </div>
          </div>

          {/* Admission Surge */}
          <div>
            <div className="flex items-center justify-between mb-2">
              <label htmlFor="whatif-admission-surge" className="text-sm text-slate-700">Admission Surge</label>
              <span className="text-sm text-slate-900">
                {admissionSurge > 0 ? '+' : ''}{admissionSurge}%
              </span>
            </div>
            <input
              id="whatif-admission-surge"
              type="range"
              min="-50"
              max="100"
              value={admissionSurge}
              onChange={(e) => setAdmissionSurge(Number(e.target.value))}
              className="w-full h-2 bg-slate-200 rounded-lg appearance-none cursor-pointer accent-cyan-600"
            />
            <div className="flex justify-between text-xs text-slate-500 mt-1">
              <span>-50%</span>
              <span>0%</span>
              <span>+100%</span>
            </div>
          </div>

          {/* Temperature */}
          <div>
            <div className="flex items-center justify-between mb-2">
              <label htmlFor="whatif-temperature" className="text-sm text-slate-700">Outside Temperature</label>
              <span className="text-sm text-slate-900">{temperature}°C</span>
            </div>
            <input
              id="whatif-temperature"
              type="range"
              min="-5"
              max="35"
              value={temperature}
              onChange={(e) => setTemperature(Number(e.target.value))}
              className="w-full h-2 bg-slate-200 rounded-lg appearance-none cursor-pointer accent-cyan-600"
            />
            <div className="flex justify-between text-xs text-slate-500 mt-1">
              <span>-5°C</span>
              <span>15°C</span>
              <span>35°C</span>
            </div>
          </div>

          {/* Staff Availability */}
          <div>
            <div className="flex items-center justify-between mb-2">
              <label htmlFor="whatif-staff-availability" className="text-sm text-slate-700">Staff Availability</label>
              <span className="text-sm text-slate-900">{staffAvailability}%</span>
            </div>
            <input
              id="whatif-staff-availability"
              type="range"
              min="50"
              max="100"
              value={staffAvailability}
              onChange={(e) => setStaffAvailability(Number(e.target.value))}
              className="w-full h-2 bg-slate-200 rounded-lg appearance-none cursor-pointer accent-cyan-600"
            />
            <div className="flex justify-between text-xs text-slate-500 mt-1">
              <span>50%</span>
              <span>75%</span>
              <span>100%</span>
            </div>
          </div>

          <button
            onClick={handleRun}
            disabled={running}
            className="w-full flex items-center justify-center gap-2 bg-cyan-600 text-white py-3 rounded-lg hover:bg-cyan-700 transition-colors disabled:opacity-60"
          >
            <Play className="w-4 h-4" />
            {running ? 'Running…' : 'Run Simulation'}
          </button>

          {error && (
            <div className="text-xs text-rose-700 bg-rose-50 border border-rose-200 rounded-lg p-3">
              Simulation failed: {error}
            </div>
          )}
        </div>

        {/* Projections Panel */}
        <div className="space-y-6">
          <div className="bg-white rounded-lg border border-slate-200 p-6">
            <h2 className="text-slate-900 mb-4">Projected Impact</h2>
            
            <div className="space-y-4">
              <div className="p-4 bg-cyan-50 rounded-lg">
                <p className="text-sm text-slate-600 mb-1">Predicted Admissions (24h)</p>
                <div className="flex items-baseline gap-2">
                  <span className="text-3xl text-slate-900">{projections.admissions}</span>
                  <span className={`text-sm ${
                    projections.admissions - baselineAdmissions > 0 ? 'text-amber-600' : 'text-green-600'
                  }`}>
                    ({projections.admissions - baselineAdmissions > 0 ? '+' : ''}
                    {projections.admissions - baselineAdmissions} / {
                      ((projections.admissions - baselineAdmissions) / baselineAdmissions * 100).toFixed(1)
                    }%)
                  </span>
                </div>
              </div>

              <div className="p-4 bg-amber-50 rounded-lg">
                <p className="text-sm text-slate-600 mb-1">ICU Occupancy</p>
                <div className="flex items-baseline gap-2">
                  <span className="text-3xl text-slate-900">{projections.icu}%</span>
                  <span className={`text-sm ${
                    Number(projections.icu) - baselineICU > 0 ? 'text-amber-600' : 'text-green-600'
                  }`}>
                    ({Number(projections.icu) - baselineICU > 0 ? '+' : ''}
                    {(Number(projections.icu) - baselineICU).toFixed(1)}%)
                  </span>
                </div>
                {Number(projections.icu) >= 95 && (
                  <p className="text-xs text-amber-700 mt-2">⚠ Critical capacity warning</p>
                )}
              </div>

              <div className="p-4 bg-rose-50 rounded-lg">
                <p className="text-sm text-slate-600 mb-1">Staff Load Index</p>
                <div className="flex items-baseline gap-2">
                  <span className="text-3xl text-slate-900">{projections.staffLoad}</span>
                  <span className="text-sm text-slate-500">/10</span>
                  <span className={`text-sm ${
                    Number(projections.staffLoad) - baselineStaffLoad > 0 ? 'text-amber-600' : 'text-green-600'
                  }`}>
                    ({Number(projections.staffLoad) - baselineStaffLoad > 0 ? '+' : ''}
                    {(Number(projections.staffLoad) - baselineStaffLoad).toFixed(1)})
                  </span>
                </div>
                {Number(projections.staffLoad) >= 8 && (
                  <p className="text-xs text-rose-700 mt-2">⚠ High burnout risk</p>
                )}
              </div>
            </div>
          </div>

          {/* Recommendations */}
          <div className="bg-white rounded-lg border border-slate-200 p-6">
            <h2 className="text-slate-900 mb-3">Scenario Recommendations</h2>
            <div className="space-y-2">
              {Number(projections.icu) >= 95 && (
                <div className="text-sm text-slate-700 p-3 bg-amber-50 rounded-lg">
                  • Activate overflow protocol and prepare surge capacity
                </div>
              )}
              {Number(projections.staffLoad) >= 8 && (
                <div className="text-sm text-slate-700 p-3 bg-rose-50 rounded-lg">
                  • Call in additional staff to manage increased workload
                </div>
              )}
              {projections.admissions > baselineAdmissions + 30 && (
                <div className="text-sm text-slate-700 p-3 bg-cyan-50 rounded-lg">
                  • Prepare emergency department for admission surge
                </div>
              )}
              {projections.admissions <= baselineAdmissions && 
               Number(projections.icu) < 90 && 
               Number(projections.staffLoad) < 7 && (
                <div className="text-sm text-slate-700 p-3 bg-green-50 rounded-lg">
                  ✓ Resources appear adequate for this scenario
                </div>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}