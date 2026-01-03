import { LucideIcon } from 'lucide-react';

interface QuickActionButtonProps {
  icon: LucideIcon;
  label: string;
  variant?: 'primary' | 'secondary' | 'danger';
  onClick?: () => void;
}

export function QuickActionButton({ icon: Icon, label, variant = 'secondary', onClick }: QuickActionButtonProps) {
  const variantStyles = {
    primary: 'bg-cyan-600 text-white hover:bg-cyan-700 border-cyan-600',
    secondary: 'bg-white text-slate-700 hover:bg-slate-50 border-slate-300',
    danger: 'bg-rose-600 text-white hover:bg-rose-700 border-rose-600',
  };

  return (
    <button
      onClick={onClick}
      className={`flex items-center gap-2 px-4 py-2.5 rounded-lg border transition-colors ${variantStyles[variant]}`}
    >
      <Icon className="w-4 h-4" />
      <span className="text-sm">{label}</span>
    </button>
  );
}
