import type { Theme } from '../types.ts'
import { Icon, type IconName } from './Icon.tsx'

const options: { value: Theme; label: string; icon: IconName }[] = [
  { value: 'system', label: 'System', icon: 'monitor' },
  { value: 'light', label: 'Light', icon: 'sun' },
  { value: 'dark', label: 'Dark', icon: 'moon' },
]

export function ThemeSwitch({ theme, onChange }: { theme: Theme; onChange: (theme: Theme) => void }) {
  return (
    <fieldset className="theme">
      <legend className="sr-only">Theme</legend>
      {options.map(({ value, label, icon }) => (
        <label key={value} className="theme__option" title={label}>
          <input
            className="sr-only"
            type="radio"
            name="theme"
            value={value}
            checked={theme === value}
            onChange={() => onChange(value)}
          />
          <Icon name={icon} />
          <span>{label}</span>
        </label>
      ))}
    </fieldset>
  )
}
