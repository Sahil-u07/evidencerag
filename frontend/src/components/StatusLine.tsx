import type { ReactNode } from 'react'
import { API_BASE } from '../api.ts'
import type { HealthState } from '../types.ts'
import { Icon, type IconName } from './Icon.tsx'

type Look = { tone: string; icon: IconName; title: string; fix: ReactNode }

function describe(health: HealthState, demo: boolean): Look {
  if (demo) return { tone: 'warning', icon: 'info', title: 'Demo', fix: 'The backend is not used.' }

  switch (health.kind) {
    case 'connecting':
      return { tone: 'neutral', icon: 'spinner', title: 'Connecting', fix: `Checking the backend at ${API_BASE}.` }
    case 'offline':
      return {
        tone: 'error',
        icon: 'x-circle',
        title: 'Backend offline',
        fix: (
          <>
            Start the API so it answers at {API_BASE}: run <code>uvicorn app.main:app</code> in <code>backend</code>.
          </>
        ),
      }
    case 'model-missing':
      return {
        tone: 'warning',
        icon: 'warning',
        title: 'Model unavailable',
        fix: (
          <>
            Make sure Ollama is running, then run <code>ollama pull {health.model}</code>.
          </>
        ),
      }
    case 'ready':
      return { tone: 'ready', icon: 'check-circle', title: 'Ready', fix: `Answers use ${health.model}.` }
  }
}

export function StatusLine({ health, demo }: { health: HealthState; demo: boolean }) {
  const { tone, icon, title, fix } = describe(health, demo)
  return (
    <div className={`status status--${tone}`}>
      <Icon name={icon} className={icon === 'spinner' ? 'spin' : undefined} />
      <span>
        <strong>{title}</strong> <span className="status__fix">{fix}</span>
      </span>
    </div>
  )
}
