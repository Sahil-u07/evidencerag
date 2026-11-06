import { Component, type ReactNode } from 'react'
import { Icon } from './Icon.tsx'

export class ErrorBoundary extends Component<{ children: ReactNode }, { failed: boolean }> {
  state = { failed: false }

  static getDerivedStateFromError() {
    return { failed: true }
  }

  render() {
    if (!this.state.failed) return this.props.children
    return (
      <main className="crash" role="alert">
        <h1 className="crash__title">
          <Icon name="x-circle" size={20} />
          Something went wrong on this page
        </h1>
        <p>Your documents are safe. Reloading clears the conversation.</p>
        <button type="button" className="btn btn--primary" onClick={() => window.location.reload()}>
          <Icon name="retry" />
          Reload
        </button>
      </main>
    )
  }
}
