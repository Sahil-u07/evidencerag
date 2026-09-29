type SettingsPanelProps = {
  open: boolean;
  onClose: () => void;
  apiBase: string;
  theme: "dark" | "light";
  onThemeChange: (theme: "dark" | "light") => void;
  demoMode: boolean;
  onDemoModeChange: (enabled: boolean) => void;
};

export default function SettingsPanel({
  open,
  onClose,
  apiBase,
  theme,
  onThemeChange,
  demoMode,
  onDemoModeChange,
}: SettingsPanelProps) {
  if (!open) {
    return null;
  }

  return (
    <>
      <button
        type="button"
        className="settings-backdrop"
        aria-label="Close settings"
        onClick={onClose}
      />

      <aside
        className="settings-panel"
        aria-label="Settings"
      >
        <div className="settings-header">
          <div>
            <span className="sidebar-eyebrow">
              Preferences
            </span>

            <h2>Settings</h2>
          </div>

          <button
            type="button"
            className="sidebar-close"
            onClick={onClose}
            aria-label="Close settings"
          >
            ×
          </button>
        </div>

        <div className="settings-content">
          <section className="settings-section">
            <h3>Appearance</h3>

            <p>
              Choose the interface appearance that works
              best for your screen.
            </p>

            <div className="theme-options">
              <button
                type="button"
                className={
                  theme === "dark"
                    ? "theme-option selected"
                    : "theme-option"
                }
                onClick={() =>
                  onThemeChange("dark")
                }
              >
                <span className="theme-preview dark-preview" />

                <span>
                  <strong>Dark</strong>
                  <small>
                    Projector-friendly
                  </small>
                </span>
              </button>

              <button
                type="button"
                className={
                  theme === "light"
                    ? "theme-option selected"
                    : "theme-option"
                }
                onClick={() =>
                  onThemeChange("light")
                }
              >
                <span className="theme-preview light-preview" />

                <span>
                  <strong>Light</strong>
                  <small>
                    Bright environments
                  </small>
                </span>
              </button>
            </div>
          </section>

          <section className="settings-section">
            <h3>Connection</h3>

            <p>
              EvidenceRAG uses this backend to index
              documents and answer questions.
            </p>

            <div className="settings-field">
              <label htmlFor="backend-address">
                Backend address
              </label>

              <input
                id="backend-address"
                type="text"
                value={apiBase}
                readOnly
              />

              <span className="field-status">
                ● Configured
              </span>
            </div>
          </section>

          <section className="settings-section">
            <h3>Demo mode</h3>

            <p>
              Use a local example conversation without
              sending requests to the backend.
            </p>

            <label className="toggle-row">
              <span>
                <strong>Offline demo</strong>
                <small>
                  Works without the API
                </small>
              </span>

              <input
                type="checkbox"
                checked={demoMode}
                onChange={(event) =>
                  onDemoModeChange(
                    event.target.checked,
                  )
                }
              />

              <span
                className="toggle-control"
                aria-hidden="true"
              />
            </label>
          </section>

          <section className="settings-section model-section">
            <h3>Models</h3>

            <div className="model-row">
              <span>Embeddings</span>
              <strong>all-MiniLM-L6-v2</strong>
            </div>

            <div className="model-row">
              <span>Reranking</span>
              <strong>
                MiniLM cross-encoder
              </strong>
            </div>

            <div className="model-row">
              <span>Generation</span>
              <strong>Local Ollama</strong>
            </div>

            <div className="model-row">
              <span>Verification</span>
              <strong>NLI + evidence overlap</strong>
            </div>
          </section>
        </div>

        <div className="settings-footer">
          <span>EvidenceRAG</span>
          <span>Local document intelligence</span>
        </div>
      </aside>
    </>
  );
}