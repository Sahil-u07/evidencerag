import { useCallback, useEffect, useRef, useState } from 'react'
import { Composer } from './components/Composer.tsx'
import { EmptyState } from './components/EmptyState.tsx'
import { EvidencePanel } from './components/EvidencePanel.tsx'
import { Icon } from './components/Icon.tsx'
import { Overlay } from './components/Overlay.tsx'
import { Sidebar } from './components/Sidebar.tsx'
import { Thread } from './components/Thread.tsx'
import { TopBar } from './components/TopBar.tsx'
import { TurnView } from './components/TurnView.tsx'
import { UploadControl } from './components/UploadControl.tsx'
import { useChat } from './hooks/useChat.ts'
import { useDocuments } from './hooks/useDocuments.ts'
import { useEvidence } from './hooks/useEvidence.ts'
import { useFileDrop } from './hooks/useFileDrop.ts'
import { useHealth } from './hooks/useHealth.ts'
import { useTheme } from './hooks/useTheme.ts'
import { useViewport } from './hooks/useViewport.ts'

const DEFAULT_MAX_UPLOAD_MB = 10

export default function App() {
  const [demo, setDemo] = useState(false)
  const [question, setQuestion] = useState('')
  const [drawerOpen, setDrawerOpen] = useState(false)
  const input = useRef<HTMLTextAreaElement>(null)

  const documents = useDocuments(demo)
  const indexing = documents.uploading !== null || documents.deleting !== null
  const { health, refresh } = useHealth(demo || indexing)
  const chat = useChat(demo, refresh)
  const viewport = useViewport()
  const evidence = useEvidence(viewport)
  const { theme, setTheme } = useTheme()

  const online = health.kind === 'ready' || health.kind === 'model-missing'
  const maxUploadMb = online ? health.maxUploadMb : DEFAULT_MAX_UPLOAD_MB
  const { reload, upload: uploadFile } = documents
  const { running } = chat
  const uploadFiles = useCallback((files: File[]) => void uploadFile(files, maxUploadMb, running), [uploadFile, maxUploadMb, running])
  const dragging = useFileDrop(uploadFiles)

  useEffect(() => {
    if (online) void reload()
  }, [online, reload])

  const noDocuments = documents.loaded && documents.documents.length === 0

  const clearAll = () => {
    chat.clear()
    evidence.reset()
    setQuestion('')
  }

  const newChat = () => {
    clearAll()
    setDrawerOpen(false)
    input.current?.focus()
  }

  const send = () => {
    evidence.select(chat.send(question.trim()))
    setQuestion('')
  }

  const fillSample = (sample: string) => {
    setQuestion(sample)
    input.current?.focus()
  }

  const upload = (
    <UploadControl
      demo={demo}
      uploading={documents.uploading}
      deleting={documents.deleting !== null}
      streaming={running}
      error={documents.uploadError}
      maxUploadMb={maxUploadMb}
      onFiles={uploadFiles}
    />
  )

  const sidebar = (
    <Sidebar
      brandTag={viewport === 'phone' ? 'h2' : 'h1'}
      demo={demo}
      streaming={running}
      documents={documents}
      stale={!demo && health.kind === 'offline'}
      upload={upload}
      theme={theme}
      onTheme={setTheme}
      onNewChat={newChat}
      onDemoChange={(next) => {
        clearAll()
        setDemo(next)
      }}
    />
  )

  const selectedTurn = chat.turns.find((turn) => turn.id === evidence.selectedTurn) ?? null
  const hoveredId = evidence.hovered?.turn === evidence.selectedTurn ? (evidence.hovered?.id ?? null) : null
  const evidencePanel = (inline: boolean) => (
    <EvidencePanel turn={selectedTurn} target={evidence.target} hoveredId={inline ? null : hoveredId} inline={inline} />
  )

  return (
    <div className="app">
      <a className="skip-link" href="#question">
        Skip to the question box
      </a>
      {demo && (
        <p className="demo-banner">
          <Icon name="info" />
          Demo data. Everything here is made up and nothing was checked.
        </p>
      )}

      <div className="columns">
        {viewport !== 'phone' && <aside className="column column--left">{sidebar}</aside>}

        <main className="column column--main">
          <TopBar health={health} demo={demo} phone={viewport === 'phone'} onOpenDocuments={() => setDrawerOpen(true)} />

          <Thread turns={chat.turns}>
            {chat.turns.length === 0 && (
              <EmptyState
                noDocuments={noDocuments}
                hasDocuments={documents.documents.length > 0}
                upload={viewport === 'phone' ? upload : null}
                onSample={fillSample}
              />
            )}
            {chat.turns.map((turn) => (
              <TurnView
                key={turn.id}
                turn={turn}
                selected={turn.id === evidence.selectedTurn}
                evidenceOpen={evidence.open}
                viewport={viewport}
                activeId={evidence.target?.id ?? null}
                busy={running || indexing}
                inlineEvidence={evidencePanel(true)}
                onCite={(id) => evidence.cite(turn.id, id)}
                onHover={(id) => evidence.hover(turn.id, id)}
                onShowEvidence={() => evidence.show(turn.id)}
                onHideEvidence={evidence.close}
                onRetry={() => {
                  evidence.select(turn.id)
                  chat.retry(turn)
                }}
              />
            ))}
          </Thread>

          <Composer
            value={question}
            onChange={setQuestion}
            onSend={send}
            onStop={chat.stop}
            running={running}
            health={health}
            demo={demo}
            indexing={indexing}
            noDocuments={noDocuments}
            inputRef={input}
          />
        </main>

        {viewport === 'desktop' && <aside className="column column--right">{evidencePanel(false)}</aside>}
      </div>

      {viewport === 'phone' && drawerOpen && (
        <Overlay label="Documents" side="left" onClose={() => setDrawerOpen(false)}>
          {sidebar}
        </Overlay>
      )}
      {viewport === 'tablet' && evidence.open && (
        <Overlay label="Evidence" side="right" onClose={evidence.close}>
          {evidencePanel(false)}
        </Overlay>
      )}

      {dragging && (
        <div className="drop-hint">
          <p>
            <Icon name="upload" size={20} />
            Drop the file to upload it
          </p>
        </div>
      )}

      <div className="sr-only" role="status">
        {chat.announcement}
      </div>
    </div>
  )
}
