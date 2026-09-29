import {
  lazy,
  Suspense,
  useEffect,
  useRef,
  useState,
} from "react";

import type {
  FormEvent,
} from "react";

import "./App.css";

const Scene3D = lazy(() => import("./components/Scene3D"));

// 3D is skipped for reduced-motion, small/touch screens and low-core devices (CSS gradient fallback)
const ENABLE_3D =
  typeof window !== "undefined" &&
  !window.matchMedia("(prefers-reduced-motion: reduce)").matches &&
  window.innerWidth >= 768 &&
  (navigator.hardwareConcurrency || 4) > 2;

import ChatThread from "./components/ChatThread";
import DocumentSidebar from "./components/DocumentSidebar";
import EmptyState from "./components/EmptyState";
import SettingsPanel from "./components/SettingsPanel";

import type {
  ChatMessage,
  DocumentItem,
  EvidenceSource,
  ProgressStage,
  QueryResult,
  VerificationState,
} from "./types";

const API_BASE =
  import.meta.env.VITE_API_BASE ||
  "http://127.0.0.1:8000";

const DEMO_DOCUMENTS: DocumentItem[] = [
  {
    name: "retrieval-notes.md",
    status: "Ready",
  },
  {
    name: "rag-architecture.md",
    status: "Ready",
  },
  {
    name: "evaluation-notes.md",
    status: "Ready",
  },
];

const INITIAL_STAGES: ProgressStage[] = [
  {
    i: 1,
    ms: 0,
    label: "Searching your documents",
    complete: false,
  },
  {
    i: 2,
    ms: 0,
    label: "Finding supporting passages",
    complete: false,
  },
  {
    i: 3,
    ms: 0,
    label: "Checking the answer",
    complete: false,
  },
];

export default function App() {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [query, setQuery] = useState("");
  const [documents, setDocuments] =
    useState<DocumentItem[]>([]);
  const [loading, setLoading] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState("");
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [settingsOpen, setSettingsOpen] =
    useState(false);

  const [theme, setTheme] = useState<"dark" | "light">(
    "dark",
  );

  const [demoMode, setDemoMode] = useState(false);
  const [online, setOnline] = useState<"checking" | "online" | "offline">("checking");
  const [dragging, setDragging] = useState(false);

  const inputRef =
    useRef<HTMLTextAreaElement | null>(null);
  const fileRef = useRef<HTMLInputElement | null>(null);

  useEffect(() => {
    const storedTheme = window.localStorage.getItem(
      "evidencerag-theme",
    );

    if (
      storedTheme === "dark" ||
      storedTheme === "light"
    ) {
      setTheme(storedTheme);
    }

    const storedDemo =
      window.localStorage.getItem(
        "evidencerag-demo",
      );

    if (storedDemo === "true") {
      setDemoMode(true);
    }

  }, []);

  useEffect(() => {
    void loadDocuments();
  }, [demoMode]);

  useEffect(() => {
    window.localStorage.setItem(
      "evidencerag-theme",
      theme,
    );
  }, [theme]);

  useEffect(() => {
    window.localStorage.setItem(
      "evidencerag-demo",
      String(demoMode),
    );
  }, [demoMode]);

  useEffect(() => {
    const handleKeyboard = (event: KeyboardEvent) => {
      if (
        (event.ctrlKey || event.metaKey) &&
        event.key === "/"
      ) {
        event.preventDefault();
        inputRef.current?.focus();
      }

      if (event.key === "Escape") {
        setSidebarOpen(false);
        setSettingsOpen(false);
      }
    };

    window.addEventListener(
      "keydown",
      handleKeyboard,
    );

    return () =>
      window.removeEventListener(
        "keydown",
        handleKeyboard,
      );
  }, []);

  useEffect(() => {
    let depth = 0;
    const files = (e: DragEvent) => Boolean(e.dataTransfer?.types.includes("Files"));
    const enter = (e: DragEvent) => { if (files(e)) { depth++; setDragging(true); } };
    const leave = (e: DragEvent) => { if (files(e) && --depth <= 0) { depth = 0; setDragging(false); } };
    const over = (e: DragEvent) => { if (files(e)) e.preventDefault(); };
    const drop = (e: DragEvent) => {
      if (!files(e)) return;
      e.preventDefault(); depth = 0; setDragging(false);
      const file = e.dataTransfer?.files?.[0];
      if (file) void uploadDocument(file);
    };
    window.addEventListener("dragenter", enter);
    window.addEventListener("dragleave", leave);
    window.addEventListener("dragover", over);
    window.addEventListener("drop", drop);
    return () => {
      window.removeEventListener("dragenter", enter);
      window.removeEventListener("dragleave", leave);
      window.removeEventListener("dragover", over);
      window.removeEventListener("drop", drop);
    };
  }, [demoMode]);

  useEffect(() => {
    const t = inputRef.current;
    if (!t) return;
    t.style.height = "auto";
    t.style.height = `${Math.min(t.scrollHeight, 168)}px`;
  }, [query]);

  async function loadDocuments() {
    if (demoMode) {
      setDocuments(DEMO_DOCUMENTS);
      return;
    }

    try {
      const response = await fetch(
        `${API_BASE}/documents`,
      );

      if (!response.ok) {
        throw new Error(
          "Could not load documents.",
        );
      }

      setOnline("online");
      const data = await response.json();

      const items = Array.isArray(data)
        ? data
        : Array.isArray(data.documents)
          ? data.documents
          : [];

      setDocuments(
        items.map((item: any) => ({
          name:
            typeof item === "string"
              ? item
              : item.name || item.filename,
          status:
            item.status ||
            "Ready",
          size: item.size,
        })),
      );
    } catch {
      setOnline("offline");
      setDocuments([]);
    }
  }

  async function askQuestion(
    event?: FormEvent,
    override?: string,
  ) {
    event?.preventDefault();

    const cleanQuery = (override ?? query).trim();

    if (!cleanQuery || loading) {
      return;
    }

    setError("");
    setQuery("");
    setLoading(true);

    const userMessage: ChatMessage = {
      id: crypto.randomUUID(),
      role: "user",
      content: cleanQuery,
    };

    const assistantId = crypto.randomUUID();

    const assistantMessage: ChatMessage = {
      id: assistantId,
      role: "assistant",
      content: "",
      query: cleanQuery,
      loading: true,
      stages: INITIAL_STAGES.map((stage) => ({
        ...stage,
      })),
      sources: [],
      verification: null,
      responseTimeMs: null,
    };

    setMessages((current) => [
      ...current,
      userMessage,
      assistantMessage,
    ]);

    const startedAt = performance.now();

    try {
      if (demoMode) {
        await runDemoQuery(
          cleanQuery,
          assistantId,
          startedAt,
        );
      } else {
        await runLiveQuery(
          cleanQuery,
          assistantId,
          startedAt,
        );
      }
    } catch (requestError) {
      console.error(requestError);

      setOnline("offline");
      setError(
        "I couldn't connect to the EvidenceRAG backend.",
      );

      updateAssistant(
        assistantId,
        {
          loading: false,
          error: true,
          content:
            `Nothing answered at ${API_BASE}. Check that the API is running, then retry.`,
          responseTimeMs:
            performance.now() - startedAt,
        },
      );
    } finally {
      setLoading(false);
    }
  }

  async function runLiveQuery(
    cleanQuery: string,
    assistantId: string,
    startedAt: number,
  ) {
    const streamed = await tryStreamingQuery(
      cleanQuery,
      assistantId,
      startedAt,
    );

    if (streamed) {
      return;
    }

    const normal = await tryNormalQuery(
      cleanQuery,
    );

    if (normal) {
      const elapsed =
        performance.now() - startedAt;

      updateAssistant(assistantId, {
        loading: false,
        content: cleanAnswer(normal.answer),
        sources: normal.sources,
        verification: normal.verification,
        stages: completedStages(elapsed),
        responseTimeMs: elapsed,
      });

      return;
    }

    throw new Error(
      "All query endpoints failed.",
    );
  }

  async function tryStreamingQuery(
    cleanQuery: string,
    assistantId: string,
    startedAt: number,
  ): Promise<boolean> {
    try {
      const response = await fetch(
        `${API_BASE}/query/stream`,
        {
          method: "POST",
          headers: {
            "Content-Type":
              "application/json",
          },
          body: JSON.stringify({
            query: cleanQuery,
            top_k: 5,
          }),
        },
      );

      if (!response.ok || !response.body) {
        return false;
      }

      const reader =
        response.body.getReader();

      const decoder = new TextDecoder();

      let buffer = "";
      let answer = "";
      let sources: EvidenceSource[] = [];
      let verification:
        | VerificationState
        | null = null;

      while (true) {
        const { done, value } =
          await reader.read();

        if (done) {
          break;
        }

        buffer += decoder.decode(value, {
          stream: true,
        });

        const events =
          buffer.split("\n\n");

        buffer = events.pop() || "";

        for (const event of events) {
          const parsed =
            parseSseEvent(event);

          if (!parsed) {
            continue;
          }

          if (
            parsed.type === "token"
          ) {
            const token =
              String(parsed.data.t || "");

            answer += token;

            updateAssistant(
              assistantId,
              {
                content:
                  cleanAnswer(answer),
                loading: true,
              },
            );
          }

          if (
            parsed.type === "stage"
          ) {
            const stageData =
              parsed.data;

            updateAssistant(
              assistantId,
              {
                stages:
                  updateStagesFromEvent(
                    stageData,
                  ),
              },
            );
          }

          if (
            parsed.type === "sources"
          ) {
            sources =
              normalizeSources(
                parsed.data.items,
              );

            updateAssistant(
              assistantId,
              {
                sources,
              },
            );
          }

          if (
            parsed.type ===
            "verification"
          ) {
            verification =
              normalizeVerification(
                parsed.data,
              );

            updateAssistant(
              assistantId,
              {
                verification,
              },
            );
          }
        }
      }

      const elapsed =
        performance.now() - startedAt;

      updateAssistant(
        assistantId,
        {
          loading: false,
          content: cleanAnswer(answer),
          sources,
          verification,
          stages: completedStages(elapsed),
          responseTimeMs: elapsed,
        },
      );

      return true;
    } catch {
      return false;
    }
  }

  async function tryNormalQuery(
    cleanQuery: string,
  ): Promise<QueryResult | null> {
    const endpoints = [
      "/query",
      "/ask",
    ];

    for (const endpoint of endpoints) {
      try {
        const response = await fetch(
          `${API_BASE}${endpoint}`,
          {
            method: "POST",
            headers: {
              "Content-Type":
                "application/json",
            },
            body: JSON.stringify({
              query: cleanQuery,
              top_k: 5,
            }),
          },
        );

        if (!response.ok) {
          continue;
        }

        const data =
          await response.json();

        return normalizeQueryResponse(
          data,
        );
      } catch {
        continue;
      }
    }

    return null;
  }

  async function runDemoQuery(
    cleanQuery: string,
    assistantId: string,
    startedAt: number,
  ) {
    const stages =
      INITIAL_STAGES.map(
        (stage) => ({
          ...stage,
        }),
      );

    for (let index = 0; index < 3; index++) {
      await sleep(350);

      stages[index] = {
        ...stages[index],
        complete: true,
        ms: 300 + index * 120,
      };

      updateAssistant(
        assistantId,
        {
          stages: stages.map(
            (stage) => ({
              ...stage,
            }),
          ),
        },
      );
    }

    const sources: EvidenceSource[] = [
      {
        id: 1,
        file: "retrieval-notes.md",
        page: null,
        score: 0.94,
        label: "Best match",
        text:
          "BM25 is a probabilistic lexical ranking function that considers query-term frequency, document length, and the rarity of terms across the collection.",
        used: true,
      },
      {
        id: 2,
        file: "rag-architecture.md",
        page: null,
        score: 0.82,
        label: "Good match",
        text:
          "Hybrid retrieval combines lexical and semantic retrieval signals to improve coverage across different types of queries.",
        used: false,
      },
    ];

    const answer =
      cleanQuery
        .toLowerCase()
        .includes("bm25")
        ? "BM25 is a probabilistic lexical ranking function that considers query-term frequency, document length, and the rarity of terms across the collection. [Evidence 1]"
        : "The indexed documents contain supporting information for this question. [Evidence 1]";

    const verification: VerificationState = {
      supported: true,
      total: 1,
      supportedClaims: 1,
      reason:
        "The answer was checked against the retrieved passages.",
    };

    const elapsed =
      performance.now() - startedAt;

    updateAssistant(
      assistantId,
      {
        loading: false,
        content: answer,
        sources,
        verification,
        stages,
        responseTimeMs: elapsed,
      },
    );
  }

  async function uploadDocument(
    file: File,
  ) {
    if (demoMode) {
      setDocuments((current) => [
        ...current,
        {
          name: file.name,
          status: "Ready",
        },
      ]);

      return;
    }

    setUploading(true);
    setError("");

    try {
      const formData =
        new FormData();

      formData.append(
        "file",
        file,
      );

      const response = await fetch(
        `${API_BASE}/documents/upload`,
        {
          method: "POST",
          body: formData,
        },
      );

      if (!response.ok) {
        const data =
          await safeJson(response);

        throw new Error(
          data?.detail ||
            "Document upload failed.",
        );
      }

      await loadDocuments();
    } catch (uploadError) {
      console.error(uploadError);

      setError(
        uploadError instanceof Error
          ? uploadError.message
          : "Document upload failed.",
      );
    } finally {
      setUploading(false);
    }
  }

  async function deleteDocument(
    name: string,
  ) {
    if (demoMode) {
      setDocuments((current) =>
        current.filter(
          (document) =>
            document.name !== name,
        ),
      );

      return;
    }

    setError("");

    try {
      const response = await fetch(
        `${API_BASE}/documents/${encodeURIComponent(
          name,
        )}`,
        {
          method: "DELETE",
        },
      );

      if (!response.ok) {
        throw new Error(
          "Could not delete the document.",
        );
      }

      await loadDocuments();
    } catch (deleteError) {
      console.error(deleteError);

      setError(
        "Could not delete the document.",
      );
    }
  }

  function updateAssistant(
    id: string,
    patch: Partial<ChatMessage>,
  ) {
    setMessages((current) =>
      current.map((message) =>
        message.id === id
          ? {
              ...message,
              ...patch,
            }
          : message,
      ),
    );
  }

  function handleExampleQuestion(
    question: string,
  ) {
    setQuery(question);
    window.setTimeout(() => {
      inputRef.current?.focus();
    }, 0);
  }

  function handleAddDocument() {
    setSidebarOpen(true);
  }

  function handleThemeChange(
    nextTheme: "dark" | "light",
  ) {
    setTheme(nextTheme);
  }

  const hasMessages = messages.length > 0;
  const last = [...messages].reverse().find((m) => m.role === "assistant");
  // Scene stage comes from real pipeline state: 1 searching · 2 finding · 3 checking · 4 error
  const stage = last?.error
    ? 4
    : last?.loading
      ? Math.min(3, (last.stages?.filter((s) => s.complete).length ?? 0) + 1)
      : 0;
  const status = demoMode ? "demo" : online;
  const statusLabel = { online: "Live", offline: "Offline", demo: "Demo", checking: "Connecting" }[status];

  return (
    <div className={`app-shell ${theme === "light" ? "theme-light" : ""}`}>
      {ENABLE_3D && (
        <Suspense fallback={null}>
          <Scene3D stage={stage} />
        </Suspense>
      )}
      <div className="veil" aria-hidden="true" />

      <header className="app-header">
        <div className="header-left">
          <button type="button" className="icon-button" onClick={() => setSidebarOpen(true)} aria-label="Open documents">
            <span /><span /><span />
          </button>
          <button type="button" className="brand" onClick={() => { setMessages([]); setError(""); }} aria-label="EvidenceRAG home">
            <span className="brand-mark"><span /><span /><span /></span>
            <span className="brand-name">EvidenceRAG</span>
          </button>
        </div>

        <div className="header-actions">
          <button
            type="button"
            className="pill"
            data-status={status}
            onClick={() => setDemoMode((c) => !c)}
            title={demoMode ? "Switch to live backend" : "Switch to demo mode"}
            aria-label={`Backend status: ${statusLabel}. Click to toggle demo mode`}
          >
            <i className="dot" />{statusLabel}
          </button>
          <button type="button" className="round" onClick={() => setTheme(theme === "dark" ? "light" : "dark")} aria-label={`Switch to ${theme === "dark" ? "light" : "dark"} theme`}>
            {theme === "dark" ? "☀" : "☾"}
          </button>
          <button type="button" className="header-button" onClick={() => setSettingsOpen(true)}>Settings</button>
        </div>
      </header>

      <main className="app-main">
        {!hasMessages ? (
          <EmptyState onExampleQuestion={handleExampleQuestion} onAddDocument={handleAddDocument} hasDocuments={documents.length > 0} />
        ) : (
          <ChatThread messages={messages} onRetry={(q) => void askQuestion(undefined, q)} onDemo={() => { setDemoMode(true); setError(""); }} />
        )}
      </main>

      <form className="composer" onSubmit={askQuestion}>
        <div className="composer-inner">
          <div className="composer-row">
            <input ref={fileRef} type="file" accept=".pdf,.txt,.md" hidden onChange={(e) => { const f = e.target.files?.[0]; if (f) void uploadDocument(f); e.target.value = ""; }} />
            <button type="button" className="clip" onClick={() => fileRef.current?.click()} disabled={uploading} aria-label="Add a document (PDF, TXT, MD)" title="Add document">
              <svg viewBox="0 0 24 24" width="20" height="20" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round"><path d="M21 11.5l-8.6 8.6a5.5 5.5 0 01-7.8-7.8l8.9-8.9a3.7 3.7 0 015.2 5.2l-8.9 8.9a1.8 1.8 0 01-2.6-2.6l8.2-8.2" /></svg>
            </button>
            <textarea
              ref={inputRef}
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter" && !e.shiftKey) {
                  e.preventDefault();
                  if (!loading) void askQuestion();
                }
              }}
              placeholder={documents.length > 0 ? "Ask a question about your documents…" : "Ask a question or add a document…"}
              rows={1}
              aria-label="Ask EvidenceRAG"
              readOnly={loading}
            />
            <button type="submit" className="send-button" disabled={loading || !query.trim()} aria-label="Ask question">
              {loading ? <span className="spin" /> : <svg viewBox="0 0 24 24" width="20" height="20" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round"><path d="M12 19V5M5 12l7-7 7 7" /></svg>}
            </button>
          </div>
          <span className="composer-hint">Enter to ask · Shift+Enter for new line</span>
        </div>
      </form>

      {error && (
        <div className="toast" role="alert">
          <span>{error}</span>
          <button type="button" onClick={() => setError("")} aria-label="Dismiss error">×</button>
        </div>
      )}

      {dragging && (
        <div className="drop" aria-hidden="true">
          <div className="drop-card"><i /><i /><i /><strong>Drop to add document</strong><span>PDF, TXT or Markdown</span></div>
        </div>
      )}

      <DocumentSidebar documents={documents} open={sidebarOpen} onClose={() => setSidebarOpen(false)} onUpload={uploadDocument} onDelete={deleteDocument} uploading={uploading} error={error} />
      <SettingsPanel open={settingsOpen} onClose={() => setSettingsOpen(false)} apiBase={API_BASE} theme={theme} onThemeChange={handleThemeChange} demoMode={demoMode} onDemoModeChange={setDemoMode} />
    </div>
  );
}

function normalizeQueryResponse(
  data: any,
): QueryResult {
  const rawSources =
    data?.sources ||
    data?.evidence ||
    [];

  const answer =
    data?.answer?.answer ||
    data?.answer ||
    "I could not find a verified answer in the indexed documents.";

  const citedIds =
    extractCitationIds(answer);

  const sources =
    normalizeSources(
      rawSources,
      citedIds,
    );

  const verification =
    normalizeVerification(
      data?.verification,
    );

  return {
    answer: cleanAnswer(answer),
    sources,
    verification,
  };
}

function normalizeSources(
  rawSources: any,
  citedIds: Set<number> = new Set<number>(),
): EvidenceSource[] {
  if (!Array.isArray(rawSources)) {
    return [];
  }

  return rawSources.map(
    (item: any, index: number) => {
      const id =
        Number(
          item?.id ??
            item?.evidence_id ??
            index + 1,
        );

      const score =
        Number(
          item?.score ??
            item?.relevance ??
            item?.reranker_score ??
            0,
        );

      return {
        id,
        file:
          item?.file ||
          item?.source ||
          item?.filename ||
          "Unknown document",
        page:
          item?.page === undefined ||
          item?.page === null
            ? null
            : Number(item.page),
        score,
        label:
          getRelevanceLabel(
            score,
            index,
          ),
        text:
          item?.text ||
          item?.chunk ||
          item?.content ||
          "",
        used:
          citedIds.size === 0
            ? index === 0
            : citedIds.has(id),
      };
    },
  );
}

function normalizeVerification(
  raw: any,
): VerificationState {
  if (!raw) {
    return {
      supported: false,
      total: 0,
      supportedClaims: 0,
      reason:
        "No verification details were returned.",
    };
  }

  const total = Number(
    raw.total ??
      raw.total_claims ??
      raw.claims ??
      0,
  );

  const supportedClaims =
    Number(
      raw.supportedClaims ??
        raw.supported_claims ??
        (raw.supported
          ? total || 1
          : 0),
    );

  return {
    supported: Boolean(
      raw.supported,
    ),
    total:
      total ||
      (raw.supported ? 1 : 0),
    supportedClaims,
    reason:
      raw.reason ||
      (raw.supported
        ? "The answer was checked against the retrieved passages."
        : "Some statements could not be fully verified."),
  };
}

function extractCitationIds(
  text: string,
): Set<number> {
  const ids = new Set<number>();

  const pattern =
    /\[\s*Evidence\s*(\d+)\s*\]/gi;

  let match: RegExpExecArray | null;

  while (
    (match =
      pattern.exec(text)) !== null
  ) {
    ids.add(Number(match[1]));
  }

  return ids;
}

function cleanAnswer(
  text: string,
) {
  return String(text || "")
    .replace(
      /\[\s*Evidence\s*(\d+)\s*\]/gi,
      "[Evidence $1]",
    )
    .replace(
      /```[\s\S]*?```/g,
      "",
    )
    .trim();
}

function getRelevanceLabel(
  score: number,
  index: number,
): EvidenceSource["label"] {
  if (
    score >= 0.75 ||
    index === 0
  ) {
    return "Best match";
  }

  if (
    score >= 0.45 ||
    index <= 2
  ) {
    return "Good match";
  }

  return "Weak match";
}

function completedStages(
  elapsed: number,
): ProgressStage[] {
  const total =
    Math.max(elapsed, 300);

  return [
    {
      i: 1,
      ms: Math.round(
        total * 0.4,
      ),
      label:
        "Searching your documents",
      complete: true,
    },
    {
      i: 2,
      ms: Math.round(
        total * 0.35,
      ),
      label:
        "Finding supporting passages",
      complete: true,
    },
    {
      i: 3,
      ms: Math.round(
        total * 0.25,
      ),
      label:
        "Checking the answer",
      complete: true,
    },
  ];
}

function updateStagesFromEvent(
  data: any,
): ProgressStage[] {
  const stageNumber =
    Number(
      data?.i ??
        data?.stage ??
        data?.index ??
        1,
    );

  const ms =
    Number(
      data?.ms ??
        data?.elapsed_ms ??
        0,
    );

  return INITIAL_STAGES.map(
    (stage) => ({
      ...stage,
      complete:
        stage.i <= stageNumber,
      ms:
        stage.i <= stageNumber
          ? ms
          : 0,
    }),
  );
}

function parseSseEvent(
  event: string,
): {
  type: string;
  data: any;
} | null {
  const lines =
    event.split("\n");

  let eventType = "message";
  let data = "";

  for (const line of lines) {
    if (
      line.startsWith("event:")
    ) {
      eventType =
        line.slice(6).trim();
    }

    if (
      line.startsWith("data:")
    ) {
      data +=
        line.slice(5).trim();
    }
  }

  if (!data) {
    return null;
  }

  try {
    return {
      type: eventType,
      data: JSON.parse(data),
    };
  } catch {
    return {
      type: eventType,
      data: {
        t: data,
      },
    };
  }
}

async function safeJson(
  response: Response,
) {
  try {
    return await response.json();
  } catch {
    return null;
  }
}

function sleep(
  milliseconds: number,
) {
  return new Promise<void>(
    (resolve) =>
      window.setTimeout(
        resolve,
        milliseconds,
      ),
  );
}