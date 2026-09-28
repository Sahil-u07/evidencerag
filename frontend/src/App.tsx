import {
  Suspense,
  useEffect,
  useMemo,
  useRef,
  useState,
  type ChangeEvent,
  type ReactNode,
} from 'react'
import { Canvas, useFrame } from '@react-three/fiber'
import {
  Float,
  Line,
  MeshTransmissionMaterial,
  OrbitControls,
  PerspectiveCamera,
  Sparkles,
  Text,
} from '@react-three/drei'
import * as THREE from 'three'
import './App.css'

type Evidence = {
  evidence_id: number
  source: string
  chunk_id: string
  page: number | null
  score: number
  text: string
}

type SearchResponse = {
  query: string
  results: Evidence[]
}

type AskResponse = {
  query: string
  answer: string
  evidence: Evidence[]
  verification: {
    supported: boolean
    cited_evidence: number[]
    unsupported_claims: string[]
    reason: string
  }
  metrics: {
    query: string
    top_k: number
    retrieved_evidence_count: number
    cited_evidence_count: number
    verification_supported: boolean
    latency_ms: number
  }
}

const API_BASE = 'http://localhost:8000'

function Icon({
  children,
  size = 18,
}: {
  children: ReactNode
  size?: number
}) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.7"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
    >
      {children}
    </svg>
  )
}

function EvidenceNode({
  position,
  label,
  color,
  delay,
}: {
  position: [number, number, number]
  label: string
  color: string
  delay: number
}) {
  const group = useRef<THREE.Group>(null)

  useFrame((state) => {
    if (!group.current) return

    const t = state.clock.elapsedTime + delay

    group.current.position.y =
      position[1] + Math.sin(t * 0.7) * 0.12

    group.current.rotation.y = t * 0.25
    group.current.rotation.z =
      Math.sin(t * 0.4) * 0.08
  })

  return (
    <group ref={group} position={position}>
      <mesh>
        <icosahedronGeometry args={[0.22, 1]} />

        <meshStandardMaterial
          color={color}
          emissive={color}
          emissiveIntensity={1.8}
          roughness={0.25}
          metalness={0.75}
        />
      </mesh>

      <mesh scale={1.5}>
        <icosahedronGeometry args={[0.22, 1]} />

        <meshBasicMaterial
          color={color}
          transparent
          opacity={0.09}
          wireframe
        />
      </mesh>

      <Text
        position={[0, -0.42, 0.22]}
        fontSize={0.115}
        color="#ddd6ff"
        anchorX="center"
        anchorY="middle"
        outlineWidth={0.012}
        outlineColor="#09070f"
        depthOffset={-1}
      >
        {label.toUpperCase()}
      </Text>
    </group>
  )
}

function EvidenceCore({
  active,
}: {
  active: boolean
}) {
  const group = useRef<THREE.Group>(null)

  useFrame((state) => {
    if (!group.current) return

    group.current.rotation.y =
      state.clock.elapsedTime * 0.18

    group.current.rotation.x =
      Math.sin(
        state.clock.elapsedTime * 0.4,
      ) * 0.08
  })

  return (
    <group ref={group}>
      <mesh>
        <sphereGeometry
          args={[0.7, 64, 64]}
        />

        <MeshTransmissionMaterial
          backside
          samples={8}
          thickness={0.45}
          roughness={0.08}
          transmission={1}
          ior={1.45}
          chromaticAberration={0.04}
          distortion={0.12}
          distortionScale={0.18}
          temporalDistortion={0.08}
          color="#7665e6"
        />
      </mesh>

      <mesh scale={1.15}>
        <sphereGeometry
          args={[0.7, 32, 32]}
        />

        <meshBasicMaterial
          color="#7868ed"
          transparent
          opacity={
            active
              ? 0.1
              : 0.045
          }
          wireframe
        />
      </mesh>

      <mesh scale={0.34}>
        <sphereGeometry
          args={[0.7, 32, 32]}
        />

        <meshBasicMaterial
          color="#a395ff"
          transparent
          opacity={
            active
              ? 0.48
              : 0.28
          }
        />
      </mesh>

      <pointLight
        color="#725cf1"
        intensity={
          active
            ? 6
            : 3
        }
        distance={7}
      />
    </group>
  )
}

function NetworkScene({
  active,
}: {
  active: boolean
}) {
  const nodes = useMemo(
    () => [
      {
        position: [
          -2.65,
          1.25,
          -0.2,
        ] as [
          number,
          number,
          number,
        ],
        label: 'dense',
        color: '#6858df',
        delay: 0,
      },
      {
        position: [
          -2.9,
          -1.2,
          -0.4,
        ] as [
          number,
          number,
          number,
        ],
        label: 'bm25',
        color: '#8572ef',
        delay: 1.8,
      },
      {
        position: [
          2.7,
          1.35,
          -0.4,
        ] as [
          number,
          number,
          number,
        ],
        label: 'rerank',
        color: '#917ef9',
        delay: 0.9,
      },
      {
        position: [
          2.8,
          -1.15,
          -0.2,
        ] as [
          number,
          number,
          number,
        ],
        label: 'verify',
        color: '#67c995',
        delay: 2.7,
      },
    ],
    [],
  )

  return (
    <>
      <PerspectiveCamera
        makeDefault
        position={[
          0,
          0.15,
          7.5,
        ]}
        fov={42}
      />

      <ambientLight intensity={0.32} />

      <pointLight
        position={[
          0,
          2.8,
          3,
        ]}
        intensity={5}
        color="#6e5af0"
      />

      <pointLight
        position={[
          -3.5,
          -2,
          2,
        ]}
        intensity={2}
        color="#3d7af0"
      />

      <Sparkles
        count={110}
        scale={[
          7,
          5,
          4,
        ]}
        size={1.2}
        speed={0.22}
        opacity={0.45}
        color="#8677dc"
      />

      <Float
        speed={1.25}
        rotationIntensity={0.08}
        floatIntensity={0.3}
      >
        <EvidenceCore
          active={active}
        />
      </Float>

      {nodes.map((node) => (
        <EvidenceNode
          key={node.label}
          {...node}
        />
      ))}

      <Line
        points={[
          [
            -2.65,
            1.25,
            -0.2,
          ],
          [
            -0.95,
            0.35,
            0,
          ],
          [
            0,
            0,
            0,
          ],
        ]}
        color="#5346a8"
        transparent
        opacity={0.46}
        lineWidth={1}
      />

      <Line
        points={[
          [
            -2.9,
            -1.2,
            -0.4,
          ],
          [
            -1.0,
            -0.35,
            0,
          ],
          [
            0,
            0,
            0,
          ],
        ]}
        color="#6758c4"
        transparent
        opacity={0.5}
        lineWidth={1}
      />

      <Line
        points={[
          [
            0,
            0,
            0,
          ],
          [
            1.15,
            0.45,
            -0.05,
          ],
          [
            2.7,
            1.35,
            -0.4,
          ],
        ]}
        color="#6f5fd4"
        transparent
        opacity={0.48}
        lineWidth={1}
      />

      <Line
        points={[
          [
            0,
            0,
            0,
          ],
          [
            1.1,
            -0.42,
            -0.05,
          ],
          [
            2.8,
            -1.15,
            -0.2,
          ],
        ]}
        color="#4d9c79"
        transparent
        opacity={0.34}
        lineWidth={1}
      />

      <OrbitControls
        enablePan={false}
        enableZoom={false}
        minPolarAngle={
          Math.PI / 2.7
        }
        maxPolarAngle={
          Math.PI / 1.9
        }
        autoRotate
        autoRotateSpeed={0.22}
      />
    </>
  )
}

function ThreeHero({
  active,
}: {
  active: boolean
}) {
  return (
    <div className="hero-3d">
      <Canvas
        dpr={[1, 1.8]}
        gl={{
          antialias: true,
          alpha: true,
          powerPreference:
            'high-performance',
        }}
      >
        <Suspense fallback={null}>
          <NetworkScene
            active={active}
          />
        </Suspense>
      </Canvas>

      <div className="hero-scanline" />
    </div>
  )
}

function App() {
  const [query, setQuery] =
    useState('')

  const [answer, setAnswer] =
    useState<AskResponse | null>(
      null,
    )

  const [searchResults, setSearchResults] =
    useState<SearchResponse | null>(
      null,
    )

  const [loading, setLoading] =
    useState(false)

  const [uploading, setUploading] =
    useState(false)

  const [error, setError] =
    useState('')

  const [activeTab, setActiveTab] =
    useState<
      'research' | 'sources'
    >('research')

  async function askQuestion() {
    if (
      !query.trim() ||
      loading
    ) {
      return
    }

    setLoading(true)
    setError('')
    setAnswer(null)
    setSearchResults(null)
    setActiveTab('research')

    try {
      const response =
        await fetch(
          `${API_BASE}/ask`,
          {
            method: 'POST',
            headers: {
              'Content-Type':
                'application/json',
            },
            body: JSON.stringify({
              query:
                query.trim(),
              top_k: 5,
            }),
          },
        )

      if (!response.ok) {
        throw new Error(
          'The RAG API could not process this question.',
        )
      }

      const data: AskResponse =
        await response.json()

      setAnswer(data)
    } catch (err) {
      setError(
        err instanceof Error
          ? err.message
          : 'Something went wrong while contacting the API.',
      )
    } finally {
      setLoading(false)
    }
  }

  async function searchEvidence() {
    if (
      !query.trim() ||
      loading
    ) {
      return
    }

    setLoading(true)
    setError('')
    setAnswer(null)
    setSearchResults(null)
    setActiveTab('sources')

    try {
      const response =
        await fetch(
          `${API_BASE}/search`,
          {
            method: 'POST',
            headers: {
              'Content-Type':
                'application/json',
            },
            body: JSON.stringify({
              query:
                query.trim(),
              top_k: 5,
            }),
          },
        )

      if (!response.ok) {
        throw new Error(
          'The retrieval service could not complete the search.',
        )
      }

      const data: SearchResponse =
        await response.json()

      setSearchResults(data)
    } catch (err) {
      setError(
        err instanceof Error
          ? err.message
          : 'Something went wrong while searching.',
      )
    } finally {
      setLoading(false)
    }
  }

  async function uploadDocument(
    event: ChangeEvent<HTMLInputElement>,
  ) {
    const file =
      event.target.files?.[0]

    if (!file) {
      return
    }

    setUploading(true)
    setError('')

    try {
      const formData =
        new FormData()

      formData.append(
        'file',
        file,
      )

      const response =
        await fetch(
          `${API_BASE}/documents/upload`,
          {
            method: 'POST',
            body: formData,
          },
        )

      if (!response.ok) {
        throw new Error(
          'Document upload failed.',
        )
      }
    } catch (err) {
      setError(
        err instanceof Error
          ? err.message
          : 'Document upload failed.',
      )
    } finally {
      setUploading(false)
      event.target.value = ''
    }
  }

  function handleSubmit(
    event: React.FormEvent,
  ) {
    event.preventDefault()
    void askQuestion()
  }

  const hasResult =
    Boolean(
      answer ||
        searchResults,
    )

  useEffect(() => {
    const handler = (
      event: KeyboardEvent,
    ) => {
      if (
        (event.ctrlKey ||
          event.metaKey) &&
        event.key === '/'
      ) {
        event.preventDefault()

        document
          .querySelector<HTMLTextAreaElement>(
            '.command-input textarea',
          )
          ?.focus()
      }
    }

    window.addEventListener(
      'keydown',
      handler,
    )

    return () =>
      window.removeEventListener(
        'keydown',
        handler,
      )
  }, [])

  return (
    <div className="app">
      <div className="ambient ambient-one" />
      <div className="ambient ambient-two" />

      <header className="topbar">
        <div className="brand">
          <div className="brand-symbol">
            <span />
            <span />
            <span />
          </div>

          <div>
            <div className="brand-title">
              EvidenceRAG
            </div>

            <div className="brand-caption">
              DOCUMENT INTELLIGENCE ENGINE
            </div>
          </div>
        </div>

        <div className="topbar-actions">
          <div className="engine-chip">
            <span className="live-dot" />
            LOCAL ENGINE
          </div>

          <a
            href="https://github.com/Sahil-u07/evidencerag"
            target="_blank"
            rel="noreferrer"
            className="icon-button"
          >
            <Icon size={17}>
              <path d="M15 22v-4a4.8 4.8 0 0 0-1-3.5c3.3-.4 6.7-1.6 6.7-7A5.5 5.5 0 0 0 19.2 4 5.1 5.1 0 0 0 19 1s-1.2-.4-4 1.3a13.4 13.4 0 0 0-6 0C6.2.6 5 1 5 1a5.1 5.1 0 0 0-.2 3A5.5 5.5 0 0 0 3.3 7.5c0 5.4 3.4 6.6 6.7 7A4.8 4.8 0 0 0 9 18v4" />
              <path d="M9 18c-4.5 2-5-2-7-2" />
            </Icon>
          </a>
        </div>
      </header>

      <main>
        <section className="hero-section">
          <ThreeHero
            active={
              loading ||
              hasResult
            }
          />

          <div className="hero-content">
            <div className="hero-kicker">
              <span />
              HYBRID RETRIEVAL · GROUNDED GENERATION
            </div>

            <h1>
              Search knowledge.
              <br />
              <span>
                Surface proof.
              </span>
            </h1>

            <p>
              A local-first document
              intelligence layer that
              finds relevant passages,
              reranks them, generates
              an answer, and checks
              whether the answer is
              actually supported.
            </p>

            <div className="hero-stats">
              <div>
                <strong>
                  05
                </strong>

                <span>
                  stages
                </span>
              </div>

              <div>
                <strong>
                  02
                </strong>

                <span>
                  retrievers
                </span>
              </div>

              <div>
                <strong>
                  01
                </strong>

                <span>
                  evidence layer
                </span>
              </div>
            </div>
          </div>

          <div className="hero-corner hero-corner-left">
            <span>
              SYS / READY
            </span>
          </div>

          <div className="hero-corner hero-corner-right">
            <span>
              127.0.0.1
            </span>
          </div>
        </section>

        <section className="workspace">
          <div className="workspace-tabs">
            <button
              type="button"
              className={
                activeTab ===
                'research'
                  ? 'tab active'
                  : 'tab'
              }
              onClick={() =>
                setActiveTab(
                  'research',
                )
              }
            >
              Research
            </button>

            <button
              type="button"
              className={
                activeTab ===
                'sources'
                  ? 'tab active'
                  : 'tab'
              }
              onClick={() => {
                setActiveTab(
                  'sources',
                )

                if (
                  query.trim()
                ) {
                  void searchEvidence()
                }
              }}
            >
              Evidence
            </button>

            <label className="upload-control">
              <Icon size={14}>
                <path d="M12 16V4" />
                <path d="m7 9 5-5 5 5" />
                <path d="M4 20h16" />
              </Icon>

              {uploading
                ? 'Indexing...'
                : 'Add document'}

              <input
                type="file"
                accept=".pdf,.txt,.md"
                onChange={
                  uploadDocument
                }
                disabled={
                  uploading
                }
              />
            </label>
          </div>

          <div className="command-panel">
            <div className="command-header">
              <div className="command-status">
                <span />
                QUERY CONSOLE
              </div>

              <div className="command-meta">
                HYBRID / RRF / RERANK
              </div>
            </div>

            <form
              onSubmit={
                handleSubmit
              }
            >
              <div className="command-input">
                <div className="prompt-symbol">
                  ›
                </div>

                <textarea
                  value={query}
                  onChange={(event) =>
                    setQuery(
                      event.target
                        .value,
                    )
                  }
                  onKeyDown={(
                    event,
                  ) => {
                    if (
                      event.key ===
                        'Enter' &&
                      (event.ctrlKey ||
                        event.metaKey)
                    ) {
                      event.preventDefault()
                      void askQuestion()
                    }
                  }}
                  placeholder="Ask your indexed knowledge base..."
                  rows={2}
                />

                <button
                  type="submit"
                  disabled={
                    loading ||
                    !query.trim()
                  }
                  className="execute-button"
                >
                  {loading ? (
                    <span className="execute-loading">
                      <span />
                      <span />
                      <span />
                    </span>
                  ) : (
                    <>
                      RUN
                      <span>
                        ↗
                      </span>
                    </>
                  )}
                </button>
              </div>
            </form>

            <div className="command-footer">
              <div className="prompt-suggestions">
                <span>
                  EXAMPLES
                </span>

                <button
                  type="button"
                  onClick={() =>
                    setQuery(
                      'What are the main findings in these documents?',
                    )
                  }
                >
                  main findings
                </button>

                <button
                  type="button"
                  onClick={() =>
                    setQuery(
                      'What evidence supports the key conclusions?',
                    )
                  }
                >
                  supporting evidence
                </button>

                <button
                  type="button"
                  onClick={() =>
                    setQuery(
                      'What are the most important limitations?',
                    )
                  }
                >
                  limitations
                </button>
              </div>

              <div className="shortcut">
                CTRL + ENTER
              </div>
            </div>
          </div>

          <div className="pipeline-strip">
            <div className="pipeline-title">
              <span>
                01
              </span>

              EXECUTION GRAPH
            </div>

            <div className="pipeline-nodes">
              {[
                [
                  'Query',
                  'Input',
                ],
                [
                  'Hybrid',
                  'Dense + BM25',
                ],
                [
                  'RRF',
                  'Fusion',
                ],
                [
                  'Rerank',
                  'Cross-encoder',
                ],
                [
                  'Verify',
                  'Grounding',
                ],
              ].map(
                (
                  [
                    title,
                    subtitle,
                  ],
                  index,
                ) => (
                  <div
                    className="pipeline-node"
                    key={title}
                  >
                    <div className="pipeline-node-number">
                      0
                      {index + 1}
                    </div>

                    <div className="pipeline-node-body">
                      <strong>
                        {title}
                      </strong>

                      <span>
                        {subtitle}
                      </span>
                    </div>

                    {index <
                      4 && (
                      <div className="pipeline-connector">
                        <span />
                      </div>
                    )}
                  </div>
                ),
              )}
            </div>
          </div>

          {error && (
            <div className="error-box">
              <div className="error-mark">
                !
              </div>

              <div>
                <strong>
                  ENGINE ERROR
                </strong>

                <p>
                  {error}
                </p>
              </div>
            </div>
          )}

          {answer && (
            <section className="result-section">
              <div className="result-section-heading">
                <div>
                  <span className="section-code">
                    RESULT / 01
                  </span>

                  <h2>
                    Grounded response
                  </h2>
                </div>

                <div
                  className={
                    answer
                      .verification
                      .supported
                      ? 'verification-state verified'
                      : 'verification-state failed'
                  }
                >
                  <span />

                  {answer
                    .verification
                    .supported
                    ? 'VERIFIED'
                    : 'NOT VERIFIED'}
                </div>
              </div>

              <div className="answer-result-grid">
                <article className="answer-result-card">
                  <div className="answer-result-bar">
                    <span>
                      GENERATED ANSWER
                    </span>

                    <span>
                      {answer.query}
                    </span>
                  </div>

                  <div className="answer-result-body">
                    {answer.answer}
                  </div>

                  <div className="telemetry-row">
                    <div>
                      <span>
                        RETRIEVED
                      </span>

                      <strong>
                        {
                          answer
                            .metrics
                            .retrieved_evidence_count
                        }
                      </strong>
                    </div>

                    <div>
                      <span>
                        CITED
                      </span>

                      <strong>
                        {
                          answer
                            .metrics
                            .cited_evidence_count
                        }
                      </strong>
                    </div>

                    <div>
                      <span>
                        LATENCY
                      </span>

                      <strong>
                        {answer.metrics.latency_ms.toFixed(
                          0,
                        )}
                        ms
                      </strong>
                    </div>
                  </div>
                </article>

                <aside className="verification-card">
                  <span className="section-code">
                    VERIFICATION
                  </span>

                  <div
                    className={
                      answer
                        .verification
                        .supported
                        ? 'verification-orb success'
                        : 'verification-orb'
                    }
                  >
                    <div>
                      {answer
                        .verification
                        .supported
                        ? '✓'
                        : '!'}
                    </div>
                  </div>

                  <strong>
                    {answer
                      .verification
                      .supported
                      ? 'Evidence aligned'
                      : 'Evidence insufficient'}
                  </strong>

                  <p>
                    {answer
                      .verification
                      .supported
                      ? 'The generated response passed the grounding layer.'
                      : answer
                          .verification
                          .reason ||
                        'The response did not meet the grounding threshold.'}
                  </p>
                </aside>
              </div>

              <EvidenceList
                evidence={
                  answer.evidence
                }
              />
            </section>
          )}

          {searchResults && (
            <section className="result-section">
              <div className="result-section-heading">
                <div>
                  <span className="section-code">
                    RETRIEVAL / 01
                  </span>

                  <h2>
                    Retrieved evidence
                  </h2>
                </div>

                <span className="results-count">
                  {
                    searchResults
                      .results
                      .length
                  }{' '}
                  passages
                </span>
              </div>

              <EvidenceList
                evidence={
                  searchResults.results
                }
              />
            </section>
          )}

          {!hasResult &&
            !loading && (
              <section className="system-grid">
                <div className="system-card">
                  <div className="system-card-top">
                    <span>
                      ARCHITECTURE
                    </span>

                    <span>
                      01
                    </span>
                  </div>

                  <div className="system-visual">
                    <div className="system-ring ring-a" />
                    <div className="system-ring ring-b" />
                    <div className="system-dot" />
                  </div>

                  <h3>
                    Retrieval is a system.
                  </h3>

                  <p>
                    Dense similarity
                    and lexical matching
                    enter the same
                    retrieval graph before
                    rank fusion and
                    semantic reranking.
                  </p>
                </div>

                <div className="system-card">
                  <div className="system-card-top">
                    <span>
                      GUARDRAIL
                    </span>

                    <span>
                      02
                    </span>
                  </div>

                  <div className="guardrail-visual">
                    <div className="guardrail-line" />

                    <div className="guardrail-node">
                      <Icon size={17}>
                        <path d="M12 3 5 6v5c0 4.5 2.8 8 7 10 4.2-2 7-5.5 7-10V6z" />
                        <path d="m9 12 2 2 4-4" />
                      </Icon>
                    </div>
                  </div>

                  <h3>
                    Generation needs proof.
                  </h3>

                  <p>
                    Answers are checked
                    against the evidence
                    that was actually
                    retrieved instead of
                    treating generated text
                    as ground truth.
                  </p>
                </div>

                <div className="system-card system-card-small">
                  <div className="system-card-top">
                    <span>
                      ENGINE
                    </span>

                    <span>
                      03
                    </span>
                  </div>

                  <div className="engine-readout">
                    <strong>
                      LOCAL
                    </strong>

                    <span>
                      INFERENCE
                    </span>

                    <div className="readout-bar">
                      <i />
                      <i />
                      <i />
                      <i />
                      <i />
                      <i />
                      <i />
                      <i />
                    </div>
                  </div>

                  <h3>
                    Private by design.
                  </h3>

                  <p>
                    Retrieval, reranking,
                    verification, and
                    generation can run
                    locally.
                  </p>
                </div>
              </section>
            )}
        </section>
      </main>
    </div>
  )
}

function EvidenceList({
  evidence,
}: {
  evidence: Evidence[]
}) {
  return (
    <div className="evidence-section">
      <div className="evidence-section-heading">
        <div>
          <span className="section-code">
            SOURCE MATERIAL
          </span>

          <h3>
            Evidence traces
          </h3>
        </div>

        <span>
          {evidence.length} chunks
        </span>
      </div>

      <div className="evidence-list">
        {evidence.map(
          (item, index) => (
            <article
              className="evidence-card"
              key={`${item.chunk_id}-${index}`}
            >
              <div className="evidence-number">
                {String(
                  item.evidence_id,
                ).padStart(
                  2,
                  '0',
                )}
              </div>

              <div className="evidence-main">
                <div className="evidence-meta">
                  <span>
                    {item.source}
                  </span>

                  {item.page !==
                    null && (
                    <>
                      <i />
                      <span>
                        PAGE{' '}
                        {item.page}
                      </span>
                    </>
                  )}

                  <div className="score">
                    SCORE{' '}
                    {item.score.toFixed(
                      3,
                    )}
                  </div>
                </div>

                <p>
                  {item.text}
                </p>

                <div className="evidence-footer">
                  <span>
                    {item.chunk_id}
                  </span>

                  <span>
                    RETRIEVED PASSAGE
                  </span>
                </div>
              </div>
            </article>
          ),
        )}
      </div>
    </div>
  )
}

export default App