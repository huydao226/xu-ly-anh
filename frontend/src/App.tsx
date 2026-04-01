import { useEffect, useRef, useState } from 'react'
import './App.css'

type Severity = 'normal' | 'warning' | 'critical'

type ConfigResponse = {
  app_name: string
  frame_cooldown_ms: number
  detection_features: string[]
}

type DetectionEvent = {
  code: string
  label: string
  severity: Severity
  score: number
  details: Record<string, unknown>
}

type SessionSummary = {
  session_id: string
  status: string
  frame_count: number
  risk_score: number
  warning_count: number
  critical_count: number
  last_event_at: string | null
  started_at: string
  stopped_at: string | null
  current_severity: Severity
}

type DetectionMetrics = {
  faces_detected: number
  phone_detected: boolean
  book_detected: boolean
  yaw_ratio: number | null
  pitch_ratio: number | null
  eye_line_angle: number | null
  face_box: { x: number; y: number; w: number; h: number } | null
  detector_notes: string[]
}

type FrameResponse = {
  severity: Severity
  risk_score: number
  session: SessionSummary
  metrics: DetectionMetrics
  events: DetectionEvent[]
  annotated_frame: string | null
}

type SessionReadResponse = {
  session: SessionSummary
  recent_events: DetectionEvent[]
}

type LogItem = DetectionEvent & {
  id: string
  observedAt: string
  screenshot: string | null
}

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? 'http://127.0.0.1:8000'

const severityLabel: Record<Severity, string> = {
  normal: 'Normal',
  warning: 'Warning',
  critical: 'Critical',
}

function formatMetric(value: number | null, digits = 2) {
  if (value === null || Number.isNaN(value)) {
    return '--'
  }
  return value.toFixed(digits)
}

function formatTime(value: string | null) {
  if (!value) {
    return '--'
  }
  return new Intl.DateTimeFormat('en-GB', {
    hour: '2-digit',
    minute: '2-digit',
    second: '2-digit',
  }).format(new Date(value))
}

async function fetchJson<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE_URL}${path}`, {
    headers: {
      'Content-Type': 'application/json',
      ...(init?.headers ?? {}),
    },
    ...init,
  })
  if (!response.ok) {
    const text = await response.text()
    throw new Error(text || `Request failed: ${response.status}`)
  }
  return response.json() as Promise<T>
}

function App() {
  const videoRef = useRef<HTMLVideoElement | null>(null)
  const canvasRef = useRef<HTMLCanvasElement | null>(null)
  const streamRef = useRef<MediaStream | null>(null)
  const timerRef = useRef<number | null>(null)
  const sessionIdRef = useRef<string | null>(null)

  const [operatorName, setOperatorName] = useState('Demo Operator')
  const [cameraReady, setCameraReady] = useState(false)
  const [backendHealthy, setBackendHealthy] = useState(false)
  const [loadingCamera, setLoadingCamera] = useState(false)
  const [running, setRunning] = useState(false)
  const [busy, setBusy] = useState(false)
  const [awaitingFirstFrame, setAwaitingFirstFrame] = useState(false)
  const [statusText, setStatusText] = useState('Open camera to begin local monitoring demo.')
  const [config, setConfig] = useState<ConfigResponse | null>(null)
  const [session, setSession] = useState<SessionSummary | null>(null)
  const [metrics, setMetrics] = useState<DetectionMetrics | null>(null)
  const [annotatedFrame, setAnnotatedFrame] = useState<string | null>(null)
  const [events, setEvents] = useState<LogItem[]>([])

  useEffect(() => {
    let cancelled = false
    let intervalId: number | null = null

    async function bootstrap() {
      try {
        await fetchJson('/health')
        if (!cancelled) {
          setBackendHealthy(true)
        }
        const configResponse = await fetchJson<ConfigResponse>('/api/config')
        if (!cancelled) {
          setConfig(configResponse)
        }
      } catch (error) {
        if (!cancelled) {
          setBackendHealthy(false)
          setStatusText((current) =>
            running ? current : `Backend unavailable: ${(error as Error).message}`,
          )
        }
      }
    }

    void bootstrap()
    intervalId = window.setInterval(() => {
      void bootstrap()
    }, 5000)

    return () => {
      cancelled = true
      if (intervalId !== null) {
        window.clearInterval(intervalId)
      }
    }
  }, [running])

  useEffect(() => {
    return () => {
      stopLoop()
      stopCameraStream()
    }
  }, [])

  async function openCamera() {
    setLoadingCamera(true)
    setStatusText('Requesting webcam access...')
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          width: { ideal: 1280 },
          height: { ideal: 720 },
          facingMode: 'user',
        },
        audio: false,
      })
      streamRef.current = stream
      const video = videoRef.current
      if (video) {
        video.srcObject = stream
        await new Promise<void>((resolve, reject) => {
          let settled = false
          const timeoutId = window.setTimeout(() => {
            if (!settled) {
              settled = true
              reject(new Error('Camera stream did not become ready in time.'))
            }
          }, 4000)
          const finish = () => {
            if (settled) {
              return
            }
            if (video.videoWidth > 0 && video.videoHeight > 0) {
              settled = true
              window.clearTimeout(timeoutId)
              resolve()
            }
          }
          video.onloadedmetadata = finish
          video.oncanplay = finish
          void video.play().then(finish).catch((error) => {
            if (!settled) {
              settled = true
              window.clearTimeout(timeoutId)
              reject(error)
            }
          })
        })
      }
      setCameraReady(true)
      setStatusText('Camera ready. Start a monitoring session when you are ready.')
    } catch (error) {
      setStatusText(`Camera access failed: ${(error as Error).message}`)
      setCameraReady(false)
    } finally {
      setLoadingCamera(false)
    }
  }

  function stopCameraStream() {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => track.stop())
      streamRef.current = null
    }
    setCameraReady(false)
  }

  function stopLoop() {
    if (timerRef.current !== null) {
      window.clearInterval(timerRef.current)
      timerRef.current = null
    }
  }

  function captureFrame() {
    const video = videoRef.current
    const canvas = canvasRef.current
    if (!video || !canvas || video.videoWidth === 0 || video.videoHeight === 0) {
      return null
    }
    canvas.width = video.videoWidth
    canvas.height = video.videoHeight
    const ctx = canvas.getContext('2d')
    if (!ctx) {
      return null
    }
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height)
    return canvas.toDataURL('image/jpeg', 0.82)
  }

  async function analyzeOnce() {
    if (!sessionIdRef.current || busy) {
      return
    }
    const frame = captureFrame()
    if (!frame) {
      setStatusText('Camera stream is live, but the current frame is not ready yet. Hold for a moment and keep the video visible.')
      return
    }
    setBusy(true)
    setAwaitingFirstFrame(true)
    try {
      const response = await fetchJson<FrameResponse>(`/api/session/${sessionIdRef.current}/frame`, {
        method: 'POST',
        body: JSON.stringify({ frame }),
      })
      setSession(response.session)
      setMetrics(response.metrics)
      setAnnotatedFrame(response.annotated_frame)
      setStatusText(
        response.events.length
          ? `${severityLabel[response.severity]}: ${response.events[0].label}`
          : 'Monitoring without active cheating signals.',
      )
      if (response.events.length) {
        const observedAt = new Date().toISOString()
        const screenshot = response.annotated_frame ?? frame
        setEvents((current) => {
          const next = [
            ...response.events.map((event, index) => ({
              ...event,
              observedAt,
              id: `${observedAt}-${event.code}-${index}`,
              screenshot,
            })),
            ...current,
          ]
          return next.slice(0, 24)
        })
      }
    } catch (error) {
      setStatusText(`Frame analysis failed: ${(error as Error).message}`)
    } finally {
      setAwaitingFirstFrame(false)
      setBusy(false)
    }
  }

  async function startSession() {
    if (!cameraReady) {
      setStatusText('Open the laptop camera first.')
      return
    }
    try {
      const response = await fetchJson<SessionReadResponse>('/api/session/start', {
        method: 'POST',
        body: JSON.stringify({ operator_name: operatorName || undefined }),
      })
      sessionIdRef.current = response.session.session_id
      setSession(response.session)
      setEvents([])
      setAnnotatedFrame(null)
      setAwaitingFirstFrame(true)
      setRunning(true)
      setStatusText('Monitoring session is live.')
      const intervalMs = config?.frame_cooldown_ms ?? 700
      stopLoop()
      timerRef.current = window.setInterval(() => {
        void analyzeOnce()
      }, intervalMs)
      void analyzeOnce()
    } catch (error) {
      setStatusText(`Unable to start session: ${(error as Error).message}`)
    }
  }

  async function stopSession() {
    stopLoop()
    const currentSessionId = sessionIdRef.current
    setRunning(false)
    if (!currentSessionId) {
      return
    }
    try {
      const response = await fetchJson<SessionReadResponse>(`/api/session/${currentSessionId}/stop`, {
        method: 'POST',
        body: JSON.stringify({ reason: 'manual-stop' }),
      })
      setSession(response.session)
      setStatusText('Monitoring session stopped.')
    } catch (error) {
      setStatusText(`Unable to stop session cleanly: ${(error as Error).message}`)
    }
  }

  const activeSeverity = session?.current_severity ?? 'normal'
  const currentSignal = events[0]?.label ?? 'No active suspicious signal in the current session.'
  const detectorNotesText = metrics?.detector_notes.join(', ') || 'YOLO and heuristic detector are both available.'

  return (
    <div className="app-shell">
      <div className="app-frame">
        <header className="topbar">
          <div>
            <p className="eyebrow">One-Camera Anti-Cheat Demo</p>
            <h1>AutoOEP-style local monitoring dashboard</h1>
            <p className="intro">
              Single laptop camera demo using OpenCV face heuristics, YOLO object detection, and live event scoring.
            </p>
          </div>
          <div className={`health-badge ${backendHealthy ? 'healthy' : 'unhealthy'}`}>
            <span className="health-dot" />
            {backendHealthy ? 'Backend ready' : 'Backend offline'}
          </div>
        </header>

        <main className="dashboard">
          <section className="hero-grid">
            <article className="panel intro-panel">
              <div className="panel-header">
                <div>
                  <p className="panel-kicker">Monitoring Overview</p>
                  <h2>What the current demo is watching</h2>
                </div>
                <div className={`severity-pill ${activeSeverity}`}>{severityLabel[activeSeverity]}</div>
              </div>

              <div className={`signal-banner ${activeSeverity}`}>
                <span className="signal-label">Current signal</span>
                <strong>{currentSignal}</strong>
              </div>

              <div className="overview-stats">
                <div>
                  <span>Session status</span>
                  <strong>{running ? 'Monitoring live' : 'Idle'}</strong>
                </div>
                <div>
                  <span>Sample interval</span>
                  <strong>{config?.frame_cooldown_ms ?? 600} ms</strong>
                </div>
                <div>
                  <span>Last event</span>
                  <strong>{formatTime(session?.last_event_at ?? null)}</strong>
                </div>
              </div>

              <div className="feature-cloud">
                {(config?.detection_features ?? []).map((feature) => (
                  <span key={feature} className="feature-chip">
                    {feature}
                  </span>
                ))}
              </div>
            </article>

            <section className="control-panel panel">
              <div className="panel-header">
                <div>
                  <p className="panel-kicker">Session Control</p>
                  <h2>Prepare the demo run</h2>
                </div>
              </div>

              <label className="field">
                <span>Operator label</span>
                <input value={operatorName} onChange={(event) => setOperatorName(event.target.value)} placeholder="Proctor station 01" />
              </label>

              <div className="button-row">
                <button onClick={() => void openCamera()} disabled={loadingCamera || cameraReady}>
                  {loadingCamera ? 'Opening camera...' : cameraReady ? 'Camera ready' : 'Open camera'}
                </button>
                <button className="secondary" onClick={() => void startSession()} disabled={!cameraReady || running}>
                  Start monitoring
                </button>
                <button className="ghost" onClick={() => void stopSession()} disabled={!running}>
                  Stop session
                </button>
              </div>

              <div className="status-box">
                <p className="status-title">Live status</p>
                <p>{statusText}</p>
              </div>
            </section>
          </section>

          <section className="metrics-grid">
            <article className="panel metric-card">
              <p className="metric-label">Backend</p>
              <strong>{backendHealthy ? 'Online' : 'Offline'}</strong>
              <span>Frontend checks health and config every 5 seconds.</span>
            </article>
            <article className="panel metric-card">
              <p className="metric-label">Risk score</p>
              <strong>{session?.risk_score.toFixed(1) ?? '0.0'}</strong>
              <span>Continuous risk score accumulated from suspicious events.</span>
            </article>
            <article className="panel metric-card">
              <p className="metric-label">Faces</p>
              <strong>{metrics?.faces_detected ?? 0}</strong>
              <span>{metrics?.phone_detected ? 'Phone visible in frame' : 'No phone visible right now'}</span>
            </article>
            <article className="panel metric-card">
              <p className="metric-label">Yaw ratio</p>
              <strong>{formatMetric(metrics?.yaw_ratio ?? null)}</strong>
              <span>Higher absolute values suggest turning away from the screen.</span>
            </article>
            <article className="panel metric-card">
              <p className="metric-label">Pitch ratio</p>
              <strong>{formatMetric(metrics?.pitch_ratio ?? null)}</strong>
              <span>Higher values indicate sustained looking down.</span>
            </article>
          </section>

          <section className="camera-grid">
            <article className="panel">
              <div className="panel-header">
                <div>
                  <p className="panel-kicker">Raw Camera</p>
                  <h2>Exam candidate view</h2>
                </div>
                <span className="mini-tag">{cameraReady ? 'Live' : 'Idle'}</span>
              </div>
              <div className="video-frame">
                <video ref={videoRef} playsInline muted autoPlay />
              </div>
            </article>

            <article className="panel">
              <div className="panel-header">
                <div>
                  <p className="panel-kicker">Annotated Output</p>
                  <h2>Backend analysis overlay</h2>
                </div>
                <span className="mini-tag">{busy ? 'Analyzing' : 'Synced'}</span>
              </div>
              <div className="video-frame annotated">
                {annotatedFrame ? (
                  <img src={annotatedFrame} alt="Annotated detection output" />
                ) : (
                  <p>
                    {awaitingFirstFrame
                      ? 'Waiting for the first analyzed frame from the backend. Keep the camera visible and allow a second for capture.'
                      : 'Annotated stream will appear here after the first frame is analyzed.'}
                  </p>
                )}
              </div>
            </article>
          </section>

          <section className="bottom-grid">
            <article className="panel summary-panel">
              <div className="panel-header">
                <div>
                  <p className="panel-kicker">Monitoring Summary</p>
                  <h2>Session telemetry</h2>
                </div>
              </div>
              <dl className="summary-grid">
                <div>
                  <dt>Session ID</dt>
                  <dd>{session?.session_id ?? '--'}</dd>
                </div>
                <div>
                  <dt>Frames analyzed</dt>
                  <dd>{session?.frame_count ?? 0}</dd>
                </div>
                <div>
                  <dt>Warnings</dt>
                  <dd>{session?.warning_count ?? 0}</dd>
                </div>
                <div>
                  <dt>Critical</dt>
                  <dd>{session?.critical_count ?? 0}</dd>
                </div>
                <div>
                  <dt>Last event</dt>
                  <dd>{formatTime(session?.last_event_at ?? null)}</dd>
                </div>
                <div>
                  <dt>Detector notes</dt>
                  <dd>{detectorNotesText}</dd>
                </div>
              </dl>
            </article>

            <article className="panel guide-panel">
              <div className="panel-header">
                <div>
                  <p className="panel-kicker">Detection Logic</p>
                  <h2>How the current demo catches cheating</h2>
                </div>
              </div>
              <ul className="guide-list">
                <li>Face missing or face leaves frame for a sustained moment.</li>
                <li>Multiple faces visible in the same camera frame.</li>
                <li>Eye offset and head yaw suggest the user is looking away.</li>
                <li>Pitch ratio suggests the user keeps looking down.</li>
                <li>YOLO flags a visible phone or a book-like object in frame.</li>
              </ul>
            </article>

            <article className="panel log-panel wide-panel">
              <div className="panel-header">
                <div>
                  <p className="panel-kicker">Event Log</p>
                  <h2>Observed suspicious signals</h2>
                </div>
              </div>
              <div className="log-list">
                {events.length === 0 ? (
                  <div className="empty-state">No suspicious events yet. Start a session and move away from the screen or show a phone to generate demo logs.</div>
                ) : (
                  events.map((event) => (
                    <div key={event.id} className={`log-item ${event.severity}`}>
                      <div className="log-shot">
                        {event.screenshot ? (
                          <img src={event.screenshot} alt={`${event.label} capture`} />
                        ) : (
                          <div className="log-shot-empty">No image</div>
                        )}
                      </div>
                      <div className="log-copy">
                        <p>{event.label}</p>
                        <span>{event.code}</span>
                      </div>
                      <div className="log-meta">
                        <strong>{severityLabel[event.severity]}</strong>
                        <span>{formatTime(event.observedAt)}</span>
                      </div>
                    </div>
                  ))
                )}
              </div>
            </article>
          </section>
        </main>

        <canvas ref={canvasRef} className="hidden-canvas" />
      </div>
    </div>
  )
}

export default App
