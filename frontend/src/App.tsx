import { useEffect, useRef, useState } from 'react'
import './App.css'

type Severity = 'normal' | 'warning' | 'critical'
type MonitorMode = 'classic' | 'oep'
type ScreenMode = 'classic' | 'oep'

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

type OepPredictionScore = {
  label: string
  confidence: number
}

type OepSessionSummary = {
  session_id: string
  status: string
  frame_count: number
  buffer_size: number
  started_at: string
  stopped_at: string | null
  last_prediction: string | null
  last_confidence: number | null
}

type OepConfigResponse = {
  model_name: string
  sequence_frames: number
  required_frames: number
  frame_width: number
  labels: string[]
}

type OepFrameResponse = {
  session: OepSessionSummary
  ready: boolean
  prediction_label: string | null
  confidence: number | null
  probabilities: OepPredictionScore[]
  annotated_frame: string | null
  features: number[]
  face_box: { x: number; y: number; w: number; h: number } | null
  status_text: string
}

type OepSessionReadResponse = {
  session: OepSessionSummary
  probabilities: OepPredictionScore[]
}

type LogItem = DetectionEvent & {
  id: string
  observedAt: string
  screenshot: string | null
}

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? 'http://127.0.0.1:8000'
const OEP_API_BASE_URL = import.meta.env.VITE_OEP_API_BASE_URL ?? 'http://127.0.0.1:8001'
const CLASSIC_IDLE_STATUS = 'Open camera to begin local monitoring demo.'
const OEP_IDLE_STATUS = 'Open camera, then start the new OEP temporal monitor to compare the trained model.'
const OEP_FEATURE_LABELS = [
  'Brightness',
  'Motion',
  'Edge density',
  'Face present',
  'Face area',
  'Face center X',
  'Face center Y',
  'Eye pair present',
  'Eye distance',
  'Yaw proxy',
  'Pitch proxy',
  'Eye open',
  'Lower-face texture',
  'Multiple faces',
  'Upper body present',
  'Upper body area',
  'Upper body center X',
  'Upper body center Y',
  'Face-body relation',
]
const OEP_LABELS: Record<string, string> = {
  normal: 'Normal',
  'absence/offscreen': 'Absence / Offscreen',
  suspicious_action: 'Suspicious Action',
  device: 'Device Use',
}

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

function formatPercent(value: number | null, digits = 1) {
  if (value === null || Number.isNaN(value)) {
    return '--'
  }
  return `${(value * 100).toFixed(digits)}%`
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

function formatOepLabel(value: string | null | undefined) {
  if (!value) {
    return '--'
  }
  return OEP_LABELS[value] ?? value
}

async function fetchJsonFrom<T>(baseUrl: string, path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${baseUrl}${path}`, {
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
  const activeMonitorModeRef = useRef<MonitorMode | null>(null)
  const classicSessionIdRef = useRef<string | null>(null)
  const oepSessionIdRef = useRef<string | null>(null)
  const classicBusyRef = useRef(false)
  const oepBusyRef = useRef(false)

  const [screen, setScreen] = useState<ScreenMode>('classic')
  const [operatorName, setOperatorName] = useState('Demo Operator')
  const [cameraReady, setCameraReady] = useState(false)
  const [backendHealthy, setBackendHealthy] = useState(false)
  const [oepHealthy, setOepHealthy] = useState(false)
  const [loadingCamera, setLoadingCamera] = useState(false)
  const [activeMonitorMode, setActiveMonitorMode] = useState<MonitorMode | null>(null)

  const [busy, setBusy] = useState(false)
  const [awaitingFirstFrame, setAwaitingFirstFrame] = useState(false)
  const [statusText, setStatusText] = useState(CLASSIC_IDLE_STATUS)
  const [config, setConfig] = useState<ConfigResponse | null>(null)
  const [session, setSession] = useState<SessionSummary | null>(null)
  const [metrics, setMetrics] = useState<DetectionMetrics | null>(null)
  const [annotatedFrame, setAnnotatedFrame] = useState<string | null>(null)
  const [events, setEvents] = useState<LogItem[]>([])

  const [oepBusy, setOepBusy] = useState(false)
  const [oepAwaitingFirstFrame, setOepAwaitingFirstFrame] = useState(false)
  const [oepStatusText, setOepStatusText] = useState(OEP_IDLE_STATUS)
  const [oepConfig, setOepConfig] = useState<OepConfigResponse | null>(null)
  const [oepSession, setOepSession] = useState<OepSessionSummary | null>(null)
  const [oepAnnotatedFrame, setOepAnnotatedFrame] = useState<string | null>(null)
  const [oepProbabilities, setOepProbabilities] = useState<OepPredictionScore[]>([])
  const [oepFeatures, setOepFeatures] = useState<number[]>([])
  const [oepReady, setOepReady] = useState(false)

  useEffect(() => {
    let cancelled = false
    let intervalId: number | null = null

    async function bootstrapClassic() {
      try {
        await fetchJsonFrom(API_BASE_URL, '/health')
        const configResponse = await fetchJsonFrom<ConfigResponse>(API_BASE_URL, '/api/config')
        if (!cancelled) {
          setBackendHealthy(true)
          setConfig(configResponse)
        }
      } catch (error) {
        if (!cancelled) {
          setBackendHealthy(false)
          if (activeMonitorModeRef.current !== 'classic') {
            setStatusText(`Backend unavailable: ${(error as Error).message}`)
          }
        }
      }
    }

    async function bootstrapOep() {
      try {
        await fetchJsonFrom(OEP_API_BASE_URL, '/health')
        const configResponse = await fetchJsonFrom<OepConfigResponse>(OEP_API_BASE_URL, '/api/config')
        if (!cancelled) {
          setOepHealthy(true)
          setOepConfig(configResponse)
        }
      } catch (error) {
        if (!cancelled) {
          setOepHealthy(false)
          if (activeMonitorModeRef.current !== 'oep') {
            setOepStatusText(`OEP service unavailable: ${(error as Error).message}`)
          }
        }
      }
    }

    async function bootstrap() {
      await Promise.allSettled([bootstrapClassic(), bootstrapOep()])
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
  }, [])

  useEffect(() => {
    return () => {
      stopLoop()
      stopCameraStream()
    }
  }, [])

  function stopLoop() {
    if (timerRef.current !== null) {
      window.clearInterval(timerRef.current)
      timerRef.current = null
    }
  }

  async function openCamera() {
    setLoadingCamera(true)
    const waitingMessage =
      screen === 'oep' ? 'Requesting webcam access for the OEP monitor...' : 'Requesting webcam access...'
    if (screen === 'oep') {
      setOepStatusText(waitingMessage)
    } else {
      setStatusText(waitingMessage)
    }

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
      setOepStatusText('Camera ready. Start monitor with new method to fill the temporal sequence.')
    } catch (error) {
      const message = `Camera access failed: ${(error as Error).message}`
      setStatusText(message)
      setOepStatusText(message)
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

  async function analyzeClassicOnce() {
    if (!classicSessionIdRef.current || classicBusyRef.current) {
      return
    }

    const frame = captureFrame()
    if (!frame) {
      setStatusText('Camera stream is live, but the current frame is not ready yet. Hold for a moment and keep the video visible.')
      return
    }

    classicBusyRef.current = true
    setBusy(true)
    setAwaitingFirstFrame(true)

    try {
      const response = await fetchJsonFrom<FrameResponse>(
        API_BASE_URL,
        `/api/session/${classicSessionIdRef.current}/frame`,
        {
          method: 'POST',
          body: JSON.stringify({ frame }),
        },
      )
      if (activeMonitorModeRef.current !== 'classic') {
        return
      }

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
      if (activeMonitorModeRef.current === 'classic') {
        setStatusText(`Frame analysis failed: ${(error as Error).message}`)
      }
    } finally {
      setAwaitingFirstFrame(false)
      classicBusyRef.current = false
      setBusy(false)
    }
  }

  async function analyzeOepOnce() {
    if (!oepSessionIdRef.current || oepBusyRef.current) {
      return
    }

    const frame = captureFrame()
    if (!frame) {
      setOepStatusText('Camera stream is live, but the temporal model is still waiting for a visible frame.')
      return
    }

    oepBusyRef.current = true
    setOepBusy(true)
    setOepAwaitingFirstFrame(true)

    try {
      const response = await fetchJsonFrom<OepFrameResponse>(
        OEP_API_BASE_URL,
        `/api/session/${oepSessionIdRef.current}/frame`,
        {
          method: 'POST',
          body: JSON.stringify({ frame }),
        },
      )
      if (activeMonitorModeRef.current !== 'oep') {
        return
      }

      setOepSession(response.session)
      setOepAnnotatedFrame(response.annotated_frame)
      setOepProbabilities(response.probabilities)
      setOepFeatures(response.features)
      setOepReady(response.ready)
      setOepStatusText(response.status_text)
    } catch (error) {
      if (activeMonitorModeRef.current === 'oep') {
        setOepStatusText(`OEP analysis failed: ${(error as Error).message}`)
      }
    } finally {
      setOepAwaitingFirstFrame(false)
      oepBusyRef.current = false
      setOepBusy(false)
    }
  }

  async function stopClassicSession(options?: { preserveStatus?: boolean }) {
    stopLoop()
    classicBusyRef.current = false
    setBusy(false)
    setAwaitingFirstFrame(false)

    if (activeMonitorModeRef.current === 'classic') {
      activeMonitorModeRef.current = null
      setActiveMonitorMode(null)
    }

    const currentSessionId = classicSessionIdRef.current
    classicSessionIdRef.current = null
    if (!currentSessionId) {
      if (!options?.preserveStatus) {
        setStatusText('Monitoring session stopped.')
      }
      return
    }

    try {
      const response = await fetchJsonFrom<SessionReadResponse>(
        API_BASE_URL,
        `/api/session/${currentSessionId}/stop`,
        {
          method: 'POST',
          body: JSON.stringify({ reason: 'manual-stop' }),
        },
      )
      setSession(response.session)
      if (!options?.preserveStatus) {
        setStatusText('Monitoring session stopped.')
      }
    } catch (error) {
      if (!options?.preserveStatus) {
        setStatusText(`Unable to stop session cleanly: ${(error as Error).message}`)
      }
    }
  }

  async function stopOepSession(options?: { preserveStatus?: boolean }) {
    stopLoop()
    oepBusyRef.current = false
    setOepBusy(false)
    setOepAwaitingFirstFrame(false)

    if (activeMonitorModeRef.current === 'oep') {
      activeMonitorModeRef.current = null
      setActiveMonitorMode(null)
    }

    const currentSessionId = oepSessionIdRef.current
    oepSessionIdRef.current = null
    if (!currentSessionId) {
      if (!options?.preserveStatus) {
        setOepStatusText('OEP monitoring session stopped.')
      }
      return
    }

    try {
      const response = await fetchJsonFrom<OepSessionReadResponse>(
        OEP_API_BASE_URL,
        `/api/session/${currentSessionId}/stop`,
        {
          method: 'POST',
          body: JSON.stringify({ reason: 'manual-stop' }),
        },
      )
      setOepSession(response.session)
      if (!options?.preserveStatus) {
        setOepStatusText('OEP monitoring session stopped.')
      }
    } catch (error) {
      if (!options?.preserveStatus) {
        setOepStatusText(`Unable to stop OEP session cleanly: ${(error as Error).message}`)
      }
    }
  }

  async function stopActiveSession() {
    if (activeMonitorModeRef.current === 'classic') {
      await stopClassicSession()
      return
    }
    if (activeMonitorModeRef.current === 'oep') {
      await stopOepSession()
    }
  }

  async function startSession() {
    if (!cameraReady) {
      setStatusText('Open the laptop camera first.')
      return
    }

    if (activeMonitorModeRef.current === 'oep') {
      await stopOepSession({ preserveStatus: true })
    } else {
      stopLoop()
    }

    try {
      const response = await fetchJsonFrom<SessionReadResponse>(API_BASE_URL, '/api/session/start', {
        method: 'POST',
        body: JSON.stringify({ operator_name: operatorName || undefined }),
      })

      classicSessionIdRef.current = response.session.session_id
      activeMonitorModeRef.current = 'classic'
      setActiveMonitorMode('classic')
      setScreen('classic')
      setSession(response.session)
      setMetrics(null)
      setEvents([])
      setAnnotatedFrame(null)
      setAwaitingFirstFrame(true)
      setStatusText('Monitoring session is live.')

      const intervalMs = config?.frame_cooldown_ms ?? 700
      stopLoop()
      timerRef.current = window.setInterval(() => {
        void analyzeClassicOnce()
      }, intervalMs)
      void analyzeClassicOnce()
    } catch (error) {
      setStatusText(`Unable to start session: ${(error as Error).message}`)
    }
  }

  async function startOepSession() {
    if (!cameraReady) {
      setOepStatusText('Open the laptop camera first.')
      return
    }

    if (activeMonitorModeRef.current === 'classic') {
      await stopClassicSession({ preserveStatus: true })
    } else {
      stopLoop()
    }

    try {
      const response = await fetchJsonFrom<OepSessionReadResponse>(OEP_API_BASE_URL, '/api/session/start', {
        method: 'POST',
        body: JSON.stringify({ operator_name: operatorName || undefined }),
      })

      oepSessionIdRef.current = response.session.session_id
      activeMonitorModeRef.current = 'oep'
      setActiveMonitorMode('oep')
      setScreen('oep')
      setOepSession(response.session)
      setOepProbabilities([])
      setOepFeatures([])
      setOepAnnotatedFrame(null)
      setOepReady(false)
      setOepAwaitingFirstFrame(true)
      setOepStatusText('OEP monitoring session is live. Collecting temporal context.')

      const intervalMs = 450
      stopLoop()
      timerRef.current = window.setInterval(() => {
        void analyzeOepOnce()
      }, intervalMs)
      void analyzeOepOnce()
    } catch (error) {
      setOepStatusText(`Unable to start OEP session: ${(error as Error).message}`)
    }
  }

  const activeSeverity = session?.current_severity ?? 'normal'
  const currentSignal = events[0]?.label ?? 'No active suspicious signal in the current session.'
  const detectorNotesText = metrics?.detector_notes.join(', ') || 'YOLO and heuristic detector are both available.'
  const overlayPlaceholderText = statusText.startsWith('Frame analysis failed:')
    ? statusText
    : awaitingFirstFrame
      ? 'Waiting for the first analyzed frame from the backend. Keep the camera visible and allow a second for capture.'
      : 'Annotated stream will appear here after the first frame is analyzed.'

  const oepOverlayPlaceholderText = oepStatusText.startsWith('OEP analysis failed:')
    ? oepStatusText
    : oepAwaitingFirstFrame
      ? 'Collecting the first frames for the temporal model. Keep your face visible for a few seconds.'
      : 'The OEP overlay will appear here after the sequence buffer starts filling.'

  const currentOepPredictionLabel = oepSession?.last_prediction ?? null
  const currentOepPredictionText = currentOepPredictionLabel
    ? `${formatOepLabel(currentOepPredictionLabel)} (${formatPercent(oepSession?.last_confidence ?? null)})`
    : 'No prediction yet'
  const currentStatusText = screen === 'oep' ? oepStatusText : statusText
  const activeFrameCount =
    activeMonitorMode === 'oep' ? (oepSession?.frame_count ?? 0) : (session?.frame_count ?? 0)
  const activeBufferText =
    activeMonitorMode === 'oep'
      ? `${oepSession?.buffer_size ?? 0}/${oepConfig?.sequence_frames ?? 16}`
      : '--'

  return (
    <div className="app-shell">
      <div className="app-frame">
        <header className="topbar">
          <div>
            <p className="eyebrow">One-Camera Anti-Cheat Demo</p>
            <h1>{screen === 'oep' ? 'OEP temporal monitor screen' : 'AutoOEP-style local monitoring dashboard'}</h1>
            <p className="intro">
              {screen === 'oep'
                ? 'Independent monitor powered by the newly trained OEP temporal model. Compare its sequence-based predictions against the classic heuristic monitor.'
                : 'Single laptop camera demo using OpenCV face heuristics, YOLO object detection, and live event scoring.'}
            </p>
          </div>

          <div className="health-stack">
            <div className={`health-badge ${backendHealthy ? 'healthy' : 'unhealthy'}`}>
              <span className="health-dot" />
              Classic backend {backendHealthy ? 'ready' : 'offline'}
            </div>
            <div className={`health-badge ${oepHealthy ? 'healthy' : 'unhealthy'}`}>
              <span className="health-dot" />
              OEP service {oepHealthy ? 'ready' : 'offline'}
            </div>
          </div>
        </header>

        <main className="workspace">
          <div className="dashboard">
            {screen === 'oep' ? (
              <>
                <section className="camera-grid camera-grid-priority">
                  <article className="panel">
                    <div className="panel-header">
                      <div>
                        <p className="panel-kicker">Raw Camera</p>
                        <h2>Candidate input stream</h2>
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
                        <p className="panel-kicker">OEP Overlay</p>
                        <h2>Trained temporal model output</h2>
                      </div>
                      <span className="mini-tag">{oepBusy ? 'Analyzing' : 'Synced'}</span>
                    </div>
                    <div className="video-frame annotated">
                      {oepAnnotatedFrame ? (
                        <img src={oepAnnotatedFrame} alt="OEP temporal model overlay" />
                      ) : (
                        <p>{oepOverlayPlaceholderText}</p>
                      )}
                    </div>
                  </article>
                </section>

                <section className="hero-grid oep-hero-grid">
                  <article className="panel intro-panel">
                    <div className="panel-header">
                      <div>
                        <p className="panel-kicker">Monitoring</p>
                        <h2>New method live state</h2>
                      </div>
                      <div className={`severity-pill ${oepReady ? 'warning' : 'normal'}`}>
                        {oepReady ? 'Predicting' : 'Buffering'}
                      </div>
                    </div>

                    <div className={`signal-banner ${oepReady ? 'warning' : 'normal'}`}>
                      <span className="signal-label">Current prediction</span>
                      <strong>{currentOepPredictionText}</strong>
                    </div>

                    <div className="overview-stats">
                      <div>
                        <span>Model</span>
                        <strong>{oepConfig?.model_name ?? 'oep_webcam_lstm_v1'}</strong>
                      </div>
                      <div>
                        <span>Frames buffered</span>
                        <strong>
                          {oepSession?.buffer_size ?? 0}/{oepConfig?.sequence_frames ?? 16}
                        </strong>
                      </div>
                      <div>
                        <span>Prediction state</span>
                        <strong>{oepReady ? 'Ready' : 'Collecting sequence'}</strong>
                      </div>
                    </div>

                    <div className="overview-metrics overview-metrics-four">
                      <article className="metric-card metric-card-inline">
                        <p className="metric-label">OEP backend</p>
                        <strong>{oepHealthy ? 'Online' : 'Offline'}</strong>
                        <span>Health and config are refreshed every 5 seconds.</span>
                      </article>
                      <article className="metric-card metric-card-inline">
                        <p className="metric-label">Last label</p>
                        <strong>{formatOepLabel(oepSession?.last_prediction ?? null)}</strong>
                        <span>Most recent temporal class predicted by the trained model.</span>
                      </article>
                      <article className="metric-card metric-card-inline">
                        <p className="metric-label">Confidence</p>
                        <strong>{formatPercent(oepSession?.last_confidence ?? null)}</strong>
                        <span>Softmax confidence from the current temporal sequence.</span>
                      </article>
                      <article className="metric-card metric-card-inline">
                        <p className="metric-label">Required frames</p>
                        <strong>{oepConfig?.required_frames ?? 8}</strong>
                        <span>Minimum number of frames needed before inference starts.</span>
                      </article>
                    </div>
                  </article>

                  <article className="panel">
                    <div className="panel-header">
                      <div>
                        <p className="panel-kicker">Top Scores</p>
                        <h2>Class probability ranking</h2>
                      </div>
                      <button className="ghost compact-button" onClick={() => setScreen('classic')}>
                        Back to dashboard
                      </button>
                    </div>

                    <div className="probability-list">
                      {oepProbabilities.length === 0 ? (
                        <div className="empty-state">No prediction yet. Keep the camera visible until the temporal buffer reaches the required length.</div>
                      ) : (
                        oepProbabilities.map((item) => (
                          <div key={item.label} className="probability-item">
                            <div>
                              <p>{formatOepLabel(item.label)}</p>
                              <span>{formatPercent(item.confidence)}</span>
                            </div>
                            <div className="probability-bar">
                              <span style={{ width: `${Math.max(item.confidence * 100, 4)}%` }} />
                            </div>
                          </div>
                        ))
                      )}
                    </div>
                  </article>
                </section>

                <section className="bottom-grid oep-bottom-grid">
                  <article className="panel summary-panel">
                    <div className="panel-header">
                      <div>
                        <p className="panel-kicker">Temporal Summary</p>
                        <h2>Session telemetry</h2>
                      </div>
                    </div>
                    <dl className="summary-grid">
                      <div>
                        <dt>Session ID</dt>
                        <dd>{oepSession?.session_id ?? '--'}</dd>
                      </div>
                      <div>
                        <dt>Frames analyzed</dt>
                        <dd>{oepSession?.frame_count ?? 0}</dd>
                      </div>
                      <div>
                        <dt>Buffer fill</dt>
                        <dd>
                          {oepSession?.buffer_size ?? 0}/{oepConfig?.sequence_frames ?? 16}
                        </dd>
                      </div>
                      <div>
                        <dt>Last confidence</dt>
                        <dd>{formatPercent(oepSession?.last_confidence ?? null)}</dd>
                      </div>
                      <div>
                        <dt>Status</dt>
                        <dd>{oepSession?.status ?? 'idle'}</dd>
                      </div>
                      <div>
                        <dt>Labels</dt>
                        <dd>{oepConfig?.labels.map((label) => formatOepLabel(label)).join(', ') ?? '--'}</dd>
                      </div>
                    </dl>
                  </article>

                  <article className="panel wide-panel">
                    <div className="panel-header">
                      <div>
                        <p className="panel-kicker">Frame Features</p>
                        <h2>Live feature vector sent to the temporal model</h2>
                      </div>
                    </div>
                    <div className="feature-grid">
                      {OEP_FEATURE_LABELS.map((label, index) => (
                        <article key={label} className="metric-card metric-card-inline">
                          <p className="metric-label">{label}</p>
                          <strong>{formatMetric(oepFeatures[index] ?? null, 3)}</strong>
                          <span>Feature {index + 1} of the current frame sequence.</span>
                        </article>
                      ))}
                    </div>
                  </article>
                </section>
              </>
            ) : (
              <>
                <section className="camera-grid camera-grid-priority">
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
                        <p>{overlayPlaceholderText}</p>
                      )}
                    </div>
                  </article>
                </section>

                <section className="hero-grid hero-grid-single">
                  <article className="panel intro-panel">
                    <div className="panel-header">
                      <div>
                        <p className="panel-kicker">Monitoring</p>
                        <h2>Live monitoring state</h2>
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
                        <strong>{activeMonitorMode === 'classic' ? 'Monitoring live' : 'Idle'}</strong>
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

                    <div className="overview-metrics">
                      <article className="metric-card metric-card-inline">
                        <p className="metric-label">Backend</p>
                        <strong>{backendHealthy ? 'Online' : 'Offline'}</strong>
                        <span>Frontend checks health and config every 5 seconds.</span>
                      </article>
                      <article className="metric-card metric-card-inline">
                        <p className="metric-label">Risk score</p>
                        <strong>{session?.risk_score.toFixed(1) ?? '0.0'}</strong>
                        <span>Continuous risk score accumulated from suspicious events.</span>
                      </article>
                      <article className="metric-card metric-card-inline">
                        <p className="metric-label">Faces</p>
                        <strong>{metrics?.faces_detected ?? 0}</strong>
                        <span>{metrics?.phone_detected ? 'Phone visible in frame' : 'No phone visible right now'}</span>
                      </article>
                      <article className="metric-card metric-card-inline">
                        <p className="metric-label">Yaw ratio</p>
                        <strong>{formatMetric(metrics?.yaw_ratio ?? null)}</strong>
                        <span>Higher absolute values suggest turning away from the screen.</span>
                      </article>
                      <article className="metric-card metric-card-inline">
                        <p className="metric-label">Pitch ratio</p>
                        <strong>{formatMetric(metrics?.pitch_ratio ?? null)}</strong>
                        <span>Higher values indicate sustained looking down.</span>
                      </article>
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
              </>
            )}
          </div>

          <aside className="control-dock panel">
            <div className="control-dock-header">
              <div>
                <p className="panel-kicker">Quick Control</p>
                <h2>Session dock</h2>
              </div>
              <div className={`severity-pill ${activeMonitorMode === 'oep' ? 'warning' : activeSeverity}`}>
                {activeMonitorMode === 'oep' ? 'OEP live' : severityLabel[activeSeverity]}
              </div>
            </div>

            <label className="field">
              <span>Operator label</span>
              <input value={operatorName} onChange={(event) => setOperatorName(event.target.value)} placeholder="Proctor station 01" />
            </label>

            <div className="dock-button-stack">
              <button onClick={() => void openCamera()} disabled={loadingCamera || cameraReady}>
                {loadingCamera ? 'Opening camera...' : cameraReady ? 'Camera ready' : 'Open camera'}
              </button>
              <button className="secondary" onClick={() => void startSession()} disabled={!cameraReady || activeMonitorMode === 'classic'}>
                Start monitoring
              </button>
              <button className="tertiary" onClick={() => void startOepSession()} disabled={!cameraReady || activeMonitorMode === 'oep'}>
                Start monitor with new method
              </button>
              <button className="ghost" onClick={() => void stopActiveSession()} disabled={!activeMonitorMode}>
                Stop session
              </button>
            </div>

            <div className="dock-mini-stats dock-mini-stats-wide">
              <div>
                <span>Status</span>
                <strong>{activeMonitorMode ? 'Live' : 'Idle'}</strong>
              </div>
              <div>
                <span>Screen</span>
                <strong>{screen === 'oep' ? 'New method' : 'Classic'}</strong>
              </div>
              <div>
                <span>Frames</span>
                <strong>{activeFrameCount}</strong>
              </div>
              <div>
                <span>Buffer</span>
                <strong>{activeBufferText}</strong>
              </div>
            </div>

            <div className="dock-service-list">
              <div>
                <span>Classic backend</span>
                <strong>{backendHealthy ? 'Online' : 'Offline'}</strong>
              </div>
              <div>
                <span>OEP service</span>
                <strong>{oepHealthy ? 'Online' : 'Offline'}</strong>
              </div>
            </div>

            <div className="status-box dock-status-box">
              <p className="status-title">Live status</p>
              <p>{currentStatusText}</p>
            </div>

            {screen === 'classic' ? (
              <button className="ghost compact-button dock-screen-button" onClick={() => setScreen('oep')}>
                Open new method screen
              </button>
            ) : (
              <button className="ghost compact-button dock-screen-button" onClick={() => setScreen('classic')}>
                Back to classic dashboard
              </button>
            )}
          </aside>
        </main>

        <canvas ref={canvasRef} className="hidden-canvas" />
      </div>
    </div>
  )
}

export default App
