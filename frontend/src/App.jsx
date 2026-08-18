import { useState, useEffect, useRef } from 'react'
import './App.css'

// Dynamic API URL for backend deployment (Render), fallback to localhost for development
const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://127.0.0.1:5000'

function App() {
  const [data, setData] = useState({
    total_vehicles: 0,
    green_time: 10,
    breakdown: { Car: 0, Motorcycle: 0, Bus: 0, Truck: 0 }
  })
  const [isCameraOn, setIsCameraOn] = useState(false)
  const [isProcessing, setIsProcessing] = useState(false)
  
  const videoRef = useRef(null)
  const streamRef = useRef(null)
  const overlayCanvasRef = useRef(null)

  const startCamera = async () => {
    try {
      console.log("Requesting user media...")
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: 640, height: 480 }
      })
      
      if (videoRef.current) {
        videoRef.current.srcObject = stream
        try {
          await videoRef.current.play()
        } catch (playErr) {
          console.error("Error calling play():", playErr)
        }
      }
      
      streamRef.current = stream
      setIsCameraOn(true)
    } catch (err) {
      console.error("Error accessing webcam:", err)
      alert("Could not access camera. Please make sure camera permission is granted.")
    }
  }

  const stopCamera = () => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(track => track.stop())
      streamRef.current = null
    }
    if (videoRef.current) {
      videoRef.current.srcObject = null
    }
    setIsCameraOn(false)
    setData({
      total_vehicles: 0,
      green_time: 10,
      breakdown: { Car: 0, Motorcycle: 0, Bus: 0, Truck: 0 }
    })
    const canvas = overlayCanvasRef.current
    if (canvas) {
      const ctx = canvas.getContext('2d')
      ctx.clearRect(0, 0, canvas.width, canvas.height)
    }
  }

  useEffect(() => {
    startCamera()
    return () => {
      if (streamRef.current) {
        streamRef.current.getTracks().forEach(track => track.stop())
      }
    }
  }, [])

  // Process frames periodically when camera is active
  useEffect(() => {
    if (!isCameraOn) return

    let active = true
    const captureFrame = async () => {
      if (!active || !isCameraOn) return
      
      const video = videoRef.current
      if (video && video.readyState === video.HAVE_ENOUGH_DATA) {
        setIsProcessing(true)
        
        const targetWidth = 480
        const scale = video.videoWidth ? targetWidth / video.videoWidth : 1
        const targetHeight = Math.round((video.videoHeight || 360) * scale)
        
        const canvas = document.createElement('canvas')
        canvas.width = targetWidth
        canvas.height = targetHeight
        const ctx = canvas.getContext('2d')
        ctx.drawImage(video, 0, 0, targetWidth, targetHeight)
        
        canvas.toBlob(async (blob) => {
          if (!blob) {
            setIsProcessing(false)
            if (active) setTimeout(captureFrame, 300)
            return
          }
          
          const formData = new FormData()
          formData.append('image', blob, 'frame.jpg')
          
          try {
            const response = await fetch(`${API_BASE_URL}/process_frame`, {
              method: 'POST',
              body: formData
            })
            const result = await response.json()
            
            if (active && result) {
              setData({
                total_vehicles: result.total_vehicles,
                green_time: result.green_time,
                breakdown: result.breakdown
              })
              
              // Draw bounding boxes on transparent canvas overlay over smooth 60 FPS video
              const overlay = overlayCanvasRef.current
              if (overlay && videoRef.current) {
                overlay.width = videoRef.current.clientWidth || 640
                overlay.height = videoRef.current.clientHeight || 480
                const oCtx = overlay.getContext('2d')
                oCtx.clearRect(0, 0, overlay.width, overlay.height)
                
                if (result.boxes && result.img_width && result.img_height) {
                  const scaleX = overlay.width / result.img_width
                  const scaleY = overlay.height / result.img_height
                  
                  result.boxes.forEach(box => {
                    const x = box.x1 * scaleX
                    const y = box.y1 * scaleY
                    const w = (box.x2 - box.x1) * scaleX
                    const h = (box.y2 - box.y1) * scaleY
                    
                    const color = box.class === 'Car' ? '#10b981' : 
                                 (box.class === 'Motorcycle' ? '#ef4444' : 
                                 (box.class === 'Bus' ? '#3b82f6' : '#f59e0b'))
                    
                    oCtx.strokeStyle = color
                    oCtx.lineWidth = 3
                    oCtx.strokeRect(x, y, w, h)
                    
                    oCtx.fillStyle = color
                    const text = `${box.class} ${Math.round(box.confidence * 100)}%`
                    oCtx.font = 'bold 13px Inter, sans-serif'
                    const textWidth = oCtx.measureText(text).width
                    oCtx.fillRect(x, Math.max(0, y - 22), textWidth + 8, 20)
                    
                    oCtx.fillStyle = '#ffffff'
                    oCtx.fillText(text, x + 4, Math.max(14, y - 7))
                  })
                }
              }
            }
          } catch (error) {
            console.error("Error processing webcam frame on backend:", error)
          } finally {
            setIsProcessing(false)
            if (active) setTimeout(captureFrame, 300)
          }
        }, 'image/jpeg', 0.5)
      } else {
        if (video && video.paused && video.srcObject) {
          video.play().catch(e => console.error("Play retry error:", e))
        }
        if (active) setTimeout(captureFrame, 300)
      }
    }

    const timeoutId = setTimeout(captureFrame, 300)

    return () => {
      active = false
      clearTimeout(timeoutId)
    }
  }, [isCameraOn])

  return (
    <div className="dashboard-container">
      <header>
        <div className="logo">
          <span className={`dot ${isCameraOn ? 'live' : ''}`}></span>
          <h1>Smart Traffic AI Controller</h1>
        </div>
        <div className="status">{isCameraOn ? 'System Active' : 'System Paused'}</div>
      </header>

      <main>
        <section className="video-section">
          <div className="camera-controls">
            <h2>Live Camera Feed</h2>
            <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
              {isProcessing && (
                <div className="loading-indicator">
                  <div className="spinner"></div>
                  <span>Analyzing...</span>
                </div>
              )}
              {isCameraOn ? (
                <button className="btn-control stop" onClick={stopCamera}>
                  🛑 Stop Camera
                </button>
              ) : (
                <button className="btn-control" onClick={startCamera}>
                  📷 Start Camera
                </button>
              )}
            </div>
          </div>
          
          <div className="video-wrapper" style={{ position: 'relative', display: isCameraOn ? 'block' : 'none' }}>
            <video
              ref={videoRef}
              autoPlay
              playsInline
              muted
              style={{ width: '100%', display: 'block', borderRadius: '12px' }}
            />
            <canvas
              ref={overlayCanvasRef}
              style={{
                position: 'absolute',
                top: 0,
                left: 0,
                width: '100%',
                height: '100%',
                pointerEvents: 'none',
                borderRadius: '12px'
              }}
            />
          </div>
          
          {!isCameraOn && (
            <div className="camera-placeholder">
              <h3>Camera Offline</h3>
              <p>Click "Start Camera" to grant webcam access. The application will capture your camera stream in real-time, detect vehicles using YOLOv8, and calculate optimal traffic light timings.</p>
              <button className="btn-control" onClick={startCamera}>
                📷 Start Camera
              </button>
            </div>
          )}
        </section>

        <section className="data-section">
          <div className="kpi-grid">
            <div className="card kpi-card blue-card">
              <h3>Total Vehicles</h3>
              <div className="kpi-value">{data.total_vehicles}</div>
              <p>Detected in current frame</p>
            </div>
            
            <div className="card kpi-card green-card">
              <h3>Allocated Green Time</h3>
              <div className="kpi-value time-value">{data.green_time}s</div>
              <p>Based on traffic density</p>
            </div>
          </div>

          <div className="card details-card">
            <h3>Vehicle Density Breakdown</h3>
            <ul className="breakdown-list">
              <li><span className="vehicle-name">🚗 Cars</span> <span className="vehicle-count">{data.breakdown.Car || 0}</span></li>
              <li><span className="vehicle-name">🏍️ Motorcycles</span> <span className="vehicle-count">{data.breakdown.Motorcycle || 0}</span></li>
              <li><span className="vehicle-name">🚌 Buses</span> <span className="vehicle-count">{data.breakdown.Bus || 0}</span></li>
              <li><span className="vehicle-name">🚚 Trucks</span> <span className="vehicle-count">{data.breakdown.Truck || 0}</span></li>
            </ul>
          </div>
        </section>
      </main>
    </div>
  )
}

export default App
