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
  const [processedImage, setProcessedImage] = useState(null)
  const [isProcessing, setIsProcessing] = useState(false)
  
  const videoRef = useRef(null)
  const streamRef = useRef(null)

  const startCamera = async () => {
    try {
      console.log("Requesting user media...")
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: 640, height: 480 }
      })
      
      console.log("User media stream obtained. Binding to video element.")
      if (videoRef.current) {
        videoRef.current.srcObject = stream
        // Explicitly trigger play to ensure readyState updates
        try {
          await videoRef.current.play()
          console.log("Video playing successfully.")
        } catch (playErr) {
          console.error("Error calling play() on video element:", playErr)
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
    console.log("Stopping camera stream.")
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(track => track.stop())
      streamRef.current = null
    }
    if (videoRef.current) {
      videoRef.current.srcObject = null
    }
    setIsCameraOn(false)
    setProcessedImage(null)
    setData({
      total_vehicles: 0,
      green_time: 10,
      breakdown: { Car: 0, Motorcycle: 0, Bus: 0, Truck: 0 }
    })
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
        
        const canvas = document.createElement('canvas')
        canvas.width = video.videoWidth
        canvas.height = video.videoHeight
        const ctx = canvas.getContext('2d')
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height)
        
        canvas.toBlob(async (blob) => {
          if (!blob) {
            setIsProcessing(false)
            if (active) setTimeout(captureFrame, 800)
            return
          }
          
          const formData = new FormData()
          formData.append('image', blob, 'frame.jpg')
          
          try {
            console.log("Uploading frame to backend...")
            const response = await fetch(`${API_BASE_URL}/process_frame`, {
              method: 'POST',
              body: formData
            })
            const result = await response.json()
            
            if (active && result.annotated_image) {
              console.log("Frame processed successfully. Vehicles detected:", result.total_vehicles)
              setProcessedImage(result.annotated_image)
              setData({
                total_vehicles: result.total_vehicles,
                green_time: result.green_time,
                breakdown: result.breakdown
              })
            }
          } catch (error) {
            console.error("Error processing webcam frame on backend:", error)
          } finally {
            setIsProcessing(false)
            if (active) setTimeout(captureFrame, 800)
          }
        }, 'image/jpeg', 0.7)
      } else {
        // If video is paused, force play it
        if (video && video.paused) {
          console.log("Video is paused in loop, attempting to play...")
          video.play().catch(e => console.error("Play retry error:", e))
        }
        console.log("Waiting for video data... Current readyState:", video ? video.readyState : 'No Video')
        if (active) setTimeout(captureFrame, 300)
      }
    }

    const timeoutId = setTimeout(captureFrame, 800)

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
          
          {isCameraOn ? (
            <div className="video-wrapper">
              <video
                ref={videoRef}
                autoPlay
                playsInline
                muted
                style={{ display: processedImage ? 'none' : 'block', width: '100%', objectFit: 'cover' }}
              />
              {processedImage && (
                <img src={processedImage} alt="Annotated Live Feed" />
              )}
            </div>
          ) : (
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
