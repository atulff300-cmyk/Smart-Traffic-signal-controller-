import { useState, useEffect } from 'react'
import './App.css'

function App() {
  const [data, setData] = useState({
    total_vehicles: 0,
    green_time: 10,
    breakdown: { Car: 0, Motorcycle: 0, Bus: 0, Truck: 0 }
  })

  useEffect(() => {
    // Fetch data from Flask API every 1 second
    const interval = setInterval(() => {
      fetch('http://127.0.0.1:5000/data')
        .then(res => res.json())
        .then(resData => setData(resData))
        .catch(err => console.error("API Fetch Error:", err))
    }, 1000)
    
    return () => clearInterval(interval)
  }, [])

  return (
    <div className="dashboard-container">
      <header>
        <div className="logo">
          <span className="dot live"></span>
          <h1>Smart Traffic AI Controller</h1>
        </div>
        <div className="status">System Active</div>
      </header>

      <main>
        <section className="video-section">
          <h2>Live Camera Feed</h2>
          <div className="video-wrapper">
            <img src="http://127.0.0.1:5000/video_feed" alt="Live Feed" />
          </div>
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
