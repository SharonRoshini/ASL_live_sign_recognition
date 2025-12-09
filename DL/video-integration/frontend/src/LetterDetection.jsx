import React, { useState, useRef, useEffect } from 'react'
import axios from 'axios'
import './App.css'
import { API_BASE } from './config'

function LetterDetection() {
  // Session management
  const sessionIdRef = useRef(null)
  
  // Video capture state
  const [isCapturing, setIsCapturing] = useState(false)
  const videoRef = useRef(null)
  const canvasRef = useRef(null)
  const streamRef = useRef(null)
  const captureIntervalRef = useRef(null)
  const isCapturingRef = useRef(false)
  
  // Letter detection state
  const [currentLetter, setCurrentLetter] = useState('') // Current prediction
  const [stableLetter, setStableLetter] = useState('') // Committed letter
  const [hasHand, setHasHand] = useState(false)
  const [isProcessing, setIsProcessing] = useState(false)
  const [handLandmarks, setHandLandmarks] = useState(null) // For drawing hand skeleton
  
  // Error state
  const [error, setError] = useState(null)
  
  // Cleanup on unmount
  useEffect(() => {
    return () => {
      stopCapture()
    }
  }, [])
  
  // ==================== DRAW HAND SKELETON ON CANVAS (REAL-TIME) ====================
  // MediaPipe hand connections (21 landmarks)
  const HAND_CONNECTIONS = [
    [0, 1], [1, 2], [2, 3], [3, 4],        // Thumb
    [0, 5], [5, 6], [6, 7], [7, 8],        // Index finger
    [0, 9], [9, 10], [10, 11], [11, 12],   // Middle finger
    [0, 13], [13, 14], [14, 15], [15, 16], // Ring finger
    [0, 17], [17, 18], [18, 19], [19, 20], // Pinky
    [5, 9], [9, 13], [13, 17]              // Palm connections
  ]
  
  useEffect(() => {
    if (!canvasRef.current || !videoRef.current || !isCapturing) {
      return
    }
    
    const canvas = canvasRef.current
    const video = videoRef.current
    const ctx = canvas.getContext('2d')
    
    if (!ctx) {
      console.error('Failed to get canvas context')
      return
    }
    
    // Set canvas size to match video
    const updateCanvasSize = () => {
      if (video.videoWidth && video.videoHeight) {
        canvas.width = video.videoWidth
        canvas.height = video.videoHeight
        console.log('Canvas size updated:', canvas.width, 'x', canvas.height)
      }
    }
    
    // Initial size update
    if (video.videoWidth && video.videoHeight) {
      updateCanvasSize()
    } else {
      // Wait for video to load
      video.addEventListener('loadedmetadata', updateCanvasSize, { once: true })
    }
    
    let animationFrameId = null
    
    const drawFrame = () => {
      if (!isCapturingRef.current) {
        return
      }
      
      // Make sure canvas has valid dimensions
      if (!canvas.width || !canvas.height) {
        if (video.videoWidth && video.videoHeight) {
          canvas.width = video.videoWidth
          canvas.height = video.videoHeight
        } else {
          animationFrameId = requestAnimationFrame(drawFrame)
          return
        }
      }
      
      // Update canvas size if video size changed
      if (video.videoWidth && video.videoHeight && 
          (canvas.width !== video.videoWidth || canvas.height !== video.videoHeight)) {
        canvas.width = video.videoWidth
        canvas.height = video.videoHeight
      }
      
      // Clear canvas
      ctx.clearRect(0, 0, canvas.width, canvas.height)
      
      // Draw hand landmarks (Mediapipe-style green skeleton)
      if (handLandmarks && Array.isArray(handLandmarks) && handLandmarks.length === 21) {
        
        // Draw dark-green connections (0,150,0)
        ctx.strokeStyle = 'rgb(0,150,0)'
        ctx.lineWidth = 2
        ctx.lineCap = 'round'
        ctx.lineJoin = 'round'
        
        HAND_CONNECTIONS.forEach(([start, end]) => {
          const s = handLandmarks[start]
          const e = handLandmarks[end]
          if (!s || !e) return
          
          const x1 = s.x * canvas.width
          const y1 = s.y * canvas.height
          const x2 = e.x * canvas.width
          const y2 = e.y * canvas.height
          
          ctx.beginPath()
          ctx.moveTo(x1, y1)
          ctx.lineTo(x2, y2)
          ctx.stroke()
        })
        
        // Draw bright-green dots (0,255,0)
        ctx.fillStyle = 'rgb(0,255,0)'
        handLandmarks.forEach(pt => {
          if (!pt) return
          const x = pt.x * canvas.width
          const y = pt.y * canvas.height
          
          ctx.beginPath()
          ctx.arc(x, y, 4, 0, Math.PI * 2)
          ctx.fill()
        })
      }
      
      
      // Continue drawing loop
      if (isCapturingRef.current) {
        animationFrameId = requestAnimationFrame(drawFrame)
      }
    }
    
    // Start drawing loop
    drawFrame()
    
    // Cleanup
    return () => {
      if (animationFrameId) {
        cancelAnimationFrame(animationFrameId)
      }
      video.removeEventListener('loadedmetadata', updateCanvasSize)
    }
  }, [handLandmarks, isCapturing])
  
  // ==================== VIDEO CAPTURE ====================
  const startCapture = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: 640, height: 480 },
        audio: false,
      })
      
      streamRef.current = stream
      
      if (!videoRef.current) {
        setError('Video element not found')
        return
      }
      
      const video = videoRef.current
      video.srcObject = stream
      
      await new Promise((resolve, reject) => {
        const onLoadedMetadata = () => {
          video.removeEventListener('loadedmetadata', onLoadedMetadata)
          resolve()
        }
        const onError = (err) => {
          video.removeEventListener('error', onError)
          reject(err)
        }
        if (video.readyState >= 1) {
          onLoadedMetadata()
          return
        }
        video.addEventListener('loadedmetadata', onLoadedMetadata)
        video.addEventListener('error', onError)
        video.play().catch(reject)
      })
      
      await new Promise(resolve => setTimeout(resolve, 500))
      
      isCapturingRef.current = true
      setIsCapturing(true)
      setError(null)
      
      // Capture and process frames at ~15 FPS (every 66ms) to reduce load
      // This gives backend more time to process each frame
      captureIntervalRef.current = setInterval(() => {
        captureFrame()
      }, 66)
      
      // Also capture immediately
      captureFrame()
      
    } catch (err) {
      console.error('Camera access error:', err)
      setError('Failed to access camera. Please check permissions.')
      setIsCapturing(false)
    }
  }
  
  const stopCapture = () => {
    if (captureIntervalRef.current) {
      clearInterval(captureIntervalRef.current)
      captureIntervalRef.current = null
    }
    
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(track => track.stop())
      streamRef.current = null
    }
    
    if (videoRef.current) {
      videoRef.current.srcObject = null
    }
    
    isCapturingRef.current = false
    setIsCapturing(false)
  }
  
  const captureFrame = () => {
    const video = videoRef.current
    
    if (!video || !isCapturingRef.current) {
      return
    }
    
    if (video.readyState < 2 || !video.videoWidth || !video.videoHeight) {
      return
    }
    
    try {
      const canvas = document.createElement('canvas')
      canvas.width = video.videoWidth
      canvas.height = video.videoHeight
      const ctx = canvas.getContext('2d')
      
      if (!ctx) {
        return
      }
      
      ctx.drawImage(video, 0, 0, canvas.width, canvas.height)
      // Use lower quality for faster encoding/decoding (0.7 instead of 0.8)
      const frameData = canvas.toDataURL('image/jpeg', 0.7)
      
      if (!frameData || frameData.length < 100) {
        return
      }
      
      // Process frame asynchronously
      processFrame(frameData).catch(err => {
        console.error('Frame processing error:', err)
      })
      
    } catch (err) {
      console.error('Frame capture error:', err)
    }
  }
  
  // Use a queue system to process frames without blocking
  const frameQueueRef = useRef([])
  const isProcessingRef = useRef(false)
  
  const processFrame = async (frameData) => {
    // Add frame to queue
    frameQueueRef.current.push(frameData)
    
    // Process queue if not already processing
    if (isProcessingRef.current) {
      return
    }
    
    processFrameQueue()
  }
  
  const processFrameQueue = async () => {
    if (frameQueueRef.current.length === 0) {
      isProcessingRef.current = false
      return
    }
    
    isProcessingRef.current = true
    
    // Get the latest frame (skip older ones for real-time feel)
    // Clear the queue and only process the most recent frame
    const frames = frameQueueRef.current
    frameQueueRef.current = []
    const latestFrame = frames[frames.length - 1]
    
    // If there were many queued frames, log it for debugging
    if (frames.length > 3) {
      console.log(`[LETTER] Processing latest frame (skipped ${frames.length - 1} older frames)`)
    }
    
    try {
      const requestData = {
        frame: latestFrame
      }
      if (sessionIdRef.current) {
        requestData.session_id = sessionIdRef.current
      }
      
      const response = await axios.post(`${API_BASE}/api/letter-detection/process-frame`, requestData, {
        headers: {
          'Content-Type': 'application/json',
        },
        timeout: 5000, // Increased timeout for reliable frame processing
      })
      
      if (response.data) {
        if (response.data.session_id && !sessionIdRef.current) {
          sessionIdRef.current = response.data.session_id
        }
        
        // Debug: Log hand landmarks
        if (response.data.hand_landmarks) {
          console.log('Received hand landmarks:', response.data.hand_landmarks.length, 'landmarks')
        }
        
        // Update current prediction
        if (response.data.current_letter !== undefined) {
          setCurrentLetter(response.data.current_letter || '')
        }
        
        // Check if a new letter was committed
        if (response.data.letter_committed) {
          const newLetter = response.data.current_letter
          if (newLetter && newLetter !== ' ') {
            setStableLetter(newLetter)
            // Clear after a moment
            setTimeout(() => setStableLetter(''), 1000)
          }
        }
        if (response.data.has_hand !== undefined) {
          setHasHand(response.data.has_hand)
        }
        
        // Update hand landmarks for real-time canvas drawing
        if (response.data.hand_landmarks && Array.isArray(response.data.hand_landmarks) && response.data.hand_landmarks.length === 21) {
          setHandLandmarks(response.data.hand_landmarks)
          console.log('✅ Hand landmarks received:', response.data.hand_landmarks.length, 'landmarks')
          console.log('Sample landmark:', response.data.hand_landmarks[0])
        } else {
          if (response.data.has_hand) {
            console.log('⚠️ Hand detected but no landmarks:', response.data)
          }
          setHandLandmarks(null)
        }
        
      }
    } catch (err) {
      // Only log timeout errors if they're frequent (to avoid spam)
      if (err.code === 'ECONNABORTED' || err.message?.includes('timeout')) {
        // Silently handle timeouts - they're expected if backend is slow
        // Only log if it's a persistent issue
        if (Math.random() < 0.1) { // Log ~10% of timeouts to avoid spam
          console.warn('[LETTER] Frame processing timeout (backend may be slow)')
        }
      } else {
        console.error('Frame processing error:', err)
      }
    } finally {
      // Continue processing queue with a small delay
      setTimeout(() => {
        isProcessingRef.current = false
        processFrameQueue()
      }, 10) // Small delay to prevent overwhelming
    }
  }
  
  return (
    <div className="letter-detection-grid">
      {/* Left Column - Camera */}
      <div className="letter-detection-camera-column">
        {/* Video Container with Canvas Overlay */}
        <div className="live-section">
          <div className="video-container" style={{ position: 'relative', width: '100%', maxWidth: '100%', margin: '0 auto', background: '#000', borderRadius: '10px', overflow: 'hidden', aspectRatio: '4/3' }}>
            <video
              ref={videoRef}
              autoPlay
              playsInline
              muted
              className="video-preview"
              style={{ width: '100%', height: '100%', objectFit: 'cover', display: 'block' }}
            />
            <canvas
              ref={canvasRef}
              style={{
                position: 'absolute',
                top: 0,
                left: 0,
                width: '100%',
                height: '100%',
                pointerEvents: 'none',
                zIndex: 10,
                display: isCapturing ? 'block' : 'none'
              }}
            />
            {!streamRef.current && (
              <div className="video-placeholder">
                <p>Camera preview will appear here</p>
                <p style={{ fontSize: '0.9em', color: '#999', marginTop: '10px' }}>
                  Click "Start Live Detection" to begin
                </p>
              </div>
            )}
          </div>
          
          <div className="capture-controls">
            {!isCapturing ? (
              <button
                onClick={startCapture}
                className="capture-button start"
              >
                Start Live Detection
              </button>
            ) : (
              <>
                <button
                  onClick={stopCapture}
                  className="capture-button stop"
                >
                  Stop Detection
                </button>
                <div style={{ 
                  marginTop: '10px',
                  padding: '8px 16px',
                  background: '#d1ecf1',
                  borderRadius: '8px',
                  color: '#0c5460',
                  fontSize: '0.9em',
                  fontWeight: '600'
                }}>
                  <span style={{ 
                    display: 'inline-block',
                    width: '10px',
                    height: '10px',
                    borderRadius: '50%',
                    background: '#28a745',
                    marginRight: '8px'
                  }}></span>
                  Detecting letters in real-time...
                </div>
              </>
            )}
          </div>
        </div>
      </div>

      {/* Right Column - Detection Results */}
      <div className="letter-detection-results-column">
        <div className="result-section">
          <h2>Detection Results</h2>
          
          <div className="result-card">
            {/* Current Prediction vs Stable Letter */}
            <div style={{ 
              display: 'grid',
              gridTemplateColumns: '1fr 1fr',
              gap: '15px',
              marginBottom: '20px'
            }}>
              <div style={{ 
                background: '#fff3cd',
                padding: '20px',
                borderRadius: '10px',
                border: '2px solid #ffc107',
                textAlign: 'center'
              }}>
                <div style={{ 
                  fontSize: '0.9em', 
                  color: '#856404',
                  marginBottom: '10px',
                  fontWeight: '600'
                }}>
                  Current Prediction
                </div>
                <div style={{ 
                  fontSize: '3em', 
                  fontWeight: 'bold',
                  color: '#856404',
                  fontFamily: 'monospace',
                  minHeight: '60px',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center'
                }}>
                  {currentLetter || (hasHand ? '?' : '-')}
                </div>
                <div style={{ fontSize: '0.8em', color: '#856404', marginTop: '5px' }}>
                  (Updates in real-time)
                </div>
              </div>
              
              <div style={{ 
                background: stableLetter ? '#d4edda' : '#f8f9fa',
                padding: '20px',
                borderRadius: '10px',
                border: stableLetter ? '2px solid #28a745' : '2px solid #dee2e6',
                textAlign: 'center',
                transition: 'all 0.3s ease'
              }}>
                <div style={{ 
                  fontSize: '0.9em', 
                  color: stableLetter ? '#155724' : '#666',
                  marginBottom: '10px',
                  fontWeight: '600'
                }}>
                  Stable Letter
                </div>
                <div style={{ 
                  fontSize: '3em', 
                  fontWeight: 'bold',
                  color: stableLetter ? '#28a745' : '#999',
                  fontFamily: 'monospace',
                  minHeight: '60px',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center'
                }}>
                  {stableLetter || '-'}
                </div>
                <div style={{ fontSize: '0.8em', color: stableLetter ? '#155724' : '#666', marginTop: '5px' }}>
                  (Committed after 12 frames)
                </div>
              </div>
            </div>
            
            <div className="result-item">
              <span className="label">Hand Detected:</span>
              <span className="value" style={{ 
                color: hasHand ? '#28a745' : '#dc3545',
                fontWeight: 'bold'
              }}>
                {hasHand ? '✓ Yes - Show your hand to the camera' : '✗ No - Position your hand in view'}
              </span>
            </div>
          </div>
        </div>
        
        {error && (
          <div className="error-message">
            <strong>Error:</strong> {error}
          </div>
        )}
      </div>
    </div>
  )
}

export default LetterDetection
