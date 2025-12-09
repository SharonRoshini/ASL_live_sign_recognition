import React, { useState, useRef, useEffect } from 'react'
import axios from 'axios'
import { Hands } from '@mediapipe/hands'
import { ThemeToggle } from './components/theme-toggle'
import './App.css'
import LetterDetection from './LetterDetection'
import { API_BASE } from './config'

// Audio Play Button Component
const AudioPlayButton = ({ audioUrl, label, size = 'medium' }) => {
  const [isPlaying, setIsPlaying] = useState(false);
  const audioRef = useRef(null);

  const handlePlay = () => {
    if (audioRef.current && audioUrl) {
      if (isPlaying) {
        audioRef.current.pause();
        audioRef.current.currentTime = 0;
        setIsPlaying(false);
      } else {
        audioRef.current.play();
        setIsPlaying(true);
      }
    }
  };

  const handleEnded = () => {
    setIsPlaying(false);
  };

  // Show button even if audioUrl is not available (disabled state)
  if (!audioUrl) {
    return (
      <button
        disabled
        style={{
          width: size === 'small' ? '28px' : size === 'large' ? '40px' : '32px',
          height: size === 'small' ? '28px' : size === 'large' ? '40px' : '32px',
          borderRadius: '50%',
          border: 'none',
          background: '#cccccc',
          color: 'white',
          cursor: 'not-allowed',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          padding: 0,
          opacity: 0.5,
          marginLeft: '8px'
        }}
        title="Audio not available"
      >
        <span style={{ fontSize: size === 'small' ? '16px' : size === 'large' ? '24px' : '20px' }}>🔇</span>
      </button>
    );
  }

  const iconSize = size === 'small' ? '16px' : size === 'large' ? '24px' : '20px';
  const buttonSize = size === 'small' ? '28px' : size === 'large' ? '40px' : '32px';

  return (
    <div style={{ display: 'inline-flex', alignItems: 'center', gap: '8px', marginLeft: '8px' }}>
      <button
        onClick={handlePlay}
        style={{
          width: buttonSize,
          height: buttonSize,
          borderRadius: '50%',
          border: 'none',
          background: isPlaying ? '#dc3545' : '#28a745',
          color: 'white',
          cursor: 'pointer',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          transition: 'all 0.2s',
          boxShadow: '0 2px 4px rgba(0,0,0,0.2)',
          padding: 0
        }}
        onMouseEnter={(e) => {
          e.target.style.transform = 'scale(1.1)';
          e.target.style.boxShadow = '0 4px 8px rgba(0,0,0,0.3)';
        }}
        onMouseLeave={(e) => {
          e.target.style.transform = 'scale(1)';
          e.target.style.boxShadow = '0 2px 4px rgba(0,0,0,0.2)';
        }}
        title={isPlaying ? 'Stop audio' : `Play ${label} audio`}
      >
        {isPlaying ? (
          <span style={{ fontSize: iconSize }}>⏸</span>
        ) : (
          <span style={{ fontSize: iconSize }}>▶</span>
        )}
      </button>
      <audio
        ref={audioRef}
        src={audioUrl}
        onEnded={handleEnded}
        onError={() => {
          setIsPlaying(false);
          console.error('Error playing audio:', audioUrl);
        }}
      />
    </div>
  );
};

function App() {
  const [activeTab, setActiveTab] = useState('upload') // 'upload', 'live', or 'letter-detection'
  
  // Upload state
  const [file, setFile] = useState(null)
  const [uploading, setUploading] = useState(false)
  
  // Live capture state
  const [isCapturing, setIsCapturing] = useState(false)
  const [capturedFrames, setCapturedFrames] = useState([])
  const [processing, setProcessing] = useState(false)
  const [recognizedWords, setRecognizedWords] = useState([]) // Array of recognized words during capture
  const [batchFrameCount, setBatchFrameCount] = useState(0) // Frame count in current batch (for UI)
  const [handsDetected, setHandsDetected] = useState(false) // Hands detection state (for UI)
  const [streamActive, setStreamActive] = useState(false) // Track if video stream is active (for UI visibility)
  const videoRef = useRef(null)
  const streamRef = useRef(null)
  const captureIntervalRef = useRef(null)
  const isCapturingRef = useRef(false) // Ref to track capturing state for closure
  const frameBatchRef = useRef([]) // Current batch of frames (from hands appeared to hands disappeared)
  const allRecognizedWordsRef = useRef([]) // Accumulate all recognized words during capture session
  const pendingBatchPromisesRef = useRef([]) // Track all pending batch processing promises
  const frameCountRef = useRef(0) // Total frame count since capture started
  const totalFramesCapturedRef = useRef(0) // Total frames captured across all batches
  const handsRef = useRef(null) // MediaPipe Hands instance
  const handsDetectionIntervalRef = useRef(null) // Interval for hand detection
  const handsDetectedRef = useRef(false) // Track if hands are currently detected
  const noHandsFrameCountRef = useRef(0) // Count consecutive frames without hands
  const HANDS_DETECTION_THRESHOLD = 1 // Stop immediately when hands disappear
  const MIN_FRAMES_TO_PROCESS = 10 // Minimum frames needed to process
  
  // Hand detection state refs (must be declared before useEffect that uses them)
  const handDetectionResultRef = useRef(null) // Store latest hand detection result
  const handDetectionProcessingRef = useRef(false) // Track if MediaPipe is currently processing
  const handDetectionTimeoutRef = useRef(null) // Timeout for processing safety
  
  // Shared state
  const [result, setResult] = useState(null)
  const [error, setError] = useState(null)
  const [countdown, setCountdown] = useState(null) // null, 3, 2, 1, or 'capturing'
  const [isInCountdown, setIsInCountdown] = useState(false)
  
  // Debug: Log result changes
  useEffect(() => {
    if (result) {
      console.log('Result updated:', result)
      console.log('Audio URLs:', result.audio_urls)
      console.log('Sentence:', result.sentence)
      console.log('Translated:', result.translated_text)
    }
  }, [result])
  
  
  const [modelStatus, setModelStatus] = useState(null)

  // Reset processing state and clear errors when switching tabs
  React.useEffect(() => {
    // Clear error when switching tabs to prevent showing errors from other tabs
    setError(null)
    if (activeTab === 'live') {
      setProcessing(false)
      setIsCapturing(false)
    }
  }, [activeTab])

  React.useEffect(() => {
    // Check model status on load
    axios.get(`${API_BASE}/model-status`)
      .then(response => {
        setModelStatus(response.data)
      })
      .catch(err => {
        console.error('Failed to get model status:', err)
      })
    
    // Initialize MediaPipe Hands with continuous detection
    const initializeHands = async () => {
      try {
        const hands = new Hands({
          locateFile: (file) => {
            return `https://cdn.jsdelivr.net/npm/@mediapipe/hands@0.4/${file}`
          }
        })
        
        hands.setOptions({
          maxNumHands: 2,
          modelComplexity: 0, // Use simpler model for speed
          minDetectionConfidence: 0.5,
          minTrackingConfidence: 0.5
        })
        
        // Set up continuous hand detection callback
        hands.onResults((results) => {
          const hasHands = results.multiHandLandmarks && 
                          results.multiHandLandmarks.length > 0
          handDetectionResultRef.current = hasHands
          handDetectionProcessingRef.current = false // Mark processing as complete
          
          // Clear any pending timeout
          if (handDetectionTimeoutRef.current) {
            clearTimeout(handDetectionTimeoutRef.current)
            handDetectionTimeoutRef.current = null
          }
        })
        
        handsRef.current = hands
        console.log('[HANDS] MediaPipe Hands initialized with continuous detection')
      } catch (error) {
        console.error('[HANDS] Error initializing MediaPipe Hands:', error)
      }
    }
    
    initializeHands()
    
    // Cleanup on unmount
    return () => {
      stopCapture()
      if (handsRef.current) {
        handsRef.current.close()
        handsRef.current = null
      }
      if (handsDetectionIntervalRef.current) {
        clearInterval(handsDetectionIntervalRef.current)
        handsDetectionIntervalRef.current = null
      }
    }
  }, [])

  // Debug: Log frame count changes
  React.useEffect(() => {
    if (capturedFrames.length > 0) {
      console.log(`Frame count updated: ${capturedFrames.length} frames`)
    }
  }, [capturedFrames.length])

  // ==================== UPLOAD MODE ====================
  const handleFileChange = (e) => {
    const selectedFile = e.target.files[0]
    if (selectedFile) {
      setFile(selectedFile)
      setResult(null)
      setError(null)
    }
  }

  const handleUpload = async () => {
    if (!file) {
      setError('Please select a video file')
      return
    }

    setUploading(true)
    setError(null)
    setResult(null)

    const formData = new FormData()
    formData.append('video', file)

    try {
      const response = await axios.post(`${API_BASE}/process-video`, formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
        timeout: 120000, // 2 minutes timeout
      })

      setResult(response.data)
      console.log('Recognition result:', response.data)
      console.log('Audio URLs in result:', response.data?.audio_urls)
    } catch (err) {
      console.error('Upload error:', err)
      setError(err.response?.data?.error || err.message || 'Failed to process video')
    } finally {
      setUploading(false)
    }
  }

  // ==================== LIVE CAPTURE MODE ====================
  const startCountdown = () => {
    return new Promise((resolve) => {
      setIsInCountdown(true)
      setCountdown(3)
      
      const countdownInterval = setInterval(() => {
        setCountdown((prev) => {
          if (prev === 1) {
            clearInterval(countdownInterval)
            setIsInCountdown(false)
            setCountdown(null)
            setTimeout(() => resolve(), 100) // Small delay before resolving
            return null
          }
          return prev - 1
        })
      }, 1000)
    })
  }

  const startCapture = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: 640, height: 480 },
        audio: false,
      })
      
      streamRef.current = stream
      setStreamActive(true) // Update state to trigger re-render and show video
      
      if (!videoRef.current) {
        setError('Video element not found')
        return
      }
      
      const video = videoRef.current
      video.srcObject = stream
      
      // Wait for video metadata to load with timeout
      await new Promise((resolve, reject) => {
        const timeout = setTimeout(() => {
          video.removeEventListener('loadedmetadata', onLoadedMetadata)
          video.removeEventListener('error', onError)
          reject(new Error('Video metadata loading timeout'))
        }, 10000) // 10 second timeout
        
        const onLoadedMetadata = () => {
          clearTimeout(timeout)
          video.removeEventListener('loadedmetadata', onLoadedMetadata)
          video.removeEventListener('error', onError)
          console.log('[START] Video metadata loaded:', video.videoWidth, 'x', video.videoHeight)
          console.log('[START] Video readyState:', video.readyState)
          resolve()
        }
        
        const onError = (err) => {
          clearTimeout(timeout)
          video.removeEventListener('loadedmetadata', onLoadedMetadata)
          video.removeEventListener('error', onError)
          reject(err)
        }
        
        // If already loaded, resolve immediately
        if (video.readyState >= 1) {
          console.log('[START] Video already has metadata')
          clearTimeout(timeout)
          onLoadedMetadata()
          return
        }
        
        video.addEventListener('loadedmetadata', onLoadedMetadata)
        video.addEventListener('error', onError)
        
        // Start playing
        video.play().then(() => {
          console.log('[START] Video play() resolved')
        }).catch((err) => {
          clearTimeout(timeout)
          reject(err)
        })
      })
      
      // Wait for video to be ready with retries
      let retries = 0
      const maxRetries = 10
      while (retries < maxRetries) {
        await new Promise(resolve => setTimeout(resolve, 200))
        
        console.log(`[START] Retry ${retries + 1}/${maxRetries} - readyState:`, video.readyState, 'dimensions:', video.videoWidth, 'x', video.videoHeight)
        
        // Check if video is ready (readyState >= 1 means we have metadata)
        if (video.readyState >= 1 && video.videoWidth > 0 && video.videoHeight > 0) {
          console.log('[START] Video is ready!')
          break
        }
        
        retries++
      }
      
      // Final check - be more lenient (readyState >= 1 is enough)
      if (video.readyState < 1) {
        throw new Error('Video stream not initialized. Please check camera permissions.')
      }
      
      // If dimensions are still 0, that's okay - they might be set later
      if (!video.videoWidth || !video.videoHeight) {
        console.warn('[START] Video dimensions not yet available, but stream is active')
      }
      
      isCapturingRef.current = true
      setIsCapturing(true)
      setProcessing(false) // Reset processing state when starting new capture
      setCapturedFrames([])
      setRecognizedWords([]) // Clear previous words
      frameBatchRef.current = [] // Clear batch buffer
      allRecognizedWordsRef.current = [] // Reset accumulated words for new session
      pendingBatchPromisesRef.current = [] // Clear pending batch promises
      frameCountRef.current = 0 // Reset frame counter
      totalFramesCapturedRef.current = 0 // Reset total frames captured
      handsDetectedRef.current = false
      setHandsDetected(false)
      noHandsFrameCountRef.current = 0
      setBatchFrameCount(0)
      handDetectionResultRef.current = null // Reset hand detection result
      setError(null)
      setResult(null)
      
      console.log('[START] Starting countdown before capture...')
      
      // Start with countdown (3, 2, 1)
      await startCountdown()
      
      console.log('[START] Countdown complete, starting hand detection and frame capture...')
      
      // Small delay to ensure MediaPipe Hands is ready
      await new Promise(resolve => setTimeout(resolve, 300))
      
      // Start continuous hand detection (check every 200ms to avoid overwhelming MediaPipe)
      handsDetectionIntervalRef.current = setInterval(() => {
        if (!videoRef.current || !handsRef.current) return
        
        const video = videoRef.current
        if (video.readyState < 2) return
        
        // Only send frame if MediaPipe is not currently processing
        if (handDetectionProcessingRef.current) {
          return // Skip this frame, wait for previous to finish
        }
        
        // Send frame to MediaPipe Hands for continuous detection
        try {
          const canvas = document.createElement('canvas')
          canvas.width = video.videoWidth
          canvas.height = video.videoHeight
          const ctx = canvas.getContext('2d')
          ctx.drawImage(video, 0, 0, canvas.width, canvas.height)
          
          handDetectionProcessingRef.current = true // Mark as processing
          
          // Set a timeout to reset processing flag if MediaPipe gets stuck (3 seconds max)
          if (handDetectionTimeoutRef.current) {
            clearTimeout(handDetectionTimeoutRef.current)
          }
          handDetectionTimeoutRef.current = setTimeout(() => {
            if (handDetectionProcessingRef.current) {
              console.warn('[HANDS] Processing timeout - resetting flag')
              handDetectionProcessingRef.current = false
              handDetectionTimeoutRef.current = null
            }
          }, 3000)
          
          try {
            handsRef.current.send({ image: canvas })
          } catch (error) {
            console.error('[HANDS] Error sending frame to MediaPipe:', error)
            handDetectionProcessingRef.current = false // Reset on error
            if (handDetectionTimeoutRef.current) {
              clearTimeout(handDetectionTimeoutRef.current)
              handDetectionTimeoutRef.current = null
            }
          }
        } catch (error) {
          console.error('[HANDS] Error preparing frame:', error)
          handDetectionProcessingRef.current = false // Reset on error
        }
        
        // Check the latest detection result
        const hasHands = handDetectionResultRef.current
        
        if (hasHands) {
          // Hands detected
          if (!handsDetectedRef.current) {
            console.log('[HANDS] Hands detected - starting new batch')
            handsDetectedRef.current = true
            setHandsDetected(true)
            noHandsFrameCountRef.current = 0
            // Start a new batch (clear current batch, but keep allRecognizedWordsRef)
            frameBatchRef.current = []
            setBatchFrameCount(0)
            
            // Start frame capture interval if not already running
            if (!captureIntervalRef.current) {
              captureIntervalRef.current = setInterval(() => {
                if (handsDetectedRef.current && isCapturingRef.current) {
                  captureFrame()
                }
              }, 100) // 10 FPS
            }
          }
        } else {
          // No hands detected - stop capturing frames immediately
          if (handsDetectedRef.current) {
            noHandsFrameCountRef.current++
            
            // Stop immediately when hands disappear (threshold = 1)
            if (noHandsFrameCountRef.current >= HANDS_DETECTION_THRESHOLD) {
              console.log('[HANDS] No hands detected - processing current batch')
              handsDetectedRef.current = false
              setHandsDetected(false)
              
              // Stop frame capture immediately
              if (captureIntervalRef.current) {
                clearInterval(captureIntervalRef.current)
                captureIntervalRef.current = null
              }
              
              // Process the current batch (variable size) and add word to recognizedWords
              // Track the promise so we can wait for it when stopping
              const batchPromise = processCurrentBatch()
              pendingBatchPromisesRef.current.push(batchPromise)
              
              // Remove from pending list when done
              batchPromise.finally(() => {
                const index = pendingBatchPromisesRef.current.indexOf(batchPromise)
                if (index > -1) {
                  pendingBatchPromisesRef.current.splice(index, 1)
                }
              })
            }
          }
        }
      }, 200) // Check for hands every 200ms (gives MediaPipe time to process)
      
      console.log('[START] Hand detection started')
      
    } catch (err) {
      console.error('Camera access error:', err)
      setError('Failed to access camera. Please check permissions.')
      setIsCapturing(false)
    }
  }

  // Process current batch when hands disappear (variable size)
  const processCurrentBatch = async () => {
    const currentBatch = [...frameBatchRef.current] // Copy batch to avoid race conditions
    
    if (currentBatch.length < MIN_FRAMES_TO_PROCESS) {
      console.log(`[BATCH] Not enough frames in current batch (${currentBatch.length} frames, need at least ${MIN_FRAMES_TO_PROCESS})`)
      frameBatchRef.current = []
      setBatchFrameCount(0)
      return
    }
    
    // Clear the batch immediately so next batch can start
    const batchSize = currentBatch.length
    frameBatchRef.current = []
    setBatchFrameCount(0)
    
    // Track total frames captured
    totalFramesCapturedRef.current += batchSize
    
    console.log(`[BATCH] Processing batch of ${batchSize} frames (Total frames captured: ${totalFramesCapturedRef.current})`)
    
    try {
      // Process immediately without waiting - this speeds up recognition
      const wordData = await processFrameBatch(currentBatch)
      
      if (wordData) {
        // Add frame count to word data
        wordData.frameCount = batchSize
        // Add to accumulated recognized words
        allRecognizedWordsRef.current.push(wordData)
        console.log(`[BATCH] Recognized word: ${wordData.word} (confidence: ${(wordData.confidence * 100).toFixed(1)}%) - Total words: ${allRecognizedWordsRef.current.length}`)
        
        // Update UI immediately to show recognized words as they come in
        setRecognizedWords([...allRecognizedWordsRef.current])
      } else {
        console.log(`[BATCH] No word recognized from ${currentBatch.length} frames`)
      }
    } catch (err) {
      console.error('[BATCH] Error processing batch:', err)
    }
  }

  // Form sentence from all accumulated words when Stop Capture is clicked
  const processAllCapturedFrames = async () => {
    // Process any remaining frames in current batch
    if (frameBatchRef.current.length >= MIN_FRAMES_TO_PROCESS) {
      const batchPromise = processCurrentBatch()
      pendingBatchPromisesRef.current.push(batchPromise)
      batchPromise.finally(() => {
        const index = pendingBatchPromisesRef.current.indexOf(batchPromise)
        if (index > -1) {
          pendingBatchPromisesRef.current.splice(index, 1)
        }
      })
    }
    
    // Wait for ALL pending batch processing to complete before forming sentence
    console.log(`[STOP] Waiting for ${pendingBatchPromisesRef.current.length} pending batch(es) to complete...`)
    await Promise.all(pendingBatchPromisesRef.current)
    console.log(`[STOP] All batches processed`)
    
    const allRecognizedWords = allRecognizedWordsRef.current
    
    if (allRecognizedWords.length === 0) {
      console.log(`[STOP] No words recognized during capture session`)
      setProcessing(false) // Reset processing when no words found
      return
    }
    
    console.log(`[STOP] Forming sentence from ${allRecognizedWords.length} recognized words`)
    setProcessing(true)
    setError(null)
    
    try {
      // Form sentence from ALL accumulated words and generate TTS
      const words = allRecognizedWords.map(w => w.word)
      console.log(`[STOP] Words to form sentence:`, words)
      
      const response = await axios.post(`${API_BASE}/form-sentence-and-tts`, {
        words: words
      }, {
        headers: {
          'Content-Type': 'application/json',
        },
        timeout: 120000,
      })
      
      if (response.data) {
        // Calculate total frames - use totalFramesCapturedRef if available, otherwise sum batch frames
        const calculatedTotalFrames = totalFramesCapturedRef.current > 0 
          ? totalFramesCapturedRef.current 
          : allRecognizedWords.reduce((sum, w) => sum + (w.frameCount || 0), 0)
        
        console.log(`[STOP] Total frames captured: ${calculatedTotalFrames} (from ${allRecognizedWords.length} words)`)
        
        setResult({
          sentence: response.data.sentence,
          translated_text: response.data.translated_text,
          audio_urls: response.data.audio_urls,
          words: allRecognizedWords, // Pass full word objects with confidence
          totalWords: allRecognizedWords.length,
          totalFrames: calculatedTotalFrames
        })
        console.log(`[STOP] Sentence formed: ${response.data.sentence}`)
        console.log(`[STOP] Audio URLs:`, response.data.audio_urls)
      }
    } catch (err) {
      console.error('[STOP] Error forming sentence:', err)
      setError('Failed to form sentence or generate audio')
    } finally {
      setProcessing(false)
    }
  }

  const stopCapture = async () => {
    console.log('[STOP] Stopping capture...')
    
    // Immediately set processing to true and stop capturing to hide the button
    setIsCapturing(false)
    isCapturingRef.current = false
    setProcessing(true)
    
    // Stop hand detection
    if (handsDetectionIntervalRef.current) {
      clearInterval(handsDetectionIntervalRef.current)
      handsDetectionIntervalRef.current = null
      console.log('[STOP] Hand detection stopped')
    }
    
    // Reset processing flag and clear timeout
    handDetectionProcessingRef.current = false
    handDetectionResultRef.current = null
    if (handDetectionTimeoutRef.current) {
      clearTimeout(handDetectionTimeoutRef.current)
      handDetectionTimeoutRef.current = null
    }
    
    // Form sentence from all accumulated words (processCurrentBatch handles remaining frames)
    try {
      await processAllCapturedFrames()
    } catch (err) {
      console.error('[STOP] Error processing frames:', err)
      setError('Error processing captured frames')
      setProcessing(false) // Ensure processing is reset even on error
    }
    
    // Clean up
    frameBatchRef.current = []
    setBatchFrameCount(0)
    
    if (captureIntervalRef.current) {
      clearInterval(captureIntervalRef.current)
      captureIntervalRef.current = null
      console.log('[STOP] Interval cleared')
    }
    
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(track => track.stop())
      streamRef.current = null
      setStreamActive(false) // Update state to hide video
      console.log('[STOP] Stream stopped')
    }
    
    if (videoRef.current) {
      videoRef.current.srcObject = null
    }
    
    setCountdown(null)
    setIsInCountdown(false)
    handsDetectedRef.current = false
    setHandsDetected(false)
    noHandsFrameCountRef.current = 0
    setBatchFrameCount(0)
    console.log('[STOP] Capture stopped, final frame count:', frameBatchRef.current.length)
    
  }

  const processFrameBatch = async (frames) => {
    if (frames.length === 0) return null
    
    try {
      console.log(`[BATCH] Processing batch of ${frames.length} frames`)
      
      const response = await axios.post(`${API_BASE}/process-live`, {
        frames: frames
      }, {
        headers: {
          'Content-Type': 'application/json',
        },
        timeout: 120000, // 2 minutes timeout
      })
      
      if (response.data && response.data.word) {
        const wordData = {
          word: response.data.word,
          confidence: response.data.confidence,
          timestamp: Date.now()
        }
        
        console.log(`[BATCH] Recognized word: ${wordData.word} (confidence: ${(wordData.confidence * 100).toFixed(1)}%)`)
        return wordData
      }
      return null
    } catch (err) {
      console.error('[BATCH] Error processing frame batch:', err)
      return null
    }
  }

  const captureFrame = async () => {
    const video = videoRef.current
    
    if (!video) {
      console.log('[CAPTURE] No video element')
      return
    }
    
    // CRITICAL: If we have an active interval, we should be capturing
    // Force isCapturingRef to true if interval is running (unless explicitly stopped)
    if (captureIntervalRef.current && !isCapturingRef.current) {
      console.log('[CAPTURE] WARNING: isCapturingRef was false but interval is active, forcing to true')
      isCapturingRef.current = true
    }
    
    // Only return early if we're explicitly not capturing AND no interval is running
    if (!isCapturingRef.current && !captureIntervalRef.current) {
      console.log('[CAPTURE] Not capturing (ref check) and no active interval')
      return
    }
    
    // Check if video is ready and has valid dimensions
    if (video.readyState < 2) {
      console.log('[CAPTURE] Video not ready, readyState:', video.readyState)
      return
    }
    
    const videoWidth = video.videoWidth
    const videoHeight = video.videoHeight
    
    if (!videoWidth || !videoHeight || videoWidth === 0 || videoHeight === 0) {
      console.log('[CAPTURE] Invalid dimensions:', videoWidth, 'x', videoHeight)
      return
    }
    
    try {
      const canvas = document.createElement('canvas')
      canvas.width = videoWidth
      canvas.height = videoHeight
      const ctx = canvas.getContext('2d')
      
      if (!ctx) {
        console.error('[CAPTURE] Failed to get canvas context')
        return
      }
      
      // Draw the current video frame to canvas
      ctx.drawImage(video, 0, 0, canvas.width, canvas.height)
      
      // Convert to base64
      const frameData = canvas.toDataURL('image/jpeg', 0.8)
      
      if (!frameData || frameData.length < 100) {
        console.warn('[CAPTURE] Invalid frame data, length:', frameData?.length)
        return
      }
      
      console.log('[CAPTURE] Frame captured successfully, size:', frameData.length, 'bytes')
      
      // Increment counters
      frameCountRef.current++
      
      // Update state with new frame
      setCapturedFrames(prev => {
        const newFrames = [...prev, frameData]
        const trimmed = newFrames.slice(-200)
        console.log('[CAPTURE] Total frames:', trimmed.length, 'Batch frames:', frameBatchRef.current.length)
        return trimmed
      })
      
      // Add to batch buffer (only when hands are detected)
      if (handsDetectedRef.current) {
        frameBatchRef.current.push(frameData)
        setBatchFrameCount(frameBatchRef.current.length)
      }
    } catch (err) {
      console.error('[CAPTURE] Frame capture error:', err)
    }
  }

  // handleProcessLive is no longer needed - processing happens automatically during capture

  return (
    <div className="app">
      <div className="container">
        <header className="header">
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', width: '100%' }}>
            <div style={{ flex: 1 }}>
              <h1>Gesture2Globe</h1>
              <p>Upload a video or capture live to recognize American Sign Language</p>
            </div>
            <ThemeToggle />
          </div>
        </header>

        {modelStatus && (
          <div className={`status-badge ${modelStatus.model_loaded ? 'success' : 'error'}`}>
            {modelStatus.model_loaded ? 'Ready to recognize your signs' : 'Model Not Ready'}
            {modelStatus.error && <span className="error-text">: {modelStatus.error}</span>}
          </div>
        )}

        {/* Tab Navigation */}
        <div className="tab-container">
          <button
            className={`tab-button ${activeTab === 'upload' ? 'active' : ''}`}
            onClick={() => {
              setActiveTab('upload')
              stopCapture()
              setResult(null)
              setError(null) // Clear error when switching to upload tab
              setProcessing(false) // Reset processing state
              setIsCapturing(false) // Reset capturing state
            }}
          >
            Upload Video
          </button>
          <button
            className={`tab-button ${activeTab === 'live' ? 'active' : ''}`}
            onClick={() => {
              setActiveTab('live')
              setResult(null)
              setError(null)
              setProcessing(false) // Reset processing state when switching to live tab
              setIsCapturing(false) // Reset capturing state
              setStreamActive(false) // Reset stream state
            }}
          >
            Live Capture
          </button>
          <button
            className={`tab-button ${activeTab === 'letter-detection' ? 'active' : ''}`}
            onClick={() => {
              setActiveTab('letter-detection')
              stopCapture()
              setResult(null)
              setError(null)
            }}
          >
            Letter Detection
          </button>
        </div>

        {/* Upload Tab Content */}
        {activeTab === 'upload' && (
          <div className="upload-section-grid">
            {/* Left Column - Upload Controls */}
            <div className="upload-controls-column">
              <div className="upload-section">
                <div className="file-input-wrapper">
                  <input
                    type="file"
                    id="video-upload"
                    accept="video/*"
                    onChange={handleFileChange}
                    disabled={uploading}
                    className="file-input"
                  />
                  <label htmlFor="video-upload" className="file-label">
                    {file ? file.name : 'Choose Video File'}
                  </label>
                </div>

                <button
                  onClick={handleUpload}
                  disabled={!file || uploading || !modelStatus?.model_loaded}
                  className="upload-button"
                >
                  {uploading ? 'Processing...' : 'Process Video'}
                </button>
              </div>
            </div>

            {/* Right Column - Results */}
            <div className="upload-results-column">
              {error && activeTab === 'upload' && (
                <div className="error-message">
                  <strong>Error:</strong> {error}
                </div>
              )}

              {result && activeTab === 'upload' && (
                <div className="result-section">
                  <h2>Recognition Results</h2>
                  
                  {result.words && Array.isArray(result.words) ? (
                    // Display multiple words (from live capture)
                    <div className="result-card">
                      <div className="result-item">
                        <span className="label">Total Words Recognized:</span>
                        <span className="value">{result.totalWords || (result.words ? result.words.length : 0)}</span>
                      </div>
                      
                      <div className="result-item">
                        <span className="label">Total Frames Processed:</span>
                        <span className="value">{result.totalFrames || (result.words ? result.words.reduce((sum, w) => sum + (w.frameCount || 0), 0) : 0)}</span>
                      </div>
                      
                      <div className="words-list" style={{ marginTop: '20px' }}>
                        <h3 style={{ marginBottom: '15px', fontSize: '1.1em' }}>Recognized Words:</h3>
                        <div style={{ display: 'flex', flexWrap: 'wrap', gap: '10px' }}>
                          {result.words.map((wordData, index) => {
                            // Handle both string and object formats
                            const word = typeof wordData === 'string' ? wordData : wordData.word
                            const confidence = typeof wordData === 'string' ? null : wordData.confidence
                            
                            return (
                              <div 
                                key={index}
                                style={{
                                  background: '#f8f9fa',
                                  border: '1px solid #dee2e6',
                                  borderRadius: '8px',
                                  padding: '12px 16px',
                                  minWidth: '150px',
                                  textAlign: 'center'
                                }}
                              >
                                <div style={{ fontSize: '1.2em', fontWeight: 'bold', color: '#28a745', marginBottom: '5px' }}>
                                  {word}
                                </div>
                                <div style={{ fontSize: '0.85em', color: '#6c757d' }}>
                                  {confidence !== null && confidence !== undefined 
                                    ? `${(confidence * 100).toFixed(1)}% confidence`
                                    : 'Confidence: N/A'}
                                </div>
                              </div>
                            )
                          })}
                        </div>
                      </div>
                      
                      {result.sentence && (
                        <div className="result-item" style={{ 
                          marginTop: '20px', 
                          paddingTop: '20px', 
                          borderTop: '1px solid #dee2e6',
                          display: 'flex', 
                          alignItems: 'center', 
                          flexWrap: 'wrap', 
                          gap: '8px' 
                        }}>
                          <span className="label">Formed Sentence:</span>
                          <span className="value" style={{ 
                            fontSize: '1.1em', 
                            fontWeight: 'bold', 
                            color: '#007bff',
                            display: 'flex', 
                            alignItems: 'center', 
                            gap: '8px' 
                          }}>
                            {result.sentence}
                            <AudioPlayButton 
                              audioUrl={result.audio_urls?.en || result.audio_urls?.['en']} 
                              label="English"
                              size="medium"
                            />
                          </span>
                        </div>
                      )}
                      
                      {result.translated_text && (
                        <div className="result-item" style={{ 
                          marginTop: '15px',
                          display: 'flex', 
                          alignItems: 'center', 
                          flexWrap: 'wrap', 
                          gap: '8px' 
                        }}>
                          <span className="label">Translated (Spanish):</span>
                          <span className="value" style={{ 
                            fontSize: '1em', 
                            color: '#6c757d',
                            display: 'flex', 
                            alignItems: 'center', 
                            gap: '8px' 
                          }}>
                            {result.translated_text}
                            <AudioPlayButton 
                              audioUrl={result.audio_urls?.es || result.audio_urls?.['es']} 
                              label="Spanish"
                              size="medium"
                            />
                          </span>
                        </div>
                      )}
                    </div>
                  ) : (
                    // Display single word result (from video upload)
                    <div className="result-card">
                      <div className="result-item">
                        <span className="label">Recognized Word:</span>
                        <span className="value">{result.word}</span>
                      </div>
                      
                      <div className="result-item">
                        <span className="label">Confidence:</span>
                        <span className="value">{(result.confidence * 100).toFixed(1)}%</span>
                      </div>
                      
                      {result.sentence && (
                        <div className="result-item" style={{ 
                          display: 'flex', 
                          alignItems: 'center', 
                          flexWrap: 'wrap', 
                          gap: '8px' 
                        }}>
                          <span className="label">Formed Sentence:</span>
                          <span className="value" style={{ 
                            display: 'flex', 
                            alignItems: 'center', 
                            gap: '8px' 
                          }}>
                            {result.sentence}
                            <AudioPlayButton 
                              audioUrl={result.audio_urls?.en || result.audio_urls?.['en']} 
                              label="English"
                              size="medium"
                            />
                          </span>
                        </div>
                      )}
                      
                      {result.translated_text && (
                        <div className="result-item" style={{ 
                          display: 'flex', 
                          alignItems: 'center', 
                          flexWrap: 'wrap', 
                          gap: '8px' 
                        }}>
                          <span className="label">Translated (Spanish):</span>
                          <span className="value" style={{ 
                            display: 'flex', 
                            alignItems: 'center', 
                            gap: '8px' 
                          }}>
                            {result.translated_text}
                            <AudioPlayButton 
                              audioUrl={result.audio_urls?.es || result.audio_urls?.['es']} 
                              label="Spanish"
                              size="medium"
                            />
                          </span>
                        </div>
                      )}
                    </div>
                  )}
                </div>
              )}

              {!result && !error && (
                <div className="results-placeholder">
                  <p style={{ color: 'var(--text-secondary)', textAlign: 'center', padding: '40px' }}>
                    Upload and process a video to see recognition results here
                  </p>
                </div>
              )}
            </div>
          </div>
        )}

        {/* Live Capture Tab Content */}
        {activeTab === 'live' && (
          <div className="live-section-grid">
            {/* Countdown Overlay */}
            {isInCountdown && countdown && (
              <div style={{
                position: 'fixed',
                top: '50%',
                left: '50%',
                transform: 'translate(-50%, -50%)',
                fontSize: '120px',
                fontWeight: 'bold',
                color: '#007bff',
                zIndex: 9999,
                textShadow: '0 0 30px rgba(0,123,255,0.8), 0 0 60px rgba(0,123,255,0.5)',
                pointerEvents: 'none',
                animation: 'pulse 0.5s ease-in-out',
                backgroundColor: 'rgba(0,0,0,0.7)',
                borderRadius: '50%',
                width: '200px',
                height: '200px',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center'
              }}>
                {countdown}
              </div>
            )}
            
            {/* Left Column - Camera */}
            <div className="live-camera-column">
              <div className="video-container">
                <video
                  ref={videoRef}
                  autoPlay
                  playsInline
                  muted
                  className="video-preview"
                  style={{
                    display: streamActive ? 'block' : 'none',
                    opacity: streamActive ? 1 : 0
                  }}
                  onLoadedMetadata={() => {
                    console.log('[VIDEO] Metadata loaded in video element:', videoRef.current?.videoWidth, 'x', videoRef.current?.videoHeight)
                  }}
                  onPlay={() => {
                    console.log('[VIDEO] Video started playing')
                  }}
                  onError={(e) => {
                    console.error('[VIDEO] Video error:', e)
                    setError('Failed to load video stream. Please check camera permissions.')
                  }}
                />
                {!streamActive && (
                  <div className="video-placeholder">
                    <p>Camera preview will appear here</p>
                    <p style={{ fontSize: '0.9em', color: '#999', marginTop: '10px' }}>
                      Click "Start Capture" to begin
                    </p>
                  </div>
                )}
              </div>

              <div className="capture-controls">
                {isCapturing && !processing && (
                  <>
                    <button
                      onClick={stopCapture}
                      className="capture-button stop"
                    >
                      Stop Capture
                    </button>
                    <div className="frame-count">
                      <span style={{ fontWeight: 'bold', color: handsDetected ? '#28a745' : '#ffc107' }}>●</span>
                      {' '}Frames: {batchFrameCount} {handsDetected ? '(capturing)' : '(waiting for hands)'}
                    </div>
                  </>
                )}
                
                {processing && (
                  <div className="frame-count" style={{ color: '#007bff', fontSize: '1.1em', fontWeight: 'bold' }}>
                    Processing... Please wait
                  </div>
                )}
                
                {!isCapturing && !processing && (
                  <>
                    <button
                      onClick={startCapture}
                      disabled={!modelStatus?.model_loaded}
                      className="capture-button start"
                    >
                      Start Capture
                    </button>
                  </>
                )}
              </div>
            </div>

            {/* Right Column - Results */}
            <div className="live-results-column">
              {error && activeTab === 'live' && (
                <div className="error-message">
                  <strong>Error:</strong> {error}
                </div>
              )}

              {result && (
                <div className="result-section">
                  <h2>Recognition Results</h2>
            
            {result.words && Array.isArray(result.words) ? (
              // Display multiple words (from live capture)
              <div className="result-card">
                <div className="result-item">
                  <span className="label">Total Words Recognized:</span>
                  <span className="value">{result.totalWords || (result.words ? result.words.length : 0)}</span>
                </div>
                
                <div className="result-item">
                  <span className="label">Total Frames Processed:</span>
                  <span className="value">{result.totalFrames || (result.words ? result.words.reduce((sum, w) => sum + (w.frameCount || 0), 0) : 0)}</span>
                </div>
                
                <div className="words-list" style={{ marginTop: '20px' }}>
                  <h3 style={{ marginBottom: '15px', fontSize: '1.1em' }}>Recognized Words:</h3>
                  <div style={{ display: 'flex', flexWrap: 'wrap', gap: '10px' }}>
                    {result.words.map((wordData, index) => {
                      // Handle both string and object formats
                      const word = typeof wordData === 'string' ? wordData : wordData.word
                      const confidence = typeof wordData === 'string' ? null : wordData.confidence
                      
                      return (
                        <div 
                          key={index}
                          style={{
                            background: '#f8f9fa',
                            border: '1px solid #dee2e6',
                            borderRadius: '8px',
                            padding: '12px 16px',
                            minWidth: '150px',
                            textAlign: 'center'
                          }}
                        >
                          <div style={{ fontSize: '1.2em', fontWeight: 'bold', color: '#28a745', marginBottom: '5px' }}>
                            {word}
                          </div>
                          <div style={{ fontSize: '0.85em', color: '#6c757d' }}>
                            {confidence !== null && confidence !== undefined 
                              ? `${(confidence * 100).toFixed(1)}% confidence`
                              : 'Confidence: N/A'}
                          </div>
                        </div>
                      )
                    })}
                  </div>
                </div>
                
                {result.sentence && (
                  <div className="result-item" style={{ 
                    marginTop: '20px', 
                    paddingTop: '20px', 
                    borderTop: '1px solid #dee2e6',
                    display: 'flex', 
                    alignItems: 'center', 
                    flexWrap: 'wrap', 
                    gap: '8px' 
                  }}>
                    <span className="label">Formed Sentence:</span>
                    <span className="value" style={{ 
                      fontSize: '1.1em', 
                      fontWeight: 'bold', 
                      color: '#007bff',
                      display: 'flex', 
                      alignItems: 'center', 
                      gap: '8px' 
                    }}>
                      {result.sentence}
                      <AudioPlayButton 
                        audioUrl={result.audio_urls?.en || result.audio_urls?.['en']} 
                        label="English"
                        size="medium"
                      />
                    </span>
                  </div>
                )}
                
                {result.translated_text && (
                  <div className="result-item" style={{ 
                    marginTop: '15px',
                    display: 'flex', 
                    alignItems: 'center', 
                    flexWrap: 'wrap', 
                    gap: '8px' 
                  }}>
                    <span className="label">Translated (Spanish):</span>
                    <span className="value" style={{ 
                      fontSize: '1em', 
                      color: '#6c757d',
                      display: 'flex', 
                      alignItems: 'center', 
                      gap: '8px' 
                    }}>
                      {result.translated_text}
                      <AudioPlayButton 
                        audioUrl={result.audio_urls?.es || result.audio_urls?.['es']} 
                        label="Spanish"
                        size="medium"
                      />
                    </span>
                  </div>
                )}
              </div>
            ) : (
              // Display single word result (from video upload)
              <div className="result-card">
                <div className="result-item">
                  <span className="label">Recognized Word:</span>
                  <span className="value">{result.word}</span>
                </div>
                
                <div className="result-item">
                  <span className="label">Confidence:</span>
                  <span className="value">{(result.confidence * 100).toFixed(1)}%</span>
                </div>
                
                {result.sentence && (
                  <div className="result-item" style={{ 
                    display: 'flex', 
                    alignItems: 'center', 
                    flexWrap: 'wrap', 
                    gap: '8px' 
                  }}>
                    <span className="label">Formed Sentence:</span>
                    <span className="value" style={{ 
                      display: 'flex', 
                      alignItems: 'center', 
                      gap: '8px' 
                    }}>
                      {result.sentence}
                      <AudioPlayButton 
                        audioUrl={result.audio_urls?.en || result.audio_urls?.['en']} 
                        label="English"
                        size="medium"
                      />
                    </span>
                  </div>
                )}
                
                {result.translated_text && (
                  <div className="result-item" style={{ 
                    display: 'flex', 
                    alignItems: 'center', 
                    flexWrap: 'wrap', 
                    gap: '8px' 
                  }}>
                    <span className="label">Translated (Spanish):</span>
                    <span className="value" style={{ 
                      display: 'flex', 
                      alignItems: 'center', 
                      gap: '8px' 
                    }}>
                      {result.translated_text}
                      <AudioPlayButton 
                        audioUrl={result.audio_urls?.es || result.audio_urls?.['es']} 
                        label="Spanish"
                        size="medium"
                      />
                    </span>
                  </div>
                )}
              </div>
            )}
                </div>
              )}

              {!result && !error && (
                <div className="results-placeholder">
                  <p style={{ color: 'var(--text-secondary)', textAlign: 'center', padding: '40px' }}>
                    Recognition results will appear here
                  </p>
                </div>
              )}
            </div>
          </div>
        )}

        {/* Letter Detection Tab Content */}
        {activeTab === 'letter-detection' && (
          <LetterDetection />
        )}

        {/* Only show error for other tabs */}
        {error && activeTab !== 'upload' && activeTab !== 'live' && (
          <div className="error-message">
            <strong>Error:</strong> {error}
          </div>
        )}
      </div>
    </div>
  )
}

export default App
