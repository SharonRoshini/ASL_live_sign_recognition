"""
Letter Detection Translator Module

Detects ASL letters from hand gestures:
- Static letters (A-I, K-Y): Uses ML model to predict from hand pose
- Motion letters (J, Z): Tracks fingertip movement
- Builds words from detected letters
"""

import os
import cv2
import mediapipe as mp
import numpy as np
import joblib
from typing import List, Tuple, Optional, Dict

# MediaPipe for detecting hand landmarks
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Constants
STABLE_FRAMES = 12  # Letter must be detected this many frames in a row before adding to text
MOTION_MIN_POINTS = 15  # Need at least this many points to validate J or Z motion
MOTION_MIN_PATH_LEN = 0.15  # Motion must be at least this long to count as valid



def extract_features(hand_landmarks):
    """
    Convert hand landmarks to a list of numbers for the ML model.
    Takes 21 hand points (x, y, z each) = 63 numbers total.
    """
    features = []
    for lm in hand_landmarks.landmark:
        features.extend([lm.x, lm.y, lm.z])
    return np.array(features, dtype=np.float32)


def get_current_word(text_chars: List[str]) -> str:
    """
    Get the last word being typed (in uppercase).
    Removes spaces and punctuation, returns only letters.
    """
    if not text_chars:
        return ""
    text = "".join(text_chars)
    text = text.rstrip()
    if not text:
        return ""
    parts = text.split(" ")
    last = parts[-1]
    # Keep only alphabetic chars (removes spaces, punctuation, etc.)
    word = "".join(ch for ch in last if ch.isalpha())
    return word.upper()




def compute_path_length(points: List[Tuple[float, float]]) -> float:
    """
    Calculate how far the fingertip moved.
    Adds up the distance between each point in the motion path.
    """
    if len(points) < 2:
        return 0.0
    total = 0.0
    # Sum Euclidean distances between consecutive points
    for i in range(1, len(points)):
        x1, y1 = points[i - 1]
        x2, y2 = points[i]
        dx = x2 - x1
        dy = y2 - y1
        total += (dx * dx + dy * dy) ** 0.5
    return total


class LetterPredictor:
    """
    Predicts ASL letters from hand poses using a trained ML model.
    Uses MediaPipe to detect hand landmarks, then the model predicts the letter.
    """
    
    def __init__(self, model_path: str = None):
        """
        Load the ML model. If no path given, uses default: backend/models/asl_model.joblib
        """
        if model_path is None:
            backend_dir = os.path.dirname(__file__)
            model_path = os.path.join(backend_dir, 'models', 'asl_model.joblib')
        
        self.model_path = model_path
        self.clf = None  # Classifier model
        self.le = None  # Label encoder (maps class indices to letters)
        self.hands = None  # MediaPipe hands processor
        self.loaded = False  # Whether model was successfully loaded
        self.load_error = None  # Track last load error for surfacing to API
        
        # Always initialize MediaPipe (needed for hand detection even if model fails)
        self.init_mediapipe()
        
        # Try to load model if file exists
        if os.path.exists(model_path):
            self.load_model()
        else:
            print(f"Warning: Model file not found at {model_path}")
            print("Hand detection will work, but letter prediction will be disabled.")
    
    def load_model(self):
        """Load the trained model from file. Model file should have 'model' and 'label_encoder'."""
        try:
            bundle = joblib.load(self.model_path)
            if "model" not in bundle:
                raise KeyError("Model file missing 'model' key")
            if "label_encoder" not in bundle:
                raise KeyError("Model file missing 'label_encoder' key")
            self.clf = bundle["model"]  # Trained classifier
            self.le = bundle["label_encoder"]  # Maps 0-23 to 'A'-'Z' (excluding J, Z)
            self.loaded = True
            self.load_error = None
            print(f"Loaded letter detection model. Classes: {list(self.le.classes_)}")
        except KeyError as e:
            print(f"Error loading model - missing key: {e}")
            print(f"Available keys in model file: {list(bundle.keys()) if 'bundle' in locals() else 'N/A'}")
            self.loaded = False
            self.clf = None
            self.le = None
            self.load_error = str(e)
        except Exception as e:
            print(f"Error loading model: {e}")
            import traceback
            traceback.print_exc()
            self.loaded = False
            self.clf = None
            self.le = None
            self.load_error = str(e)
    
    def ensure_model_loaded(self) -> bool:
        """
        Lazily ensure the model is loaded. This helps recover if the model file
        becomes available after startup or if the initial load failed.
        """
        if self.loaded and self.clf is not None and self.le is not None:
            return True
        
        if not os.path.exists(self.model_path):
            self.load_error = f"Model file not found at {self.model_path}"
            return False
        
        try:
            self.load_model()
        except Exception as e:
            self.load_error = str(e)
            self.loaded = False
        return self.loaded
    
    def init_mediapipe(self):
        """Set up MediaPipe to detect hands. Configured for fast real-time tracking."""
        self.hands = mp_hands.Hands(
            max_num_hands=1,
            min_detection_confidence=0.5,  # Lower for faster detection
            min_tracking_confidence=0.5,    # Lower for smoother tracking
            static_image_mode=False,
            model_complexity=0  # Use simpler model for faster processing
        )
    
    def predict_from_image(self, image: np.ndarray) -> Tuple[Optional[str], Optional[object]]:
        """
        Predict letter from a single frame.
        Returns (letter, landmarks) or (None, None) if no hand detected.
        Works for static letters only (not J or Z).
        """
        if not self.loaded or self.hands is None:
            return None, None
        
        # Convert BGR to RGB (MediaPipe needs RGB)
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        result = self.hands.process(rgb)
        
        if result.multi_hand_landmarks:
            hand_landmarks = result.multi_hand_landmarks[0]
            # Get features and predict letter
            feat = extract_features(hand_landmarks).reshape(1, -1)
            pred_idx = self.clf.predict(feat)[0]
            predicted_letter = self.le.inverse_transform([pred_idx])[0]
            return predicted_letter, hand_landmarks
        return None, None
    
    def get_hand_landmarks(self, image: np.ndarray) -> Optional[object]:
        """Get hand landmarks from an image frame."""
        if not self.loaded or self.hands is None:
            return None
        
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        result = self.hands.process(rgb)
        
        if result.multi_hand_landmarks:
            return result.multi_hand_landmarks[0]
        return None
    
    def get_fingertip(self, image: np.ndarray) -> Optional[Tuple[float, float]]:
        """
        Get index fingertip position (x, y between 0-1).
        Used for tracking J and Z motion gestures.
        """
        if not self.loaded or self.hands is None:
            return None
        
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        result = self.hands.process(rgb)
        
        if result.multi_hand_landmarks:
            hand_landmarks = result.multi_hand_landmarks[0]
            # MediaPipe landmark 8 is the index fingertip
            tip = hand_landmarks.landmark[8]  # Index fingertip
            return (tip.x, tip.y)
        return None


class MotionTracker:
    """
    Tracks fingertip movement for J and Z letters.
    J and Z need motion tracking (not just a static pose).
    """
    
    def __init__(self):
        self.motion_mode = False  # Are we tracking motion?
        self.motion_target = None  # "J" or "Z"
        self.motion_points = []  # List of fingertip positions
    
    def start_motion_mode(self):
        """Start motion tracking mode."""
        self.motion_mode = True
        self.motion_target = None
        self.motion_points = []
    
    def set_motion_target(self, target: str):
        """Set motion target (J or Z)."""
        if target.upper() in ["J", "Z"]:
            self.motion_target = target.upper()
            self.motion_points = []
    
    def add_motion_point(self, point: Tuple[float, float]):
        """Add a motion point."""
        if self.motion_mode and self.motion_target:
            self.motion_points.append(point)
    
    def finish_motion(self) -> Optional[str]:
        """
        Check if motion is valid and return letter (J or Z) or None.
        Motion must have enough points and be long enough to count.
        """
        if not self.motion_mode or not self.motion_target:
            return None
        
        # Need enough points
        if len(self.motion_points) < MOTION_MIN_POINTS:
            self.reset()
            return None
        
        # Check if motion is long enough
        path_len = compute_path_length(self.motion_points)
        if path_len >= MOTION_MIN_PATH_LEN:
            letter = self.motion_target
            self.reset()
            return letter
        else:
            self.reset()
            return None
    
    def reset(self):
        """Reset motion tracking state."""
        self.motion_mode = False
        self.motion_target = None
        self.motion_points = []
    
    def get_state(self) -> Dict:
        """Get current motion state."""
        return {
            "motion_mode": self.motion_mode,
            "motion_target": self.motion_target,
            "motion_points_count": len(self.motion_points),
            "motion_points": self.motion_points  # Include actual points for drawing
        }


class TextBuilder:
    """
    Builds text from detected letters.
    Only adds letters after they're detected consistently (prevents flickering).
    """
    
    def __init__(self):
        self.text_chars = []  # The text we're building
        self.last_pred = ""  # Last letter predicted (not committed yet)
        self.stable_count = 0  # How many frames in a row this letter was detected
        self.committed_letter = None  # Last letter we actually added to text
    
    def add_letter(self, letter: str):
        """Add a letter to the text."""
        if letter:
            self.text_chars.append(letter)
            self.committed_letter = letter
    
    def add_space(self):
        """Add a space to the text."""
        self.text_chars.append(" ")
    
    def backspace(self):
        """Remove last character."""
        if self.text_chars:
            self.text_chars.pop()
    
    def clear(self):
        """Clear all text."""
        self.text_chars = []
        self.last_pred = ""
        self.stable_count = 0
        self.committed_letter = None
    
    def update_stability(self, predicted_letter: str) -> bool:
        """
        Track if letter is stable. Returns True when letter is added to text.
        Letter must be detected for STABLE_FRAMES frames in a row before adding.
        """
        # Count how many frames in a row we see the same letter
        if predicted_letter == self.last_pred and predicted_letter != "":
            self.stable_count += 1
        else:
            self.stable_count = 0
            if predicted_letter != self.last_pred:
                self.committed_letter = None
            self.last_pred = predicted_letter
        
        # Add letter if it's been stable long enough (and not already added)
        if self.stable_count == STABLE_FRAMES and predicted_letter != "":
            if self.committed_letter != predicted_letter:
                self.add_letter(predicted_letter)
                self.stable_count = 0
                return True
            self.stable_count = 0
        
        return False
    
    def get_text(self) -> str:
        """Get current text as string."""
        return "".join(self.text_chars)
    
    def get_current_word(self) -> str:
        """Get current word being built."""
        return get_current_word(self.text_chars)
    
    def get_state(self) -> Dict:
        """Get current text builder state."""
        return {
            "text": self.get_text(),
            "current_word": self.get_current_word(),
            "last_pred": self.last_pred,
            "stable_count": self.stable_count,
            "committed_letter": self.committed_letter
        }


class LetterDetectionSystem:
    """
    Main system that combines everything.
    Use this class to detect letters from video frames.
    """
    
    def __init__(self, model_path: str = None):
        """Initialize. If no model path given, uses default location."""
        self.predictor = LetterPredictor(model_path)
        self.motion_tracker = MotionTracker()
        self.text_builder = TextBuilder()
    
    def process_frame(self, image: np.ndarray, draw_on_frame: bool = False) -> Tuple[Dict, np.ndarray]:
        """
        Process one video frame. Main function to call for each frame.
        
        Returns:
            (result_dict, drawn_frame)
            result_dict has: current_letter, text, current_word, has_hand, etc.
            drawn_frame is the image with landmarks drawn (if draw_on_frame=True)
        """
        drawn_frame = image.copy() if draw_on_frame else None
        
        result = {
            "current_letter": "",
            "text": "",
            "current_word": "",
            "motion_state": self.motion_tracker.get_state(),
            "has_hand": False,
            "fingertip_position": None,
            "hand_landmarks": None,
            "model_loaded": self.predictor.loaded,  # Debug: indicate if model is loaded
            "model_error": self.predictor.load_error,
        }
        
        # Detect hand
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        hand_result = self.predictor.hands.process(rgb) if self.predictor.hands else None
        
        hand_landmarks_obj = None
        if hand_result and hand_result.multi_hand_landmarks:
            hand_landmarks_obj = hand_result.multi_hand_landmarks[0]
            result["has_hand"] = True
            
            # Convert landmarks to list for JSON
            landmarks_list = []
            for lm in hand_landmarks_obj.landmark:
                landmarks_list.append({"x": lm.x, "y": lm.y, "z": lm.z})
            result["hand_landmarks"] = landmarks_list
            
            # Draw landmarks on frame if requested
            if draw_on_frame and drawn_frame is not None:
                drawn_frame_rgb = cv2.cvtColor(drawn_frame, cv2.COLOR_BGR2RGB)
                mp_drawing.draw_landmarks(
                    drawn_frame_rgb,
                    hand_landmarks_obj,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=3),
                    mp_drawing.DrawingSpec(color=(0, 150, 0), thickness=2),
                )
                drawn_frame = cv2.cvtColor(drawn_frame_rgb, cv2.COLOR_RGB2BGR)
                
                # Draw motion path (red line) if tracking J or Z
                if self.motion_tracker.motion_mode and self.motion_tracker.motion_target and len(self.motion_tracker.motion_points) > 1:
                    h, w, _ = drawn_frame.shape
                    for i in range(1, len(self.motion_tracker.motion_points)):
                        x1, y1 = self.motion_tracker.motion_points[i - 1]
                        x2, y2 = self.motion_tracker.motion_points[i]
                        p1 = (int(x1 * w), int(y1 * h))
                        p2 = (int(x2 * w), int(y2 * h))
                        cv2.line(drawn_frame, p1, p2, (0, 0, 255), 2)
        
        # Predict letter or track motion
        if not self.motion_tracker.motion_mode:
            # Static mode: predict letter from hand pose
            if hand_landmarks_obj is not None:
                # Make sure model is loaded (auto-retry if it failed at startup)
                self.predictor.ensure_model_loaded()
                result["model_loaded"] = self.predictor.loaded
                result["model_error"] = self.predictor.load_error
                
                if self.predictor.loaded and self.predictor.clf is not None and self.predictor.le is not None:
                    try:
                        feat = extract_features(hand_landmarks_obj).reshape(1, -1)
                        pred_idx = self.predictor.clf.predict(feat)[0]
                        predicted_letter = str(self.predictor.le.inverse_transform([pred_idx])[0])
                        if predicted_letter:
                            result["current_letter"] = predicted_letter
                            committed = self.text_builder.update_stability(predicted_letter)
                            if committed:
                                result["letter_committed"] = True
                    except Exception as e:
                        # Only log error once to avoid spam
                        if not hasattr(self, '_prediction_error_logged'):
                            print(f"[LETTER PREDICTION ERROR] {e}")
                            import traceback
                            traceback.print_exc()
                            self._prediction_error_logged = True
                        # Continue without prediction rather than crashing
                else:
                    # Model not loaded - log once
                    if not hasattr(self, '_model_not_loaded_logged'):
                        print(f"[LETTER PREDICTION] Model not available. loaded={self.predictor.loaded}, clf={self.predictor.clf is not None}, le={self.predictor.le is not None}")
                        self._model_not_loaded_logged = True
        else:
            # Motion mode: track fingertip for J or Z
            if hand_landmarks_obj is not None:
                tip = hand_landmarks_obj.landmark[8]  # Index fingertip
                fingertip = (tip.x, tip.y)
                result["fingertip_position"] = {"x": fingertip[0], "y": fingertip[1]}
                self.motion_tracker.add_motion_point(fingertip)
                result["motion_state"]["motion_points"] = self.motion_tracker.motion_points
        
        result["text"] = self.text_builder.get_text()
        result["current_word"] = self.text_builder.get_current_word()
        
        return result, drawn_frame
    
    def start_motion_mode(self):
        """Start tracking motion for J or Z."""
        self.motion_tracker.start_motion_mode()
    
    def set_motion_target(self, target: str):
        """Set which letter to track: "J" or "Z"."""
        self.motion_tracker.set_motion_target(target)
    
    def finish_motion(self) -> Optional[str]:
        """Finish motion tracking. Returns "J" or "Z" if valid, None otherwise."""
        letter = self.motion_tracker.finish_motion()
        if letter:
            self.text_builder.add_letter(letter)
        return letter
    
    def add_space(self):
        """Add a space to the text."""
        self.text_builder.add_space()
    
    def backspace(self):
        """Delete the last character."""
        self.text_builder.backspace()
    
    def clear(self):
        """Clear all text."""
        self.text_builder.clear()
    
    def get_state(self) -> Dict:
        """Get current state (for debugging)."""
        state = self.text_builder.get_state()
        state["motion_state"] = self.motion_tracker.get_state()
        return state

