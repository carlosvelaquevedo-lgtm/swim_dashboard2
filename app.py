"""
Freestyle Swimming Technique Analyzer Pro
==========================================
AI-powered swimming technique analysis with MediaPipe pose detection.
Enhanced UI inspired by modern web design principles.
"""

import streamlit as st
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import os
import datetime
import statistics
from collections import deque
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict
import pandas as pd
import matplotlib.pyplot as plt
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak,
    Image as RLImage
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.units import inch
import io
import tempfile
import urllib.request
import zipfile

from dataclasses import dataclass

@dataclass
class Recommendation:
    title: str
    description: str
    priority: str  # "high" | "medium" | "low"


# ───────────────────────────────────────────────────────────────────
# CUSTOM CSS FOR MODERN UI
# ───────────────────────────────────────────────────────────────────

CUSTOM_CSS = """
<style>
    /* Main background gradient */
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e3a5f 50%, #0f172a 100%);
    }
    
    /* Card styling */
    .metric-card {
        background: rgba(30, 41, 59, 0.7);
        backdrop-filter: blur(10px);
        border-radius: 16px;
        padding: 20px;
        border: 1px solid rgba(100, 116, 139, 0.3);
        margin-bottom: 16px;
    }
    
    .metric-card-green {
        border-left: 4px solid #22c55e;
    }
    
    .metric-card-red {
        border-left: 4px solid #ef4444;
    }
    
    .metric-card-yellow {
        border-left: 4px solid #eab308;
    }
    
    /* Score card */
    .score-card {
        background: linear-gradient(135deg, #0891b2 0%, #2563eb 100%);
        border-radius: 16px;
        padding: 24px;
        color: white;
        margin-bottom: 24px;
    }
    
    /* Drill card */
    .drill-card {
        background: rgba(15, 23, 42, 0.6);
        border-radius: 12px;
        padding: 16px;
        border: 1px solid rgba(100, 116, 139, 0.3);
        margin-bottom: 12px;
    }
    
    /* Recommendation cards */
    .rec-high {
        background: rgba(127, 29, 29, 0.3);
        border-left: 4px solid #ef4444;
        border-radius: 12px;
        padding: 16px;
        margin-bottom: 12px;
    }
    
    .rec-medium {
        background: rgba(113, 63, 18, 0.3);
        border-left: 4px solid #eab308;
        border-radius: 12px;
        padding: 16px;
        margin-bottom: 12px;
    }
    
    .rec-low {
        background: rgba(20, 83, 45, 0.3);
        border-left: 4px solid #22c55e;
        border-radius: 12px;
        padding: 16px;
        margin-bottom: 12px;
    }
    
    /* Progress bar */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #06b6d4 0%, #3b82f6 100%);
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #06b6d4 0%, #3b82f6 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 12px 24px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 20px rgba(6, 182, 212, 0.3);
    }
    
    /* Sidebar */
    .css-1d391kg {
        background: rgba(15, 23, 42, 0.9);
    }
    
    /* Headers */
    h1, h2, h3 {
        color: #f8fafc !important;
    }
    
    /* Text */
    p, span, label {
        color: #cbd5e1;
    }
    
    /* Legend items */
    .legend-item {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        padding: 8px 12px;
        background: rgba(15, 23, 42, 0.6);
        border-radius: 8px;
        margin-right: 8px;
        margin-bottom: 8px;
    }
    
    .legend-dot {
        width: 12px;
        height: 12px;
        border-radius: 50%;
    }
    
    .legend-dot-green { background: #22c55e; }
    .legend-dot-yellow { background: #eab308; }
    .legend-dot-red { background: #ef4444; }
    .legend-dot-white { background: #ffffff; }
</style>
"""



# ───────────────────────────────────────────────────────────────────
# DATA STRUCTURES
# ───────────────────────────────────────────────────────────────────

@dataclass
class AnalysisConfig:
    """Configuration parameters for swimming analysis."""
    smoothing_window: int = 7
    elbow_min_window: int = 9
    elbow_min_prominence: float = 10.0
    min_stroke_gap_s: float = 0.5
    breath_side_threshold: float = 0.15
    min_breath_gap_s: float = 1.0
    min_breath_hold_frames: int = 4
    ideal_elbow: Tuple[float, float] = (100, 135)
    ideal_knee_underwater: Tuple[float, float] = (120, 160)
    ideal_knee_surface: Tuple[float, float] = (125, 165)
    ideal_roll_range: Tuple[float, float] = (35, 55)
    ideal_stroke_rate: Tuple[float, float] = (55, 65)
    ideal_breathing_rate: Tuple[float, float] = (25, 40)
    ideal_symmetry_max: float = 10.0
    min_detection_confidence: float = 0.6


@dataclass
class FrameMetrics:
    """Metrics collected for each video frame."""
    time_s: float
    elbow_angle: float
    knee_left: float
    knee_right: float
    symmetry: float
    score: float
    yaw_proxy: float
    breath_state: str
    body_roll: float
    phase: str
    detection_confidence: float = 1.0


@dataclass
class TrainingDrill:
    """Represents a recommended training drill."""
    title: str
    description: str
    sets: str
    focus: str


@dataclass
class Recommendation:
    """Represents a technique recommendation."""
    title: str
    description: str
    priority: str  # 'high', 'medium', 'low'


@dataclass
class SessionSummary:
    """Summary statistics for the analyzed swimming session."""
    duration_s: float
    avg_score: float
    avg_symmetry: float
    avg_roll: float
    max_roll_abs: float
    stroke_rate_single: float
    stroke_rate_both: float
    breaths_per_min: float
    breath_count_left: int
    breath_count_right: int
    total_strokes: int
    avg_detection_confidence: float = 1.0
    best_frame_time: Optional[float] = None
    worst_frame_time: Optional[float] = None
    drills: List[TrainingDrill] = field(default_factory=list)
    recommendations: List[Recommendation] = field(default_factory=list)


# ───────────────────────────────────────────────────────────────────
# GEOMETRY & ANALYSIS FUNCTIONS
# ───────────────────────────────────────────────────────────────────

def calculate_angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Calculate angle at point b formed by points a-b-c."""
    ba = a - b
    bc = c - b
    cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
    return np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))


def calculate_deviation(value: float, ideal_range: Tuple[float, float]) -> float:
    """Calculate how far a value is from the ideal range."""
    low, high = ideal_range
    if value < low:
        return low - value
    if value > high:
        return value - high
    return 0.0


def is_in_range(value: float, ideal_range: Tuple[float, float]) -> bool:
    """Check if value is within ideal range."""
    return ideal_range[0] <= value <= ideal_range[1]


def detect_local_minimum(window: List[float], prominence: float = 10.0) -> Tuple[bool, Optional[float]]:
    """Detect if the center of the window is a local minimum."""
    if len(window) < 3:
        return False, None
    
    center = len(window) // 2
    center_val = window[center]
    
    left_valid = all(center_val <= window[i] + prominence for i in range(center))
    right_valid = all(center_val <= window[i] + prominence for i in range(center, len(window)))
    
    return left_valid and right_valid, center_val


def calculate_yaw_proxy(nose: np.ndarray, left_shoulder: np.ndarray, 
                        right_shoulder: np.ndarray) -> float:
    """Calculate head yaw relative to shoulders (breathing indicator)."""
    dx = right_shoulder[0] - left_shoulder[0]
    if abs(dx) < 1e-6:
        return 0.0
    
    expected_nose_x = left_shoulder[0] + dx * 0.5
    return (nose[0] - expected_nose_x) / abs(dx)


def calculate_shoulder_roll(left_shoulder: np.ndarray, right_shoulder: np.ndarray) -> float:
    """Calculate body roll angle from shoulder positions."""
    dy = left_shoulder[1] - right_shoulder[1]
    dx = left_shoulder[0] - right_shoulder[0]
    
    if abs(dx) < 1e-6:
        return 90.0 if dy > 0 else -90.0
    
    return np.degrees(np.arctan2(dy, dx))


def calculate_technique_score(elbow_dev: float, symmetry: float, 
                              knee_left_dev: float, knee_right_dev: float,
                              roll_dev: float = 0.0) -> float:
    """Calculate overall technique score (0-100)."""
    raw_penalty = (
        elbow_dev * 0.35 + 
        symmetry * 0.25 + 
        abs(knee_left_dev - knee_right_dev) * 0.25 +
        roll_dev * 0.15
    )
    return max(0.0, min(100.0, 100.0 - raw_penalty))


def determine_swim_phase(wrist_y: float, shoulder_y: float, elbow_angle: float, 
                         prev_phase: str = 'Recovery') -> str:
    """
    Determine the current swimming phase based on arm position.
    Phases: Entry, Pull, Push, Recovery
    """
    # Wrist below shoulder = underwater phases
    is_underwater = wrist_y > shoulder_y
    
    if is_underwater:
        if elbow_angle > 130:
            return 'Entry'
        elif elbow_angle > 90:
            return 'Pull'
        else:
            return 'Push'
    else:
        return 'Recovery'


# ───────────────────────────────────────────────────────────────────
# VIDEO PROCESSING
# ───────────────────────────────────────────────────────────────────

class SwimAnalyzer:
    """Main class for analyzing swimming technique from video."""
    
    # MediaPipe landmark indices
    NOSE, L_EYE, R_EYE = 0, 2, 5
    L_SHOULDER, R_SHOULDER = 11, 12
    L_ELBOW, R_ELBOW = 13, 14
    L_WRIST, R_WRIST = 15, 16
    L_HIP, R_HIP = 23, 24
    L_KNEE, R_KNEE = 25, 26
    L_ANKLE, R_ANKLE = 27, 28
    
    def __init__(self, config: AnalysisConfig, is_underwater: bool = False):
        self.config = config
        self.is_underwater = is_underwater
        self.detector = self._initialize_detector()
        
        # Results storage
        self.frame_metrics: List[FrameMetrics] = []
        self.stroke_times: List[float] = []
        self.best_frame_bytes: Optional[bytes] = None
        self.worst_frame_bytes: Optional[bytes] = None
        self.best_frame_deviation = float('inf')
        self.worst_frame_deviation = -float('inf')
        
        # Smoothing buffers
        self.elbow_buffer = deque(maxlen=config.smoothing_window)
        self.knee_left_buffer = deque(maxlen=config.smoothing_window)
        self.knee_right_buffer = deque(maxlen=config.smoothing_window)
        self.roll_buffer = deque(maxlen=config.smoothing_window)
        
        # Stroke detection
        self.elbow_window = deque(maxlen=config.elbow_min_window)
        self.time_window = deque(maxlen=config.elbow_min_window)
        
        # Breathing tracking
        self.breath_count_left = 0
        self.breath_count_right = 0
        self.current_breath_side = 'N'
        self.breath_persist_counter = 0
        self.last_breath_time = -1e9
        
        # Phase tracking
        self.current_phase = 'Recovery'
        
        # Detection confidence tracking
        self.confidence_scores: List[float] = []
    
    def _initialize_detector(self) -> vision.PoseLandmarker:
        """Initialize the MediaPipe pose detector."""
        model_path = "pose_landmarker_heavy.task"
        
        if not os.path.exists(model_path):
            st.info("📥 Downloading MediaPipe model (one-time download)...")
            urllib.request.urlretrieve(
                "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
                "pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task",
                model_path
            )
            st.success("✅ Model downloaded successfully!")
        
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=self.config.min_detection_confidence,
            min_pose_presence_confidence=self.config.min_detection_confidence,
            min_tracking_confidence=self.config.min_detection_confidence,
            output_segmentation_masks=False
        )
        
        return vision.PoseLandmarker.create_from_options(options)
    
    def _extract_landmarks(self, landmarks, frame_shape: Tuple[int, int]) -> Dict[str, np.ndarray]:
        """Extract landmark positions as pixel coordinates."""
        height, width = frame_shape
        
        def to_pixel(idx: int) -> np.ndarray:
            lm = landmarks[idx]
            return np.array([lm.x * width, lm.y * height])
        
        def get_visibility(idx: int) -> float:
            return landmarks[idx].visibility if hasattr(landmarks[idx], 'visibility') else 1.0
        
        return {
            'nose': to_pixel(self.NOSE),
            'left_eye': to_pixel(self.L_EYE),
            'right_eye': to_pixel(self.R_EYE),
            'left_shoulder': to_pixel(self.L_SHOULDER),
            'right_shoulder': to_pixel(self.R_SHOULDER),
            'left_elbow': to_pixel(self.L_ELBOW),
            'right_elbow': to_pixel(self.R_ELBOW),
            'left_wrist': to_pixel(self.L_WRIST),
            'right_wrist': to_pixel(self.R_WRIST),
            'left_hip': to_pixel(self.L_HIP),
            'right_hip': to_pixel(self.R_HIP),
            'left_knee': to_pixel(self.L_KNEE),
            'right_knee': to_pixel(self.R_KNEE),
            'left_ankle': to_pixel(self.L_ANKLE),
            'right_ankle': to_pixel(self.R_ANKLE),
            'avg_visibility': (get_visibility(self.L_SHOULDER) + 
                              get_visibility(self.R_SHOULDER) +
                              get_visibility(self.L_HIP) +
                              get_visibility(self.R_HIP)) / 4.0
        }
    
    def _compute_frame_angles(self, lm: Dict) -> Tuple[float, float, float]:
        """Compute joint angles from landmarks."""
        elbow_left = calculate_angle(lm['left_shoulder'], lm['left_elbow'], lm['left_wrist'])
        elbow_right = calculate_angle(lm['right_shoulder'], lm['right_elbow'], lm['right_wrist'])
        knee_left = calculate_angle(lm['left_hip'], lm['left_knee'], lm['left_ankle'])
        knee_right = calculate_angle(lm['right_hip'], lm['right_knee'], lm['right_ankle'])
        
        # Use the arm that's more visible/active (lower elbow angle = more bent = pulling)
        elbow = min(elbow_left, elbow_right)
        
        return elbow, knee_left, knee_right
    
    def _smooth_values(self, elbow: float, knee_left: float, 
                       knee_right: float, roll: float) -> Tuple[float, float, float, float]:
        """Apply smoothing to reduce noise."""
        self.elbow_buffer.append(elbow)
        self.knee_left_buffer.append(knee_left)
        self.knee_right_buffer.append(knee_right)
        self.roll_buffer.append(roll)
        
        return (
            statistics.mean(self.elbow_buffer) if self.elbow_buffer else elbow,
            statistics.mean(self.knee_left_buffer) if self.knee_left_buffer else knee_left,
            statistics.mean(self.knee_right_buffer) if self.knee_right_buffer else knee_right,
            statistics.mean(self.roll_buffer) if self.roll_buffer else roll
        )
    
    def _detect_stroke(self, elbow_angle: float, time_s: float) -> bool:
        """Detect stroke cycles based on elbow angle local minima."""
        self.elbow_window.append(elbow_angle)
        self.time_window.append(time_s)
        
        if len(self.elbow_window) < self.config.elbow_min_window:
            return False
        
        is_minimum, _ = detect_local_minimum(list(self.elbow_window), self.config.elbow_min_prominence)
        
        if not is_minimum:
            return False
        
        center_time = list(self.time_window)[self.config.elbow_min_window // 2]
        
        if not self.stroke_times or center_time - self.stroke_times[-1] >= self.config.min_stroke_gap_s:
            self.stroke_times.append(center_time)
            return True
        
        return False
    
    def _detect_breathing(self, yaw: float, time_s: float) -> str:
        """Detect breathing side from head yaw."""
        if yaw > self.config.breath_side_threshold:
            desired_side = 'R'
        elif yaw < -self.config.breath_side_threshold:
            desired_side = 'L'
        else:
            desired_side = 'N'
        
        if desired_side == self.current_breath_side:
            self.breath_persist_counter += 1
        else:
            self.breath_persist_counter = 1
            self.current_breath_side = desired_side
        
        if (self.current_breath_side in ('L', 'R') and 
            self.breath_persist_counter >= self.config.min_breath_hold_frames and
            time_s - self.last_breath_time >= self.config.min_breath_gap_s):
            
            if self.current_breath_side == 'L':
                self.breath_count_left += 1
            else:
                self.breath_count_right += 1
            
            self.last_breath_time = time_s
        
        return self.current_breath_side
    
    def _deviation_to_color(self, deviation: float) -> Tuple[int, int, int]:
        """Convert deviation to BGR color for visualization."""
        if deviation <= 10:
            return (0, 255, 0)    # Green - Good
        elif deviation <= 20:
            return (0, 255, 255)  # Yellow/Cyan - Fair
        else:
            return (0, 0, 255)    # Red - Needs work
    
    def _draw_skeleton_with_glow(self, frame: np.ndarray, lm: Dict, 
                                  quality_score: float) -> np.ndarray:
        """Draw skeleton overlay with glow effect based on technique quality."""
        annotated = frame.copy()
        height, width = frame.shape[:2]
        
        # Determine main color based on quality
        if quality_score >= 80:
            main_color = (0, 255, 0)    # Green
            glow_color = (0, 200, 0)
        elif quality_score >= 65:
            main_color = (0, 255, 255)  # Cyan/Yellow
            glow_color = (0, 200, 200)
        else:
            main_color = (0, 0, 255)    # Red
            glow_color = (0, 0, 200)
        
        # Calculate deviations for limb-specific coloring
        ideal_knee = self.config.ideal_knee_underwater if self.is_underwater else self.config.ideal_knee_surface
        
        def get_limb_color(deviation: float) -> Tuple[int, int, int]:
            return self._deviation_to_color(deviation)
        
        # Define skeleton connections with their colors
        connections = [
            # Spine
            (lm['nose'], lm['left_shoulder'], main_color),
            (lm['nose'], lm['right_shoulder'], main_color),
            (lm['left_shoulder'], lm['right_shoulder'], main_color),
            (lm['left_shoulder'], lm['left_hip'], main_color),
            (lm['right_shoulder'], lm['right_hip'], main_color),
            (lm['left_hip'], lm['right_hip'], main_color),
            
            # Left arm
            (lm['left_shoulder'], lm['left_elbow'], main_color),
            (lm['left_elbow'], lm['left_wrist'], main_color),
            
            # Right arm
            (lm['right_shoulder'], lm['right_elbow'], main_color),
            (lm['right_elbow'], lm['right_wrist'], main_color),
            
            # Left leg
            (lm['left_hip'], lm['left_knee'], main_color),
            (lm['left_knee'], lm['left_ankle'], main_color),
            
            # Right leg
            (lm['right_hip'], lm['right_knee'], main_color),
            (lm['right_knee'], lm['right_ankle'], main_color),
        ]
        
        # Draw glow effect (thicker, semi-transparent lines)
        overlay = annotated.copy()
        for start, end, color in connections:
            cv2.line(overlay, tuple(start.astype(int)), tuple(end.astype(int)), 
                    glow_color, 8, cv2.LINE_AA)
        cv2.addWeighted(overlay, 0.3, annotated, 0.7, 0, annotated)
        
        # Draw main skeleton lines
        for start, end, color in connections:
            cv2.line(annotated, tuple(start.astype(int)), tuple(end.astype(int)), 
                    color, 3, cv2.LINE_AA)
        
        # Draw joints with white border
        joints = [
            lm['nose'], lm['left_shoulder'], lm['right_shoulder'],
            lm['left_elbow'], lm['right_elbow'], lm['left_wrist'], lm['right_wrist'],
            lm['left_hip'], lm['right_hip'], lm['left_knee'], lm['right_knee'],
            lm['left_ankle'], lm['right_ankle']
        ]
        
        for joint in joints:
            pos = tuple(joint.astype(int))
            # White border
            cv2.circle(annotated, pos, 7, (255, 255, 255), 2, cv2.LINE_AA)
            # Colored fill
            cv2.circle(annotated, pos, 5, main_color, -1, cv2.LINE_AA)
        
        # Larger circle for head
        nose_pos = tuple(lm['nose'].astype(int))
        cv2.circle(annotated, nose_pos, 12, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.circle(annotated, nose_pos, 10, main_color, -1, cv2.LINE_AA)
        
        return annotated
    
    def _draw_info_panel(self, frame: np.ndarray, metrics: FrameMetrics,
                         stroke_rate: float, breathing_rate: float) -> np.ndarray:
        """Draw semi-transparent info panel with metrics."""
        annotated = frame.copy()
        height, width = frame.shape[:2]
        
        # Panel dimensions and position
        panel_x, panel_y = 20, 20
        panel_w, panel_h = 240, 160
        
        # Determine panel border color based on score
        if metrics.score >= 80:
            border_color = (0, 255, 0)
        elif metrics.score >= 65:
            border_color = (0, 255, 255)
        else:
            border_color = (0, 0, 255)
        
        # Draw semi-transparent panel background
        overlay = annotated.copy()
        cv2.rectangle(overlay, (panel_x, panel_y), 
                     (panel_x + panel_w, panel_y + panel_h), 
                     (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.75, annotated, 0.25, 0, annotated)
        
        # Draw panel border
        cv2.rectangle(annotated, (panel_x, panel_y), 
                     (panel_x + panel_w, panel_y + panel_h), 
                     border_color, 2, cv2.LINE_AA)
        
        # Text content
        font = cv2.FONT_HERSHEY_SIMPLEX
        text_x = panel_x + 15
        text_y = panel_y + 30
        line_height = 26
        
        # Phase
        cv2.putText(annotated, f"Phase: {metrics.phase}", (text_x, text_y),
                   font, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        
        # Score
        text_y += line_height
        cv2.putText(annotated, f"Score: {int(metrics.score)}/100", (text_x, text_y),
                   font, 0.7, border_color, 2, cv2.LINE_AA)
        
        # Stroke rate
        text_y += line_height
        cv2.putText(annotated, f"Stroke Rate: {stroke_rate:.1f} spm", (text_x, text_y),
                   font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        
        # Breathing rate
        text_y += line_height - 4
        cv2.putText(annotated, f"Breathing: {breathing_rate:.1f}/min", (text_x, text_y),
                   font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        
        # Body roll
        text_y += line_height - 4
        cv2.putText(annotated, f"Body Roll: {abs(metrics.body_roll):.1f} deg", (text_x, text_y),
                   font, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
        
        # Timestamp
        time_str = f"Time: {metrics.time_s:.2f}s"
        cv2.putText(annotated, time_str, (panel_x + panel_w - 100, panel_y + panel_h - 10),
                   font, 0.4, (180, 180, 180), 1, cv2.LINE_AA)
        
        return annotated
    
    def _annotate_frame(self, frame: np.ndarray, lm: Dict, metrics: FrameMetrics,
                        stroke_rate: float, breathing_rate: float) -> np.ndarray:
        """Create fully annotated frame with skeleton and info panel."""
        # Draw skeleton with glow
        annotated = self._draw_skeleton_with_glow(frame, lm, metrics.score)
        
        # Draw info panel
        annotated = self._draw_info_panel(annotated, metrics, stroke_rate, breathing_rate)
        
        return annotated
    
    def process_video(self, input_path: str, output_path: str, 
                      progress_callback=None) -> SessionSummary:
        """Process video and generate analysis."""
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise RuntimeError("Cannot open video file")
        
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), 
                                fps, (width, height))
        
        frame_id = 0
        ideal_knee = (self.config.ideal_knee_underwater if self.is_underwater 
                     else self.config.ideal_knee_surface)
        
        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_id += 1
                if progress_callback and frame_id % 5 == 0:
                    progress_callback(min(frame_id / max(total_frames, 1), 1.0))
                
                time_s = frame_id / fps
                time_ms = int(time_s * 1000)
                
                # Run pose detection
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB,
                                   data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                result = self.detector.detect_for_video(mp_image, time_ms)
                
                if not result.pose_landmarks:
                    writer.write(frame)
                    continue
                
                # Extract landmarks
                lm = self._extract_landmarks(result.pose_landmarks[0], (height, width))
                detection_conf = lm.get('avg_visibility', 1.0)
                self.confidence_scores.append(detection_conf)
                
                # Compute angles
                elbow, knee_left, knee_right = self._compute_frame_angles(lm)
                body_roll = calculate_shoulder_roll(lm['left_shoulder'], lm['right_shoulder'])
                
                # Smooth values
                elbow_s, knee_left_s, knee_right_s, roll_s = self._smooth_values(
                    elbow, knee_left, knee_right, body_roll)
                
                # Head yaw for breathing detection
                yaw = calculate_yaw_proxy(lm['nose'], lm['left_shoulder'], lm['right_shoulder'])
                
                # Calculate deviations
                elbow_dev = calculate_deviation(elbow_s, self.config.ideal_elbow)
                knee_l_dev = calculate_deviation(knee_left_s, ideal_knee)
                knee_r_dev = calculate_deviation(knee_right_s, ideal_knee)
                roll_dev = calculate_deviation(abs(roll_s), self.config.ideal_roll_range)
                symmetry = abs(knee_left_s - knee_right_s)
                
                # Calculate score
                score = calculate_technique_score(elbow_dev, symmetry, 
                                                  knee_l_dev, knee_r_dev, roll_dev)
                
                # Determine phase
                self.current_phase = determine_swim_phase(
                    lm['left_wrist'][1], lm['left_shoulder'][1], 
                    elbow_s, self.current_phase)
                
                # Track best/worst frames during pull phase
                if self.current_phase == 'Pull':
                    elbow_ideal_center = statistics.mean(self.config.ideal_elbow)
                    current_dev = abs(elbow_s - elbow_ideal_center)
                    
                    if current_dev < self.best_frame_deviation:
                        self.best_frame_deviation = current_dev
                        _, buffer = cv2.imencode('.jpg', frame)
                        self.best_frame_bytes = buffer.tobytes()
                    
                    if current_dev > self.worst_frame_deviation:
                        self.worst_frame_deviation = current_dev
                        _, buffer = cv2.imencode('.jpg', frame)
                        self.worst_frame_bytes = buffer.tobytes()
                
                # Detect stroke and breathing
                self._detect_stroke(elbow_s, time_s)
                breath_state = self._detect_breathing(yaw, time_s)
                
                # Create metrics
                metrics = FrameMetrics(
                    time_s=time_s,
                    elbow_angle=elbow_s,
                    knee_left=knee_left_s,
                    knee_right=knee_right_s,
                    symmetry=symmetry,
                    score=score,
                    yaw_proxy=yaw,
                    breath_state=breath_state,
                    body_roll=roll_s,
                    phase=self.current_phase,
                    detection_confidence=detection_conf
                )
                self.frame_metrics.append(metrics)
                
                # Calculate current rates
                stroke_rate = self._calculate_current_stroke_rate()
                breathing_rate = self._calculate_current_breathing_rate(time_s)
                
                # Annotate and write frame
                annotated = self._annotate_frame(frame, lm, metrics, stroke_rate, breathing_rate)
                writer.write(annotated)
        
        finally:
            cap.release()
            writer.release()
        
        return self._generate_summary()
    
    def _calculate_current_stroke_rate(self) -> float:
        """Calculate strokes per minute."""
        if len(self.stroke_times) < 2:
            return 0.0
        duration = self.stroke_times[-1] - self.stroke_times[0]
        if duration < 0.1:
            return 0.0
        return 60.0 * (len(self.stroke_times) - 1) / duration
    
    def _calculate_current_breathing_rate(self, current_time: float) -> float:
        """Calculate breaths per minute."""
        total_breaths = self.breath_count_left + self.breath_count_right
        minutes = current_time / 60.0
        return total_breaths / max(minutes, 1e-6)
    
    def _generate_drills(self, summary_data: dict) -> List[TrainingDrill]:
        """Generate personalized training drills."""
        drills = []
        
        # Always include some core drills
        drills.append(TrainingDrill(
            title="High Elbow Catch",
            description="Fingertip drag drill. Keep elbow high during recovery, "
                       "focus on early vertical forearm entry.",
            sets="4 x 50m",
            focus="Early vertical forearm"
        ))
        
        if summary_data['symmetry'] > 10:
            drills.append(TrainingDrill(
                title="Leg Symmetry Work",
                description="Single-arm freestyle alternating every 25m. "
                           "Focus on maintaining even kick tempo.",
                sets="8 x 25m",
                focus="Even propulsion"
            ))
        
        if summary_data['breath_ratio'] < 0.6:  # Asymmetric breathing
            drills.append(TrainingDrill(
                title="Breathing Pattern Correction",
                description="Practice bilateral breathing every 3 strokes for 200m. "
                           "Maintain head position neutral during non-breath strokes.",
                sets="4 x 50m",
                focus="Balance and symmetry"
            ))
        
        if summary_data['max_roll'] > 55:
            drills.append(TrainingDrill(
                title="Body Roll Control Drill",
                description="6-kick switch drill. Push off wall on side, 6 kicks, "
                           "then rotate to other side. Aim for 45° maximum roll.",
                sets="6 x 25m",
                focus="Maintaining 45° roll angle"
            ))
        
        if summary_data['stroke_rate'] < 50:
            drills.append(TrainingDrill(
                title="Tempo Training",
                description="Use tempo trainer at 1.2-1.4 sec/stroke. "
                           "Focus on quick hand entry and catch.",
                sets="4 x 100m",
                focus="Increasing stroke rate"
            ))
        
        return drills[:4]  # Return top 4 drills
    
    def _generate_recommendations(self, summary_data: dict) -> List[Recommendation]:
        """Generate prioritized recommendations."""
        recs = []
        
        # Check breathing pattern
        if summary_data['breath_ratio'] < 0.6:
            dominant = "left" if summary_data['breaths_left'] > summary_data['breaths_right'] else "right"
            recs.append(Recommendation(
                title="Breathing Pattern Asymmetry",
                description=f"You favor {dominant}-side breathing "
                           f"({summary_data['breaths_left']} vs {summary_data['breaths_right']}). "
                           "Practice bilateral breathing to balance body rotation.",
                priority="high"
            ))
        
        # Check symmetry
        if summary_data['symmetry'] > 15:
            recs.append(Recommendation(
                title="Leg Symmetry Needs Work",
                description=f"Average leg angle difference is {summary_data['symmetry']:.1f}°. "
                           "Focus on maintaining consistent kick depth and tempo.",
                priority="high"
            ))
        elif summary_data['symmetry'] <= 10:
            recs.append(Recommendation(
                title="Excellent Leg Symmetry",
                description="Kick consistency is within ideal range. "
                           "Maintain current kick technique.",
                priority="low"
            ))
        
        # Check body roll
        if summary_data['max_roll'] > 55:
            recs.append(Recommendation(
                title="Excessive Body Roll",
                description=f"Maximum roll of {summary_data['max_roll']:.1f}° exceeds ideal (35-55°). "
                           "Work on core stability during rotation.",
                priority="high"
            ))
        elif 35 <= summary_data['avg_roll'] <= 55:
            recs.append(Recommendation(
                title="Excellent Body Roll",
                description="Body rotation is within ideal range (35-55°). "
                           "Good hip-driven rotation.",
                priority="low"
            ))
        
        # Check score
        if summary_data['score'] >= 80:
            recs.append(Recommendation(
                title="Strong Technique Foundation",
                description=f"Score of {summary_data['score']:.0f}/100 indicates solid fundamentals. "
                           "Focus on race-pace work and endurance.",
                priority="medium"
            ))
        elif summary_data['score'] < 70:
            recs.append(Recommendation(
                title="Technique Development Needed",
                description=f"Score of {summary_data['score']:.0f}/100 suggests room for improvement. "
                           "Consider working with a coach on fundamentals.",
                priority="high"
            ))
        
        return recs
    
    def _generate_summary(self) -> SessionSummary:
        """Generate session summary with drills and recommendations."""
        if not self.frame_metrics:
            raise ValueError("No metrics to summarize")
        
        duration = self.frame_metrics[-1].time_s
        scores = [m.score for m in self.frame_metrics]
        symmetries = [m.symmetry for m in self.frame_metrics]
        rolls = [m.body_roll for m in self.frame_metrics]
        
        stroke_rate_single = self._calculate_current_stroke_rate()
        stroke_rate_both = 2.0 * stroke_rate_single
        breathing_rate = self._calculate_current_breathing_rate(duration)
        
        avg_score = statistics.mean(scores)
        avg_symmetry = statistics.mean(symmetries)
        avg_roll = abs(statistics.mean(rolls))
        max_roll = max(abs(r) for r in rolls)
        
        total_breaths = self.breath_count_left + self.breath_count_right
        breath_ratio = (min(self.breath_count_left, self.breath_count_right) / 
                       max(total_breaths, 1))
        
        avg_confidence = (statistics.mean(self.confidence_scores) 
                         if self.confidence_scores else 1.0)
        
        # Data for generating drills and recommendations
        summary_data = {
            'score': avg_score,
            'symmetry': avg_symmetry,
            'avg_roll': avg_roll,
            'max_roll': max_roll,
            'stroke_rate': stroke_rate_single,
            'breath_ratio': breath_ratio,
            'breaths_left': self.breath_count_left,
            'breaths_right': self.breath_count_right
        }
        
        drills = self._generate_drills(summary_data)
        recommendations = self._generate_recommendations(summary_data)
        
        return SessionSummary(
            duration_s=duration,
            avg_score=avg_score,
            avg_symmetry=avg_symmetry,
            avg_roll=avg_roll,
            max_roll_abs=max_roll,
            stroke_rate_single=stroke_rate_single,
            stroke_rate_both=stroke_rate_both,
            breaths_per_min=breathing_rate,
            breath_count_left=self.breath_count_left,
            breath_count_right=self.breath_count_right,
            total_strokes=len(self.stroke_times),
            avg_detection_confidence=avg_confidence,
            drills=drills,
            recommendations=recommendations
        )


# ───────────────────────────────────────────────────────────────────
# VISUALIZATION & REPORTING
# ───────────────────────────────────────────────────────────────────

def generate_plots(analyzer: SwimAnalyzer, config: AnalysisConfig) -> io.BytesIO:
    """Generate analysis plots."""
    metrics = analyzer.frame_metrics
    times = [m.time_s for m in metrics]
    
    # Set style
    plt.style.use('dark_background')
    fig = plt.figure(figsize=(14, 12))
    gs = fig.add_gridspec(5, 1, hspace=0.4)
    
    # Colors
    cyan = '#06b6d4'
    green = '#22c55e'
    red = '#ef4444'
    yellow = '#eab308'
    purple = '#a855f7'
    blue = '#3b82f6'
    
    # Plot 1: Joint Angles
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(times, [m.elbow_angle for m in metrics], label="Elbow", 
             color=cyan, linewidth=1.5)
    ax1.plot(times, [m.knee_left for m in metrics], label="Knee L", 
             color=green, linewidth=1.5)
    ax1.plot(times, [m.knee_right for m in metrics], label="Knee R", 
             color=red, linewidth=1.5)
    ax1.axhspan(config.ideal_elbow[0], config.ideal_elbow[1], 
                alpha=0.15, color=cyan, label='Ideal Elbow')
    ax1.set_ylabel("Angle (°)", color='white')
    ax1.set_title("Joint Angles Over Time", fontweight='bold', color='white')
    ax1.legend(loc="upper right", framealpha=0.9)
    ax1.grid(True, alpha=0.2)
    ax1.set_facecolor('#1e293b')
    
    # Plot 2: Score & Symmetry
    ax2 = fig.add_subplot(gs[1, 0])
    ax2_twin = ax2.twinx()
    ax2.plot(times, [m.score for m in metrics], label="Score", 
             color=green, linewidth=1.5)
    ax2_twin.plot(times, [m.symmetry for m in metrics], label="Symmetry", 
                  color=yellow, linewidth=1.5)
    ax2.set_ylabel("Score (0-100)", color=green)
    ax2_twin.set_ylabel("Symmetry (°)", color=yellow)
    ax2.set_title("Technique Score & Leg Symmetry", fontweight='bold', color='white')
    ax2.legend(loc="upper left")
    ax2_twin.legend(loc="upper right")
    ax2.grid(True, alpha=0.2)
    ax2.set_facecolor('#1e293b')
    
    # Plot 3: Body Roll
    ax3 = fig.add_subplot(gs[2, 0])
    ax3.plot(times, [m.body_roll for m in metrics], label="Body Roll", 
             color=purple, linewidth=1.5)
    ax3.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax3.axhspan(-config.ideal_roll_range[1], config.ideal_roll_range[1], 
                color=green, alpha=0.1, label='Ideal Range')
    ax3.set_ylabel("Roll Angle (°)", color='white')
    ax3.set_title("Body Roll Analysis", fontweight='bold', color='white')
    ax3.legend()
    ax3.grid(True, alpha=0.2)
    ax3.set_facecolor('#1e293b')
    
    # Plot 4: Head Yaw (Breathing)
    ax4 = fig.add_subplot(gs[3, 0])
    ax4.plot(times, [m.yaw_proxy for m in metrics], label="Head Yaw", 
             color='#ec4899', linewidth=1.5)
    ax4.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax4.axhline(config.breath_side_threshold, color=blue, 
                linestyle=':', alpha=0.7, label='Breath Threshold')
    ax4.axhline(-config.breath_side_threshold, color=blue, 
                linestyle=':', alpha=0.7)
    ax4.set_ylabel("Yaw Proxy", color='white')
    ax4.set_title("Head Rotation Pattern (Breathing)", fontweight='bold', color='white')
    ax4.legend()
    ax4.grid(True, alpha=0.2)
    ax4.set_facecolor('#1e293b')
    
    # Plot 5: Events Timeline
    ax5 = fig.add_subplot(gs[4, 0])
    ax5.plot(times, [m.score for m in metrics], color=green, 
             alpha=0.4, linewidth=1)
    
    # Mark strokes
    for stroke_time in analyzer.stroke_times:
        ax5.axvline(stroke_time, color=cyan, linestyle="--", 
                    alpha=0.7, linewidth=1)
    
    # Mark breaths
    last_state = 'N'
    for m in metrics:
        if m.breath_state in ('L', 'R') and m.breath_state != last_state:
            color = "#ff9500" if m.breath_state == 'L' else blue
            ax5.axvline(m.time_s, color=color, linestyle=":", 
                        alpha=0.8, linewidth=1)
        last_state = m.breath_state
    
    ax5.set_xlabel("Time (seconds)", fontweight='bold', color='white')
    ax5.set_ylabel("Score", color='white')
    ax5.set_title("Events Timeline (strokes: cyan | breaths L: orange, R: blue)", 
                  fontweight='bold', color='white')
    ax5.set_ylim(0, 110)
    ax5.grid(True, alpha=0.2)
    ax5.set_facecolor('#1e293b')
    
    fig.patch.set_facecolor('#0f172a')
    
    buffer = io.BytesIO()
    plt.savefig(buffer, format="png", dpi=150, bbox_inches="tight", 
                facecolor='#0f172a', edgecolor='none')
    plt.close(fig)
    buffer.seek(0)
    return buffer


def generate_pdf_report(analyzer: SwimAnalyzer, summary: SessionSummary, 
                        config: AnalysisConfig, filename: str, 
                        plot_buffer: Optional[io.BytesIO] = None) -> io.BytesIO:
    """Generate PDF report."""
    buffer = io.BytesIO()
    pdf = SimpleDocTemplate(buffer, pagesize=letter, 
                           topMargin=0.5*inch, bottomMargin=0.5*inch)
    
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name='CustomTitle', parent=styles['Title'], 
                              fontSize=24, textColor=colors.HexColor('#06b6d4'), 
                              spaceAfter=20))
    styles.add(ParagraphStyle(name='SectionHeader', parent=styles['Heading2'], 
                              fontSize=14, textColor=colors.HexColor('#22c55e'), 
                              spaceBefore=15, spaceAfter=10))
    styles.add(ParagraphStyle(name='DrillTitle', parent=styles['Normal'],
                              fontSize=11, textColor=colors.HexColor('#06b6d4'),
                              fontName='Helvetica-Bold'))
    
    story = []
    
    # Title
    story.append(Paragraph("🏊 Freestyle Swimming Technique Analysis", styles['CustomTitle']))
    story.append(Spacer(1, 0.2*inch))
    
    # Session Info
    story.append(Paragraph("Session Information", styles['SectionHeader']))
    session_data = [
        ['Video File:', filename],
        ['Duration:', f"{summary.duration_s:.1f} seconds ({summary.duration_s/60:.1f} minutes)"],
        ['Total Strokes:', str(summary.total_strokes)],
        ['Detection Quality:', f"{summary.avg_detection_confidence*100:.1f}%"],
        ['Analysis Date:', datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")]
    ]
    session_table = Table(session_data, colWidths=[2*inch, 4*inch])
    session_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#1e293b')),
        ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey)
    ]))
    story.append(session_table)
    story.append(Spacer(1, 0.3*inch))
    
    # Performance Metrics
    story.append(Paragraph("Performance Metrics", styles['SectionHeader']))
    
    def status_cell(value, ideal_range, is_max=False):
        if is_max:
            in_range = value <= ideal_range[1]
        else:
            in_range = ideal_range[0] <= value <= ideal_range[1]
        return '✓ Good' if in_range else '⚠ Check'
    
    metrics_data = [
        ['Metric', 'Value', 'Ideal Range', 'Status'],
        ['Overall Technique Score', f"{summary.avg_score:.1f} / 100", '70+', 
         '✓ Good' if summary.avg_score >= 70 else '⚠ Needs Work'],
        ['Stroke Rate (single arm)', f"{summary.stroke_rate_single:.1f} spm", 
         f"{config.ideal_stroke_rate[0]}-{config.ideal_stroke_rate[1]}",
         status_cell(summary.stroke_rate_single, config.ideal_stroke_rate)],
        ['Breathing Rate', f"{summary.breaths_per_min:.1f} /min",
         f"{config.ideal_breathing_rate[0]}-{config.ideal_breathing_rate[1]}",
         status_cell(summary.breaths_per_min, config.ideal_breathing_rate)],
        ['Left / Right Breaths', f"{summary.breath_count_left} / {summary.breath_count_right}", 
         'Balanced', '✓ Good' if abs(summary.breath_count_left - summary.breath_count_right) <= 5 else '⚠ Asymmetric'],
        ['Leg Symmetry', f"{summary.avg_symmetry:.1f}°", f"< {config.ideal_symmetry_max}°",
         '✓ Good' if summary.avg_symmetry < config.ideal_symmetry_max else '⚠ Check'],
        ['Avg Body Roll', f"{summary.avg_roll:.1f}°", 
         f"{config.ideal_roll_range[0]}-{config.ideal_roll_range[1]}°",
         status_cell(summary.avg_roll, config.ideal_roll_range)],
        ['Max Body Roll', f"{summary.max_roll_abs:.1f}°", f"< {config.ideal_roll_range[1]}°",
         '✓ Good' if summary.max_roll_abs <= config.ideal_roll_range[1] else '⚠ Excessive'],
    ]
    
    metrics_table = Table(metrics_data, colWidths=[2.2*inch, 1.5*inch, 1.5*inch, 1.3*inch])
    metrics_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#06b6d4')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 10),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 10),
        ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#f1f5f9')),
        ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#94a3b8')),
        ('FONTSIZE', (0, 1), (-1, -1), 9),
    ]))
    story.append(metrics_table)
    story.append(Spacer(1, 0.3*inch))
    
    # Training Drills
    if summary.drills:
        story.append(Paragraph("Recommended Training Drills", styles['SectionHeader']))
        for i, drill in enumerate(summary.drills, 1):
            story.append(Paragraph(f"{i}. {drill.title}", styles['DrillTitle']))
            story.append(Paragraph(drill.description, styles['Normal']))
            story.append(Paragraph(f"<i>Sets: {drill.sets} | Focus: {drill.focus}</i>", 
                                  styles['Normal']))
            story.append(Spacer(1, 0.1*inch))
    
    # Recommendations
    if summary.recommendations:
        story.append(Paragraph("Technique Recommendations", styles['SectionHeader']))
        for rec in summary.recommendations:
            priority_color = {'high': '#ef4444', 'medium': '#eab308', 'low': '#22c55e'}
            story.append(Paragraph(
                f"<b>[{rec.priority.upper()}]</b> {rec.title}", styles['Normal']))
            story.append(Paragraph(rec.description, styles['Normal']))
            story.append(Spacer(1, 0.1*inch))
    
    # Charts
    if plot_buffer and plot_buffer.getvalue():
        story.append(PageBreak())
        story.append(Paragraph("Time-Series Analysis", styles['SectionHeader']))
        story.append(Spacer(1, 0.1*inch))
        plot_image = RLImage(plot_buffer)
        plot_image.drawWidth = 7*inch
        plot_image.drawHeight = 6*inch
        story.append(plot_image)
    
    pdf.build(story)
    buffer.seek(0)
    return buffer


def export_to_csv(analyzer: SwimAnalyzer) -> io.BytesIO:
    """Export frame metrics to CSV."""
    data = {
        'time_s': [m.time_s for m in analyzer.frame_metrics],
        'elbow_angle_deg': [m.elbow_angle for m in analyzer.frame_metrics],
        'knee_left_deg': [m.knee_left for m in analyzer.frame_metrics],
        'knee_right_deg': [m.knee_right for m in analyzer.frame_metrics],
        'symmetry_deg': [m.symmetry for m in analyzer.frame_metrics],
        'score': [m.score for m in analyzer.frame_metrics],
        'yaw_proxy': [m.yaw_proxy for m in analyzer.frame_metrics],
        'breath_state': [m.breath_state for m in analyzer.frame_metrics],
        'body_roll_deg': [m.body_roll for m in analyzer.frame_metrics],
        'phase': [m.phase for m in analyzer.frame_metrics],
        'detection_confidence': [m.detection_confidence for m in analyzer.frame_metrics]
    }
    df = pd.DataFrame(data)
    buffer = io.BytesIO()
    df.to_csv(buffer, index=False, float_format="%.3f")
    buffer.seek(0)
    return buffer


def create_results_bundle(video_path: str, csv_buffer: io.BytesIO, 
                          pdf_buffer: io.BytesIO, plot_buffer: Optional[io.BytesIO],
                          timestamp: str) -> io.BytesIO:
    """Create ZIP bundle of all results."""
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zipf:
        with open(video_path, 'rb') as f:
            zipf.writestr(f"annotated_swim_{timestamp}.mp4", f.read())
        zipf.writestr(f"swim_data_{timestamp}.csv", csv_buffer.getvalue())
        zipf.writestr(f"swim_report_{timestamp}.pdf", pdf_buffer.getvalue())
        if plot_buffer and plot_buffer.getvalue():
            zipf.writestr(f"analysis_charts_{timestamp}.png", plot_buffer.getvalue())
    zip_buffer.seek(0)
    return zip_buffer

# ───────────────────────────────────────────────────────────────────
# STREAMLIT UI COMPONENTS (FIXED – valid Python)
# ───────────────────────────────────────────────────────────────────

def render_metric_card(label: str, value: str, ideal: str,
                       in_range: bool, icon: str = "📊"):
    status_class = "metric-card-green" if in_range else "metric-card-red"
    status_icon = "✓" if in_range else "⚠"

    st.markdown(f"""
    <div class="metric-card {status_class}">
        <div style="font-size: 24px; margin-bottom: 8px;">{icon}</div>
        <div style="color: #94a3b8; font-size: 14px;">{label}</div>
        <div style="color: white; font-size: 28px; font-weight: bold;">{value}</div>
        <div style="color: #64748b; font-size: 12px;">Ideal: {ideal} {status_icon}</div>
    </div>
    """, unsafe_allow_html=True)


def render_score_card(score: float):
    st.markdown(f"""
    <div class="score-card">
        <div style="display:flex;justify-content:space-between;align-items:center;">
            <h2 style="margin:0;font-size:24px;">Overall Technique Score</h2>
            <div style="font-size:48px;font-weight:bold;">
                {score:.0f}<span style="font-size:24px;opacity:0.75;">/100</span>
            </div>
        </div>
        <div style="background:rgba(255,255,255,0.2);border-radius:8px;height:12px;margin-top:16px;">
            <div style="background:white;width:{score}%;height:12px;border-radius:8px;"></div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_drill_card(drill: TrainingDrill, index: int):
    st.markdown(f"""
    <div class="drill-card">
        <div style="display:flex;gap:12px;">
            <div style="background:#06b6d4;color:white;width:28px;height:28px;
                        border-radius:50%;display:flex;align-items:center;
                        justify-content:center;font-weight:bold;">
                {index}
            </div>
            <div>
                <div style="color:white;font-weight:600;font-size:16px;">{drill.title}</div>
                <div style="color:#94a3b8;font-size:14px;margin-top:4px;">{drill.description}</div>
                <div style="color:#64748b;font-size:12px;margin-top:8px;">
                    {drill.sets} • {drill.focus}
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_recommendation_card(rec: Recommendation):
    rec_class = f"rec-{rec.priority}"
    colors = {"high": "#ef4444", "medium": "#eab308", "low": "#22c55e"}

    st.markdown(f"""
    <div class="{rec_class}">
        <div style="color:white;font-weight:600;font-size:16px;">{rec.title}</div>
        <div style="color:#94a3b8;font-size:14px;margin-top:4px;">{rec.description}</div>
        <span style="display:inline-block;margin-top:8px;padding:4px 8px;
                     font-size:11px;border-radius:4px;
                     background:{colors[rec.priority]}33;
                     color:{colors[rec.priority]};">
            {rec.priority.upper()}
        </span>
    </div>
    """, unsafe_allow_html=True)


def render_legend():
    st.markdown("""
    <div style="display:flex;flex-wrap:wrap;gap:8px;margin-top:12px;">
        <div class="legend-item">
            <div class="legend-dot legend-dot-green"></div>
            <span>Good Form</span>
        </div>
        <div class="legend-item">
            <div class="legend-dot legend-dot-yellow"></div>
            <span>Fair Form</span>
        </div>
        <div class="legend-item">
            <div class="legend-dot legend-dot-red"></div>
            <span>Needs Work</span>
        </div>
        <div class="legend-item">
            <div class="legend-dot legend-dot-white"></div>
            <span>Joint Markers</span>
        </div>
    </div>
    """, unsafe_allow_html=True)


def main():
    st.set_page_config(
        page_title="Freestyle Swimming Analyzer Pro",
        page_icon="🏊",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    st.markdown("""
    <div style="text-align:center;padding:20px 0;">
        <div style="display:flex;justify-content:center;gap:12px;">
            <span style="font-size:40px;">🏊</span>
            <h1 style="
                margin:0;
                font-size:36px;
                background:linear-gradient(135deg,#06b6d4,#3b82f6);
                -webkit-background-clip:text;
                -webkit-text-fill-color:transparent;
            ">
                Freestyle Swim Analyzer Pro
            </h1>
        </div>
        <p style="color:#94a3b8;">
            AI-powered swimming technique analysis
        </p>
    </div>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Upload your swimming video",
        type=["mp4", "mov", "avi", "mkv"]
    )

    if uploaded_file is None:
        st.markdown("""
        <div style="
            background:rgba(30,41,59,0.7);
            border-radius:16px;
            padding:40px;
            text-align:center;
            border:2px dashed #06b6d4;
        ">
            <div style="font-size:48px;">📤</div>
            <h3 style="color:white;">Upload Your Swimming Video</h3>
            <p style="color:#94a3b8;">MP4, MOV, AVI, MKV supported</p>
        </div>
        """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
