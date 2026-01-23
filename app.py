import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import os
import datetime
import statistics
from collections import deque
from dataclasses import dataclass
from typing import List, Tuple, Optional
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


# ───────────────────────────────────────────────────────────────────
# DATA STRUCTURES
# ───────────────────────────────────────────────────────────────────

@dataclass
class AnalysisConfig:
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
    ideal_roll_abs_max: float = 55.0
    min_detection_confidence: float = 0.6


@dataclass
class FrameMetrics:
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


@dataclass
class SessionSummary:
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
    best_frame_time: Optional[float] = None
    worst_frame_time: Optional[float] = None


# ───────────────────────────────────────────────────────────────────
# GEOMETRY & ANALYSIS FUNCTIONS
# ───────────────────────────────────────────────────────────────────

def calculate_angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    ba = a - b
    bc = c - b
    cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
    return np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))


def calculate_deviation(value: float, ideal_range: Tuple[float, float]) -> float:
    low, high = ideal_range
    if value < low:
        return low - value
    if value > high:
        return value - high
    return 0.0


def detect_local_minimum(window: List[float], prominence: float = 10.0) -> Tuple[bool, Optional[float]]:
    if len(window) < 3:
        return False, None
    
    center = len(window) // 2
    center_val = window[center]
    
    left_valid = all(center_val <= window[i] + prominence for i in range(center))
    right_valid = all(center_val <= window[i] + prominence for i in range(center, len(window)))
    
    return left_valid and right_valid, center_val


def calculate_yaw_proxy(nose: np.ndarray, left_shoulder: np.ndarray, 
                        right_shoulder: np.ndarray) -> float:
    dx = right_shoulder[0] - left_shoulder[0]
    if abs(dx) < 1e-6:
        return 0.0
    
    expected_nose_x = left_shoulder[0] + dx * 0.5
    return (nose[0] - expected_nose_x) / abs(dx)


def calculate_shoulder_roll(left_shoulder: np.ndarray, right_shoulder: np.ndarray) -> float:
    dy = left_shoulder[1] - right_shoulder[1]
    dx = left_shoulder[0] - right_shoulder[0]
    
    if abs(dx) < 1e-6:
        return 90.0 if dy > 0 else -90.0
    
    return np.degrees(np.arctan2(dy, dx))


def calculate_technique_score(elbow_dev: float, symmetry: float, 
                              knee_left_dev: float, knee_right_dev: float) -> float:
    raw_penalty = (
        elbow_dev * 0.4 + 
        symmetry * 0.3 + 
        abs(knee_left_dev - knee_right_dev) * 0.3
    )
    return max(0.0, min(100.0, 100.0 - raw_penalty))


# ───────────────────────────────────────────────────────────────────
# VIDEO PROCESSING
# ───────────────────────────────────────────────────────────────────

class SwimAnalyzer:
    # MediaPipe landmark indices
    NOSE, L_SHOULDER, R_SHOULDER = 0, 11, 12
    L_ELBOW, L_WRIST = 13, 15
    L_HIP, L_KNEE, L_ANKLE = 23, 25, 27
    R_HIP, R_KNEE, R_ANKLE = 24, 26, 28
    
    def __init__(self, config: AnalysisConfig, is_underwater: bool = False):
        self.config = config
        self.is_underwater = is_underwater
        self.detector = self._initialize_detector()
        
        self.frame_metrics: List[FrameMetrics] = []
        self.stroke_times: List[float] = []
        self.best_frame_bytes: Optional[bytes] = None
        self.worst_frame_bytes: Optional[bytes] = None
        self.best_frame_deviation = float('inf')
        self.worst_frame_deviation = -float('inf')
        
        self.elbow_buffer = deque(maxlen=config.smoothing_window)
        self.knee_left_buffer = deque(maxlen=config.smoothing_window)
        self.knee_right_buffer = deque(maxlen=config.smoothing_window)
        
        self.elbow_window = deque(maxlen=config.elbow_min_window)
        self.time_window = deque(maxlen=config.elbow_min_window)
        
        self.breath_count_left = 0
        self.breath_count_right = 0
        self.current_breath_side = 'N'
        self.breath_persist_counter = 0
        self.last_breath_time = -1e9
    
    def _initialize_detector(self) -> vision.PoseLandmarker:
        model_path = "pose_landmarker_heavy.task"
        
        if not os.path.exists(model_path):
            st.info("Downloading MediaPipe model (one-time download)...")
            urllib.request.urlretrieve(
                "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
                "pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task",
                model_path
            )
        
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=self.config.min_detection_confidence,
            min_pose_presence_confidence=self.config.min_detection_confidence,
            min_tracking_confidence=self.config.min_detection_confidence
        )
        
        return vision.PoseLandmarker.create_from_options(options)
    
    def _extract_landmarks(self, landmarks, frame_shape: Tuple[int, int]) -> dict:
        height, width = frame_shape
        
        def to_pixel(idx: int) -> np.ndarray:
            lm = landmarks[idx]
            return np.array([lm.x * width, lm.y * height])
        
        return {
            'nose': to_pixel(self.NOSE),
            'left_shoulder': to_pixel(self.L_SHOULDER),
            'right_shoulder': to_pixel(self.R_SHOULDER),
            'left_elbow': to_pixel(self.L_ELBOW),
            'left_wrist': to_pixel(self.L_WRIST),
            'left_hip': to_pixel(self.L_HIP),
            'left_knee': to_pixel(self.L_KNEE),
            'left_ankle': to_pixel(self.L_ANKLE),
            'right_hip': to_pixel(self.R_HIP),
            'right_knee': to_pixel(self.R_KNEE),
            'right_ankle': to_pixel(self.R_ANKLE)
        }
    
    def _compute_frame_angles(self, lm: dict) -> Tuple[float, float, float]:
        elbow = calculate_angle(lm['left_shoulder'], lm['left_elbow'], lm['left_wrist'])
        knee_left = calculate_angle(lm['left_hip'], lm['left_knee'], lm['left_ankle'])
        knee_right = calculate_angle(lm['right_hip'], lm['right_knee'], lm['right_ankle'])
        return elbow, knee_left, knee_right
    
    def _smooth_angles(self, elbow: float, knee_left: float, knee_right: float) -> Tuple[float, float, float]:
        self.elbow_buffer.append(elbow)
        self.knee_left_buffer.append(knee_left)
        self.knee_right_buffer.append(knee_right)
        
        elbow_smooth = statistics.mean(self.elbow_buffer) if self.elbow_buffer else elbow
        knee_left_smooth = statistics.mean(self.knee_left_buffer) if self.knee_left_buffer else knee_left
        knee_right_smooth = statistics.mean(self.knee_right_buffer) if self.knee_right_buffer else knee_right
        
        return elbow_smooth, knee_left_smooth, knee_right_smooth
    
    def _detect_stroke(self, elbow_angle: float, time_s: float) -> bool:
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
    
    def _annotate_frame(self, frame: np.ndarray, lm: dict, metrics: FrameMetrics,
                        stroke_rate: float, breathing_rate: float) -> np.ndarray:
        annotated = frame.copy()
        
        ideal_knee = (self.config.ideal_knee_underwater if self.is_underwater 
                      else self.config.ideal_knee_surface)
        
        elbow_dev = calculate_deviation(metrics.elbow_angle, self.config.ideal_elbow)
        knee_l_dev = calculate_deviation(metrics.knee_left, ideal_knee)
        knee_r_dev = calculate_deviation(metrics.knee_right, ideal_knee)
        
        arm_color = self._deviation_to_color(elbow_dev)
        leg_l_color = self._deviation_to_color(knee_l_dev)
        leg_r_color = self._deviation_to_color(knee_r_dev)
        
        skeleton_lines = [
            (lm['left_shoulder'], lm['left_elbow'], arm_color),
            (lm['left_elbow'], lm['left_wrist'], arm_color),
            (lm['left_hip'], lm['left_knee'], leg_l_color),
            (lm['left_knee'], lm['left_ankle'], leg_l_color),
            (lm['right_hip'], lm['right_knee'], leg_r_color),
            (lm['right_knee'], lm['right_ankle'], leg_r_color)
        ]
        
        for start, end, color in skeleton_lines:
            cv2.line(annotated, tuple(start.astype(int)), tuple(end.astype(int)), color, 3)
            cv2.circle(annotated, tuple(start.astype(int)), 5, color, -1)
            cv2.circle(annotated, tuple(end.astype(int)), 5, color, -1)
        
        y_pos = 30
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        overlays = [
            (f"Phase: {metrics.phase}", (255, 255, 255), 0.8),
            (f"Score: {int(metrics.score)}/100", (0, 255, 0), 0.8),
            (f"Stroke Rate: {stroke_rate:.1f} spm", (255, 255, 255), 0.7),
            (f"Breathing: {breathing_rate:.1f}/min", (255, 255, 255), 0.7),
            (f"Body Roll: {metrics.body_roll:.1f}°", (255, 255, 0), 0.7)
        ]
        
        for text, color, scale in overlays:
            cv2.putText(annotated, text, (30, y_pos), font, scale, color, 2)
            y_pos += 25 if scale < 0.8 else 30
        
        return annotated
    
    @staticmethod
    def _deviation_to_color(deviation: float) -> Tuple[int, int, int]:
        if deviation <= 10:
            return (0, 255, 0)
        elif deviation <= 20:
            return (0, 255, 255)
        else:
            return (0, 0, 255)
    
    def process_video(self, input_path: str, output_path: str, progress_callback=None) -> SessionSummary:
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise RuntimeError("Cannot open video file")
        
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
        
        frame_id = 0
        ideal_knee = self.config.ideal_knee_underwater if self.is_underwater else self.config.ideal_knee_surface
        
        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_id += 1
                if progress_callback and frame_id % 10 == 0:
                    progress_callback(min(frame_id / max(total_frames, 1), 1.0))
                
                time_s = frame_id / fps
                time_ms = int(time_s * 1000)
                
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB,
                                    data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                result = self.detector.detect_for_video(mp_image, time_ms)
                
                if not result.pose_landmarks:
                    writer.write(frame)
                    continue
                
                lm = self._extract_landmarks(result.pose_landmarks[0], (height, width))
                
                elbow, knee_left, knee_right = self._compute_frame_angles(lm)
                elbow_s, knee_left_s, knee_right_s = self._smooth_angles(elbow, knee_left, knee_right)
                
                body_roll = calculate_shoulder_roll(lm['left_shoulder'], lm['right_shoulder'])
                yaw = calculate_yaw_proxy(lm['nose'], lm['left_shoulder'], lm['right_shoulder'])
                
                elbow_dev = calculate_deviation(elbow_s, self.config.ideal_elbow)
                knee_l_dev = calculate_deviation(knee_left_s, ideal_knee)
                knee_r_dev = calculate_deviation(knee_right_s, ideal_knee)
                symmetry = abs(knee_left_s - knee_right_s)
                
                score = calculate_technique_score(elbow_dev, symmetry, knee_l_dev, knee_r_dev)
                
                phase = 'Pull' if lm['left_wrist'][1] > lm['left_shoulder'][1] else 'Recovery'
                
                if phase == 'Pull':
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
                
                self._detect_stroke(elbow_s, time_s)
                breath_state = self._detect_breathing(yaw, time_s)
                
                metrics = FrameMetrics(
                    time_s=time_s,
                    elbow_angle=elbow_s,
                    knee_left=knee_left_s,
                    knee_right=knee_right_s,
                    symmetry=symmetry,
                    score=score,
                    yaw_proxy=yaw,
                    breath_state=breath_state,
                    body_roll=body_roll,
                    phase=phase
                )
                self.frame_metrics.append(metrics)
                
                stroke_rate = self._calculate_current_stroke_rate()
                breathing_rate = self._calculate_current_breathing_rate(time_s)
                
                annotated = self._annotate_frame(frame, lm, metrics, stroke_rate, breathing_rate)
                writer.write(annotated)
        
        finally:
            cap.release()
            writer.release()
        
        return self._generate_summary()
    
    def _calculate_current_stroke_rate(self) -> float:
        if len(self.stroke_times) < 2:
            return 0.0
        duration = self.stroke_times[-1] - self.stroke_times[0]
        if duration < 0.1:
            return 0.0
        return 60.0 * (len(self.stroke_times) - 1) / duration
    
    def _calculate_current_breathing_rate(self, current_time: float) -> float:
        total_breaths = self.breath_count_left + self.breath_count_right
        minutes = current_time / 60.0
        return total_breaths / max(minutes, 1e-6)
    
    def _generate_summary(self) -> SessionSummary:
        if not self.frame_metrics:
            raise ValueError("No metrics to summarize")
        
        duration = self.frame_metrics[-1].time_s
        scores = [m.score for m in self.frame_metrics]
        symmetries = [m.symmetry for m in self.frame_metrics]
        rolls = [m.body_roll for m in self.frame_metrics]
        
        stroke_rate_single = self._calculate_current_stroke_rate()
        stroke_rate_both = 2.0 * stroke_rate_single
        breathing_rate = self._calculate_current_breathing_rate(duration)
        
        return SessionSummary(
            duration_s=duration,
            avg_score=statistics.mean(scores),
            avg_symmetry=statistics.mean(symmetries),
            avg_roll=statistics.mean(rolls),
            max_roll_abs=max(abs(r) for r in rolls),
            stroke_rate_single=stroke_rate_single,
            stroke_rate_both=stroke_rate_both,
            breaths_per_min=breathing_rate,
            breath_count_left=self.breath_count_left,
            breath_count_right=self.breath_count_right,
            total_strokes=len(self.stroke_times),
            best_frame_time=self.frame_metrics[0].time_s if self.best_frame_bytes else None,
            worst_frame_time=self.frame_metrics[0].time_s if self.worst_frame_bytes else None
        )


# ───────────────────────────────────────────────────────────────────
# REPORTING & VISUALIZATION
# ───────────────────────────────────────────────────────────────────

def generate_plots(analyzer: SwimAnalyzer, config: AnalysisConfig) -> io.BytesIO:
    metrics = analyzer.frame_metrics
    times = [m.time_s for m in metrics]
    
    fig = plt.figure(figsize=(14, 11))
    gs = fig.add_gridspec(5, 1, hspace=0.35)
    
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(times, [m.elbow_angle for m in metrics], label="Elbow", color="#1f77b4", linewidth=1.5)
    ax1.plot(times, [m.knee_left for m in metrics], label="Knee L", color="#2ca02c", linewidth=1.5)
    ax1.plot(times, [m.knee_right for m in metrics], label="Knee R", color="#d62728", linewidth=1.5)
    ax1.axhspan(config.ideal_elbow[0], config.ideal_elbow[1], alpha=0.1, color='blue', label='Ideal Elbow')
    ax1.set_ylabel("Angle (°)")
    ax1.set_title("Joint Angles Over Time", fontweight='bold')
    ax1.legend(loc="upper right", framealpha=0.9)
    ax1.grid(True, alpha=0.3)
    
    ax2 = fig.add_subplot(gs[1, 0])
    ax2_twin = ax2.twinx()
    ax2.plot(times, [m.score for m in metrics], label="Score", color="#2ca02c", linewidth=1.5)
    ax2_twin.plot(times, [m.symmetry for m in metrics], label="Symmetry", color="#ff7f0e", linewidth=1.5)
    ax2.set_ylabel("Score (0-100)", color="#2ca02c")
    ax2_twin.set_ylabel("Symmetry (°)", color="#ff7f0e")
    ax2.set_title("Technique Score & Leg Symmetry", fontweight='bold')
    ax2.legend(loc="upper left")
    ax2_twin.legend(loc="upper right")
    ax2.grid(True, alpha=0.3)
    
    ax3 = fig.add_subplot(gs[2, 0])
    ax3.plot(times, [m.body_roll for m in metrics], label="Body Roll", color="#9467bd", linewidth=1.5)
    ax3.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax3.axhspan(-config.ideal_roll_abs_max, config.ideal_roll_abs_max, color='green', alpha=0.08, label='Ideal Range')
    ax3.set_ylabel("Roll Angle (°)")
    ax3.set_title("Body Roll Analysis", fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    ax4 = fig.add_subplot(gs[3, 0])
    ax4.plot(times, [m.yaw_proxy for m in metrics], label="Head Yaw", color="#e377c2", linewidth=1.5)
    ax4.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax4.axhline(config.breath_side_threshold, color='blue', linestyle=':', alpha=0.5, label='Breath Threshold')
    ax4.axhline(-config.breath_side_threshold, color='blue', linestyle=':', alpha=0.5)
    ax4.set_ylabel("Yaw Proxy")
    ax4.set_title("Head Rotation Pattern", fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    ax5 = fig.add_subplot(gs[4, 0])
    ax5.plot(times, [m.score for m in metrics], color="#2ca02c", alpha=0.3, linewidth=1)
    
    for stroke_time in analyzer.stroke_times:
        ax5.axvline(stroke_time, color="#00c8ff", linestyle="--", alpha=0.6, linewidth=1)
    
    last_state = 'N'
    for i, m in enumerate(metrics):
        if m.breath_state in ('L', 'R') and m.breath_state != last_state:
            color = "#ff9500" if m.breath_state == 'L' else "#00a6ff"
            ax5.axvline(m.time_s, color=color, linestyle=":", alpha=0.8, linewidth=1)
        last_state = m.breath_state
    
    ax5.set_xlabel("Time (seconds)", fontweight='bold')
    ax5.set_ylabel("Score")
    ax5.set_title("Events Timeline (strokes: cyan dashed, breaths: L=orange / R=blue dotted)", fontweight='bold')
    ax5.set_ylim(0, 110)
    ax5.grid(True, alpha=0.3)
    
    buffer = io.BytesIO()
    plt.savefig(buffer, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    buffer.seek(0)
    return buffer


def generate_pdf_report(analyzer: SwimAnalyzer, summary: SessionSummary, 
                        config: AnalysisConfig, filename: str, 
                        plot_buffer: Optional[io.BytesIO] = None) -> io.BytesIO:
    buffer = io.BytesIO()
    pdf = SimpleDocTemplate(buffer, pagesize=letter, topMargin=0.5*inch, bottomMargin=0.5*inch)
    
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name='CustomTitle', parent=styles['Title'], fontSize=24,
                              textColor=colors.HexColor('#1f77b4'), spaceAfter=20))
    styles.add(ParagraphStyle(name='SectionHeader', parent=styles['Heading2'], fontSize=14,
                              textColor=colors.HexColor('#2ca02c'), spaceBefore=15, spaceAfter=10))
    
    story = []
    
    story.append(Paragraph("Freestyle Swimming Technique Analysis", styles['CustomTitle']))
    story.append(Spacer(1, 0.2*inch))
    
    story.append(Paragraph("Session Information", styles['SectionHeader']))
    session_data = [
        ['Video File:', filename],
        ['Duration:', f"{summary.duration_s:.1f} seconds ({summary.duration_s/60:.1f} minutes)"],
        ['Total Strokes:', str(summary.total_strokes)],
        ['Analysis Date:', datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")]
    ]
    session_table = Table(session_data, colWidths=[2*inch, 4*inch])
    session_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#f0f0f0')),
        ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey)
    ]))
    story.append(session_table)
    story.append(Spacer(1, 0.3*inch))
    
    story.append(Paragraph("Performance Metrics", styles['SectionHeader']))
    
    score_color = colors.green if summary.avg_score >= 80 else colors.orange if summary.avg_score >= 60 else colors.red
    metrics_data = [
        ['Metric', 'Value', 'Status'],
        ['Overall Technique Score', f"{summary.avg_score:.1f} / 100", ''],
        ['Stroke Rate (single arm)', f"{summary.stroke_rate_single:.1f} spm", ''],
        ['Stroke Rate (both arms)', f"{summary.stroke_rate_both:.1f} spm", ''],
        ['Breathing Rate', f"{summary.breaths_per_min:.1f} breaths/min", ''],
        ['Left Breaths', str(summary.breath_count_left), ''],
        ['Right Breaths', str(summary.breath_count_right), ''],
    ]
    
    metrics_table = Table(metrics_data, colWidths=[2.5*inch, 2*inch, 1.5*inch])
    metrics_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1f77b4')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 11),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 10),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('FONTSIZE', (0, 1), (-1, -1), 10),
    ]))
    story.append(metrics_table)
    story.append(Spacer(1, 0.3*inch))
    
    story.append(Paragraph("Biomechanical Analysis", styles['SectionHeader']))
    bio_data = [
        ['Parameter', 'Average', 'Ideal Range', 'Assessment'],
        ['Leg Symmetry', f"{summary.avg_symmetry:.1f}°", '< 10°', 
         'Good' if summary.avg_symmetry < 10 else 'Fair' if summary.avg_symmetry < 15 else 'Needs Work'],
        ['Body Roll (avg)', f"{summary.avg_roll:.1f}°", '35-55°', 
         'Good' if 35 <= abs(summary.avg_roll) <= 55 else 'Check Form'],
        ['Max Body Roll', f"{summary.max_roll_abs:.1f}°", '< 55°', 
         'Good' if summary.max_roll_abs <= 55 else 'Excessive']
    ]
    
    bio_table = Table(bio_data, colWidths=[2*inch, 1.5*inch, 1.5*inch, 1.5*inch])
    bio_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2ca02c')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 11),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 10),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
    ]))
    story.append(bio_table)
    story.append(Spacer(1, 0.3*inch))
    
    story.append(Paragraph("Recommended Training Focus", styles['SectionHeader']))
    for rec in generate_recommendations(summary, config):
        story.append(Paragraph(rec, styles['Normal']))
        story.append(Spacer(1, 0.1*inch))
    
    if plot_buffer and plot_buffer.getvalue():
        story.append(PageBreak())
        story.append(Paragraph("Time-Series Analysis", styles['SectionHeader']))
        story.append(Spacer(1, 0.1*inch))
        plot_image = RLImage(plot_buffer)
        plot_image.drawWidth = 7*inch
        plot_image.drawHeight = 5.5*inch
        story.append(plot_image)
    
    pdf.build(story)
    buffer.seek(0)
    return buffer


def generate_recommendations(summary: SessionSummary, config: AnalysisConfig) -> List[str]:
    recs = []
    if summary.avg_score < 70:
        recs.append("**High-Elbow Catch Drill**: Practice fingertip drag or catch-up drill to develop early vertical forearm (EVF).")
    if summary.avg_symmetry > 15:
        recs.append("**Symmetry Development**: Practice single-arm freestyle, alternating sides every 25 m.")
    if summary.max_roll_abs > config.ideal_roll_abs_max + 10:
        recs.append("**Body Roll Control**: Practice 6-kick switch drill. Aim for 45° maximum roll.")
    if summary.breaths_per_min > 40:
        recs.append("**Breathing Efficiency**: High breathing rate detected. Try bilateral breathing every 3 strokes.")
    if summary.stroke_rate_single < 40:
        recs.append("**Increase Tempo**: Use tempo trainer at 1.2–1.4 sec/stroke.")
    if not recs:
        recs.append("**Progressive Overload**: Technique is solid — gradually increase volume and intensity.")
    return recs


def export_to_csv(analyzer: SwimAnalyzer) -> io.BytesIO:
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
        'phase': [m.phase for m in analyzer.frame_metrics]
    }
    df = pd.DataFrame(data)
    buffer = io.BytesIO()
    df.to_csv(buffer, index=False, float_format="%.2f")
    buffer.seek(0)
    return buffer


def create_results_bundle(video_path: str, csv_buffer: io.BytesIO, 
                          pdf_buffer: io.BytesIO, plot_buffer: Optional[io.BytesIO],
                          timestamp: str) -> io.BytesIO:
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
# STREAMLIT APP
# ───────────────────────────────────────────────────────────────────

def main():
    st.set_page_config(page_title="Freestyle Swimming Analyzer", layout="wide", page_icon="🏊")
    
    st.title("Freestyle Swimming Technique Analyzer")
    st.write("Upload side-view freestyle swimming video for biomechanical analysis and recommendations.")
    
    with st.sidebar:
        st.header("Analysis Settings")
        is_underwater = st.checkbox("Underwater footage", value=False)
        min_confidence = st.slider("Min detection confidence", 0.3, 0.9, 0.6, 0.05)
        smoothing_window = st.slider("Smoothing window (frames)", 3, 15, 7, 2)
    
    uploaded_file = st.file_uploader("Upload video", type=["mp4", "mov", "avi"])
    
    if uploaded_file is not None:
        if st.button("Analyze Video", type="primary"):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_in:
                tmp_in.write(uploaded_file.read())
                input_path = tmp_in.name
            
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = tempfile.mktemp(suffix=".mp4")
            
            try:
                config = AnalysisConfig(
                    min_detection_confidence=min_confidence,
                    smoothing_window=smoothing_window
                )
                analyzer = SwimAnalyzer(config, is_underwater=is_underwater)
                
                progress = st.progress(0)
                status = st.empty()
                
                def update(p):
                    progress.progress(p)
                    status.text(f"Processing: {int(p*100)}%")
                
                summary = analyzer.process_video(input_path, output_path, update)
                
                progress.empty()
                status.empty()
                
                st.success("Analysis complete!")
                st.video(output_path)
                
                st.subheader("Key Metrics")
                cols = st.columns(4)
                cols[0].metric("Technique Score", f"{summary.avg_score:.1f}/100")
                cols[1].metric("Stroke Rate", f"{summary.stroke_rate_single:.1f} spm")
                cols[2].metric("Breathing Rate", f"{summary.breaths_per_min:.1f}/min")
                cols[3].metric("Max Roll", f"{summary.max_roll_abs:.1f}°")
                
                st.subheader("Recommendations")
                for r in generate_recommendations(summary, config):
                    st.markdown(r)
                
                with st.spinner("Preparing files..."):
                    csv_buf = export_to_csv(analyzer)
                    plot_buf = generate_plots(analyzer, config)
                    pdf_buf = generate_pdf_report(analyzer, summary, config, uploaded_file.name, plot_buf)
                    zip_buf = create_results_bundle(output_path, csv_buf, pdf_buf, plot_buf, timestamp)
                
                st.download_button(
                    "Download Full Analysis (ZIP)",
                    zip_buf,
                    f"swim_analysis_{timestamp}.zip",
                    "application/zip",
                    use_container_width=True
                )
            
            except Exception as e:
                st.error(f"Error: {str(e)}")
            
            finally:
                for p in [input_path, output_path]:
                    if os.path.exists(p):
                        try:
                            os.unlink(p)
                        except:
                            pass


if __name__ == "__main__":
    main()
