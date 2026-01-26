# ===============================================================
# Freestyle Swimming Analyzer Pro — production-ready app.py
# Streamlit • Safe ordering • No NameErrors
# ===============================================================

# ---------------------------
# Imports
# ---------------------------
import streamlit as st
from dataclasses import dataclass
from typing import List

# ---------------------------
# Data Models (safe stubs)
# ---------------------------
@dataclass
class Recommendation:
    title: str
    description: str
    priority: str  # "high" | "medium" | "low"


@dataclass
class TrainingDrill:
    title: str
    description: str
    sets: str
    focus: str


# ---------------------------
# Global CSS (MUST be defined before main)
# ---------------------------
CUSTOM_CSS = """
<style>
.stApp {
    background: linear-gradient(135deg, #0f172a 0%, #1e3a5f 50%, #0f172a 100%);
}

.metric-card {
    background: rgba(30, 41, 59, 0.7);
    backdrop-filter: blur(10px);
    border-radius: 16px;
    padding: 20px;
    border: 1px solid rgba(100, 116, 139, 0.3);
    margin-bottom: 16px;
}

.metric-card-green { border-left: 4px solid #22c55e; }
.metric-card-red   { border-left: 4px solid #ef4444; }

.score-card {
    background: linear-gradient(135deg, #0891b2 0%, #2563eb 100%);
    border-radius: 16px;
    padding: 24px;
    color: white;
    margin-bottom: 24px;
}

.drill-card {
    background: rgba(15, 23, 42, 0.6);
    border-radius: 12px;
    padding: 16px;
    border: 1px solid rgba(100, 116, 139, 0.3);
    margin-bottom: 12px;
}

.rec-high   { background: rgba(127,29,29,.3); border-left:4px solid #ef4444; padding:16px; border-radius:12px; }
.rec-medium { background: rgba(113,63,18,.3); border-left:4px solid #eab308; padding:16px; border-radius:12px; }
.rec-low    { background: rgba(20,83,45,.3);  border-left:4px solid #22c55e; padding:16px; border-radius:12px; }

.legend-item {
    display:inline-flex;
    align-items:center;
    gap:8px;
    padding:8px 12px;
    background:rgba(15,23,42,.6);
    border-radius:8px;
}

.legend-dot {
    width:12px;
    height:12px;
    border-radius:50%;
}
.legend-dot-green { background:#22c55e; }
.legend-dot-yellow{ background:#eab308; }
.legend-dot-red   { background:#ef4444; }
.legend-dot-white { background:#ffffff; }

.stButton > button {
    background: linear-gradient(135deg, #06b6d4 0%, #3b82f6 100%);
    color: white;
    border: none;
    border-radius: 12px;
    padding: 12px 24px;
    font-weight: 600;
}
</style>
"""

# ---------------------------
# UI Components
# ---------------------------
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
                <div style="color:white;font-weight:600;">{drill.title}</div>
                <div style="color:#94a3b8;font-size:14px;">{drill.description}</div>
                <div style="color:#64748b;font-size:12px;margin-top:6px;">
                    {drill.sets} • {drill.focus}
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_legend():
    st.markdown("""
    <div style="display:flex;flex-wrap:wrap;gap:8px;margin-top:12px;">
        <div class="legend-item"><div class="legend-dot legend-dot-green"></div><span>Good Form</span></div>
        <div class="legend-item"><div class="legend-dot legend-dot-yellow"></div><span>Fair Form</span></div>
        <div class="legend-item"><div class="legend-dot legend-dot-red"></div><span>Needs Work</span></div>
        <div class="legend-item"><div class="legend-dot legend-dot-white"></div><span>Joint Markers</span></div>
    </div>
    """, unsafe_allow_html=True)


# ---------------------------
# Main App
# ---------------------------
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
        <p style="color:#94a3b8;">AI-powered swimming technique analysis</p>
    </div>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Upload your swimming video",
        type=["mp4", "mov", "avi", "mkv"]
    )

    if uploaded_file is None:
        st.markdown("""
        <div style="background:rgba(30,41,59,.7);border-radius:16px;
                    padding:40px;text-align:center;border:2px dashed #06b6d4;">
            <div style="font-size:48px;">📤</div>
            <h3 style="color:white;">Upload Your Swimming Video</h3>
            <p style="color:#94a3b8;">MP4, MOV, AVI, MKV supported</p>
        </div>
        """, unsafe_allow_html=True)
        return

    st.video(uploaded_file)

    if st.button("🎯 Analyze Technique", use_container_width=True):
        st.success("✅ Demo analysis complete (engine placeholder)")

        st.markdown("### 💡 Recommendations")
        recs: List[Recommendation] = [
            Recommendation("Improve Body Roll", "Increase hip rotation during recovery.", "high"),
            Recommendation("Breathing Timing", "Exhale fully underwater.", "medium"),
            Recommendation("Kick Consistency", "Maintain steady flutter kick.", "low"),
        ]
        for r in recs:
            render_recommendation_card(r)

        st.markdown("### 🏋️ Training Drills")
        drills: List[TrainingDrill] = [
            TrainingDrill("Side Kick", "Improve balance and rotation.", "4×50m", "Body roll"),
            TrainingDrill("3-3-3 Drill", "Sync breathing and stroke.", "6×25m", "Breathing"),
        ]
        for i, d in enumerate(drills, 1):
            render_drill_card(d, i)

        render_legend()


# ---------------------------
# Entry Point
# ---------------------------
if __name__ == "__main__":
    main()
