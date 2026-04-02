"""
Affective Mental Analytics — Professional Edition
Stack: Streamlit · DeepFace · OpenCV · Plotly
"""

import os
import time
import warnings
from collections import Counter
from datetime import datetime

import cv2
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from deepface import DeepFace

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Affective Analytics",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# THEME & GLOBAL STYLES
# ─────────────────────────────────────────────
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=DM+Mono:wght@400;500&display=swap');

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
    }

    /* ── Background ── */
    .stApp {
        background: #f4f6f8;
        color: #1a2332;
    }

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {
        background: #ffffff;
        border-right: 1px solid #dce3ec;
    }

    /* ── Cards ── */
    .card {
        background: #ffffff;
        border: 1px solid #dce3ec;
        border-radius: 12px;
        padding: 24px;
        margin-bottom: 16px;
        box-shadow: 0 1px 4px rgba(0,0,0,.06);
    }

    /* ── Stat tiles ── */
    .stat-tile {
        background: #ffffff;
        border: 1px solid #dce3ec;
        border-radius: 10px;
        padding: 18px 22px;
        margin-bottom: 12px;
        box-shadow: 0 1px 3px rgba(0,0,0,.05);
        transition: border-color .2s, box-shadow .2s;
    }
    .stat-tile:hover {
        border-color: #1a73e8;
        box-shadow: 0 2px 8px rgba(26,115,232,.12);
    }
    .stat-label {
        font-size: 10px;
        font-weight: 600;
        letter-spacing: .10em;
        text-transform: uppercase;
        color: #6b7a8d;
        margin-bottom: 4px;
    }
    .stat-value {
        font-size: 28px;
        font-weight: 600;
        color: #1a2332;
    }
    .stat-sub {
        font-size: 12px;
        color: #6b7a8d;
        margin-top: 2px;
    }

    /* ── Emotion badge ── */
    .badge {
        display: inline-block;
        padding: 4px 14px;
        border-radius: 20px;
        font-size: 13px;
        font-weight: 600;
        letter-spacing: .04em;
        text-transform: capitalize;
    }

    /* ── Section headers ── */
    .section-header {
        font-size: 10px;
        font-weight: 700;
        letter-spacing: .14em;
        text-transform: uppercase;
        color: #9aa5b4;
        margin-bottom: 16px;
        padding-bottom: 8px;
        border-bottom: 1px solid #dce3ec;
    }

    /* ── Page title ── */
    .page-title {
        font-size: 22px;
        font-weight: 600;
        color: #1a2332;
        display: flex;
        align-items: center;
        gap: 10px;
    }
    .page-subtitle {
        font-size: 13px;
        color: #6b7a8d;
        margin-top: 2px;
    }

    /* ── Buttons ── */
    div[data-testid="stButton"] > button {
        border-radius: 8px;
        font-family: 'DM Sans', sans-serif;
        font-weight: 500;
        font-size: 14px;
        padding: 8px 20px;
        border: 1px solid #dce3ec;
        background: #ffffff;
        color: #1a2332;
        box-shadow: 0 1px 3px rgba(0,0,0,.06);
        transition: all .2s;
    }
    div[data-testid="stButton"] > button:hover {
        background: #f0f4ff;
        border-color: #1a73e8;
        color: #1a73e8;
        box-shadow: 0 2px 6px rgba(26,115,232,.15);
    }

    /* ── Progress timeline entry ── */
    .log-entry {
        display: flex;
        align-items: center;
        gap: 12px;
        padding: 10px 0;
        border-bottom: 1px solid #edf1f5;
        font-size: 13px;
        color: #3d4f63;
        font-family: 'DM Mono', monospace;
    }
    .log-entry:last-child { border-bottom: none; }
    .log-dot {
        width: 8px; height: 8px;
        border-radius: 50%;
        flex-shrink: 0;
    }

    /* ── Metric override ── */
    [data-testid="stMetric"] {
        background: #ffffff;
        border: 1px solid #dce3ec;
        border-radius: 10px;
        padding: 14px 18px;
    }

    /* ── Recommendation item ── */
    .rec-item {
        background: #f0f6ff;
        border-left: 3px solid #1a73e8;
        border-radius: 0 8px 8px 0;
        padding: 10px 16px;
        margin-bottom: 8px;
        font-size: 14px;
        color: #1a2332;
    }

    /* ── Download button ── */
    div[data-testid="stDownloadButton"] > button {
        background: #e8f0fe;
        border: 1px solid #1a73e8;
        color: #1a73e8;
    }
    div[data-testid="stDownloadButton"] > button:hover {
        background: #d2e3fc;
    }

    /* ── Hide default Streamlit chrome ── */
    #MainMenu, footer { visibility: hidden; }
    .block-container { padding-top: 2rem; padding-bottom: 2rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────
EMOTION_META: dict[str, dict] = {
    "happy":    {"color": "#3fb950", "icon": "😊"},
    "neutral":  {"color": "#8b949e", "icon": "😐"},
    "sad":      {"color": "#388bfd", "icon": "😢"},
    "angry":    {"color": "#f85149", "icon": "😠"},
    "surprise": {"color": "#d29922", "icon": "😲"},
    "fear":     {"color": "#a371f7", "icon": "😨"},
    "disgust":  {"color": "#3fb950", "icon": "🤢"},
}

RECOMMENDATIONS: dict[str, list[str]] = {
    "happy":    ["Channel this energy into creative work", "Strengthen social connections"],
    "sad":      ["Reach out to someone you trust", "Gentle physical movement can help"],
    "angry":    ["Try box breathing: 4s in, hold 4s, out 4s", "Take a brief walk to reset"],
    "fear":     ["Ground yourself — name 5 things you can see", "Consider speaking with a professional"],
    "surprise": ["Pause before reacting to new information", "Journal what triggered the response"],
    "disgust":  ["Identify the underlying cause calmly", "Set a healthy boundary if needed"],
    "neutral":  ["Set a small, meaningful goal for today", "Explore something outside your routine"],
}

TRACKING_INTERVAL_SEC = 2
TRACKING_SAMPLES      = 15

# ─────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────
_defaults = {
    "history":    [],
    "timestamps": [],
    "capturing":  False,
    "frame":      None,
}
for _k, _v in _defaults.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def detect_emotion(frame: np.ndarray) -> tuple[str | None, dict | None]:
    """Run DeepFace analysis; return (dominant, scores_dict)."""
    try:
        result = DeepFace.analyze(
            frame,
            actions=["emotion"],
            enforce_detection=False,
            silent=True,
        )
        if isinstance(result, list):
            result = result[0]
        return result["dominant_emotion"], result["emotion"]
    except Exception:
        return None, None


def emotion_badge(emotion: str) -> str:
    meta  = EMOTION_META.get(emotion, {"color": "#8b949e", "icon": "🔍"})
    color = meta["color"]
    icon  = meta["icon"]
    return (
        f'<span class="badge" style="background:{color}22;color:{color};'
        f'border:1px solid {color}55">{icon} {emotion}</span>'
    )


def build_bar_chart(scores: dict) -> go.Figure:
    emotions = list(scores.keys())
    values   = [round(v, 1) for v in scores.values()]
    colors   = [EMOTION_META.get(e, {}).get("color", "#8b949e") for e in emotions]

    fig = go.Figure(
        go.Bar(
            x=emotions,
            y=values,
            marker=dict(color=colors, opacity=0.85),
            hovertemplate="<b>%{x}</b><br>%{y:.1f}%<extra></extra>",
        )
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#f9fafb",
        font=dict(family="DM Sans", color="#6b7a8d", size=12),
        margin=dict(l=0, r=0, t=10, b=0),
        xaxis=dict(showgrid=False, tickfont=dict(size=11)),
        yaxis=dict(
            showgrid=True,
            gridcolor="#e5eaf0",
            gridwidth=1,
            ticksuffix="%",
            range=[0, 100],
        ),
        height=240,
    )
    return fig


def build_timeline_chart(history: list, timestamps: list) -> go.Figure:
    labels = [EMOTION_META.get(e, {}).get("icon", "") + " " + e for e in history]
    colors = [EMOTION_META.get(e, {}).get("color", "#8b949e") for e in history]

    fig = go.Figure(
        go.Scatter(
            x=timestamps,
            y=history,
            mode="markers+lines",
            marker=dict(color=colors, size=10, line=dict(width=1, color="#ffffff")),
            line=dict(color="#dce3ec", width=2, dash="dot"),
            hovertemplate="<b>%{y}</b><br>%{x}<extra></extra>",
        )
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#f9fafb",
        font=dict(family="DM Sans", color="#6b7a8d", size=12),
        margin=dict(l=0, r=0, t=10, b=0),
        xaxis=dict(showgrid=False, tickformat="%H:%M:%S"),
        yaxis=dict(showgrid=True, gridcolor="#e5eaf0"),
        height=220,
    )
    return fig


def build_pie_chart(counts: Counter) -> go.Figure:
    labels  = list(counts.keys())
    values  = list(counts.values())
    colors  = [EMOTION_META.get(e, {}).get("color", "#8b949e") for e in labels]

    fig = go.Figure(
        go.Pie(
            labels=labels,
            values=values,
            marker=dict(colors=colors, line=dict(color="#ffffff", width=2)),
            hole=0.55,
            hovertemplate="<b>%{label}</b><br>%{percent}<extra></extra>",
        )
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="DM Sans", color="#6b7a8d", size=12),
        margin=dict(l=0, r=0, t=10, b=0),
        legend=dict(
            orientation="v",
            font=dict(size=12),
            bgcolor="rgba(0,0,0,0)",
        ),
        height=280,
    )
    return fig


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        '<div class="page-title">🧠 Affective<br>Analytics</div>'
        '<div class="page-subtitle">Facial emotion intelligence</div>',
        unsafe_allow_html=True,
    )
    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown('<div class="section-header">Session Controls</div>', unsafe_allow_html=True)
    if st.button("🔄  Reset Session", use_container_width=True):
        for k, v in _defaults.items():
            st.session_state[k] = v if not isinstance(v, list) else []
        st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-header">Configuration</div>', unsafe_allow_html=True)
    interval = st.slider("Capture interval (s)", 1, 5, TRACKING_INTERVAL_SEC)
    samples  = st.slider("Tracking samples", 5, 30, TRACKING_SAMPLES)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(
        '<div style="font-size:11px;color:#9aa5b4;">Powered by DeepFace · OpenCV · Streamlit</div>',
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
st.markdown(
    """
    <div class="card" style="display:flex;justify-content:space-between;align-items:center;padding:16px 24px">
      <div>
        <div class="page-title">Affective Mental Analytics</div>
        <div class="page-subtitle">Real-time facial emotion recognition &amp; wellbeing intelligence</div>
      </div>
      <div style="font-size:11px;color:#9aa5b4;text-align:right;">
        DeepFace Engine<br>
        <span style="color:#1e8a3e;">● Active</span>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────
# MAIN COLUMNS
# ─────────────────────────────────────────────
col_cam, col_stats = st.columns([3, 2], gap="large")

# ── CAMERA COLUMN ─────────────────────────────
with col_cam:
    st.markdown('<div class="section-header">Camera Input</div>', unsafe_allow_html=True)

    img_file = st.camera_input("", label_visibility="collapsed")

    if img_file:
        raw   = np.frombuffer(img_file.getvalue(), np.uint8)
        frame = cv2.imdecode(raw, cv2.IMREAD_COLOR)
        st.session_state.frame = frame

    # Single-frame analysis
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-header">Single-Frame Analysis</div>', unsafe_allow_html=True)

    if st.button("🔍  Analyze Current Frame", use_container_width=True):
        if st.session_state.frame is not None:
            with st.spinner("Running emotion analysis…"):
                dominant, scores = detect_emotion(st.session_state.frame)

            if dominant and scores:
                st.markdown(
                    f'<div class="card" style="text-align:center;padding:16px">'
                    f'<div class="stat-label">Detected Emotion</div>'
                    f'<div style="margin:10px 0">{emotion_badge(dominant)}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
                st.plotly_chart(build_bar_chart(scores), use_container_width=True)
            else:
                st.warning("No face detected. Please ensure your face is visible.")
        else:
            st.info("Capture an image first using the camera above.")

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-header">Continuous Tracking</div>', unsafe_allow_html=True)

    c_start, c_stop = st.columns(2)
    if c_start.button("▶  Start Tracking", use_container_width=True):
        st.session_state.capturing  = True
        st.session_state.history    = []
        st.session_state.timestamps = []

    if c_stop.button("■  Stop Tracking", use_container_width=True):
        st.session_state.capturing = False

    # Tracking progress log placeholder
    log_placeholder = st.empty()


# ── STATS COLUMN ──────────────────────────────
with col_stats:
    st.markdown('<div class="section-header">Live Statistics</div>', unsafe_allow_html=True)

    history = st.session_state.history

    if history:
        cnt     = Counter(history)
        top_em  = cnt.most_common(1)[0][0]
        top_meta = EMOTION_META.get(top_em, {"color": "#8b949e", "icon": "🔍"})

        # Stat tiles
        st.markdown(
            f'<div class="stat-tile">'
            f'<div class="stat-label">Total Samples</div>'
            f'<div class="stat-value">{len(history)}</div>'
            f'<div class="stat-sub">frames analyzed</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<div class="stat-tile">'
            f'<div class="stat-label">Dominant Emotion</div>'
            f'<div class="stat-value" style="color:{top_meta["color"]}">'
            f'{top_meta["icon"]} {top_em.capitalize()}'
            f'</div>'
            f'<div class="stat-sub">{cnt[top_em]} occurrences '
            f'({cnt[top_em]/len(history)*100:.0f}%)</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
        current_em   = history[-1]
        current_meta = EMOTION_META.get(current_em, {"color": "#8b949e", "icon": "🔍"})
        st.markdown(
            f'<div class="stat-tile">'
            f'<div class="stat-label">Most Recent</div>'
            f'<div class="stat-value" style="font-size:20px">'
            f'{emotion_badge(current_em)}'
            f'</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

        # Mini distribution
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="section-header">Emotion Distribution</div>', unsafe_allow_html=True)
        for emo, count in cnt.most_common():
            meta  = EMOTION_META.get(emo, {"color": "#8b949e"})
            pct   = count / len(history) * 100
            st.markdown(
                f'<div style="margin-bottom:10px">'
                f'<div style="display:flex;justify-content:space-between;'
                f'font-size:12px;margin-bottom:4px">'
                f'<span>{meta.get("icon","🔍")} {emo.capitalize()}</span>'
                f'<span style="color:#8b949e">{count} · {pct:.0f}%</span></div>'
                f'<div style="background:#edf1f5;border-radius:4px;height:6px">'
                f'<div style="width:{pct}%;background:{meta["color"]};'
                f'height:6px;border-radius:4px;transition:width .3s"></div>'
                f'</div></div>',
                unsafe_allow_html=True,
            )
    else:
        st.markdown(
            '<div class="card" style="text-align:center;padding:40px;color:#9aa5b4">'
            '<div style="font-size:32px;margin-bottom:12px">📷</div>'
            '<div>No data collected yet.<br>Capture an image and run analysis.</div>'
            '</div>',
            unsafe_allow_html=True,
        )

# ─────────────────────────────────────────────
# CONTINUOUS TRACKING LOOP
# ─────────────────────────────────────────────
if st.session_state.capturing:
    log_entries: list[str] = []

    for i in range(samples):
        if not st.session_state.capturing:
            break

        if st.session_state.frame is not None:
            dominant, _ = detect_emotion(st.session_state.frame)

            if dominant:
                st.session_state.history.append(dominant)
                ts = datetime.now()
                st.session_state.timestamps.append(ts)

                meta  = EMOTION_META.get(dominant, {"color": "#8b949e", "icon": "🔍"})
                entry = (
                    f'<div class="log-entry">'
                    f'<div class="log-dot" style="background:{meta["color"]}"></div>'
                    f'<span style="color:#9aa5b4">[{ts.strftime("%H:%M:%S")}]</span>'
                    f'&nbsp;{meta["icon"]} <b>{dominant}</b>'
                    f'&nbsp;<span style="color:#9aa5b4">— sample {i+1}/{samples}</span>'
                    f'</div>'
                )
                log_entries.append(entry)
                log_placeholder.markdown(
                    '<div class="card">' + "".join(log_entries) + "</div>",
                    unsafe_allow_html=True,
                )

        time.sleep(interval)

    st.session_state.capturing = False
    st.rerun()

# ─────────────────────────────────────────────
# TIMELINE CHART
# ─────────────────────────────────────────────
if len(st.session_state.history) >= 2:
    st.markdown("---")
    st.markdown('<div class="section-header">Emotion Timeline</div>', unsafe_allow_html=True)
    st.plotly_chart(
        build_timeline_chart(st.session_state.history, st.session_state.timestamps),
        use_container_width=True,
    )

# ─────────────────────────────────────────────
# FULL REPORT
# ─────────────────────────────────────────────
if st.session_state.history:
    st.markdown("---")
    st.markdown('<div class="section-header">Session Report</div>', unsafe_allow_html=True)

    if st.button("📊  Generate Full Report", use_container_width=False):
        history    = st.session_state.history
        timestamps = st.session_state.timestamps
        cnt        = Counter(history)
        dominant   = cnt.most_common(1)[0][0]

        r1, r2 = st.columns(2, gap="large")

        with r1:
            st.markdown("**Distribution Overview**")
            st.plotly_chart(build_pie_chart(cnt), use_container_width=True)

            # Data table
            df = (
                pd.DataFrame(cnt.items(), columns=["Emotion", "Count"])
                .assign(Percentage=lambda d: (d["Count"] / d["Count"].sum() * 100).round(1))
                .sort_values("Count", ascending=False)
                .reset_index(drop=True)
            )
            st.dataframe(
                df.style.format({"Percentage": "{:.1f}%"}),
                use_container_width=True,
                hide_index=True,
            )

        with r2:
            st.markdown("**Wellbeing Recommendations**")
            recs = RECOMMENDATIONS.get(dominant, ["Maintain awareness of your emotional state."])
            for rec in recs:
                st.markdown(f'<div class="rec-item">→ {rec}</div>', unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("**Session Summary**")
            duration = (
                (timestamps[-1] - timestamps[0]).total_seconds()
                if len(timestamps) >= 2 else 0
            )
            m, s = divmod(int(duration), 60)
            st.markdown(
                f'<div class="card" style="background:#f9fafb">'
                f'<div class="stat-label">Duration</div>'
                f'<div class="stat-value" style="font-size:22px">{m:02d}:{s:02d}</div>'
                f'<div class="stat-label" style="margin-top:16px">Total Samples</div>'
                f'<div class="stat-value" style="font-size:22px">{len(history)}</div>'
                f'<div class="stat-label" style="margin-top:16px">Primary State</div>'
                f'<div style="margin-top:6px">{emotion_badge(dominant)}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

        # Download
        lines = [
            "AFFECTIVE ANALYTICS — SESSION REPORT",
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Duration : {m:02d}m {s:02d}s",
            f"Samples  : {len(history)}",
            f"Dominant : {dominant.upper()}",
            "",
            "EMOTION BREAKDOWN",
            "-" * 30,
        ] + [
            f"{e:<10} {c:>4} samples  ({c/len(history)*100:5.1f}%)"
            for e, c in cnt.most_common()
        ] + [
            "",
            "RECOMMENDATIONS",
            "-" * 30,
        ] + [f"• {r}" for r in RECOMMENDATIONS.get(dominant, [])]

        if timestamps:
            lines += ["", "SAMPLE LOG", "-" * 30] + [
                f"[{t.strftime('%H:%M:%S')}] {e}"
                for t, e in zip(timestamps, history)
            ]

        st.download_button(
            "⬇  Download Report (.txt)",
            data="\n".join(lines),
            file_name=f"affective_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain",
        )
