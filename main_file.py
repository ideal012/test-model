import streamlit as st
# ตั้งค่าหน้าเว็บ
st.set_page_config(page_title="Automatic Scheduler", layout="wide")
st.title("🎓 Automatic Course Scheduler")

# ==========================================
# 1. User Config (UI Side)
# ==========================================
st.sidebar.header("Configuration")
schedule_mode_desc = {
    1: "Compact Mode (09:00 - 16:00)",
    2: "Flexible Mode (08:30 - 19:00)"
}
SCHEDULE_MODE = st.sidebar.radio(
    "Select Scheduling Mode:",
    options=[1, 2],
    format_func=lambda x: schedule_mode_desc[x]
)
