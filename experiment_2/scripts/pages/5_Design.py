# ============================================================
# Page 5 — Antenna Design & CST Simulation Results
# ============================================================

import os
import sys
import pandas as pd
import streamlit as st

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ── Page Config ───────────────────────────────────────────────
st.set_page_config(
    page_title="Antenna Design",
    page_icon="📡",
    layout="wide"
)

# ── Base Directory ────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMAGES_DIR = os.path.join(BASE_DIR, '../antenna_design_images')

# ── Sidebar ───────────────────────────────────────────────────
st.sidebar.title("📡 Antenna ML Predictor")
st.sidebar.markdown("---")
st.sidebar.markdown("**Antenna Specs:**")
st.sidebar.markdown("- Type: Microstrip Patch")
st.sidebar.markdown("- Frequency: 2.4 GHz")
st.sidebar.markdown("- Substrate: FR4 (εr=4.3)")
st.sidebar.markdown("- S11: -31.77 dB")
st.sidebar.markdown("- Gain: 6.15 dBi")

# ── Page Content ──────────────────────────────────────────────
st.title("📡 Antenna Design & CST Simulation Results")
st.markdown("Rectangular Microstrip Patch Antenna with Inset Feed designed in CST Studio Suite.")
st.markdown("---")

# ── Antenna Specifications ────────────────────────────────────
st.subheader("📋 Antenna Specifications")
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Type", "Microstrip Patch")
    st.metric("Target Frequency", "2.4 GHz")
    st.metric("Achieved Frequency", "2.401 GHz")
with col2:
    st.metric("Substrate", "FR4")
    st.metric("Dielectric Constant", "εr = 4.3")
    st.metric("Substrate Height", "1.6 mm")
with col3:
    st.metric("S11", "-31.77 dB")
    st.metric("Gain", "6.15 dBi")
    st.metric("Tool", "CST Studio Suite")

st.markdown("---")

# ── Antenna Dimensions ────────────────────────────────────────
st.subheader("📐 Final Antenna Dimensions")
params_data = {
    'Parameter': ['Patch Length (PL)', 'Patch Width (PW)',
                  'Inset Notch Length (INL)', 'Feed Line Length (ML)',
                  'Substrate Height (SH)', 'Ground Plane'],
    'Value': ['29.45 mm', '38.39 mm', '7.84 mm',
              '17.29 mm', '1.6 mm', '60 × 50 mm'],
    'Description': [
        'Controls resonant frequency',
        'Controls bandwidth',
        'Controls impedance matching',
        'Feed line from port to patch',
        'FR4 substrate thickness',
        'Full ground plane'
    ]
}
params_df = pd.DataFrame(params_data)
st.dataframe(params_df, use_container_width=True)

st.markdown("---")

# ── 3D Antenna Model ──────────────────────────────────────────
st.subheader("🏗️ 3D Antenna Model")
try:
    st.image(os.path.join(IMAGES_DIR, 'antenna_3d.png'),
            caption='Rectangular Microstrip Patch Antenna with Inset Feed — CST Studio Suite',
            use_container_width=True)
except:
    st.warning("Antenna 3D model image not found")

st.markdown("---")

# ── S11 Result ────────────────────────────────────────────────
st.subheader("📉 S11 Return Loss Result")
col1, col2 = st.columns([2, 1])

with col1:
    try:
        st.image(os.path.join(IMAGES_DIR, 's11_graph.png'),
                caption='S11 vs Frequency — Resonance at 2.401 GHz, S11 = -28.43 dB',
                use_container_width=True)
    except:
        st.warning("S11 graph image not found")

with col2:
    st.markdown("**Key Results:**")
    st.success("✅ Resonant Frequency: 2.401 GHz")
    st.success("✅ S11 = -28.43 dB")
    st.success("✅ Well below -10 dB threshold")
    st.info("**What this means:**\nOnly 0.14% of input power is reflected back. 99.86% is radiated — excellent impedance matching!")

st.markdown("---")

# ── Radiation Pattern ─────────────────────────────────────────
st.subheader("📡 Radiation Pattern")
col1, col2 = st.columns(2)

with col1:
    try:
        st.image(os.path.join(IMAGES_DIR, 'radiation_pattern_2d.png'),
                caption='2D Radiation Pattern — Main lobe: 6.15 dBi at 179°',
                use_container_width=True)
    except:
        st.warning("2D radiation pattern image not found")

with col2:
    try:
        st.image(os.path.join(IMAGES_DIR, 'radiation_pattern_3d.png'),
                caption='3D Radiation Pattern — Broadside radiation',
                use_container_width=True)
    except:
        st.warning("3D radiation pattern image not found")

st.markdown("**Radiation Pattern Results:**")
col_a, col_b, col_c = st.columns(3)
with col_a:
    st.metric("Gain", "6.15 dBi")
with col_b:
    st.metric("Main Lobe Direction", "179°")
with col_c:
    st.metric("Side Lobe Level", "-11.8 dB")

st.markdown("---")

# ── E-field Distribution ──────────────────────────────────────
st.subheader("⚡ E-Field Distribution")
col1, col2 = st.columns([2, 1])

with col1:
    try:
        st.image(os.path.join(IMAGES_DIR, 'efield.png'),
                caption='E-Field Distribution at 2.401 GHz — Maximum: 9900.09 V/m',
                use_container_width=True)
    except:
        st.warning("E-field image not found")

with col2:
    st.markdown("**E-Field Analysis:**")
    st.info("🔴 **Red:** Maximum field at feed point")
    st.info("🟢 **Green:** High field at patch edges (radiating edges)")
    st.info("🔵 **Blue:** Low field in patch center")
    st.markdown("This confirms correct antenna operation — field concentrates at radiating edges.")

st.markdown("---")

# ── Simulation Summary ────────────────────────────────────────
st.subheader("✅ Simulation Summary")
summary_data = {
    'Parameter': ['Resonant Frequency', 'S11', 'Gain',
                  'Main Lobe Direction', 'Side Lobe Level',
                  'Radiation Efficiency', 'Total Efficiency'],
    'Value': ['2.401 GHz', '-28.43 dB', '6.15 dBi',
              '179°', '-11.8 dB', '-3.352 dB', '-3.358 dB'],
    'Status': ['✅ Excellent', '✅ Excellent', '✅ Good',
               '✅ Broadside', '✅ Acceptable',
               '✅ Good', '✅ Good']
}
summary_df = pd.DataFrame(summary_data)
st.dataframe(summary_df, use_container_width=True)