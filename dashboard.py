import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO
from PIL import Image
import io

# Page config
st.set_page_config(
    page_title="Uncertainty-Aware CV System",
    page_icon="🎯",
    layout="wide"
)

# Load model
@st.cache_resource
def load_model():
    return YOLO('yolov8n.pt')

model = load_model()

# Load predictions data
@st.cache_data
def load_predictions():
    return pd.read_csv('predictions.csv')

df = load_predictions()

# --- SIDEBAR ---
st.sidebar.title("⚙️ System Controls")
st.sidebar.markdown("---")

threshold = st.sidebar.slider(
    "Confidence Threshold",
    min_value=0.1,
    max_value=0.99,
    value=0.40,
    step=0.01,
    help="Detections above this threshold are flagged for review"
)

fn_cost = st.sidebar.slider(
    "False Negative Cost Weight",
    min_value=1,
    max_value=20,
    value=5,
    help="How much worse is missing a threat vs a false alarm?"
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 Dataset Stats")
st.sidebar.metric("Total Predictions", len(df))
st.sidebar.metric("Overall Accuracy", f"{df['correct'].mean():.1%}")
st.sidebar.metric("ECE Score", "0.1271")

# --- MAIN PAGE ---
st.title("🎯 Uncertainty-Aware Computer Vision")
st.markdown("### Human-in-the-Loop Safety Review System")
st.markdown("---")

# Top metrics
col1, col2, col3, col4 = st.columns(4)

flagged = df[df['confidence'] >= threshold]
not_flagged = df[df['confidence'] < threshold]

fp = ((flagged['correct'] == 0)).sum()
fn = ((not_flagged['correct'] == 1)).sum()
tp = ((flagged['correct'] == 1)).sum()

with col1:
    st.metric("Flagged for Review", len(flagged))
with col2:
    st.metric("False Positives", fp)
with col3:
    st.metric("Missed Threats", fn)
with col4:
    total_cost = fp * 1 + fn * fn_cost
    st.metric("Total Cost", total_cost)

st.markdown("---")

# Two columns layout
left, right = st.columns(2)

with left:
    st.markdown("### 📈 Calibration Analysis")
    
    # Reliability diagram
    n_bins = 10
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_accuracy = []
    bin_confidence = []
    bin_counts = []

    for i in range(n_bins):
        low = bin_edges[i]
        high = bin_edges[i+1]
        mask = (df['confidence'] >= low) & (df['confidence'] < high)
        bin_df = df[mask]
        if len(bin_df) > 0:
            bin_accuracy.append(bin_df['correct'].mean())
            bin_confidence.append(bin_df['confidence'].mean())
            bin_counts.append(len(bin_df))
        else:
            bin_accuracy.append(0)
            bin_confidence.append((low + high) / 2)
            bin_counts.append(0)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration', linewidth=2)
    ax.bar(bin_confidence, bin_accuracy, width=0.08, alpha=0.7,
           color='steelblue', label='Model Calibration', edgecolor='black')
    ax.axvline(x=threshold, color='red', linestyle='--',
               linewidth=2, label=f'Current Threshold: {threshold:.2f}')
    ax.set_xlabel('Confidence Score')
    ax.set_ylabel('Actual Accuracy')
    ax.set_title('Reliability Diagram')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)
    plt.close()

with right:
    st.markdown("### 💰 Threshold Cost Analysis")
    
    thresholds = np.linspace(0.1, 0.99, 200)
    costs = []
    for t in thresholds:
        f = df['confidence'] >= t
        nf = ~f
        fp_t = ((f) & (df['correct'] == 0)).sum()
        fn_t = ((nf) & (df['correct'] == 1)).sum()
        costs.append(fp_t * 1 + fn_t * fn_cost)

    optimal_idx = np.argmin(costs)
    optimal_t = thresholds[optimal_idx]

    fig2, ax2 = plt.subplots(figsize=(6, 4))
    ax2.plot(thresholds, costs, color='darkred', linewidth=2)
    ax2.axvline(x=threshold, color='red', linestyle='--',
                linewidth=2, label=f'Current: {threshold:.2f}')
    ax2.axvline(x=optimal_t, color='green', linestyle='--',
                linewidth=2, label=f'Optimal: {optimal_t:.2f}')
    ax2.set_xlabel('Threshold')
    ax2.set_ylabel('Total Cost')
    ax2.set_title('Cost vs Threshold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    st.pyplot(fig2)
    plt.close()

st.markdown("---")

# Live detection section
st.markdown("### 🔍 Live Detection & Human Review")
st.markdown("Upload an image to run detection and review flagged objects.")

uploaded_file = st.file_uploader("Upload an image", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    
    # Run detection
    with st.spinner("Running detection..."):
        results = model(image, conf=0.1)
    
    col_img, col_detections = st.columns(2)
    
    with col_img:
        st.markdown("**Detection Results**")
        result_img = results[0].plot()
        st.image(result_img, channels="BGR", use_container_width=True)
    
    with col_detections:
        st.markdown("**Detections requiring review**")
        
        boxes = results[0].boxes
        if len(boxes) == 0:
            st.info("No detections found.")
        else:
            for i, box in enumerate(boxes):
                confidence = float(box.conf)
                class_id = int(box.cls)
                class_name = results[0].names[class_id]
                
                needs_review = confidence >= threshold
                
                if needs_review:
                    status = "🔴 FLAGGED FOR REVIEW"
                    color = "red"
                else:
                    status = "🟢 Below threshold"
                    color = "green"
                
                with st.expander(f"{class_name} — Confidence: {confidence:.2f} | {status}"):
                    st.progress(confidence)
                    st.markdown(f"**Class:** {class_name}")
                    st.markdown(f"**Confidence:** {confidence:.3f}")
                    st.markdown(f"**Threshold:** {threshold:.2f}")
                    
                    if needs_review:
                        st.warning("This detection requires human review before any action.")
                        col_a, col_b = st.columns(2)
                        with col_a:
                            if st.button(f"✅ Confirm", key=f"confirm_{i}"):
                                st.success("Confirmed as real threat.")
                        with col_b:
                            if st.button(f"❌ Dismiss", key=f"dismiss_{i}"):
                                st.info("Dismissed as false positive.")
                    else:
                        st.info("Below threshold — no review needed.")

st.markdown("---")
st.markdown("*Uncertainty-Aware CV System | ICME Data Science Project*")