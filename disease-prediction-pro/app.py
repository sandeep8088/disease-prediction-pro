import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

import joblib
import pandas as pd
import streamlit as st

from recommend import DiseaseInfo

st.set_page_config(
    page_title="Disease Prediction System",
    page_icon="🩺",
    layout="centered",
    menu_items={"Get Help": None, "Report a bug": None, "About": None},
)

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model")


@st.cache_resource
def load_artifacts():
    model = joblib.load(os.path.join(MODEL_DIR, "best_model.pkl"))
    label_encoder = joblib.load(os.path.join(MODEL_DIR, "label_encoder.pkl"))
    feature_cols = joblib.load(os.path.join(MODEL_DIR, "feature_columns.pkl"))
    info = DiseaseInfo()
    return model, label_encoder, feature_cols, info


model, label_encoder, feature_cols, info = load_artifacts()


def pretty(symptom: str) -> str:
    return symptom.replace("_", " ").capitalize()


symptom_display_map = {pretty(s): s for s in feature_cols}

st.title("🩺 Disease Prediction System")

selected_display = st.multiselect(
    "Select your symptoms:",
    options=sorted(symptom_display_map.keys()),
    placeholder="Start typing, e.g. 'Fever'",
)
selected_symptoms = [symptom_display_map[s] for s in selected_display]

if st.button("Predict", type="primary"):
    if not selected_symptoms:
        st.warning("Please select at least one symptom.")
    else:
        input_vector = pd.DataFrame(
            [[1 if s in selected_symptoms else 0 for s in feature_cols]],
            columns=feature_cols,
        )

        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(input_vector)[0]
            top_idx = probs.argsort()[::-1][:3]
            top_diseases = [
                (label_encoder.inverse_transform([i])[0], probs[i]) for i in top_idx
            ]
        else:
            pred = model.predict(input_vector)[0]
            top_diseases = [(label_encoder.inverse_transform([pred])[0], 1.0)]

        primary_disease, primary_conf = top_diseases[0]

        st.subheader(f"Most likely: {primary_disease}")
        st.progress(min(float(primary_conf), 1.0), text=f"{primary_conf:.0%} confidence")

        if len(top_diseases) > 1:
            with st.expander("Other possible matches"):
                for disease, conf in top_diseases[1:]:
                    if conf > 0.01:
                        st.write(f"- {disease} ({conf:.0%})")

        st.markdown("### About this condition")
        st.write(info.get_description(primary_disease))

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### Precautions")
            for p in info.get_precautions(primary_disease):
                st.write(f"- {p}")
        with col2:
            st.markdown("### Suggested medications")
            for m in info.get_medications(primary_disease):
                st.write(f"- {m}")

        st.markdown("### Dietary suggestions")
        st.write(", ".join(info.get_diet(primary_disease)))

        st.warning(
            "This tool is for educational purposes only and does not replace "
            "professional medical diagnosis. Please consult a doctor for any "
            "health concerns."
        )
