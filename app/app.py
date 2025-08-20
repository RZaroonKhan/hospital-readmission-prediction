import json
import io
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import joblib

# Paths & loading
ART_DIR = Path("notebooks/artifacts") 
CFG_PATH = ART_DIR / "deployment_config.json"

st.set_page_config(page_title="Readmission Risk Scorer", layout="wide")

@st.cache_resource(show_spinner=False)
def load_config_and_model():
    if not CFG_PATH.exists():
        raise FileNotFoundError(f"Config not found at {CFG_PATH}")
    with open(CFG_PATH, "r") as f:
        cfg = json.load(f)

    model_path = ART_DIR / cfg["model"]
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at {model_path}")

    model = joblib.load(model_path)
    threshold = float(cfg.get("threshold", 0.5))
    return cfg, model, threshold

def get_expected_columns_from_pipeline(pipeline):
    """Extract the original expected column names (cats/nums) from ColumnTransformer."""
    try:
        ct = pipeline.named_steps["prep"]  # ColumnTransformer
    except Exception:
        return None, None
    # Find by transformer name to be robust to ordering
    name_to_cols = {name: cols for name, trans, cols in ct.transformers_ if name != "remainder"}
    cat_cols = name_to_cols.get("cat", [])
    num_cols = name_to_cols.get("num", [])
    # Ensure list types
    cat_cols = list(cat_cols) if cat_cols is not None else []
    num_cols = list(num_cols) if num_cols is not None else []
    return cat_cols, num_cols

def coerce_types(df, cat_cols, num_cols):
    dfc = df.copy()
    # Coerce numeric
    for c in num_cols:
        if c in dfc.columns:
            dfc[c] = pd.to_numeric(dfc[c], errors="coerce")
    # Coerce categoricals
    for c in cat_cols:
        if c in dfc.columns:
            dfc[c] = dfc[c].astype("object")
    return dfc

# UI
st.title("🏥 30-Day Readmission Risk")
st.caption("Calibrated ML model from Phase 4 (LogReg/RandomForest via scikit-learn).")

# Load model + config
try:
    cfg, model, saved_thr = load_config_and_model()
    st.success(f"Loaded model: `{cfg['model']}` | Default threshold: {saved_thr:.3f}")
except Exception as e:
    st.error(f"Could not load model/config: {e}")
    st.stop()

# Grab expected columns
cat_cols, num_cols = get_expected_columns_from_pipeline(model)
expected_cols = list(cat_cols) + list(num_cols) if cat_cols or num_cols else None

with st.sidebar:
    st.header("Settings")
    thr = st.slider("Decision threshold", 0.0, 1.0, float(saved_thr), 0.005,
                    help="Predictions ≥ threshold → positive (readmit within 30 days).")
    st.write("Model:", cfg.get("model", ""))
    st.write("Saved PR AUC:", cfg.get("metric_pr_auc", ""))
    st.write("Saved ROC AUC:", cfg.get("metric_roc_auc", ""))

st.subheader("Upload CSV")
st.write("CSV should include the **same feature columns** used to train the model")

# =========================
# Required feature columns
# =========================
required_cols = [
    # Demographics
    "race", "gender", "age",
    # Encounter / hospital info
    "admission_type_id", "discharge_disposition_id", "admission_source_id",
    # Utilization / counts
    "time_in_hospital", "num_lab_procedures", "num_procedures", "num_medications",
    "number_outpatient", "number_emergency", "number_inpatient", "number_diagnoses",
    # Diagnoses (categorical strings from ICD groups)
    "diag_1", "diag_2", "diag_3",
    # Labs / diabetes flags
    "A1Cresult", "change", "diabetesMed",
    # Medication columns (categorical: No/Steady/Up/Down)
    "metformin", "repaglinide", "nateglinide", "chlorpropamide", "glimepiride",
    "acetohexamide", "glipizide", "glyburide", "tolbutamide", "pioglitazone",
    "rosiglitazone", "acarbose", "miglitol", "troglitazone", "tolazamide",
    "examide", "citoglipton", "insulin", "glyburide-metformin",
    "glipizide-metformin", "glimepiride-pioglitazone", "metformin-rosiglitazone",
    "metformin-pioglitazone"
]

st.subheader("Required columns")
st.write("Your CSV must include these **exact** columns:")
st.code(", ".join(required_cols), language="text")

# Offer a downloadable template with headers only
import io
template_csv = io.StringIO()
pd.DataFrame(columns=required_cols).to_csv(template_csv, index=False)
st.download_button("Download empty CSV template (headers only)",
                   data=template_csv.getvalue(),
                   file_name="readmission_features_template.csv",
                   mime="text/csv")

# =========================
# Generate a sample CSV
# =========================
st.subheader("Need a sample CSV?")
st.write("Generate a small CSV with the exact feature columns expected by the model.")

with st.expander("Generate sample from cleaned data"):
    # Path to clean dataset (adjust if yours is elsewhere)
    phase2_path = Path("../data/diabetic_data_clean.csv")
    n_default = 20

    if not phase2_path.exists():
        st.error(f"Could not find {phase2_path}. Make sure your file is present.")
    else:
        n_rows = st.number_input("Number of rows", min_value=5, max_value=1000, value=n_default, step=5,
                                 help="How many random rows to include in the sample CSV.")
        seed = st.number_input("Random seed", min_value=0, max_value=10_000, value=42, step=1)

        if st.button("Create sample CSV"):
            try:
                df_phase2 = pd.read_csv(phase2_path)

                # Drop identifiers + target to match training features
                drop_cols = [c for c in ["encounter_id", "patient_nbr", "readmitted", "readmit_30"] if c in df_phase2.columns]
                X_all = df_phase2.drop(columns=drop_cols)

                # If we can infer expected columns from the fitted pipeline, enforce order/subset
                if expected_cols:
                    missing = [c for c in expected_cols if c not in X_all.columns]
                    if missing:
                        st.warning(f"The Phase 2 file is missing expected columns (showing first 10): {missing[:10]}")
                        # proceed with intersection
                        cols_use = [c for c in expected_cols if c in X_all.columns]
                    else:
                        cols_use = expected_cols
                else:
                    cols_use = X_all.columns.tolist()

                # Random sample (or all if small)
                n_take = min(int(n_rows), len(X_all))
                sample = X_all.sample(n_take, random_state=int(seed))[cols_use].copy()

                # Coerce types to match model expectations (optional but helpful)
                if expected_cols:
                    sample = coerce_types(sample, cat_cols, num_cols)

                # Offer download
                import io
                buf = io.StringIO()
                sample.to_csv(buf, index=False)
                st.success(f"Sample created with {n_take} rows and {len(cols_use)} columns.")
                st.download_button(
                    "⬇️ Download sample_inference.csv",
                    data=buf.getvalue(),
                    file_name="sample_inference.csv",
                    mime="text/csv"
                )

                st.write("Preview:")
                st.dataframe(sample.head(10), use_container_width=True)

            except Exception as e:
                st.error(f"Failed to create sample: {e}")


# =========================
# Enter a single patient manually
# =========================
with st.expander("Or enter a single patient manually"):
    with st.form("single_patient_form"):
        # Categorical option sets (based on the UCI diabetes dataset values)
        age_buckets = ["[0-10)","[10-20)","[20-30)","[30-40)","[40-50)","[50-60)","[60-70)","[70-80)","[80-90)","[90-100)"]
        yn_med4 = ["No","Steady","Up","Down"]  # for medication columns
        a1c_opts = ["None","Norm",">7",">8"]
        change_opts = ["No","Ch"]
        diabmed_opts = ["No","Yes"]
        race_opts = ["Caucasian","AfricanAmerican","Asian","Hispanic","Other","?"]
        gender_opts = ["Male","Female","Unknown/Invalid"] 

        # Layout: two columns for compactness
        c1, c2 = st.columns(2)

        with c1:
            race = st.selectbox("race", race_opts, index=0)
            gender = st.selectbox("gender", gender_opts, index=1)
            age = st.selectbox("age", age_buckets, index=7)
            admission_type_id = st.number_input("admission_type_id", min_value=1, max_value=8, value=1, step=1)
            discharge_disposition_id = st.number_input("discharge_disposition_id", min_value=1, max_value=30, value=1, step=1)
            admission_source_id = st.number_input("admission_source_id", min_value=1, max_value=25, value=1, step=1)
            time_in_hospital = st.number_input("time_in_hospital", min_value=1, max_value=14, value=4, step=1)
            num_lab_procedures = st.number_input("num_lab_procedures", min_value=0, max_value=150, value=44, step=1)
            num_procedures = st.number_input("num_procedures", min_value=0, max_value=6, value=0, step=1)
            num_medications = st.number_input("num_medications", min_value=0, max_value=100, value=16, step=1)
            number_outpatient = st.number_input("number_outpatient", min_value=0, max_value=20, value=0, step=1)
            number_emergency = st.number_input("number_emergency", min_value=0, max_value=20, value=0, step=1)
            number_inpatient = st.number_input("number_inpatient", min_value=0, max_value=20, value=0, step=1)
            number_diagnoses = st.number_input("number_diagnoses", min_value=1, max_value=16, value=8, step=1)
            diag_1 = st.text_input("diag_1 (e.g., 250.0 or 428)", value="250.00")
            diag_2 = st.text_input("diag_2", value="401.9")
            diag_3 = st.text_input("diag_3", value="414")

        with c2:
            A1Cresult = st.selectbox("A1Cresult", a1c_opts, index=0)
            change = st.selectbox("change", change_opts, index=0)
            diabetesMed = st.selectbox("diabetesMed", diabmed_opts, index=1)

            # Med flags (No/Steady/Up/Down)
            metformin = st.selectbox("metformin", yn_med4, index=0)
            repaglinide = st.selectbox("repaglinide", yn_med4, index=0)
            nateglinide = st.selectbox("nateglinide", yn_med4, index=0)
            chlorpropamide = st.selectbox("chlorpropamide", yn_med4, index=0)
            glimepiride = st.selectbox("glimepiride", yn_med4, index=0)
            acetohexamide = st.selectbox("acetohexamide", yn_med4, index=0)
            glipizide = st.selectbox("glipizide", yn_med4, index=0)
            glyburide = st.selectbox("glyburide", yn_med4, index=0)
            tolbutamide = st.selectbox("tolbutamide", yn_med4, index=0)
            pioglitazone = st.selectbox("pioglitazone", yn_med4, index=0)
            rosiglitazone = st.selectbox("rosiglitazone", yn_med4, index=0)
            acarbose = st.selectbox("acarbose", yn_med4, index=0)
            miglitol = st.selectbox("miglitol", yn_med4, index=0)
            troglitazone = st.selectbox("troglitazone", yn_med4, index=0)
            tolazamide = st.selectbox("tolazamide", yn_med4, index=0)
            examide = st.selectbox("examide", yn_med4, index=0)
            citoglipton = st.selectbox("citoglipton", yn_med4, index=0)
            insulin = st.selectbox("insulin", yn_med4, index=0)
            glyburide_metformin = st.selectbox("glyburide-metformin", yn_med4, index=0)
            glipizide_metformin = st.selectbox("glipizide-metformin", yn_med4, index=0)
            glimepiride_pioglitazone = st.selectbox("glimepiride-pioglitazone", yn_med4, index=0)
            metformin_rosiglitazone = st.selectbox("metformin-rosiglitazone", yn_med4, index=0)
            metformin_pioglitazone = st.selectbox("metformin-pioglitazone", yn_med4, index=0)

        submitted = st.form_submit_button("Predict for this patient")
        if submitted:
            row = {
                # left column
                "race": race, "gender": gender, "age": age,
                "admission_type_id": int(admission_type_id),
                "discharge_disposition_id": int(discharge_disposition_id),
                "admission_source_id": int(admission_source_id),
                "time_in_hospital": int(time_in_hospital),
                "num_lab_procedures": int(num_lab_procedures),
                "num_procedures": int(num_procedures),
                "num_medications": int(num_medications),
                "number_outpatient": int(number_outpatient),
                "number_emergency": int(number_emergency),
                "number_inpatient": int(number_inpatient),
                "number_diagnoses": int(number_diagnoses),
                "diag_1": diag_1, "diag_2": diag_2, "diag_3": diag_3,
                # right column
                "A1Cresult": A1Cresult, "change": change, "diabetesMed": diabetesMed,
                "metformin": metformin, "repaglinide": repaglinide, "nateglinide": nateglinide,
                "chlorpropamide": chlorpropamide, "glimepiride": glimepiride, "acetohexamide": acetohexamide,
                "glipizide": glipizide, "glyburide": glyburide, "tolbutamide": tolbutamide,
                "pioglitazone": pioglitazone, "rosiglitazone": rosiglitazone, "acarbose": acarbose,
                "miglitol": miglitol, "troglitazone": troglitazone, "tolazamide": tolazamide,
                "examide": examide, "citoglipton": citoglipton, "insulin": insulin,
                "glyburide-metformin": glyburide_metformin,
                "glipizide-metformin": glipizide_metformin,
                "glimepiride-pioglitazone": glimepiride_pioglitazone,
                "metformin-rosiglitazone": metformin_rosiglitazone,
                "metformin-pioglitazone": metformin_pioglitazone
            }

            # Align to expected training columns if we could infer them
            if expected_cols:
                missing = [c for c in expected_cols if c not in row]
                if missing:
                    st.error(f"Internal error: form missing columns: {missing[:10]}")
                    st.stop()
                X_one = pd.DataFrame([row])[expected_cols]
                X_one = coerce_types(X_one, cat_cols, num_cols)
            else:
                X_one = pd.DataFrame([row])

            # Predict with current threshold slider (thr)
            try:
                p = float(model.predict_proba(X_one)[:, 1][0])
            except Exception as e:
                st.error(f"Prediction failed: {e}")
                st.stop()

            yhat = int(p >= thr)
            st.success(f"Predicted probability of 30-day readmission: **{p:.3f}** "
                       f"→ decision at threshold {thr:.3f}: **{yhat}** "
                       f"({'READMIT' if yhat==1 else 'NO READMIT'})")


uploaded = st.file_uploader("Choose a CSV file", type=["csv"])

if uploaded:
    try:
        df_in = pd.read_csv(uploaded)
    except Exception as e:
        st.error(f"Failed to read CSV: {e}")
        st.stop()

    st.write("Preview:")
    st.dataframe(df_in.head(10), use_container_width=True)

    # If we can infer expected columns from the pipeline, validate & align
    if expected_cols:
        missing = [c for c in expected_cols if c not in df_in.columns]
        extra = [c for c in df_in.columns if c not in expected_cols]
        if missing:
            st.error(f"Your CSV is missing required columns ({len(missing)}). "
                     f"Examples: {missing[:10]}")
            st.stop()
        # Keep only expected columns, order them
        df_use = df_in[expected_cols].copy()
        df_use = coerce_types(df_use, cat_cols, num_cols)
        if extra:
            st.info(f"Ignoring {len(extra)} extra column(s) not used by the model.")
    else:
        # Fallback: try as-is (pipeline may handle columns internally)
        st.warning("Could not infer expected columns from pipeline; proceeding with your CSV as-is.")
        df_use = df_in.copy()

    # Predict calibrated probabilities
    try:
        proba = model.predict_proba(df_use)[:, 1]
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        st.stop()

    yhat = (proba >= thr).astype(int)
    out = df_in.copy()
    out["readmit_30_proba"] = proba
    out[f"readmit_30_pred_at_{thr:.3f}"] = yhat

    # Summary
    pos_rate = float((yhat == 1).mean())
    st.subheader("Results")
    st.write(f"Predicted **positive rate** at threshold {thr:.3f}: **{pos_rate:.3%}**")
    st.dataframe(out.head(20), use_container_width=True)

    # Download
    csv_buf = io.StringIO()
    out.to_csv(csv_buf, index=False)
    st.download_button("⬇️ Download predictions as CSV", data=csv_buf.getvalue(),
                       file_name="predictions_with_risk.csv", mime="text/csv")

    # Explain a single row with SHAP if model is RandomForest
    with st.expander("🔍 Explain a single row (SHAP, optional)"):
        try:
            from sklearn.ensemble import RandomForestClassifier
            import shap
            # Try to extract underlying RF from pipeline
            clf = model.named_steps.get("clf", None)
            is_rf = isinstance(clf, RandomForestClassifier)
            if not is_rf:
                st.info("SHAP demo is available for RandomForest models.")
            else:
                idx = st.number_input("Row index to explain", min_value=0, max_value=len(df_use)-1, value=0, step=1)
                explain_row = df_use.iloc[[idx]]
                # Transform through preprocessor to tree input space
                ct = model.named_steps["prep"]
                X_tr = ct.transform(explain_row)
                explainer = shap.TreeExplainer(clf)
                X_arr = X_tr.toarray() if hasattr(X_tr, "toarray") else X_tr
                shap_vals = explainer.shap_values(X_arr)
                st.write("Top positive contributors (approximate, model input space):")
                # We won't reconstruct feature names here (OneHot expands a lot); show a force plot image
                shap_fig = shap.force_plot(explainer.expected_value[1], shap_vals[1][0], X_arr[0], matplotlib=True)
                import matplotlib.pyplot as plt
                st.pyplot(plt.gcf(), clear_figure=True)
        except Exception as e:
            st.warning(f"SHAP explanation skipped: {e}")
else:
    st.info("Upload a CSV to score readmission risk.")
