# app/app.py
import json
import io
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import joblib

# =============================
# Paths & page config
# =============================
ART_DIR = Path("notebooks/artifacts")
CFG_PATH = ART_DIR / "deployment_config.json"
PHASE2_PATH = Path("data/diabetic_data_clean.csv")
st.set_page_config(page_title="🏥 Readmission Risk Scorer", layout="wide")

# =============================
# Static schema (fallback)
# =============================
CAT_COLS_STATIC = [
    "race", "gender", "age", "medical_specialty",
    "diag_1", "diag_2", "diag_3",
    "A1Cresult", "change", "diabetesMed",
    "metformin", "repaglinide", "nateglinide", "chlorpropamide", "glimepiride",
    "acetohexamide", "glipizide", "glyburide", "tolbutamide", "pioglitazone",
    "rosiglitazone", "acarbose", "miglitol", "troglitazone", "tolazamide",
    "examide", "citoglipton", "insulin", "glyburide-metformin",
    "glipizide-metformin", "glimepiride-pioglitazone", "metformin-rosiglitazone",
    "metformin-pioglitazone"
]
NUM_COLS_STATIC = [
    "admission_type_id", "discharge_disposition_id", "admission_source_id",
    "time_in_hospital", "num_lab_procedures", "num_procedures", "num_medications",
    "number_outpatient", "number_emergency", "number_inpatient", "number_diagnoses"
]

# =============================
# Helpers
# =============================
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
    thr = float(cfg.get("threshold", 0.5))
    return cfg, model, thr

def get_expected_columns_from_pipeline(pipeline):
    """Infer original feature columns from ColumnTransformer inside pipeline, if present."""
    try:
        ct = pipeline.named_steps["prep"]
    except Exception:
        return None, None
    name_to_cols = {name: cols for name, trans, cols in ct.transformers_ if name != "remainder"}
    cat_cols = list(name_to_cols.get("cat", []) or [])
    num_cols = list(name_to_cols.get("num", []) or [])
    return cat_cols, num_cols

def coerce_to_schema(df, cats, nums):
    """Force numeric columns to numeric and cats to object to prevent float<str issues."""
    dfc = df.copy()
    for c in nums:
        if c in dfc.columns:
            dfc[c] = pd.to_numeric(dfc[c], errors="coerce")
    for c in cats:
        if c in dfc.columns:
            dfc[c] = dfc[c].astype("object")
    return dfc

def find_non_numeric(df, num_cols):
    bad = {}
    for c in num_cols:
        if c in df.columns:
            s = pd.to_numeric(df[c], errors="coerce")
            n_bad = int(s.isna().sum() - df[c].isna().sum())
            if n_bad > 0:
                bad[c] = n_bad
    return bad

# =============================
# Load model & derive columns
# =============================
st.title("🏥 30-Day Readmission Risk")
st.caption("Calibrated ML model (LogReg/RandomForest)")

try:
    cfg, model, saved_thr = load_config_and_model()
    st.success(
        f"Loaded model: `{cfg['model']}` • Saved threshold: {saved_thr:.3f} • "
        f"PR AUC: {cfg.get('metric_pr_auc','?')} • ROC AUC: {cfg.get('metric_roc_auc','?')}"
    )
except Exception as e:
    st.error(f"Could not load model/config: {e}")
    st.stop()

# Try to get the actual expected columns from the pipeline; fallback to static lists
cat_cols_pipe, num_cols_pipe = get_expected_columns_from_pipeline(model)
if (cat_cols_pipe or num_cols_pipe):
    REQUIRED_COLS_DYNAMIC = list(cat_cols_pipe) + list(num_cols_pipe)
    CATS_IN_USE = list(cat_cols_pipe)
    NUMS_IN_USE = list(num_cols_pipe)
else:
    REQUIRED_COLS_DYNAMIC = CAT_COLS_STATIC + NUM_COLS_STATIC
    CATS_IN_USE = CAT_COLS_STATIC
    NUMS_IN_USE = NUM_COLS_STATIC

with st.sidebar:
    st.header("Settings")

    # Use a slider with a stable session key so we can reset it
    st.slider(
        "Decision threshold",
        min_value=0.0, max_value=1.0, step=0.005,
        value=float(saved_thr),
        key="thr_slider",
        help="Predictions ≥ threshold → positive (30-day readmission).",
    )
    # Read the current threshold value
    thr = float(st.session_state["thr_slider"])

    # Quick actions
    cols = st.columns(2)
    with cols[0]:
        if st.button("Reset to saved"):
            st.session_state["thr_slider"] = float(saved_thr)
    with cols[1]:
        st.caption(f"Saved: **{saved_thr:.3f}**")

    st.divider()
    st.subheader("What does threshold do?")
    st.markdown(
        """
- The model outputs a **probability** for each patient (0–1).
- The **threshold** converts that probability into a **Yes/No** decision:
    - If **prob ≥ threshold** → **READMIT (positive)**
    - If **prob < threshold** → **NO READMIT (negative)**
- **Lower threshold (e.g., 0.20)** → *more* patients flagged  
  → higher **recall** (catch more true cases)  
  → lower **precision** (more false alarms)
- **Higher threshold (e.g., 0.70)** → *fewer* patients flagged  
  → higher **precision** (fewer false alarms)  
  → lower **recall** (miss more true cases)
        """
    )



# =============================
# Required columns & template
# =============================
st.subheader("Required feature columns")
st.write("Your CSV must include these **exact** columns (identifiers & targets already removed):")
st.code(", ".join(REQUIRED_COLS_DYNAMIC), language="text")

template_csv = io.StringIO()
pd.DataFrame(columns=REQUIRED_COLS_DYNAMIC).to_csv(template_csv, index=False)
st.download_button("Download empty CSV template (headers only)",
                   data=template_csv.getvalue(),
                   file_name="readmission_features_template.csv",
                   mime="text/csv")

# =============================
# Upload CSV for batch scoring (per-row predictions)
# =============================
st.subheader("Upload CSV to score (per-row)")
uploaded = st.file_uploader("Choose a CSV with the required columns", type=["csv"])

if uploaded:
    try:
        df_in = pd.read_csv(uploaded)
    except Exception as e:
        st.error(f"Failed to read CSV: {e}")
        st.stop()

    st.write("Preview of uploaded data:")
    st.dataframe(df_in.head(10), use_container_width=True)

    # Let user optionally pick an ID column to carry through
    with st.expander("Optional: choose an ID column to keep in the results"):
        id_col = st.selectbox(
            "ID column (optional)", 
            options=["(none)"] + list(df_in.columns),
            index=0
        )
        id_col = None if id_col == "(none)" else id_col

    # Validate required columns
    missing = [c for c in REQUIRED_COLS_DYNAMIC if c not in df_in.columns]
    if missing:
        st.error(f"CSV is missing required columns ({len(missing)}). Examples: {missing[:12]}")
        st.stop()

    # Align order & coerce types strictly
    df_use = df_in[REQUIRED_COLS_DYNAMIC].copy()
    badmap = find_non_numeric(df_use, NUMS_IN_USE)
    if badmap:
        st.warning(f"Found non-numeric values in numeric columns (coercing to NaN): {badmap}")
    df_use = coerce_to_schema(df_use, CATS_IN_USE, NUMS_IN_USE)

    # --- Predict per-row ---
    try:
        proba = model.predict_proba(df_use)[:, 1]              # one probability per row
        yhat = (proba >= thr).astype(int)                      # 0/1 per row
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        st.stop()

    # Build results table
    out_cols = []
    if id_col is not None and id_col in df_in.columns:
        out_cols.append(id_col)

    results = pd.DataFrame({
        **({id_col: df_in[id_col]} if id_col else {}),
        "readmit_30_proba": proba,
        f"readmit_30_pred_at_{thr:.3f}": yhat
    })

    # Optional: join back any original columns you want to display
    # Here we show only the selected ID (if any) + predictions.
    # If you want more context, uncomment next line to include everything:
    # results = pd.concat([df_in, results[["readmit_30_proba", f"readmit_30_pred_at_{thr:.3f}"]]], axis=1)

    # Sort by risk descending for convenience
    results_sorted = results.sort_values("readmit_30_proba", ascending=False).reset_index(drop=True)

    # KPIs
    pos_rate = float((yhat == 1).mean())
    st.markdown(
        f"**Predicted positive rate** at threshold {thr:.3f}: **{pos_rate:.2%}**  "
        f"• Avg risk: **{results['readmit_30_proba'].mean():.3f}**  "
        f"• Max risk: **{results['readmit_30_proba'].max():.3f}**"
    )

    # Nice display: format probability as %
    display_df = results_sorted.copy()
    display_df["risk_%"] = (display_df["readmit_30_proba"] * 100).round(1).astype(str) + "%"
    pred_col = f"readmit_30_pred_at_{thr:.3f}"
    display_df["prediction"] = np.where(display_df[pred_col] == 1, "READMIT", "NO READMIT")

    show_cols = ([id_col] if id_col else []) + ["risk_%", "prediction"]
    st.dataframe(display_df[show_cols].head(50), use_container_width=True)
    st.caption("Showing top 50 by risk. Download below to get all rows.")

    # Download full results (numerical proba retained)
    buf = io.StringIO()
    results_sorted.to_csv(buf, index=False)
    st.download_button(
        "Download full predictions CSV",
        data=buf.getvalue(),
        file_name="predictions_with_risk.csv",
        mime="text/csv"
    )

# =============================
# Single-patient manual form
# =============================
with st.expander("Or enter a single patient manually"):
    with st.form("single_patient_form"):
        age_buckets = ["[0-10)","[10-20)","[20-30)","[30-40)","[40-50)","[50-60)",
                       "[60-70)","[70-80)","[80-90)","[90-100)"]
        yn_med4 = ["No","Steady","Up","Down"]
        a1c_opts = ["None","Norm",">7",">8"]
        change_opts = ["No","Ch"]
        diabmed_opts = ["No","Yes"]
        race_opts = ["Caucasian","AfricanAmerican","Asian","Hispanic","Other","?"]
        gender_opts = ["Male","Female","Unknown/Invalid"]

        # Simple defaults; you can refine options further if you have domain maps
        med_spec_opts = ["Unknown","Cardiology","Emergency/Trauma","Family/GeneralPractice",
                         "InternalMedicine","Surgery-General","Orthopedics","Endocrinology","?"]

        c1, c2 = st.columns(2)

        with c1:
            race = st.selectbox("race", race_opts, index=0)
            gender = st.selectbox("gender", gender_opts, index=1)
            age = st.selectbox("age", age_buckets, index=7)
            medical_specialty = st.selectbox("medical_specialty", med_spec_opts, index=0)

            admission_type_id = st.number_input("admission_type_id", 1, 8, 1, 1)
            discharge_disposition_id = st.number_input("discharge_disposition_id", 1, 30, 1, 1)
            admission_source_id = st.number_input("admission_source_id", 1, 25, 1, 1)

            time_in_hospital = st.number_input("time_in_hospital", 1, 14, 4, 1)
            num_lab_procedures = st.number_input("num_lab_procedures", 0, 150, 44, 1)
            num_procedures = st.number_input("num_procedures", 0, 6, 0, 1)
            num_medications = st.number_input("num_medications", 0, 100, 16, 1)
            number_outpatient = st.number_input("number_outpatient", 0, 20, 0, 1)
            number_emergency = st.number_input("number_emergency", 0, 20, 0, 1)
            number_inpatient = st.number_input("number_inpatient", 0, 20, 0, 1)
            number_diagnoses = st.number_input("number_diagnoses", 1, 16, 8, 1)
            diag_1 = st.text_input("diag_1 (e.g., 250.00)", value="250.00")
            diag_2 = st.text_input("diag_2", value="401.9")
            diag_3 = st.text_input("diag_3", value="414")

        with c2:
            A1Cresult = st.selectbox("A1Cresult", a1c_opts, index=0)
            change = st.selectbox("change", change_opts, index=0)
            diabetesMed = st.selectbox("diabetesMed", diabmed_opts, index=1)

            def med(name): return st.selectbox(name, yn_med4, index=0)
            metformin = med("metformin"); repaglinide = med("repaglinide"); nateglinide = med("nateglinide")
            chlorpropamide = med("chlorpropamide"); glimepiride = med("glimepiride"); acetohexamide = med("acetohexamide")
            glipizide = med("glipizide"); glyburide = med("glyburide"); tolbutamide = med("tolbutamide")
            pioglitazone = med("pioglitazone"); rosiglitazone = med("rosiglitazone"); acarbose = med("acarbose")
            miglitol = med("miglitol"); troglitazone = med("troglitazone"); tolazamide = med("tolazamide")
            examide = med("examide"); citoglipton = med("citoglipton"); insulin = med("insulin")
            glyburide_metformin = med("glyburide-metformin"); glipizide_metformin = med("glipizide-metformin")
            glimepiride_pioglitazone = med("glimepiride-pioglitazone")
            metformin_rosiglitazone = med("metformin-rosiglitazone")
            metformin_pioglitazone = med("metformin-pioglitazone")

        submitted = st.form_submit_button("Predict for this patient")
        if submitted:
            row = {
                "race": race, "gender": gender, "age": age, "medical_specialty": medical_specialty,
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

            X_one = pd.DataFrame([row])
            X_one = X_one[[c for c in REQUIRED_COLS_DYNAMIC if c in X_one.columns]]
            X_one = coerce_to_schema(X_one, CATS_IN_USE, NUMS_IN_USE)

            try:
                p = float(model.predict_proba(X_one)[:, 1][0])
            except Exception as e:
                st.error(f"Prediction failed: {e}")
                st.stop()

            yhat = int(p >= thr)
            st.success(
                f"Predicted probability of 30-day readmission: **{p:.3f}**  "
                f"→ decision at threshold {thr:.3f}: **{yhat}** "
                f"({'READMIT' if yhat==1 else 'NO READMIT'})"
            )

# =============================
# Generate a sample CSV
# =============================
st.subheader("Need a sample CSV?")
with st.expander("Generate sample cleaned data"):
    if not PHASE2_PATH.exists():
        st.error(f"Could not find {PHASE2_PATH}. Place your file there.")
    else:
        n_rows = st.number_input("Number of rows", 5, 1000, 20, 5)
        seed = st.number_input("Random seed", 0, 10_000, 42, 1)
        if st.button("Create sample CSV"):
            try:
                df_phase2 = pd.read_csv(PHASE2_PATH)
                drop_cols = [c for c in ["encounter_id","patient_nbr","readmitted","readmit_30"] if c in df_phase2.columns]
                X_all = df_phase2.drop(columns=drop_cols)

                cols_final = [c for c in REQUIRED_COLS_DYNAMIC if c in X_all.columns]
                sample = X_all.sample(min(int(n_rows), len(X_all)), random_state=int(seed))[cols_final].copy()
                sample = coerce_to_schema(sample, CATS_IN_USE, NUMS_IN_USE)

                buf = io.StringIO()
                sample.to_csv(buf, index=False)
                st.success(f"Sample created with {len(sample)} rows and {len(cols_final)} columns.")
                st.download_button("⬇️ Download sample_inference.csv",
                                   data=buf.getvalue(),
                                   file_name="sample_inference.csv",
                                   mime="text/csv")
                st.write("Preview:")
                st.dataframe(sample.head(10), use_container_width=True)
            except Exception as e:
                st.error(f"Failed to create sample: {e}")

# =============================
# Optional: SHAP explanation (RandomForest)
# =============================
with st.expander("🔍 Explain a single row (SHAP, optional)"):
    try:
        from sklearn.ensemble import RandomForestClassifier
        import shap
        clf = model.named_steps.get("clf", None)
        if not isinstance(clf, RandomForestClassifier):
            st.info("SHAP demo available for RandomForest models only.")
        else:
            # Use the first row from last uploaded df_use if available; else from Phase 2
            if 'df_use' in locals() and len(df_use) > 0:
                explain_row = df_use.iloc[[0]]
            elif PHASE2_PATH.exists():
                base = pd.read_csv(PHASE2_PATH)
                drop_cols = [c for c in ["encounter_id","patient_nbr","readmitted","readmit_30"] if c in base.columns]
                base = base.drop(columns=drop_cols)
                base = base[[c for c in REQUIRED_COLS_DYNAMIC if c in base.columns]].head(1)
                explain_row = coerce_to_schema(base, CATS_IN_USE, NUMS_IN_USE)
            else:
                st.info("Upload a CSV or place the Phase 2 file to enable SHAP demo.")
                explain_row = None

            if explain_row is not None:
                ct = model.named_steps["prep"]
                X_tr = ct.transform(explain_row)
                explainer = shap.TreeExplainer(clf)
                X_arr = X_tr.toarray() if hasattr(X_tr, "toarray") else X_tr
                shap_vals = explainer.shap_values(X_arr)
                st.write("Approximate local explanation (model input space):")
                import matplotlib.pyplot as plt
                shap.force_plot(explainer.expected_value[1], shap_vals[1][0], X_arr[0], matplotlib=True)
                st.pyplot(plt.gcf(), clear_figure=True)
    except Exception as e:
        st.warning(f"SHAP explanation skipped: {e}")
