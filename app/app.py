import json
import io
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import joblib

# Paths & loading
ART_DIR = Path("notebooks/artifacts")  # adjust if you keep artifacts elsewhere
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
st.write("CSV should include the **same feature columns** used to train the model "
         "(after dropping identifiers and target in Phase 2).")

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
