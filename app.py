import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
from pathlib import Path


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="ExplainLoan AI",
    page_icon="💰",
    layout="wide"
)


# =========================================================
# LOAD MODEL
# =========================================================

MODEL_PATH = Path(__file__).parent / "notebooks" / "xgb_pipeline.pkl"

try:
    model = joblib.load(MODEL_PATH)
except Exception as e:
    st.error(f"Could not load model: {e}")
    st.stop()


# =========================================================
# TITLE
# =========================================================

st.title("💰 ExplainLoan AI")
st.subheader("AI-Based Explainable Loan Approval System")

st.write(
    "Enter applicant details to predict loan approval "
    "and understand which features influenced the decision."
)


# =========================================================
# INPUT SECTION
# =========================================================

st.markdown("## 👤 Applicant Details")

col1, col2 = st.columns(2)


with col1:

    no_of_dependents = st.number_input(
        "Number of Dependents",
        min_value=0,
        max_value=20,
        value=2
    )

    education = st.selectbox(
        "Education",
        ["Graduate", "Not Graduate"]
    )

    self_employed = st.selectbox(
        "Self Employed",
        ["Yes", "No"]
    )

    income_annum = st.number_input(
        "Annual Income",
        min_value=0,
        value=5000000
    )

    loan_amount = st.number_input(
        "Loan Amount",
        min_value=0,
        value=15000000
    )

    loan_term = st.number_input(
        "Loan Term",
        min_value=1,
        max_value=50,
        value=10
    )


with col2:

    cibil_score = st.number_input(
        "CIBIL Score",
        min_value=0,
        max_value=900,
        value=750
    )

    residential_assets_value = st.number_input(
        "Residential Assets Value",
        min_value=0,
        value=10000000
    )

    commercial_assets_value = st.number_input(
        "Commercial Assets Value",
        min_value=0,
        value=5000000
    )

    luxury_assets_value = st.number_input(
        "Luxury Assets Value",
        min_value=0,
        value=15000000
    )

    bank_asset_value = st.number_input(
        "Bank Asset Value",
        min_value=0,
        value=7000000
    )


# =========================================================
# PREDICTION
# =========================================================

if st.button("🔮 Predict Loan Status", type="primary"):

    # Create applicant dataframe
    new_applicant = pd.DataFrame([{
        "no_of_dependents": no_of_dependents,
        "education": education,
        "self_employed": self_employed,
        "income_annum": income_annum,
        "loan_amount": loan_amount,
        "loan_term": loan_term,
        "cibil_score": cibil_score,
        "residential_assets_value": residential_assets_value,
        "commercial_assets_value": commercial_assets_value,
        "luxury_assets_value": luxury_assets_value,
        "bank_asset_value": bank_asset_value
    }])


    # -----------------------------------------------------
    # Prediction
    # -----------------------------------------------------

    prediction = model.predict(new_applicant)[0]

    probabilities = model.predict_proba(new_applicant)[0]

    classes = model.classes_

    # Find probability for each class
    probability_dict = dict(zip(classes, probabilities))

    rejection_probability = probability_dict.get(0, 0)
    approval_probability = probability_dict.get(1, 0)


    # -----------------------------------------------------
    # Result
    # -----------------------------------------------------

    st.markdown("---")
    st.markdown("## 📊 Loan Prediction Result")


    if prediction == 1:

        st.success("✅ LOAN APPROVED")

    else:

        st.error("❌ LOAN REJECTED")


    col1, col2 = st.columns(2)


    with col1:

        st.metric(
            "Approval Probability",
            f"{approval_probability * 100:.2f}%"
        )


    with col2:

        st.metric(
            "Rejection Probability",
            f"{rejection_probability * 100:.2f}%"
        )


    # -----------------------------------------------------
    # Applicant Details
    # -----------------------------------------------------

    st.markdown("## 📋 Applicant Information")

    st.dataframe(
        new_applicant,
        use_container_width=True
    )


    # =====================================================
    # SHAP EXPLANATION
    # =====================================================

    st.markdown("---")
    st.markdown("## 🔍 Explainable AI — Why this decision?")


    try:

        # Get preprocessing pipeline
        preprocessor = model.named_steps["preprocessor"]

        # Get XGBoost model
        xgb_model = model.named_steps["model"]


        # Transform applicant data
        X_transformed = preprocessor.transform(new_applicant)


        # Get feature names
        feature_names = preprocessor.get_feature_names_out()


        # SHAP Tree Explainer
        explainer = shap.TreeExplainer(xgb_model)

        shap_values = explainer.shap_values(X_transformed)


        # -------------------------------------------------
        # Handle SHAP output
        # -------------------------------------------------

        if isinstance(shap_values, list):

            shap_values_for_plot = shap_values[1]

        else:

            shap_values_for_plot = shap_values


        # -------------------------------------------------
        # Feature Importance Bar Chart
        # -------------------------------------------------

        shap_importance = pd.DataFrame({
            "Feature": feature_names,
            "SHAP Value": shap_values_for_plot[0]
        })

        shap_importance["Absolute SHAP"] = (
            shap_importance["SHAP Value"].abs()
        )

        shap_importance = shap_importance.sort_values(
            "Absolute SHAP",
            ascending=False
        )


        st.write(
            "The following features had the strongest influence "
            "on this prediction:"
        )


        # Top 10 features
        top_features = shap_importance.head(10)


        fig, ax = plt.subplots(figsize=(10, 6))


        ax.barh(
            top_features["Feature"][::-1],
            top_features["SHAP Value"][::-1]
        )


        ax.set_xlabel("SHAP Value")
        ax.set_ylabel("Feature")
        ax.set_title("Feature Contribution to Loan Decision")


        plt.tight_layout()

        st.pyplot(fig)


        # -------------------------------------------------
        # Explanation Table
        # -------------------------------------------------

        st.markdown("### 📌 Feature Contributions")

        explanation_table = top_features[
            ["Feature", "SHAP Value"]
        ].copy()

        explanation_table["Impact"] = explanation_table[
            "SHAP Value"
        ].apply(
            lambda x: "Positive" if x > 0 else "Negative"
        )


        st.dataframe(
            explanation_table,
            use_container_width=True
        )


        # -------------------------------------------------
        # Simple Explanation
        # -------------------------------------------------

        top_feature = top_features.iloc[0]

        if top_feature["SHAP Value"] > 0:

            st.info(
                f"**{top_feature['Feature']}** had the strongest "
                f"positive contribution toward the prediction."
            )

        else:

            st.warning(
                f"**{top_feature['Feature']}** had the strongest "
                f"negative contribution toward the prediction."
            )


    except Exception as e:

        st.warning(
            f"SHAP explanation could not be generated: {e}"
        )