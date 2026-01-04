# import streamlit as st
# import pandas as pd
# import torch
# import joblib
# import os
# import plotly.express as px
# import altair as alt

# from sklearn.pipeline import Pipeline
# from sklearn.svm import LinearSVC
# from sklearn.feature_extraction.text import TfidfVectorizer

# # from transformers import (
# #     DistilBertTokenizerFast,
# #     DistilBertForSequenceClassification
# # )

# st.set_page_config(
#     page_title="Cyberbullying Text Classifier",
#     layout="wide"
# )

# st.title("🚨 Cyberbullying Detection App")

# st.markdown("""
# Welcome!  
# This application demonstrates **text classification for cyberbullying detection**
# using machine learning and transformer-based models.

# """)

# tab_intro, tab_eval, tab_demo = st.tabs(
#     ["📘 Introduction", "📊 Evaluation", "⛑️ Model Demo"]
# )


# with tab_intro:

#     # st.header("🚨 Cyberbullying Detection System")
#     st.subheader("✅ Objective")
#     st.markdown("""
     
#     Build an automated NLP system to **identify and classify cyberbullying content**
#     across multiple sensitive categories.
#     """)

#     st.divider()

#     col1, col2 = st.columns(2)

#     with col1:
#         st.subheader("📌 Problem Context")
#         st.markdown("""
#         - Online platforms face **high volume of harmful text**
#         - Manual moderation is:
#             - Slow
#             - Inconsistent
#             - Not scalable
#         """)

#     with col2:
#         st.subheader("🎯 Project Goals")
#         st.markdown("""
#         - Detect cyberbullying automatically
#         - Support **multi-class classification**
#         - Balance **accuracy vs inference speed**
#         """)

#     st.divider()

#     st.subheader("🏷️ Target Classes")
#     st.markdown("""
#     - **Age-based**
#     - **Gender-based**
#     - **Ethnicity-based**
#     - **Religion-based**
#     - **Other cyberbullying**
#     - **Not cyberbullying**
#     """)

#     st.divider()

#     st.subheader("🧠 Modeling Strategy")
#     st.markdown("""
#     | Model Type | Purpose |
#     |-----------|---------|
#     | TF-IDF + ML | Fast baseline & interpretability |
#     | DistilBERT | Context-aware, higher accuracy |
#     """)

#     st.info("""
#     **Why multiple models?**  
#     Baseline models provide speed and explainability,  
#     while transformer models maximize detection performance.
#     """)

# with tab_eval:
#     @st.cache_data
#     def load_results():
#         df = pd.read_csv("model_comparison.csv")
#         return df.drop(columns=["Best_Params"], errors="ignore")

#     df_compare = load_results()

#     st.subheader("📊 Model Comparison Results")

#     st.markdown("""
#     Models are evaluated using **Accuracy**, **F1 Macro**, and **F1 Weighted**
#     to handle **class imbalance** in cyberbullying data.
#     """)

#     st.divider()

#     # KPI row
#     col1, col2, col3 = st.columns(3)

#     best = df_compare.iloc[0]

#     col1.metric("🏆 Best Model", best["Model"])
#     col2.metric("🎯 Accuracy", f"{best['Accuracy']:.3f}")
#     col3.metric("⚖️ F1 Macro", f"{best['F1_Macro']:.3f}")

#     st.divider()

#     st.subheader("📋 Model Leaderboard")
#     st.dataframe(
#         df_compare,
#         use_container_width=True,
#         hide_index=True
#     )

#     st.divider()

#     df_plot = df_compare.copy()
#     df_plot["Model_Label"] = (
#         df_plot["Model"] + " (" + df_plot["Stage"] + ")"
#     )    
#     st.subheader("📈 F1 Macro Comparison")


#     chart = (
#         alt.Chart(df_plot)
#         .mark_bar()
#         .encode(
#             x=alt.X(
#                 "Model_Label:N",
#                 sort="-y",
#                 axis=alt.Axis(labelAngle=45, title="Model")
#             ),
#             y=alt.Y("F1_Macro:Q", title="F1 Macro"),
#             tooltip=["Model", "Stage", "F1_Macro"]
#         )
#     )

#     st.altair_chart(chart, use_container_width=True)

#     st.divider()

#     st.subheader("🧠 Key Observations")
#     st.markdown("""
#     - **DistilBERT** achieves the highest overall performance
#     - **Tuned ML models** significantly improve over baseline
#     - **Baseline models** remain valuable for real-time inference
#     """)

#     st.success("""
#     **Recommendation**  
#     - Use **DistilBERT** for offline moderation and batch analysis  
#     - Use **Tuned Logistic Regression** for real-time or low-latency use cases
#     """)


#     best_score_model = df_compare.iloc[0]

#     col1, col2 = st.columns(2)

#     with col1:
#         st.metric("🏆 Best Scoring Model", best_score_model["Model"])
#         st.caption("Highest validation performance")

#     with col2:
#         st.metric("🚀 Deployed Model", "DistilBERT")
#         st.caption("Used for live prediction")

#     st.info("""
#     **Note**  
#     The highest-scoring model is not always deployed.
#     Deployment decisions consider **generalization**, **robustness**, and **context awareness**.
#     """)   



# with tab_demo:
#     BASE_DIR = os.path.dirname(os.path.abspath(__file__))
#     MODEL_DIR = os.path.join(BASE_DIR, "cyberbully_distilbert_model")

#     DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

#     # -----------------------------
#     # Load model & tokenizer
#     # -----------------------------
#     @st.cache_resource
#     def load_model():
#         tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_DIR)
#         model = DistilBertForSequenceClassification.from_pretrained(MODEL_DIR)
#         model.to(DEVICE)
#         model.eval()
#         return tokenizer, model

#     tokenizer, model = load_model()

#     # Load label encoder
#     le = joblib.load("label_encoder.pkl")

#     # # Load metrics
#     # if os.path.exists("eval_metrics.json"):
#     #     with open("eval_metrics.json") as f:
#     #         metrics = json.load(f)
#     # else:
#     #     metrics = None

#     st.markdown(
#         """
#         <div style="text-align:center; margin-bottom: 20px;">
#             <h2>🛡 Cyberbully Detection</h2>
#             <p style="color: gray;">
#                 AI-powered text classification for identifying cyberbullying
#             </p>
#         </div>
#         """,
#         unsafe_allow_html=True
#     )

#     st.markdown("""
#     <style>
#     .card {
#         padding: 24px;
#         border-radius: 16px;
#         margin-bottom: 20px;
#     }

#     .header-card {
#         background: radial-gradient(circle,rgba(238, 174, 202, 1) 0%, rgba(148, 187, 233, 1) 100%);
#         text-align: left;
#     }

#     .desc-text {
#         font-size: 14px; 
#         font-weight: bold;
#         line-height: 1.6;
#         color: #0b3c5d;
#     }

#     .big-number {
#         font-size: 36px;
#         font-weight: bold;
#         color: #1f77b4;
#     }
#     </style>
#     """, unsafe_allow_html=True)
#     # -----------------------------
#     # UI
#     # -----------------------------
#     # st.title("Cyberbullying Detection")

#     st.markdown("""
#     <div class="card header-card">
#         <p class="desc-text">
#             This model analyzes text to identify potential cyberbullying categories.
#             Predictions are probabilistic and intended for decision support.
#         </p>
#     </div>
#     """, unsafe_allow_html=True)


#     text = st.text_area("Enter a short text to classify.", height=120)

#     if st.button("Predict"):
#         if not text.strip():
#             st.warning("Please enter some text.")
#         else:
#             inputs = tokenizer(
#                 text,
#                 return_tensors="pt",
#                 truncation=True,
#                 padding=True,
#                 max_length=128
#             ).to(DEVICE)

#             with torch.no_grad():
#                 outputs = model(**inputs)
#                 probs = torch.softmax(outputs.logits, dim=1).cpu().numpy()[0]

#             pred_id = probs.argmax()
#             pred_label = le.inverse_transform([pred_id])[0]

#             # --------------------------------------------------
#             # Build dataframe for pie chart
#             # --------------------------------------------------
#             df_probs = pd.DataFrame({
#                 "Class": le.classes_,
#                 "Probability": probs
#             })

#             # --------------------------------------------------
#             # Layout
#             # --------------------------------------------------
#             st.markdown("""
#             <style>
#             .pred-box {
#                 background: linear-gradient(2deg,rgba(87, 199, 133, 1) 50%, rgba(237, 221, 83, 1) 100%);
#                 padding: 1.5rem;
#                 border-radius: 12px;
#                 height: 550px;
#                 display: flex;
#                 flex-direction: column;
#                 justify-content: center;
#                 color: white;
#             }

#             .prob-box {
#                 background-color: #0f2b46;
#                 padding: 1rem;
#                 border-radius: 12px;
#                 height: 180px;   
#                 overflow-y: auto; 
#                 margin-top: 0.5rem;
#             }
#             </style>
#             """, unsafe_allow_html=True)

#             col1, col2 = st.columns([1, 1])

#             # ==========================
#             # LEFT: Prediction
#             # ==========================
#             with col1:
#                 st.markdown(
#                     f"""
#                     <div class="pred-box">
#                         <h3>Prediction: </h3>
#                         <h2>{pred_label}</h2>
#                     </div>
#                     """,
#                     unsafe_allow_html=True
#                 )

#                     # st.metric(
#                     #     label="Confidence",
#                     #     value=f"{probs[pred_id]:.3f}"
#                     # )

#             # ==========================
#             # RIGHT: Probabilities
#             # ==========================
#             with col2:
#                 with st.container(height=550, border=True):

#                     # ---- Pie chart (top) ----
#                     fig = px.pie(
#                         df_probs,
#                         names="Class",
#                         values="Probability",
#                         hole=0.5
#                     )

#                     fig.update_traces(
#                         textposition="inside",
#                         textinfo="percent"
#                     )

#                     fig.update_layout(
#                         height=200,
#                         margin=dict(t=40, b=0),
#                         showlegend=False,
#                         title="Class Probabilities"
#                     )

#                     st.plotly_chart(fig, use_container_width=True)

#                     st.write("Class Confidence Scores")

#                     for _, row in df_probs.iterrows():
#                         col_label, col_bar = st.columns([1, 3])

#                         with col_label:
#                             st.write(f"**{row['Class']}**")
#                             # st.write(f"{row['Probability']:.3f}")

#                         with col_bar:
#                             st.progress(float(row["Probability"]))

import streamlit as st
import pandas as pd
import joblib
import os
import altair as alt
import numpy as np

# ----------------------------------
# Page config
# ----------------------------------
st.set_page_config(
    page_title="Cyberbullying Text Classifier",
    layout="wide"
)

st.title("🚨 Cyberbullying Detection App")

st.markdown("""
Welcome!  
This application demonstrates **cyberbullying text classification**
using **TF-IDF + LinearSVC** for fast and interpretable inference.
""")

tab_intro, tab_eval, tab_demo = st.tabs(
    ["📘 Introduction", "📊 Evaluation", "⛑️ Model Demo"]
)

# ==================================
# INTRO TAB
# ==================================
with tab_intro:
    st.subheader("✅ Objective")
    st.markdown("""
    Build an automated NLP system to **identify and classify cyberbullying content**
    across multiple sensitive categories.
    """)

    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📌 Problem Context")
        st.markdown("""
        - High volume of harmful online text
        - Manual moderation is:
            - Slow
            - Inconsistent
            - Not scalable
        """)

    with col2:
        st.subheader("🎯 Project Goals")
        st.markdown("""
        - Automatic detection
        - Multi-class classification
        - Low-latency inference
        """)

    st.divider()

    st.subheader("🏷️ Target Classes")
    st.markdown("""
    - Age-based
    - Gender-based
    - Ethnicity-based
    - Religion-based
    - Other cyberbullying
    - Not cyberbullying
    """)

    st.divider()

    st.subheader("🧠 Modeling Strategy")
    st.markdown("""
    | Model | Purpose |
    |------|--------|
    | TF-IDF + LinearSVC | Fast, robust, production-ready |
    | DistilBERT | Higher accuracy (offline / batch use) |
    """)

    st.info("""
    **Deployment Choice**  
    LinearSVC is deployed due to **low latency**, **simplicity**,  
    and **strong performance on sparse text features**.
    """)

# ==================================
# EVALUATION TAB
# ==================================
with tab_eval:

    @st.cache_data
    def load_results():
        df = pd.read_csv("model_comparison.csv")
        return df.drop(columns=["Best_Params"], errors="ignore")

    df_compare = load_results()

    st.subheader("📊 Model Comparison Results")

    st.markdown("""
    Models are evaluated using **Accuracy**, **F1 Macro**, and **F1 Weighted**
    to handle **class imbalance**.
    """)

    st.divider()

    best = df_compare.iloc[0]

    col1, col2, col3 = st.columns(3)
    col1.metric("🏆 Best Model", best["Model"])
    col2.metric("🎯 Accuracy", f"{best['Accuracy']:.3f}")
    col3.metric("⚖️ F1 Macro", f"{best['F1_Macro']:.3f}")

    st.divider()

    st.subheader("📋 Model Leaderboard")
    st.dataframe(df_compare, use_container_width=True, hide_index=True)

    st.divider()

    df_plot = df_compare.copy()
    df_plot["Model_Label"] = df_plot["Model"] + " (" + df_plot["Stage"] + ")"

    chart = (
        alt.Chart(df_plot)
        .mark_bar()
        .encode(
            x=alt.X("Model_Label:N", sort="-y", axis=alt.Axis(labelAngle=45)),
            y="F1_Macro:Q",
            tooltip=["Model", "Stage", "F1_Macro"]
        )
    )

    st.subheader("📈 F1 Macro Comparison")
    st.altair_chart(chart, use_container_width=True)

# ==================================
# DEMO TAB (LinearSVC)
# ==================================
with tab_demo:

    @st.cache_resource
    def load_model():
        model = joblib.load("linearsvc_model.pkl")
        le = joblib.load("label_encoder.pkl")
        return model, le

    model, le = load_model()

    st.markdown("""
    <div style="text-align:center; margin-bottom: 20px;">
        <h2>🛡 Cyberbully Detection</h2>
        <p style="color: gray;">
            Fast AI-powered text classification (LinearSVC)
        </p>
    </div>
    """, unsafe_allow_html=True)

    text = st.text_area("Enter a short text to classify.", height=120)

    if st.button("Predict"):
        if not text.strip():
            st.warning("Please enter some text.")
        else:
            # ----------------------------------
            # Prediction
            # ----------------------------------
            pred_id = model.predict([text])[0]
            pred_label = le.inverse_transform([pred_id])[0]

            scores = model.decision_function([text])[0]

            # Normalize scores for display
            scores_norm = (scores - scores.min()) / (scores.max() - scores.min() + 1e-9)

            df_scores = pd.DataFrame({
                "Class": le.classes_,
                "Confidence": scores_norm
            }).sort_values("Confidence", ascending=False)

            col1, col2 = st.columns([1, 1])

            # LEFT
            with col1:
                st.markdown(
                    f"""
                    <div style="
                        background: linear-gradient(135deg,#43cea2,#185a9d);
                        padding: 2rem;
                        border-radius: 12px;
                        color: white;
                        height: 300px;
                        display: flex;
                        align-items: center;
                        justify-content: center;
                        text-align: center;
                    ">
                        <div>
                            <h3>Prediction</h3>
                            <h2>{pred_label}</h2>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            # RIGHT
            with col2:
                st.subheader("Class Confidence Scores")

                for _, row in df_scores.iterrows():
                    st.write(f"**{row['Class']}**")
                    st.progress(float(row["Confidence"]))

                st.caption("""
                Confidence scores are derived from **LinearSVC decision margins**  
                (normalized for visualization, not probabilities).
                """)

# ==================================
# FOOTER
# ==================================
st.markdown("---")
st.caption("⚠️ Educational demo — predictions are decision-support only.")
