import streamlit as st
import joblib
import pandas as pd
import numpy as np
import re
from datetime import datetime

# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="Fake News Detection System",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =====================================================
# CUSTOM CSS
# =====================================================

st.markdown("""
<style>

.main-header{
    text-align:center;
    padding:20px;
    border-radius:15px;
    background:#0E1117;
    color:white;
}

.footer{
    text-align:center;
    color:gray;
    margin-top:50px;
    font-size:14px;
}

</style>
""", unsafe_allow_html=True)

# =====================================================
# LOAD MODEL
# =====================================================

@st.cache_resource
def load_artifacts():

    model = joblib.load(
        "fake_news_model.pkl"
    )

    vectorizer = joblib.load(
        "tfidf_vectorizer.pkl"
    )

    return model, vectorizer


model, vectorizer = load_artifacts()

# =====================================================
# LABELS
# =====================================================

CLASS_LABELS = {
    0: "Fake News",
    1: "Real News"
}

CATEGORY_ICONS = {
    "Fake News": "🚨",
    "Real News": "✅"
}

# =====================================================
# TEXT CLEANING
# =====================================================

def clean_text(text):

    text = text.lower()

    text = re.sub(r'http\\S+', '', text)

    text = re.sub(r'[^a-zA-Z\\s]', ' ', text)

    text = re.sub(r'\\s+', ' ', text)

    return text.strip()

# =====================================================
# HEADER
# =====================================================

st.markdown("""
<div class='main-header'>
<h1>📰 Fake News Detection System</h1>
<p>AI Powered News Verification using Machine Learning & NLP</p>
</div>
""", unsafe_allow_html=True)

st.write("")

# =====================================================
# SIDEBAR
# =====================================================

with st.sidebar:

    st.header("Project Overview")

    st.info("""
This application predicts whether a news article is Fake News or Real News using Machine Learning and Natural Language Processing.
""")

    st.subheader("Technology Stack")

    st.markdown("""
- Python
- NLP
- TF-IDF Vectorization
- Scikit-Learn
- Streamlit
- Joblib
""")

    st.subheader("Prediction Classes")

    st.markdown("""
- 🚨 Fake News
- ✅ Real News
""")
    
    st.markdown("""
                Developed by [Sanjana Manthena]
                """, unsafe_allow_html=True)

# =====================================================
# MAIN SECTION
# =====================================================

news_text = st.text_area(
    "📰 Enter News Article",
    height=350,
    placeholder="""
Paste a news article, headline, or news content copied from the internet for analysis...
"""
)
# =====================================================
# PREDICTION
# =====================================================

if st.button(
    "🔍 Detect News",
    use_container_width=True
):

    if news_text.strip() == "":

        st.warning(
            "Please enter news content."
        )

    else:

        try:

            cleaned_text = clean_text(
                news_text
            )

            transformed_text = vectorizer.transform(
                [cleaned_text]
            )

            prediction = model.predict(
                transformed_text
            )[0]

            category = CLASS_LABELS.get(
                prediction,
                "Unknown"
            )

            icon = CATEGORY_ICONS.get(
                category,
                "📌"
            )

            st.success(
                "Prediction completed successfully."
            )

            st.markdown("---")

            col_a, col_b, col_c = st.columns(3)

            with col_a:

                st.metric(
                    label="Prediction",
                    value=f"{icon} {category}"
                )

            with col_b:

                st.metric(
                    label="Words",
                    value=len(news_text.split())
                )

            with col_c:

                st.metric(
                    label="Time",
                    value=datetime.now().strftime("%H:%M:%S")
                )

            # =====================================
            # CONFIDENCE SCORE
            # =====================================

            if hasattr(model, "predict_proba"):

                probabilities = model.predict_proba(
                    transformed_text
                )[0]

                confidence = round(
                    np.max(probabilities) * 100,
                    2
                )

                st.subheader(
                    "Confidence Score"
                )

                st.progress(
                    float(confidence) / 100
                )

                st.write(
                    f"Model Confidence: {confidence}%"
                )

                prob_df = pd.DataFrame({

                    "Class":[
                        "Fake News",
                        "Real News"
                    ],

                    "Probability":[
                        probabilities[0],
                        probabilities[1]
                    ]

                })

                st.subheader(
                    "Probability Distribution"
                )

                st.dataframe(
                    prob_df,
                    use_container_width=True,
                    hide_index=True
                )

                st.bar_chart(
                    prob_df.set_index("Class")
                )

            # =====================================
            # KEYWORDS
            # =====================================

            try:

                feature_names = (
                    vectorizer.get_feature_names_out()
                )

                row = transformed_text.toarray()[0]

                top_indices = row.argsort()[-10:]

                keywords = [
                    feature_names[i]
                    for i in top_indices
                    if row[i] > 0
                ]

                if keywords:

                    st.subheader(
                        "Important Keywords"
                    )

                    st.write(
                        ", ".join(keywords)
                    )

            except:
                pass

            # =====================================
            # ARTICLE SUMMARY
            # =====================================

            st.markdown("---")

            st.subheader(
                "Article Analysis"
            )

            summary = pd.DataFrame({

                "Metric":[
                    "Word Count",
                    "Character Count",
                    "Sentence Count",
                    "Prediction"
                ],

                "Value":[
                    len(news_text.split()),
                    len(news_text),
                    len(news_text.split(".")),
                    category
                ]

            })

            st.dataframe(
                summary,
                use_container_width=True,
                hide_index=True
            )

        except Exception as e:

            st.error(
                f"Prediction Error: {str(e)}"
            )

# =====================================================
# PROJECT INFO
# =====================================================

st.markdown("---")

st.subheader(
    "Supported Classification Labels"
)

info_df = pd.DataFrame({

    "Label":[
        "Fake News",
        "Real News"
    ],

    "Description":[
        "Misleading, fabricated, or false news content",
        "Authentic and fact-based news content"
    ]

})

st.dataframe(
    info_df,
    use_container_width=True,
    hide_index=True
)

# =====================================================
# FOOTER
# =====================================================

st.markdown("""
<div class='footer'>

Fake News Detection System<br>

Machine Learning | NLP | TF-IDF | Streamlit

</div>
""", unsafe_allow_html=True)