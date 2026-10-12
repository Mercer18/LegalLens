"""
LegalLens Streamlit Application
Interactive demonstration interface for whole-judgment classification of Indian court judgments (Civil vs. Criminal).

Architecture & Operating Constraints:
- Production inference loads the locked, pre-trained model pipeline artifact from disk.
- Zero retraining, parameter tuning, or threshold alterations from the UI.
- Input is strictly raw Indian court judgment text (no metadata).
- Handles empty or invalid inputs gracefully.
- Clear non-legal-advice disclaimer.
- Deployment-ready for Streamlit Community Cloud (runs from repository root).
"""

import sys
from pathlib import Path

# Ensure project root is in Python path for Streamlit Cloud execution
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import streamlit as st
from src.predict import load_model_pipeline, predict_judgment

# Configure Page
st.set_page_config(
    page_title="LegalLens | Indian Court Judgment Classifier",
    page_icon="⚖️",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# Custom Styling
st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .subtitle {
        font-size: 1.05rem;
        color: #475569;
        margin-bottom: 1.5rem;
    }
    .disclaimer-box {
        background-color: #F8FAFC;
        border-left: 4px solid #64748B;
        padding: 0.8rem 1rem;
        border-radius: 4px;
        font-size: 0.85rem;
        color: #334155;
        margin-top: 2rem;
    }
    .metric-card {
        background-color: #F1F5F9;
        border-radius: 6px;
        padding: 0.75rem 1rem;
        text-align: center;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner=False)
def get_cached_pipeline():
    """Loads and caches the locked production pipeline in memory."""
    return load_model_pipeline()


def main():
    st.markdown('<div class="main-title">⚖️ LegalLens</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="subtitle">NLP-Based Whole-Judgment Classifier for Indian Court Judgments (Civil vs. Criminal)</div>',
        unsafe_allow_html=True,
    )

    # Preload Model
    try:
        pipeline = get_cached_pipeline()
    except Exception as e:
        st.error(f"Failed to load locked classification model: {e}")
        st.stop()

    st.markdown("### Paste Court Judgment Text")
    st.caption("Paste the full text or operative excerpt of an Indian court judgment for category classification.")

    sample_options = {
        "Select a sample or paste custom text below...": "",
        "Sample 1: Criminal Case (Murder Conviction / Appeal)": (
            "The appellant was convicted by the learned Sessions Judge under Section 302 of the Indian Penal Code "
            "and Section 25 of the Arms Act and sentenced to rigorous imprisonment for life. The prosecution examined "
            "ten eyewitnesses whose testimony established the commission of the offence beyond reasonable doubt. "
            "The recovery of the weapon was duly proved by the investigating officer. The application for regular bail "
            "was rejected during trial. We find no illegality in the order of conviction and the criminal appeal is dismissed."
        ),
        "Sample 2: Civil Case (Specific Performance of Contract / Land Suit)": (
            "This civil regular second appeal arises out of a suit for specific performance of an agreement to sell "
            "immovable ancestral agricultural property. The plaintiff instituted the original suit in the Court of the Civil "
            "Judge Senior Division claiming decree of specific performance and execution of the sale deed. The defendant "
            "contended that time was of the essence of the contract and earnest money was forfeited. The trial court decreed "
            "the suit with costs. The First Appellate Court affirmed the findings under the Code of Civil Procedure."
        ),
        "Sample 3: Civil Case (Motor Accidents Claims Tribunal / Compensation)": (
            "This appeal is filed by the insurance company challenging the award passed by the Motor Accidents Claims "
            "Tribunal awarding compensation of rupees fifteen lakhs to the claimants on account of death in a road accident. "
            "The tribunal directed payment of compensation with interest and awarded costs. The appellant claims quantum "
            "of compensation is excessive and disputes rash and negligent driving under the Motor Vehicles Act."
        ),
    }

    selected_sample = st.selectbox(
        "Load Example Judgment:",
        options=list(sample_options.keys()),
        index=0,
    )

    default_text = sample_options[selected_sample] if selected_sample else ""

    judgment_input = st.text_area(
        label="Indian Court Judgment Text",
        value=default_text,
        height=220,
        placeholder="Enter or paste the Indian court judgment text here...",
        help="Input is restricted strictly to judgment text. No metadata (case numbers, court type, outcome) is required.",
    )

    col_btn, col_clear = st.columns([1, 4])
    with col_btn:
        classify_clicked = st.button("Classify Judgment", type="primary", use_container_width=True)

    if classify_clicked:
        if not judgment_input.strip():
            st.warning("⚠️ Please paste or enter an Indian court judgment text before classifying.")
        else:
            with st.spinner("Analyzing text and extracting legal n-gram features..."):
                result = predict_judgment(judgment_input, pipeline=pipeline)

            if not result.get("is_valid", False):
                st.error(f"❌ Input validation error: {result.get('status', 'Unknown error')}")
            else:
                pred_class = result["predicted_class"]
                decision_score = result["decision_score"]
                confidence_pct = result["confidence_prob"] * 100
                word_count = result["word_count"]
                char_count = result["char_count"]

                st.markdown("---")
                st.subheader("Classification Result")

                # Display category with visual callout
                if pred_class == "Criminal":
                    st.error(f"### Predicted Class: **Criminal**")
                else:
                    st.success(f"### Predicted Class: **Civil**")

                # Metrics Row
                m_col1, m_col2, m_col3, m_col4 = st.columns(4)
                with m_col1:
                    st.metric(
                        label="Decision Score",
                        value=f"{decision_score:+.3f}",
                        help="Linear SVM decision margin (squared hinge loss). Positive indicates Criminal, negative indicates Civil.",
                    )
                with m_col2:
                    st.metric(
                        label="Estimated Confidence",
                        value=f"{confidence_pct:.1f}%",
                        help="Calibrated probability score derived via sigmoid transformation of the decision margin.",
                    )
                with m_col3:
                    st.metric(
                        label="Word Count",
                        value=f"{word_count:,}",
                        help="Total words after domain-aware text normalization.",
                    )
                with m_col4:
                    st.metric(
                        label="Character Count",
                        value=f"{char_count:,}",
                        help="Total characters in normalized judgment text.",
                    )

                # Class Probability Breakdown
                st.markdown("#### Model Decision Distribution")
                st.progress(result["probability_criminal"])
                p_c1, p_c2 = st.columns(2)
                p_c1.caption(f"**Civil Probability:** {result['probability_civil'] * 100:.1f}%")
                p_c2.caption(f"**Criminal Probability:** {result['probability_criminal'] * 100:.1f}%")

    # Architecture & Academic Disclaimer
    st.markdown(
        """
        <div class="disclaimer-box">
            <strong>Disclaimer:</strong> LegalLens is an academic Natural Language Processing (NLP) research project 
            engineered for whole-judgment topic categorization. It operates purely as an empirical machine learning 
            text classifier (TF-IDF 1–2 ngrams + Linear Support Vector Machine) and <strong>does not provide legal advice</strong>, 
            statutory interpretation, case outcome predictions, or judicial assessment.
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
