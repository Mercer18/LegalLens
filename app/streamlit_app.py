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
from app.nlp_utils import (
    extract_top_features,
    extract_statutory_entities,
    compute_text_diagnostics,
    load_validation_cases,
)

# Configure Page
st.set_page_config(
    page_title="LegalLens | Indian Court Judgment Classifier",
    page_icon="⚖️",
    layout="centered",
    initial_sidebar_state="expanded",
)

# Load External Stylesheet
CSS_FILE = Path(__file__).resolve().parent / "styles" / "main.css"
if CSS_FILE.exists():
    with open(CSS_FILE, "r", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
else:
    # Fallback Custom Styling
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
            background-color: rgba(100, 116, 139, 0.08);
            border-left: 4px solid #64748B;
            padding: 0.8rem 1rem;
            border-radius: 4px;
            font-size: 0.85rem;
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

    # Academic Training Corpus & Benchmark Banner (Clean, no emojis)
    st.markdown(
        """
        <div class="metrics-banner">
            <span>Cleaned Data: <b>33,717 Judgments</b></span>
            <span>Trained Data: <b>26,973</b></span>
            <span>Tested Data: <b>3,372 (Frozen)</b></span>
            <span>Demo Data: <b>3,372</b></span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Preload Model
    try:
        pipeline = get_cached_pipeline()
    except Exception as e:
        st.error(f"Failed to load locked classification model: {e}")
        st.stop()

    # Initialize Session State for Demo Cases & Text Area
    if "validation_pool_cases" not in st.session_state:
        st.session_state.validation_pool_cases = load_validation_cases(n_per_category=5)
    if "selected_case_key" not in st.session_state:
        st.session_state.selected_case_key = None
    if "judgment_text_area" not in st.session_state:
        st.session_state.judgment_text_area = ""

    # Sidebar: Point-Wise Demo Judgment Library with Refresh (Option 1)
    with st.sidebar:
        st.markdown("### Indian Court Judgments Library")
        st.caption("Live cases sampled from the 3,372-record demo pool (never used to train the model):")

        if st.button("🔄 Refresh Cases", use_container_width=True, help="Load 10 fresh random court judgments from the 3,372 demo pool"):
            st.session_state.validation_pool_cases = load_validation_cases(n_per_category=5)
            st.session_state.selected_case_key = None
            st.rerun()

        crim_cases = [(k, v) for k, v in st.session_state.validation_pool_cases.items() if v.get("category") == "Criminal"]
        civ_cases = [(k, v) for k, v in st.session_state.validation_pool_cases.items() if v.get("category") == "Civil"]

        st.markdown("##### ⚖️ Criminal Cases (5)")
        for idx, (k, info) in enumerate(crim_cases, start=1):
            is_selected = (st.session_state.selected_case_key == k)
            btn_label = f"{idx}. Doc #{info['doc_id']} ({info['total_words']:,} words)"
            if is_selected:
                btn_label = f"✓ {btn_label}"

            if st.button(
                btn_label,
                key=f"btn_crim_{info['doc_id']}_{idx}",
                use_container_width=True,
                type="primary" if is_selected else "secondary",
                help=f"Click to load Indian Kanoon Document #{info['doc_id']} into the classification text box",
            ):
                st.session_state.selected_case_key = k
                st.session_state.judgment_text_area = info["text"]
                st.rerun()

        st.markdown("##### 📜 Civil Cases (5)")
        for idx, (k, info) in enumerate(civ_cases, start=6):
            is_selected = (st.session_state.selected_case_key == k)
            btn_label = f"{idx}. Doc #{info['doc_id']} ({info['total_words']:,} words)"
            if is_selected:
                btn_label = f"✓ {btn_label}"

            if st.button(
                btn_label,
                key=f"btn_civ_{info['doc_id']}_{idx}",
                use_container_width=True,
                type="primary" if is_selected else "secondary",
                help=f"Click to load Indian Kanoon Document #{info['doc_id']} into the classification text box",
            ):
                st.session_state.selected_case_key = k
                st.session_state.judgment_text_area = info["text"]
                st.rerun()

    st.markdown("### Paste Court Judgment Text")
    st.caption("Paste the full text or operative excerpt of an Indian court judgment for category classification.")

    active_case_info = None
    if st.session_state.selected_case_key and st.session_state.selected_case_key in st.session_state.validation_pool_cases:
        active_case_info = st.session_state.validation_pool_cases[st.session_state.selected_case_key]
        st.info(f"Loaded from Demo Data: **{st.session_state.selected_case_key}**")

    judgment_input = st.text_area(
        label="Indian Court Judgment Text",
        key="judgment_text_area",
        height=220,
        placeholder="Enter or paste an Indian court judgment text here, or select a case from the demo pool on the left...",
        help="Input is restricted strictly to judgment text. No metadata (case numbers, court type, outcome) is required.",
    )

    col_btn, col_clear = st.columns([1, 4])
    with col_btn:
        classify_clicked = st.button("Classify Judgment", type="primary", use_container_width=True)
    with col_clear:
        if st.button("Clear Input", use_container_width=False):
            st.session_state.judgment_text_area = ""
            st.session_state.selected_case_key = None
            st.rerun()

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

                if active_case_info and "category" in active_case_info and active_case_info["text"].strip() == judgment_input.strip():
                    expected = active_case_info["category"]
                    if pred_class.lower() == expected.lower():
                        st.caption(f"✓ Validated: Prediction matches Ground Truth in Demo Data (**{expected}**)")
                    else:
                        st.caption(f"Demo Data Ground Truth: **{expected}** | Predicted: **{pred_class}**")

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

                # 1. Feature Attribution
                with st.expander("🔍 Feature Attribution: Salient Legal N-Grams", expanded=False):
                    st.caption(
                        "Direct contribution of salient n-grams to the Linear SVM decision margin "
                        "($x_j \\cdot w_j$). Positive contributions push toward **Criminal**, negative toward **Civil**."
                    )
                    top_feats = extract_top_features(judgment_input, pipeline=pipeline, top_n=6)
                    col_crim, col_civ = st.columns(2)
                    with col_crim:
                        st.markdown("**Favors Criminal (+ Margin)**")
                        if top_feats["criminal_features"]:
                            chips_html = "".join(
                                f'<span class="token-chip-crim">{feat} ({val:+.2f})</span>'
                                for feat, val in top_feats["criminal_features"]
                            )
                            st.markdown(chips_html, unsafe_allow_html=True)
                        else:
                            st.caption("No strong criminal n-gram signals detected.")
                    with col_civ:
                        st.markdown("**Favors Civil (- Margin)**")
                        if top_feats["civil_features"]:
                            chips_html = "".join(
                                f'<span class="token-chip-civ">{feat} ({val:+.2f})</span>'
                                for feat, val in top_feats["civil_features"]
                            )
                            st.markdown(chips_html, unsafe_allow_html=True)
                        else:
                            st.caption("No strong civil n-gram signals detected.")

                # 2. Statutory Entity Extractor
                with st.expander("📜 Detected Statutory Citations & Legal Entities", expanded=False):
                    st.caption("Pattern-extracted references to Indian statutory sections, acts, and procedural codes.")
                    citations = extract_statutory_entities(judgment_input)
                    if citations:
                        chips_html = "".join(
                            f'<span class="citation-chip">{cit} <small style="opacity:0.7">({cat})</small></span>'
                            for cit, cat in citations
                        )
                        st.markdown(chips_html, unsafe_allow_html=True)
                    else:
                        st.caption("No explicit Indian statutory sections or named acts detected in the input excerpt.")

                # 3. Document Stylometry & Linguistic Diagnostics
                with st.expander("📊 Document Stylometry & Linguistic Diagnostics", expanded=False):
                    st.caption("Linguistic metrics and vocabulary coverage against the locked 40,000-feature vocabulary.")
                    diag = compute_text_diagnostics(judgment_input, pipeline=pipeline)
                    d_col1, d_col2, d_col3, d_col4 = st.columns(4)
                    with d_col1:
                        st.metric(
                            label="Lexical Diversity (TTR)",
                            value=f"{diag['ttr']:.1f}%",
                            help=f"Type-Token Ratio ({diag['unique_words']} unique words / {diag['word_count']} total words). Higher values indicate richer legal terminology.",
                        )
                    with d_col2:
                        st.metric(
                            label="Avg Sentence Length",
                            value=f"{diag['avg_sentence_len']:.1f} w/s",
                            help="Average number of words per sentence. Complex legal pleadings typically exceed 20-30 words per sentence.",
                        )
                    with d_col3:
                        st.metric(
                            label="Sentence Count",
                            value=f"{diag['sentence_count']}",
                            help="Total sentences identified in judgment text.",
                        )
                    with d_col4:
                        st.metric(
                            label="Vocabulary Match",
                            value=f"{diag['active_vocab_count']}",
                            help="Count of active n-gram features from this text matched in the locked 40,000 TF-IDF vocabulary.",
                        )

    # Architecture & Academic Disclaimer (Footer)
    st.markdown(
        """
        <div class="footer-disclaimer">
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
