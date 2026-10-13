"""
LegalLens Streamlit Application
Minimalist, elegant academic demonstration interface for Indian court judgment classification.

Architecture & Operating Constraints:
- Production inference loads the locked, pre-trained model pipeline artifact from disk.
- Zero retraining, parameter tuning, or threshold alterations from the UI.
- Input is strictly raw Indian court judgment text (no metadata).
- Handles empty or invalid inputs gracefully.
- Minimalist, calm, academic layout with three clear modes:
  1. Classify a Judgment
  2. Try an Example
  3. How the Model Works
"""

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Ensure project root is in Python path for Streamlit Cloud execution
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import streamlit as st
from src.predict import load_model_pipeline, predict_judgment
from src.preprocessing import normalize_legal_text

# Configure Page
st.set_page_config(
    page_title="LegalLens | Indian Court Judgment Classifier",
    page_icon="⚖️",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# Load External Stylesheet
CSS_FILE = Path(__file__).resolve().parent / "styles" / "main.css"
if CSS_FILE.exists():
    with open(CSS_FILE, "r", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Load Curated Demo Data
DEMO_DATA_FILE = Path(__file__).resolve().parent / "demo_cases.json"


@st.cache_data
def load_demo_cases() -> Dict[str, Any]:
    """Loads curated demo cases for demonstration modes."""
    if DEMO_DATA_FILE.exists():
        with open(DEMO_DATA_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"known_cases": [], "blind_cases": [], "guided_examples": []}


@st.cache_resource(show_spinner=False)
def get_cached_pipeline():
    """Loads and caches the locked production pipeline in memory."""
    return load_model_pipeline()


def extract_top_features(text: str, pipeline, top_n: int = 5) -> List[Dict[str, Any]]:
    """
    Safely extracts active n-gram features and their linear contributions
    from the fitted TF-IDF vectorizer and LinearSVC classifier.
    Does NOT refit or modify any model parameters.
    """
    clean_text = normalize_legal_text(text)
    if not clean_text:
        return []

    try:
        tfidf_vec = pipeline.named_steps["tfidf"]
        svc_clf = pipeline.named_steps["clf"]

        X_tfidf = tfidf_vec.transform([clean_text])
        feature_names = np.array(tfidf_vec.get_feature_names_out())
        indices = X_tfidf.nonzero()[1]
        coefs = svc_clf.coef_[0]

        extracted = []
        for idx in indices:
            fn = feature_names[idx]
            val = float(X_tfidf[0, idx])
            w = float(coefs[idx])
            contrib = float(val * w)
            extracted.append({
                "feature": fn,
                "tfidf": round(val, 4),
                "weight": round(w, 4),
                "contribution": round(contrib, 4),
                "favors": "Criminal" if contrib > 0 else "Civil",
            })

        extracted.sort(key=lambda x: abs(x["contribution"]), reverse=True)
        return extracted[:top_n]
    except Exception:
        return []


def render_prediction_output(result: Dict[str, Any], text: str, pipeline):
    """Renders compact prediction results with clear accessible category distinction."""
    pred_class = result["predicted_class"]
    decision_score = result["decision_score"]
    confidence_pct = result["confidence_prob"] * 100
    word_count = result["word_count"]
    char_count = result["char_count"]

    st.markdown("---")

    # Clear, accessible visual distinction with explicit text label
    if pred_class == "Criminal":
        st.markdown(
            '<div class="badge-criminal">Predicted category: <strong>CRIMINAL</strong></div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="badge-civil">Predicted category: <strong>CIVIL</strong></div>',
            unsafe_allow_html=True,
        )

    # Compact estimated confidence line
    st.markdown(
        f'<div class="prediction-confidence">'
        f'Estimated confidence: <strong>{confidence_pct:.1f}%</strong> '
        f'<span style="opacity: 0.7; font-size: 0.85rem;">(heuristic estimate derived from decision margin)</span>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # Compact Model Details Expander
    with st.expander("Model details"):
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Decision score", f"{decision_score:+.3f}")
        with c2:
            st.metric("Word count", f"{word_count:,}")
        with c3:
            st.metric("Character count", f"{char_count:,}")

        st.caption(
            "Note: Linear SVM classifiers optimize margin distance rather than cross-entropy loss. "
            "Estimated confidence is computed using a standard sigmoid transformation of the margin "
            "(p = 1 / (1 + exp(-|margin|))) and is not a calibrated Bayesian posterior probability."
        )

        # Compact feature influence inspection
        top_feats = extract_top_features(text, pipeline=pipeline, top_n=5)
        if top_feats:
            st.markdown("**Top Influential Language Features in Text:**")
            feat_cols = st.columns(min(len(top_feats), 5))
            for i, feat in enumerate(top_feats[:5]):
                with feat_cols[i]:
                    chip_class = "token-chip-crim" if feat["favors"] == "Criminal" else "token-chip-civ"
                    st.markdown(
                        f'<div class="{chip_class}">'
                        f'<strong>{feat["feature"]}</strong><br>'
                        f'<span style="font-size:0.75rem;">Weight: {feat["weight"]:+.2f}</span><br>'
                        f'<span style="font-size:0.75rem;">Favors: {feat["favors"]}</span>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
            st.caption("Feature weights reflect learned model associations across the training corpus, not causal legal reasoning.")


def main():
    # Minimal Header
    st.markdown(
        """
        <div class="header-container">
            <div class="main-title">⚖️ LegalLens</div>
            <div class="subtitle">NLP-Based Classification of Indian Court Judgments</div>
            <div class="methodology-pill">TF-IDF (40,000 n-grams) + Linear SVM</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Preload Model Pipeline
    try:
        pipeline = get_cached_pipeline()
    except Exception as e:
        st.error(f"Failed to load locked classification model pipeline: {e}")
        st.stop()

    demo_data = load_demo_cases()

    # Three Clean, Minimalist Navigation Modes
    tab_classify, tab_example, tab_pipeline = st.tabs([
        "Classify a Judgment",
        "Try an Example",
        "How the Model Works",
    ])

    # -------------------------------------------------------------
    # 1. CLASSIFY A JUDGMENT (Primary Experience)
    # -------------------------------------------------------------
    with tab_classify:
        st.markdown("### Classify a Judgment")
        st.caption("Paste an Indian court judgment or excerpt to classify it as Civil or Criminal.")

        # Default text if set from Example mode
        default_text = st.session_state.get("classify_input_text", "")

        user_text = st.text_area(
            "Judgment text",
            value=default_text,
            height=200,
            placeholder="Paste Indian court judgment text or excerpt here...",
            key="main_judgment_input",
            label_visibility="collapsed",
        )

        classify_clicked = st.button("Classify Judgment", key="btn_classify_main", use_container_width=True)

        if classify_clicked:
            clean_input = user_text.strip()
            if not clean_input:
                st.warning("Please paste Indian court judgment text before classifying.")
            else:
                words = clean_input.split()
                if len(words) < 25:
                    st.info(
                        "Input text is short (<25 words). Short or non-legal excerpts lack "
                        "substantial legal terminology and may result in less reliable classifications."
                    )

                with st.spinner("Analyzing text..."):
                    result = predict_judgment(clean_input, pipeline=pipeline)

                if not result.get("is_valid", False):
                    st.error(f"Input validation error: {result.get('status', 'Unknown error')}")
                else:
                    render_prediction_output(result, clean_input, pipeline)

    # -------------------------------------------------------------
    # 2. TRY AN EXAMPLE
    # -------------------------------------------------------------
    with tab_example:
        st.markdown("### Try an Example")
        st.caption("Select an illustrative judgment excerpt to test the classifier.")

        # Curated selection of 4 diverse, verified examples
        examples = [
            {
                "id": "ex_crim_homicide",
                "name": "Criminal Appeal: Homicide Conviction & Life Imprisonment",
                "desc": "Appellate challenge against Sessions conviction under Section 302 IPC.",
                "text": (
                    "Reportable in the Supreme Court of India Criminal Appellate Jurisdiction. "
                    "Hari Shankar alias Hari Shankar Sharma appellant versus State of Mysore respondent. "
                    "This appeal under the Supreme Court Enlargement of Criminal Appellate Jurisdiction Act is directed "
                    "against the judgment of the High Court convicting the appellant under Section 302 of the Indian Penal Code "
                    "and sentencing him to imprisonment for life. Both the Sessions Judge and the High Court have concurrently "
                    "found that the prosecution proved the charge beyond reasonable doubt. We find no ground to interfere "
                    "with the conviction. The criminal appeal is dismissed."
                ),
                "citation": "Supreme Court of India (1979) — Record 1745908.txt (Indian Kanoon doc/1745908/)",
            },
            {
                "id": "ex_civ_insurance",
                "name": "Civil Appeal: Motor Vehicle Insurance & Statutory Liability",
                "desc": "Statutory interpretation of insurer liability and third-party compensation.",
                "text": (
                    "Reportable in the Supreme Court of India Civil Appellate Jurisdiction. "
                    "Civil Appeal Number 3659 of 1993. Smt Mallawwa and Others appellants versus Oriental Insurance Co. Ltd. "
                    "and Others respondents. What we are concerned with in these matters is the correct interpretation of Section 95 "
                    "of the Motor Vehicles Act, 1939. The question arises in the context of the death of the owner of goods carried "
                    "in a goods vehicle, and whether the insurer is liable to pay compensation awarded to legal heirs under the award "
                    "of the Claims Tribunal. The civil appeals are disposed of accordingly."
                ),
                "citation": "Supreme Court of India (1998) — Record 788960.txt (Indian Kanoon doc/788960/)",
            },
            {
                "id": "ex_civ_property",
                "name": "Civil Suit: Property Succession & Land Partition",
                "desc": "Dispute concerning Hindu Succession Act, partition of agricultural land, and tenancy rights.",
                "text": (
                    "Petitioner Budha vs. Respondent Amilal. Date of judgment 21/12/1990. Bench S.C. Agrawal, Kuldip Singh. "
                    "The plaintiff filed a civil suit for declaration of title and permanent injunction in respect of agricultural "
                    "land claiming rights under Section 14 of the Hindu Succession Act, 1956. The learned Civil Judge decreed the suit "
                    "holding that the widow held absolute ownership. The First Appellate Court and High Court dismissed the regular "
                    "second appeal. We find no error in the interpretation of property succession law. The civil appeal is dismissed with costs."
                ),
                "citation": "Supreme Court of India (1990) — Record 1809723.txt (Indian Kanoon doc/1809723/)",
            },
            {
                "id": "ex_crim_bail",
                "name": "Criminal SLP: Economic Offence Investigation & Regular Bail",
                "desc": "Trial court and appellate proceedings concerning regular bail rejection under CrPC.",
                "text": (
                    "Petitioner Chandraswami another appellants vs. Respondent Central Bureau of Investigation. "
                    "Date of judgment 07/11/1996. Bench J.S. Verma, B.N. Kirpal. Judgment: Leave granted. The application for bail "
                    "was dismissed by the trial court. Another application for regular bail was filed by the appellants before the "
                    "learned Chief Metropolitan Magistrate, Delhi, but the same was rejected on the ground that the offences alleged "
                    "against the accused under the Indian Penal Code involve serious economic fraud and conspiracy. The prosecution "
                    "established sufficient prima facie grounds to decline bail."
                ),
                "citation": "Supreme Court of India (1996) — Record 814411.txt (Indian Kanoon doc/814411/)",
            },
        ]

        ex_titles = [f"{e['name']}" for e in examples]
        selected_ex_idx = st.selectbox(
            "Select an illustrative example:",
            options=range(len(examples)),
            format_func=lambda i: ex_titles[i],
            key="example_selector",
        )
        selected_ex = examples[selected_ex_idx]

        st.markdown(f"**Description:** {selected_ex['desc']}")

        ex_text_area = st.text_area(
            "Example text",
            value=selected_ex["text"],
            height=140,
            key=f"ex_text_{selected_ex['id']}",
            label_visibility="collapsed",
        )

        c_btn1, c_btn2 = st.columns([1, 1])
        with c_btn1:
            classify_ex = st.button("Classify This Example", key=f"btn_run_{selected_ex['id']}", use_container_width=True)

        # Optional source citation expander for academic transparency
        with st.expander("Source attribution"):
            st.caption(f"Curated from: {selected_ex['citation']}")

        if classify_ex:
            with st.spinner("Classifying example..."):
                result_ex = predict_judgment(selected_ex["text"], pipeline=pipeline)
            render_prediction_output(result_ex, selected_ex["text"], pipeline)

    # -------------------------------------------------------------
    # 3. HOW THE MODEL WORKS
    # -------------------------------------------------------------
    with tab_pipeline:
        st.markdown("### How the Model Works")
        st.caption("LegalLens classifies Indian court judgments through four structured pipeline stages:")

        # Clean, restrained vertical step layout
        st.markdown(
            """
            <div class="pipeline-step">
                <div class="pipeline-step-header">1. Normalize</div>
                <div class="pipeline-step-body">
                    Standardizes text by case-folding and collapsing whitespace while strictly preserving statutory references 
                    (e.g., <em>Section 302 IPC</em>, <em>Article 226</em>) and critical legal negations.
                </div>
            </div>
            <div class="pipeline-step">
                <div class="pipeline-step-header">2. Vectorize</div>
                <div class="pipeline-step-body">
                    Transforms normalized text into sublinear TF-IDF features across a domain vocabulary of 40,000 unigrams and bigrams, 
                    capturing legal phrases such as <em>penal code</em>, <em>bail application</em>, and <em>specific performance</em>.
                </div>
            </div>
            <div class="pipeline-step">
                <div class="pipeline-step-header">3. Classify</div>
                <div class="pipeline-step-body">
                    Computes a signed decision margin using a Linear Support Vector Machine (LinearSVC, C=1.0) with balanced 
                    class weighting to account for the corpus ratio of ~4:1 Civil to Criminal judgments.
                </div>
            </div>
            <div class="pipeline-step">
                <div class="pipeline-step-header">4. Predict</div>
                <div class="pipeline-step-body">
                    Returns the final predicted category (Civil or Criminal) along with the geometric decision score 
                    and a heuristic estimated confidence.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        with st.expander("Technical details"):
            st.markdown(
                r"""
                - **Feature representation:** TF-IDF (1–2 n-grams, `max_features=40,000`, `sublinear_tf=True`, `min_df=3`, `max_df=0.90`)
                - **Stopword configuration:** Custom stopword list preserving legal negations (`not`, `no`, `nor`, `neither`, `never`, `without`, `against`)
                - **Classifier:** Linear Support Vector Classifier (`LinearSVC`, `loss="squared_hinge"`, `C=1.0`, `class_weight="balanced"`)
                - **Decision rule:** $\text{sign}(\mathbf{w}^T \mathbf{x} + b)$, where positive margin indicates Criminal and negative margin indicates Civil.
                """
            )

    # -------------------------------------------------------------
    # ABOUT THIS PROJECT & OFFICIAL BENCHMARKS
    # -------------------------------------------------------------
    with st.expander("About this project & evaluation"):
        st.markdown(
            """
            **LegalLens** is an academic NLP research project for whole-document classification of Indian court judgments into Civil versus Criminal categories.

            #### Official Held-Out Test Benchmarks (Frozen Test Set, N = 3,372):
            - **Accuracy:** `93.59%`
            - **Macro-F1:** `0.9061`
            - **Criminal Recall:** `91.68%`
            - **Civil F1:** `95.90%` | **Criminal F1:** `85.33%`

            *These figures represent empirical held-out evaluation results on a frozen test set and do not constitute a guarantee of correctness for every individual judgment.*
            """
        )

    # Single Academic Disclaimer at Bottom
    st.markdown(
        """
        <div class="disclaimer-box">
            LegalLens is an academic NLP research project that classifies Indian court judgment text as Civil or Criminal. 
            It does not provide legal advice or predict case outcomes.
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
