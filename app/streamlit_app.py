"""
LegalLens Streamlit Application
Interactive demonstration interface for whole-judgment classification of Indian court judgments (Civil vs. Criminal).

Architecture & Operating Constraints:
- Production inference loads the locked, pre-trained model pipeline artifact from disk.
- Zero retraining, parameter tuning, or threshold alterations from the UI.
- Input is strictly raw Indian court judgment text (no metadata).
- Handles empty or invalid inputs gracefully.
- Clear non-legal-advice disclaimer and theme-aware UI.
- 4 Dedicated Interactive Modes:
  Mode A: Test Known Cases (with revealable ground truth & demo accuracy session counter)
  Mode B: Blind Case Challenge (blind judgment prediction with Indian Kanoon source link verification)
  Mode C: Analyze Your Own Text (with guided real, short illustrative, & out-of-domain presets)
  Mode D: Explore the NLP Pipeline (transparent 6-stage NLP inspection with actual feature contributions)
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


def extract_top_features(text: str, pipeline, top_n: int = 6) -> List[Dict[str, Any]]:
    """
    Safely extracts actual active n-gram features and their linear contributions
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

        # Sort by absolute contribution to the decision margin
        extracted.sort(key=lambda x: abs(x["contribution"]), reverse=True)
        return extracted[:top_n]
    except Exception:
        return []


def render_prediction_card(result: Dict[str, Any]):
    """Renders the standard prediction card with accessible badges and metrics."""
    pred_class = result["predicted_class"]
    decision_score = result["decision_score"]
    confidence_pct = result["confidence_prob"] * 100
    word_count = result["word_count"]
    char_count = result["char_count"]

    st.markdown("---")
    st.markdown("### Model Classification Output")

    # Outcome Badge
    if pred_class == "Criminal":
        st.markdown(
            '<div class="badge-criminal">Predicted Category: <strong>Criminal</strong></div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="badge-civil">Predicted Category: <strong>Civil</strong></div>',
            unsafe_allow_html=True,
        )

    # Metrics Row
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric(
            label="Decision Score",
            value=f"{decision_score:+.3f}",
            help="Linear SVM signed decision margin (squared hinge loss). Positive indicates Criminal, negative indicates Civil.",
        )
    with m2:
        st.metric(
            label="Estimated Confidence",
            value=f"{confidence_pct:.1f}%",
            help="Heuristic confidence estimate derived via sigmoid transformation of the SVM decision margin: p = 1 / (1 + exp(-margin)). Not a formally calibrated Bayesian probability.",
        )
    with m3:
        st.metric(
            label="Word Count",
            value=f"{word_count:,}",
            help="Total words in normalized judgment text.",
        )
    with m4:
        st.metric(
            label="Character Count",
            value=f"{char_count:,}",
            help="Total characters in normalized judgment text.",
        )

    # Decision Balance Distribution
    st.markdown("#### Decision Weight Distribution")
    st.progress(float(result["probability_criminal"]))
    p1, p2 = st.columns(2)
    p1.caption(f"**Civil Estimate:** {result['probability_civil'] * 100:.1f}%")
    p2.caption(f"**Criminal Estimate:** {result['probability_criminal'] * 100:.1f}%")


def main():
    # Initialize Session State for Demo Accuracy Tracker
    if "session_tested_cases" not in st.session_state:
        st.session_state["session_tested_cases"] = set()
    if "session_correct_cases" not in st.session_state:
        st.session_state["session_correct_cases"] = set()

    # App Header
    st.markdown(
        """
        <div class="header-container">
            <div class="main-title">⚖️ LegalLens</div>
            <div class="subtitle">NLP-Based Classification of Indian Court Judgments</div>
            <div>
                <span class="methodology-pill">🔬 TF-IDF (40k n-grams) + Balanced Linear SVM</span>
                <span class="methodology-pill">📊 Held-Out Test Accuracy: 93.59% | Macro-F1: 0.9061</span>
            </div>
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

    # Mode Navigation Tabs
    tab_known, tab_blind, tab_custom, tab_nlp = st.tabs([
        "📋 Mode A: Test Known Cases",
        "🎯 Mode B: Blind Case Challenge",
        "✍️ Mode C: Analyze Your Own Text",
        "🔍 Mode D: Explore the NLP Pipeline",
    ])

    # -------------------------------------------------------------
    # MODE A: TEST KNOWN CASES
    # -------------------------------------------------------------
    with tab_known:
        st.markdown("### Mode A: Test Known Cases")
        st.caption(
            "Select an authentic court record with known provenance. Run the classifier without seeing the label, "
            "inspect diagnostics, and then reveal the ground truth to check the model's accuracy."
        )

        known_cases = demo_data.get("known_cases", [])
        if known_cases:
            case_titles = [f"{c['title']}" for c in known_cases]
            selected_idx = st.selectbox(
                "Select a Curated Case:",
                options=range(len(case_titles)),
                format_func=lambda i: case_titles[i],
                key="mode_a_select",
            )
            case = known_cases[selected_idx]

            st.markdown(f"**Overview:** {case['description']}")
            st.markdown(f"**Source Attribution:** *{case['source_attribution']}*")

            # Judgment Text Area (Display only text to model)
            text_a = st.text_area(
                "Judgment Text Excerpt:",
                value=case["judgment_text"],
                height=150,
                key="mode_a_text",
            )

            # Execution Button
            col_a1, col_a2 = st.columns([1, 2])
            with col_a1:
                classify_a = st.button("⚖️ Classify Judgment", key="btn_classify_a", use_container_width=True)

            if classify_a or f"res_a_{case['demo_id']}" in st.session_state:
                # Store prediction in session state
                if classify_a:
                    st.session_state[f"res_a_{case['demo_id']}"] = predict_judgment(text_a, pipeline=pipeline)

                result_a = st.session_state[f"res_a_{case['demo_id']}"]
                render_prediction_card(result_a)

                # Reveal Ground Truth
                st.markdown("---")
                st.markdown("#### Ground Truth Verification")
                reveal_a = st.checkbox(
                    "👁️ Reveal Ground Truth & Compare",
                    key=f"reveal_a_{case['demo_id']}",
                )

                if reveal_a:
                    true_cat = case["ground_truth_category"]
                    pred_cat = result_a["predicted_class"]
                    is_correct = (pred_cat.lower() == true_cat.lower())

                    # Update Session Accuracy tracker
                    st.session_state["session_tested_cases"].add(case["demo_id"])
                    if is_correct:
                        st.session_state["session_correct_cases"].add(case["demo_id"])
                    else:
                        st.session_state["session_correct_cases"].discard(case["demo_id"])

                    if is_correct:
                        st.markdown(
                            f"""
                            <div class="truth-box-correct">
                                <strong>✅ Model Prediction Correct!</strong><br>
                                <strong>Ground Truth:</strong> {true_cat} | <strong>Model Predicted:</strong> {pred_cat}<br>
                                <em>Provenance:</em> {case['verification_notes']}
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
                    else:
                        st.markdown(
                            f"""
                            <div class="truth-box-incorrect">
                                <strong>⚠️ Model Misclassification</strong><br>
                                <strong>Ground Truth:</strong> {true_cat} | <strong>Model Predicted:</strong> {pred_cat}<br>
                                <em>Provenance:</em> {case['verification_notes']}
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                    st.markdown(f"🔗 [Inspect Original Case Record on Indian Kanoon]({case['source_url']})")

            # Session Accuracy Counter
            st.markdown("---")
            total_tested = len(st.session_state["session_tested_cases"])
            total_correct = len(st.session_state["session_correct_cases"])
            session_acc = (total_correct / total_tested * 100) if total_tested > 0 else 0.0

            sc1, sc2, sc3 = st.columns([2, 1, 1])
            sc1.markdown(f"**Demo Session Accuracy:** `{total_correct} / {total_tested}` cases verified ({session_acc:.1f}%)")
            sc1.caption("Calculated only from curated demo cases evaluated and revealed in this session.")
            with sc3:
                if st.button("Reset Session Stats", key="reset_stats_btn"):
                    st.session_state["session_tested_cases"].clear()
                    st.session_state["session_correct_cases"].clear()
                    st.rerun()

    # -------------------------------------------------------------
    # MODE B: BLIND CASE CHALLENGE
    # -------------------------------------------------------------
    with tab_blind:
        st.markdown("### Mode B: Blind Case Challenge")
        st.caption(
            "Evaluate a case without any prior indication of jurisdiction. Review the text, observe the model's "
            "predicted class, independently inspect the official court link, and then reveal the curated label."
        )

        blind_cases = demo_data.get("blind_cases", [])
        if blind_cases:
            b_titles = [f"{b['title']} — {b['description']}" for b in blind_cases]
            b_idx = st.selectbox(
                "Select a Blind Challenge Case:",
                options=range(len(b_titles)),
                format_func=lambda i: b_titles[i],
                key="mode_b_select",
            )
            b_case = blind_cases[b_idx]

            text_b = st.text_area(
                "Unlabeled Judgment Excerpt:",
                value=b_case["judgment_text"],
                height=140,
                key="mode_b_text",
            )

            col_b1, col_b2 = st.columns([1, 2])
            with col_b1:
                classify_b = st.button("⚖️ Classify Judgment", key="btn_classify_b", use_container_width=True)

            if classify_b or f"res_b_{b_case['demo_id']}" in st.session_state:
                if classify_b:
                    st.session_state[f"res_b_{b_case['demo_id']}"] = predict_judgment(text_b, pipeline=pipeline)

                result_b = st.session_state[f"res_b_{b_case['demo_id']}"]
                render_prediction_card(result_b)

                # Step 1: Open external source link
                st.markdown("---")
                st.markdown("#### Independent Verification")
                st.markdown(
                    f"1. 🔗 **[Open Source Record on Indian Kanoon ({b_case['title']})]({b_case['source_url']})**"
                )
                st.caption("Inspect the actual docket and bench order in a separate browser tab to verify the procedural jurisdiction.")

                # Step 2: Reveal ground truth
                reveal_b = st.checkbox(
                    "2. 👁️ Reveal Curated Ground Truth",
                    key=f"reveal_b_{b_case['demo_id']}",
                )

                if reveal_b:
                    true_b = b_case["ground_truth_category"]
                    pred_b = result_b["predicted_class"]
                    match_b = (pred_b.lower() == true_b.lower())

                    if match_b:
                        st.success(f"**Verified Match!** Curated Ground Truth is **{true_b}** (matches model prediction).")
                    else:
                        st.warning(f"**Discrepancy:** Curated Ground Truth is **{true_b}**, but model predicted **{pred_b}**.")

                    st.info(f"**Jurisdictional Context:** {b_case['verification_notes']}")

    # -------------------------------------------------------------
    # MODE C: ANALYZE YOUR OWN TEXT
    # -------------------------------------------------------------
    with tab_custom:
        st.markdown("### Mode C: Analyze Your Own Text")
        st.caption(
            "Paste any Indian court judgment excerpt or test illustrative statements. "
            "Use the guided presets below to explore model sensitivity and out-of-domain behavior."
        )

        guided_examples = demo_data.get("guided_examples", [])
        example_options = ["Custom Input (Clear / Empty)"] + [e["label"] for e in guided_examples]

        selected_preset = st.selectbox(
            "Load Guided Preset Example (Optional):",
            options=example_options,
            index=0,
            key="preset_select",
        )

        preset_text = ""
        if selected_preset != "Custom Input (Clear / Empty)":
            for e in guided_examples:
                if e["label"] == selected_preset:
                    preset_text = e["text"]
                    break

        user_text = st.text_area(
            "Indian Court Judgment Text:",
            value=preset_text,
            height=200,
            placeholder="Paste Indian court judgment text, appeal excerpts, or test sentences here...",
            key="custom_input_area",
            help="Prediction is restricted strictly to text. No case numbers, court types, or outcome labels are required.",
        )

        col_c1, col_c2 = st.columns([1, 4])
        with col_c1:
            classify_c = st.button("⚖️ Classify Judgment", key="btn_classify_c", use_container_width=True)

        if classify_c:
            if not user_text.strip():
                st.warning("⚠️ Please paste or enter text before classifying.")
            else:
                # Heuristic Out-of-Domain Notice
                words = user_text.strip().split()
                if len(words) < 25:
                    st.info(
                        "ℹ️ **Short / Out-of-Domain Notice:** This input is very short (<25 words) compared to "
                        "the Indian court judgments on which LegalLens was trained (median length ~2,310 words). "
                        "The classifier will evaluate learned word patterns, but predictions on short everyday sentences "
                        "do not constitute a reliable legal determination."
                    )

                with st.spinner("Analyzing text and evaluating n-gram features..."):
                    result_c = predict_judgment(user_text, pipeline=pipeline)

                if not result_c.get("is_valid", False):
                    st.error(f"❌ Input validation error: {result_c.get('status', 'Unknown error')}")
                else:
                    render_prediction_card(result_c)

    # -------------------------------------------------------------
    # MODE D: EXPLORE THE NLP PIPELINE
    # -------------------------------------------------------------
    with tab_nlp:
        st.markdown("### Mode D: Explore the NLP Pipeline")
        st.caption(
            "Inspect how an Indian court judgment moves through the real 6-stage NLP pipeline: "
            "from raw characters through statutory normalization, TF-IDF vectorization, and the Linear SVM margin."
        )

        nlp_sample_default = (
            "The appellant was convicted by the learned Sessions Judge under Section 302 of the Indian Penal Code "
            "and sentenced to rigorous imprisonment for life. Bail application was rejected. Meanwhile, the plaintiff "
            "filed a civil suit for recovery of possession with costs."
        )

        nlp_input = st.text_area(
            "Input Text for Pipeline Inspection:",
            value=nlp_sample_default,
            height=120,
            key="nlp_pipeline_text",
        )

        if st.button("🔬 Trace NLP Pipeline Stages", key="btn_trace_nlp", use_container_width=True):
            if not nlp_input.strip():
                st.warning("⚠️ Please provide text to inspect.")
            else:
                # Stage 1: Raw Input
                st.markdown(
                    f"""
                    <div class="pipeline-step-card">
                        <div class="pipeline-step-title">Stage 1: Raw Document Ingestion</div>
                        <div class="pipeline-step-desc">Accepts raw unformatted Indian court judgment text.</div>
                        <strong>Input Characters:</strong> {len(nlp_input):,} | <strong>Input Words:</strong> {len(nlp_input.split()):,}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                # Stage 2: Normalization
                clean_preview = normalize_legal_text(nlp_input)
                st.markdown(
                    f"""
                    <div class="pipeline-step-card">
                        <div class="pipeline-step-title">Stage 2: Domain-Aware Text Normalization</div>
                        <div class="pipeline-step-desc">
                            Standardizes internal whitespace and lowercases text while strictly preserving statutory references 
                            (e.g., <code>section 302</code>, <code>article 226</code>) and critical legal negations.
                        </div>
                        <code>{clean_preview[:300]}...</code>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                # Stage 3 & 4: TF-IDF Features
                tfidf_vec = pipeline.named_steps["tfidf"]
                X_vec = tfidf_vec.transform([clean_preview])
                nonzero_count = X_vec.nnz
                top_feats = extract_top_features(nlp_input, pipeline=pipeline, top_n=8)

                st.markdown(
                    f"""
                    <div class="pipeline-step-card">
                        <div class="pipeline-step-title">Stages 3 & 4: TF-IDF Unigram/Bigram Feature Transformation</div>
                        <div class="pipeline-step-desc">
                            Applies sublinear term frequency scaling across the 40,000-term legal vocabulary.
                        </div>
                        <strong>Active Non-Zero Features in Document:</strong> {nonzero_count} / 40,000 terms
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                # Display Active Features
                if top_feats:
                    st.markdown("##### Highest-Impact Features in Submitted Text:")
                    f_cols = st.columns(min(len(top_feats), 4))
                    for idx, feat in enumerate(top_feats[:4]):
                        col = f_cols[idx % 4]
                        chip_class = "token-chip-crim" if feat["favors"] == "Criminal" else "token-chip-civ"
                        col.markdown(
                            f"""
                            <div class="{chip_class}">
                                <strong>{feat['feature']}</strong><br>
                                TF-IDF: {feat['tfidf']:.3f}<br>
                                Weight: {feat['weight']:+.2f}<br>
                                Favors: {feat['favors']}
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                # Stage 5 & 6: SVM Margin & Decision
                res_nlp = predict_judgment(nlp_input, pipeline=pipeline)
                st.markdown(
                    f"""
                    <div class="pipeline-step-card">
                        <div class="pipeline-step-title">Stages 5 & 6: Linear SVM Margin & Classification Decision</div>
                        <div class="pipeline-step-desc">
                            The balanced Linear SVM computes the geometric distance from the hyperplane.
                        </div>
                        <strong>Signed Decision Score (Margin):</strong> <code>{res_nlp['decision_score']:+.4f}</code><br>
                        <strong>Estimated Confidence:</strong> <code>{res_nlp['confidence_prob'] * 100:.1f}%</code><br>
                        <strong>Final Predicted Category:</strong> <strong>{res_nlp['predicted_class']}</strong>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    # -------------------------------------------------------------
    # EXPANDABLE ABOUT / MODEL SPECIFICATION
    # -------------------------------------------------------------
    with st.expander("ℹ️ About LegalLens & Official Benchmark Metrics"):
        st.markdown(
            """
            ### Project Overview & Methodology
            **LegalLens** is an academic Natural Language Processing (NLP) research project engineered for whole-document 
            binary categorization of Indian court judgments into **Civil** versus **Criminal** jurisdictions.

            #### Architecture:
            - **Text Normalization:** Case-folding and whitespace collapsing with statutory notation preservation.
            - **Feature Representation:** TF-IDF unigram + bigram representations (`max_features=40,000`, `sublinear_tf=True`, `min_df=3`, `max_df=0.90`).
            - **Classification Engine:** Linear Support Vector Machine (`LinearSVC`, `class_weight='balanced'`, $C=1.0$).
            - **Class Balancing:** Balanced class weighting was selected to protect minority Criminal recall across the ~4:1 class imbalance.

            #### Official Held-Out Test Benchmarks (Frozen Test Partition, N = 3,372):
            - **Overall Accuracy:** `93.59%`
            - **Macro-F1 Score:** `0.9061` (remained above 0.90 on unseen data)
            - **Criminal Recall (Minority Class):** `91.68%`
            - **Civil F1-Score:** `95.90%` | **Criminal F1-Score:** `85.33%`
            - *The small validation-to-test performance drop indicates good generalization and no evidence of substantial overfitting.*
            """
        )

    # Academic Disclaimer
    st.markdown(
        """
        <div class="disclaimer-box">
            <strong>Academic Research Notice:</strong> LegalLens is an academic NLP research project that classifies 
            Indian court judgment text as Civil or Criminal. It uses statistical language features learned from labeled 
            examples and does not provide legal advice, interpret statutes, predict case outcomes, or replace professional legal judgment.
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
