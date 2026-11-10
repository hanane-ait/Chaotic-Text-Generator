"""
Lab 03 - Chaotic Text Generator
Interactive NLP test bench using:
- Plain Hidden Markov Model (HMM)
- Role-split HMM
- Trigram language model
- Viterbi POS tagger

Run:  streamlit run app.py   (from the project folder)
"""

import random
import re
from pathlib import Path

import pandas as pd
import streamlit as st

from src.pipeline import (
    HMM,
    NGramGenerator,
    load_tagged,
    split_by_document,
    generation_metrics,
    coarse,
)

# Absolute path to the corpus: works wherever Streamlit is launched from
# (local run, Streamlit Community Cloud, ...).
BASE_DIR = Path(__file__).resolve().parent
CORPUS_PATH = BASE_DIR / "data" / "corpus.csv"

MODEL_NAMES = ["Plain HMM", "Role-split HMM", "Trigram"]

METRIC_LABELS = {
    "valid_tag_pattern": "Valid tag pattern",
    "plausible_subject_verb": "Plausible subject-verb",
    "novel_sentences": "Novelty",
    "distinct_2": "Distinct-2",
}

CURVE_TEMPERATURES = [0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 2.5, 3.0]


# =================================================
# 1. PAGE CONFIGURATION
# =================================================

st.set_page_config(
    page_title="Chaotic Text Generator",
    page_icon="🧠",
    layout="wide",
)

st.title("🧠 Chaotic Text Generator")

st.caption(
    "NLP Laboratory | Hidden Markov Models, "
    "Viterbi decoding and Trigram generation"
)


# =================================================
# 2. LOAD DATA AND MODELS
# =================================================

@st.cache_resource
def get_models():
    docs = load_tagged(str(CORPUS_PATH))
    train, test = split_by_document(docs)

    hmm_plain = HMM(train)
    hmm_split = HMM(train, split_states=True)
    ng = NGramGenerator([words for words, tags in train])

    training_patterns = {" ".join(tags) for words, tags in train}

    return train, test, hmm_plain, hmm_split, ng, training_patterns


try:
    train, test, hmm_plain, hmm_split, ng, training_patterns = get_models()
except Exception as e:
    st.error(f"Error while loading corpus or models: {e}")
    st.stop()

MODELS = {
    "Plain HMM": hmm_plain,
    "Role-split HMM": hmm_split,
    "Trigram": ng,
}

st.success(
    f"Models loaded successfully | "
    f"{len(train)} training sentences | "
    f"{len(test)} test sentences"
)


# =================================================
# 3. HELPER FUNCTIONS
# =================================================

def generate_batch(model_name, temperature, seed, number, prefix_words):
    """Generate `number` sentences with one model.

    Returns (samples, table, errors):
      samples : list of (words, coarse_tags) used by generation_metrics
      table   : list of dict rows for the results table
      errors  : number of sentences that could not be generated
    """
    model = MODELS[model_name]
    rng = random.Random(int(seed))
    samples, table, errors = [], [], 0

    for i in range(int(number)):
        try:
            if model_name == "Trigram":
                words = model.sample_sentence(
                    rng, temperature, prefix=prefix_words
                )
                # No hidden states in a trigram: estimate POS with Viterbi.
                tags = hmm_plain.viterbi(words)
                displayed_tags = tags
            else:
                words, states = model.sample_with_states(rng, temperature)
                displayed_tags = states        # real hidden states
                tags = coarse(states)          # coarse POS for the metrics

            samples.append((words, tags))
            table.append({
                "ID": i + 1,
                "Sentence": " ".join(words),
                "Tag sequence": " → ".join(displayed_tags),
                "Valid pattern": (
                    "Yes" if " ".join(tags) in training_patterns else "No"
                ),
            })
        except Exception:
            errors += 1

    return samples, table, errors


def tokenize(text):
    """Lower-case and split words and punctuation: 'birds.' -> 'birds', '.'"""
    return re.findall(r"\w+|[^\w\s]", text.lower())


def check_prefix(model_name, prefix_words):
    """Return the list of prompt words unknown to the trigram vocabulary."""
    if model_name != "Trigram":
        return []
    return [w for w in prefix_words if w not in ng.vocab]


# =================================================
# 4. MODEL CONFIGURATION
# =================================================

st.header("1. Generation configuration")

col1, col2 = st.columns(2)

with col1:
    model_name = st.selectbox("Select generation model", MODEL_NAMES)

    temperature = st.slider(
        "Temperature",
        min_value=0.3,
        max_value=3.0,
        value=1.0,
        step=0.1,
        help=(
            "Low temperature produces more predictable "
            "sentences. High temperature increases randomness."
        ),
    )

with col2:
    seed = st.number_input(
        "Random seed", min_value=0, max_value=10000, value=42, step=1
    )

    number = st.number_input(
        "Number of sentences",
        min_value=10,
        max_value=500,
        value=100,
        step=10,
    )

prefix_text = st.text_input(
    "Trigram prompt (optional, used by the Trigram model only)",
    value="the",
    help="Words used as the initial context of the trigram model.",
).lower().strip()

prefix_words = prefix_text.split()


# =================================================
# 5. GENERATE SENTENCES  (results are kept in session_state)
# =================================================

st.divider()

st.header("2. Generate sentences")

if st.button(
    "🚀 Generate sentences", type="primary", use_container_width=True
):
    unknown_prefix = check_prefix(model_name, prefix_words)

    if unknown_prefix:
        st.error("Unknown prompt words: " + ", ".join(unknown_prefix))
    else:
        with st.spinner("Generating sentences..."):
            samples, table, errors = generate_batch(
                model_name, temperature, seed, number, prefix_words
            )

        if not samples:
            st.error("No sentences were generated.")
        else:
            st.session_state["generation"] = {
                "model_name": model_name,
                "temperature": temperature,
                "seed": int(seed),
                "number": int(number),
                "errors": errors,
                "metrics": generation_metrics(samples, train, hmm_plain),
                "df": pd.DataFrame(table),
            }


# =================================================
# 6. DISPLAY RESULTS  (outside the button: they survive any rerun)
# =================================================

gen = st.session_state.get("generation")

if gen is None:
    st.info("Press **Generate sentences** to see the metrics and the table.")
else:
    metrics = gen["metrics"]
    df = gen["df"]

    st.divider()
    st.header("3. Generation metrics")

    st.caption(
        f"Model: **{gen['model_name']}** | temperature {gen['temperature']:.1f} "
        f"| seed {gen['seed']} | {len(df)} sentences"
    )

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Valid tag pattern", f"{metrics['valid_tag_pattern']:.1%}")
    m2.metric(
        "Plausible subject-verb", f"{metrics['plausible_subject_verb']:.1%}"
    )
    m3.metric("Novelty", f"{metrics['novel_sentences']:.1%}")
    m4.metric("Distinct-2", f"{metrics['distinct_2']:.3f}")

    st.caption(
        "Metrics are calculated on the generated sentences. "
        "POS categories are used for HMM metric comparison."
    )

    if gen["errors"]:
        st.warning(f"{gen['errors']} sentence(s) could not be generated.")

    st.divider()
    st.header("4. Generated sentences")

    if gen["model_name"] == "Trigram":
        st.info(
            "The trigram model has no hidden HMM states. "
            "POS tags are estimated using the Viterbi algorithm."
        )
    else:
        st.info(
            "The displayed tag sequence represents the actual "
            "hidden states sampled by the HMM."
        )

    st.dataframe(df, use_container_width=True, hide_index=True)

    st.caption(f"{len(df)} sentences generated successfully.")

    st.download_button(
        label="📥 Download generated sentences (CSV)",
        data=df.to_csv(index=False).encode("utf-8-sig"),
        file_name="generated_sentences.csv",
        mime="text/csv",
        use_container_width=True,
    )


# =================================================
# 7. MODEL COMPARISON AND TEMPERATURE CURVE
# =================================================

st.divider()

st.header("5. Model comparison and temperature curve")

st.write(
    "Compare the three models with the same seed and number of sentences. "
    "The table uses the temperature chosen above; the curve sweeps "
    f"{CURVE_TEMPERATURES[0]} to {CURVE_TEMPERATURES[-1]}."
)

if st.button("📊 Compare the 3 models", use_container_width=True):
    unknown_prefix = check_prefix("Trigram", prefix_words)

    if unknown_prefix:
        st.error("Unknown prompt words: " + ", ".join(unknown_prefix))
    else:
        rows_cmp = []
        rows_curve = []

        steps = len(MODEL_NAMES) * (len(CURVE_TEMPERATURES) + 1)
        progress = st.progress(0.0)
        done = 0

        for name in MODEL_NAMES:
            # --- comparison at the selected temperature
            samples, _, _ = generate_batch(
                name, temperature, seed, number, prefix_words
            )
            m = generation_metrics(samples, train, hmm_plain)
            rows_cmp.append({"Model": name, **m})
            done += 1
            progress.progress(done / steps)

            # --- sweep over temperatures
            for t in CURVE_TEMPERATURES:
                samples, _, _ = generate_batch(
                    name, t, seed, number, prefix_words
                )
                m = generation_metrics(samples, train, hmm_plain)
                rows_curve.append({"Model": name, "Temperature": t, **m})
                done += 1
                progress.progress(done / steps)

        progress.empty()

        st.session_state["comparison"] = {
            "temperature": temperature,
            "seed": int(seed),
            "number": int(number),
            "table": pd.DataFrame(rows_cmp),
            "curve": pd.DataFrame(rows_curve),
        }

cmp_state = st.session_state.get("comparison")

if cmp_state is not None:
    st.subheader("Comparison table")

    st.caption(
        f"Temperature {cmp_state['temperature']:.1f} | "
        f"seed {cmp_state['seed']} | {cmp_state['number']} sentences per model"
    )

    cmp_df = cmp_state["table"].set_index("Model")
    cmp_view = pd.DataFrame({
        "Valid tag pattern": cmp_df["valid_tag_pattern"].map("{:.1%}".format),
        "Plausible subject-verb": cmp_df["plausible_subject_verb"].map(
            "{:.1%}".format
        ),
        "Novelty": cmp_df["novel_sentences"].map("{:.1%}".format),
        "Distinct-2": cmp_df["distinct_2"].map("{:.3f}".format),
    })

    st.dataframe(cmp_view, use_container_width=True)

    st.subheader("Metric against temperature")

    metric_key = st.selectbox(
        "Metric to plot",
        list(METRIC_LABELS),
        format_func=lambda k: METRIC_LABELS[k],
    )

    curve = cmp_state["curve"].pivot(
        index="Temperature", columns="Model", values=metric_key
    )[MODEL_NAMES]

    st.line_chart(curve)

    st.caption(
        f"{METRIC_LABELS[metric_key]} for each model, "
        f"{cmp_state['number']} sentences per point, seed {cmp_state['seed']}. "
        "Trigram uses the prompt above."
    )


# =================================================
# 8. VITERBI POS TAGGER
# =================================================

st.divider()

st.header("6. Viterbi POS tagger")

st.write(
    "Enter an English sentence to predict its "
    "Part-of-Speech (POS) tags using the Viterbi algorithm."
)

sentence = st.text_input(
    "Enter a sentence", placeholder="the children watch the birds."
)

if st.button("🔍 Analyze sentence", use_container_width=True):
    st.session_state["tagger_words"] = tokenize(sentence)

tagger_words = st.session_state.get("tagger_words")

if tagger_words is not None:
    if not tagger_words:
        st.warning("Please enter a sentence before analyzing.")
    else:
        unknown = [w for w in tagger_words if w not in hmm_plain.vocab]

        if unknown:
            st.warning(
                "Unknown words detected: "
                + ", ".join(unknown)
                + ". The HMM will use smoothing to handle them."
            )

        try:
            tags = hmm_plain.viterbi(tagger_words)

            tag_df = pd.DataFrame({
                "Word": tagger_words,
                "Viterbi POS tag": tags,
                "Vocabulary status": [
                    "Known" if w in hmm_plain.vocab else "Unknown"
                    for w in tagger_words
                ],
            })

            st.subheader("POS tagging results")

            st.dataframe(tag_df, use_container_width=True, hide_index=True)

            st.markdown(
                "**Tagged sentence:** "
                + " | ".join(
                    f"{w}/{t}" for w, t in zip(tagger_words, tags)
                )
            )
        except Exception as e:
            st.error(f"Viterbi tagging error: {e}")


# =================================================
# 9. SYNTHETIC CORPUS NOTICE
# =================================================

st.divider()

st.warning(
    "The corpus is synthetic and the grammar is tiny. "
    "Metrics such as valid tag pattern are defined relative "
    "to this corpus and do not measure the quality of real English."
)


# =================================================
# 10. FOOTER
# =================================================

st.divider()

st.caption(
    "Lab 03 - Chaotic Text Generator | NLP | HMM | Viterbi | Trigram"
)