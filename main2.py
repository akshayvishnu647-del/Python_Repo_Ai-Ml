import streamlit as st
import pandas as pd
import plotly.express as px
import json
import os
import re

try:
    import joblib
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False

st.set_page_config(
    page_title="AI Finance Automation",
    page_icon="💰",
    layout="wide"
)

CATEGORY_FILE = "categories.json"
MODEL_FILE = "category_model.pkl"
BUDGET_FILE = "budgets.json"
SAVINGS_GOAL_FILE = "savings_goal.json"
TRAINING_EXAMPLES_FILE = "training_examples.json"
BUNDLED_TRAIN_FILE = "financial_transactions_train_cleaned.csv"

# Change this if your transactions are in a different currency.
CURRENCY_SYMBOL = "₹"

# If AI confidence is below this value, the app leaves the expense
# Uncategorized instead of making an unreliable automatic decision.
ML_CONFIDENCE_THRESHOLD = 0.60

# Labels learned by the ML model -> names used by the finance dashboard.
ML_LABEL_MAP = {
    "Food": "Food and Dining",
    "Travel": "Transportation",
    "Shopping": "Shopping",
    "EMI": "EMI and Loans",
    "Investment": "Investment",
}

DEFAULT_CATEGORIES = {
    "Food and Dining": [
        "restaurant", "grocery", "coffee", "cafe", "bar",
        "takeout", "bakery", "swiggy", "zomato"
    ],
    "Transportation": [
        "gas station", "petrol", "public transit", "taxi",
        "rideshare", "uber", "ola", "car rental", "flight ticket"
    ],
    "Housing": [
        "rent", "mortgage", "home repair", "home improvement", "property tax"
    ],
    "Utilities": [
        "electricity", "water bill", "internet", "phone bill", "cable"
    ],
    "Entertainment": [
        "movie", "concert", "sports event", "game", "subscription"
    ],
    "Healthcare": [
        "doctor", "dentist", "pharmacy", "health insurance", "fitness"
    ],
    "Shopping": [
        "clothing", "electronics", "amazon", "myntra", "ebay"
    ],
    "Education": [
        "course", "tuition", "student loan"
    ],
    "EMI and Loans": [
        "emi", "loan"
    ],
    "Investment": [
        "investment", "mutual fund", "sip", "nps"
    ],
    "Uncategorized": [],
}


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------
def save_categories():
    with open(CATEGORY_FILE, "w", encoding="utf-8") as f:
        json.dump(st.session_state.categories, f, indent=2)


def save_budgets():
    with open(BUDGET_FILE, "w", encoding="utf-8") as f:
        json.dump(st.session_state.budgets, f, indent=2)


def save_savings_goal():
    with open(SAVINGS_GOAL_FILE, "w", encoding="utf-8") as f:
        json.dump(st.session_state.savings_goal, f, indent=2)


def add_keyword_to_category(category, keyword):
    """
    Store a user correction as a future keyword rule.
    This gives the app a simple human-in-the-loop learning mechanism.
    """
    keyword = str(keyword).strip().lower()
    if not keyword:
        return False

    st.session_state.categories.setdefault(category, [])

    if keyword not in st.session_state.categories[category]:
        st.session_state.categories[category].append(keyword)
        save_categories()
        return True

    return False


def load_training_examples():
    if os.path.exists(TRAINING_EXAMPLES_FILE):
        try:
            with open(TRAINING_EXAMPLES_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                return data
        except Exception:
            pass
    return []


def save_training_examples(examples):
    with open(TRAINING_EXAMPLES_FILE, "w", encoding="utf-8") as f:
        json.dump(examples, f, indent=2)


def add_training_example(text, category):
    """Persist a labeled transaction so later retraining can use it."""
    text = str(text).strip()
    category = str(category).strip()
    if not text or not category or category == "Uncategorized":
        return

    examples = load_training_examples()
    cleaned = clean_text(text)
    for item in examples:
        if clean_text(item.get("text", "")) == cleaned and item.get("category") == category:
            return

    examples.append({"text": text, "category": category})
    save_training_examples(examples)


# ---------------------------------------------------------------------------
# Text preprocessing
# IMPORTANT: This is the same preprocessing used when training category_model.pkl.
# ---------------------------------------------------------------------------
def clean_text(text):
    text = str(text)

    # Remove amounts such as "INR 48,395".
    text = re.sub(r"-?\s*INR\s*[\d,]+", " ", text, flags=re.I)

    # Remove transaction reference numbers such as "Ref 776405".
    text = re.sub(r"-?\s*Ref\s*\d+", " ", text, flags=re.I)

    # Remove remaining numbers.
    text = re.sub(r"\d+", " ", text)

    # Normalize whitespace and case.
    text = re.sub(r"\s+", " ", text).strip().lower()

    return text


@st.cache_resource
def load_ml_model():
    if not ML_AVAILABLE:
        return None

    if not os.path.exists(MODEL_FILE):
        return None

    try:
        return joblib.load(MODEL_FILE)
    except Exception as e:
        st.error(f"Could not load AI model: {e}")
        return None


# ---------------------------------------------------------------------------
# AI categorization
# ---------------------------------------------------------------------------
def predict_with_ai(text):
    """
    Returns:
        category, confidence, raw_label
    """
    model = load_ml_model()

    if model is None:
        return "Uncategorized", 0.0, None

    cleaned = clean_text(text)

    if not cleaned:
        return "Uncategorized", 0.0, None

    try:
        probabilities = model.predict_proba([cleaned])[0]
        best_index = probabilities.argmax()

        raw_label = model.classes_[best_index]
        confidence = float(probabilities[best_index])

        mapped_category = ML_LABEL_MAP.get(raw_label, raw_label)

        if confidence >= ML_CONFIDENCE_THRESHOLD:
            return mapped_category, confidence, raw_label

        return "Uncategorized", confidence, raw_label

    except Exception:
        return "Uncategorized", 0.0, None


def _append_example(rows, text, category, repeats=1):
    text = str(text).strip()
    category = str(category).strip()
    if not text or not category or category == "Uncategorized":
        return
    mapped = ML_LABEL_MAP.get(category, category)
    cleaned = clean_text(text)
    if not cleaned:
        return
    for _ in range(repeats):
        rows.append((cleaned, mapped))


def collect_training_data(include_bundled=True, extra_file=None):
    """
    Build text/label pairs from:
      1. User corrections saved on disk
      2. Currently loaded categorized expenses
      3. Keyword rules (bootstraps categories the bundled model never saw)
      4. Optional bundled training CSV
      5. Optional extra labeled CSV
    """
    rows = []

    for item in load_training_examples():
        _append_example(rows, item.get("text", ""), item.get("category", ""), repeats=8)

    if st.session_state.get("debits_df") is not None:
        labeled = st.session_state.debits_df
        labeled = labeled[labeled["Category"] != "Uncategorized"]
        for _, row in labeled.iterrows():
            repeats = 8 if row.get("Match Type") == "Manual correction" else 3
            _append_example(rows, row["Details"], row["Category"], repeats=repeats)

    for category, keywords in st.session_state.categories.items():
        if category == "Uncategorized":
            continue
        for keyword in keywords:
            _append_example(rows, keyword, category, repeats=20)

    if include_bundled and os.path.exists(BUNDLED_TRAIN_FILE):
        try:
            bundled = pd.read_csv(BUNDLED_TRAIN_FILE).dropna()
            text_col = "Transaction_Text" if "Transaction_Text" in bundled.columns else bundled.columns[0]
            label_col = "Label" if "Label" in bundled.columns else bundled.columns[1]
            cleaned = bundled[text_col].map(clean_text)
            mapped = bundled[label_col].map(
                lambda x: ML_LABEL_MAP.get(str(x).strip(), str(x).strip())
            )
            valid = cleaned.astype(bool) & mapped.ne("") & mapped.ne("Uncategorized")
            rows.extend(zip(cleaned[valid], mapped[valid]))
        except Exception as e:
            st.warning(f"Could not load bundled training data: {e}")

    if extra_file is not None:
        try:
            extra = pd.read_csv(extra_file).dropna()
            extra.columns = [col.strip() for col in extra.columns]
            text_col = next(
                (
                    col for col in extra.columns
                    if col.lower() in {
                        "transaction_text", "details", "transaction details",
                        "text", "description",
                    }
                ),
                extra.columns[0],
            )
            label_col = next(
                (
                    col for col in extra.columns
                    if col.lower() in {"label", "category"}
                ),
                extra.columns[1] if len(extra.columns) > 1 else None,
            )
            if label_col is not None:
                for _, row in extra.iterrows():
                    _append_example(rows, row[text_col], row[label_col], repeats=4)
        except Exception as e:
            st.warning(f"Could not load extra labeled CSV: {e}")

    if not rows:
        return [], []

    texts, labels = zip(*rows)
    return list(texts), list(labels)


def train_category_model(texts, labels):
    """Train TF-IDF + logistic regression and save category_model.pkl."""
    if not ML_AVAILABLE:
        raise RuntimeError("scikit-learn / joblib is not installed.")

    unique_labels = sorted(set(labels))
    if len(unique_labels) < 2:
        raise ValueError("Need labeled examples from at least 2 categories to train.")

    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1)),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
    ])

    metrics = {
        "examples": len(texts),
        "categories": unique_labels,
        "test_accuracy": None,
        "train_accuracy": None,
    }

    label_counts = pd.Series(labels).value_counts()
    can_split = len(texts) >= 20 and label_counts.min() >= 2

    if can_split:
        X_train, X_test, y_train, y_test = train_test_split(
            texts,
            labels,
            test_size=0.2,
            random_state=42,
            stratify=labels,
        )
        pipe.fit(X_train, y_train)
        metrics["test_accuracy"] = float(accuracy_score(y_test, pipe.predict(X_test)))

    # Always refit on every labeled example before saving the production model.
    pipe.fit(texts, labels)
    metrics["train_accuracy"] = float(accuracy_score(labels, pipe.predict(texts)))

    joblib.dump(pipe, MODEL_FILE)
    load_ml_model.clear()
    return metrics


def recategorize_uncategorized(df):
    """Run the trained model only on rows that still need a category."""
    df = df.copy()
    updated = 0

    for idx, row in df.iterrows():
        match_type = str(row.get("Match Type", ""))
        category = str(row.get("Category", "Uncategorized"))
        needs_model = (
            category == "Uncategorized"
            or match_type in {"Needs review", "Uncategorized"}
        )
        if not needs_model or match_type == "Manual correction":
            continue

        predicted, confidence, raw_label = predict_with_ai(str(row["Details"]))
        if predicted == "Uncategorized":
            df.at[idx, "Match Type"] = "Needs review"
            df.at[idx, "AI Confidence"] = confidence
            continue

        df.at[idx, "Category"] = predicted
        df.at[idx, "Match Type"] = f"AI: {raw_label}"
        df.at[idx, "AI Confidence"] = confidence
        updated += 1

    return df, updated


def keyword_category(text):
    """
    Pass 1: transparent user-editable rules.
    """
    details = str(text).lower()

    # Check longer phrases first to reduce accidental partial matches.
    candidates = []

    for category, keywords in st.session_state.categories.items():
        if category == "Uncategorized":
            continue

        for keyword in keywords:
            keyword = str(keyword).strip().lower()
            if keyword:
                candidates.append((len(keyword), category, keyword))

    candidates.sort(reverse=True)

    for _, category, keyword in candidates:
        if keyword in details:
            return category, keyword

    return None, None


def categorize_transactions(df):
    """
    Two-pass categorization:

    Pass 1:
        User-editable keyword rules.

    Pass 2:
        Trained ML model for transactions not matched by rules.

    The AI also stores confidence and the method used.
    """
    df = df.copy()

    df["Category"] = "Uncategorized"
    df["Match Type"] = "Uncategorized"
    df["AI Confidence"] = 0.0

    for idx, row in df.iterrows():
        details = str(row["Details"])

        # ---------------------------------------------------------------
        # Pass 1 - keyword rules
        # ---------------------------------------------------------------
        category, keyword = keyword_category(details)

        if category:
            df.at[idx, "Category"] = category
            df.at[idx, "Match Type"] = f"Keyword: {keyword}"
            df.at[idx, "AI Confidence"] = 1.0
            continue

        # ---------------------------------------------------------------
        # Pass 2 - trained AI
        # ---------------------------------------------------------------
        category, confidence, raw_label = predict_with_ai(details)

        if category != "Uncategorized":
            df.at[idx, "Category"] = category
            df.at[idx, "Match Type"] = f"AI: {raw_label}"
            df.at[idx, "AI Confidence"] = confidence
        else:
            df.at[idx, "Match Type"] = "Needs review"
            df.at[idx, "AI Confidence"] = confidence

    return df


# ---------------------------------------------------------------------------
# Transaction loading
# ---------------------------------------------------------------------------
def load_transactions(file):
    try:
        df = pd.read_csv(file)

        df.columns = [col.strip() for col in df.columns]

        if "Transaction Details" in df.columns:
            df = df.rename(columns={"Transaction Details": "Details"})

        required_columns = {"Date", "Details", "Amount"}

        missing = required_columns - set(df.columns)

        if missing:
            st.error(
                "Missing required column(s): "
                + ", ".join(sorted(missing))
            )
            return None

        # Convert amount safely.
        if pd.api.types.is_string_dtype(df["Amount"]):
            df["Amount"] = (
                df["Amount"]
                .astype(str)
                .str.replace("$", "", regex=False)
                .str.replace("₹", "", regex=False)
                .str.replace(",", "", regex=False)
                .str.replace("+", "", regex=False)
                .str.strip()
            )

        df["Amount"] = pd.to_numeric(df["Amount"], errors="coerce")
        df = df.dropna(subset=["Amount", "Details", "Date"]).copy()

        # Date conversion.
        df["Date"] = pd.to_datetime(
            df["Date"],
            dayfirst=True,
            errors="coerce"
        )

        df = df.dropna(subset=["Date"]).copy()

        # Determine debit/credit BEFORE converting amounts to absolute values.
        df["debits/credits"] = df["Amount"].apply(
            lambda x: "debit" if x < 0 else "credit"
        )

        df["Amount"] = df["Amount"].abs()

        # Automatic categorization.
        df = categorize_transactions(df)

        return df

    except Exception as e:
        st.error(f"Error loading transactions: {e}")
        return None


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
if "categories" not in st.session_state:
    if os.path.exists(CATEGORY_FILE):
        try:
            with open(CATEGORY_FILE, "r", encoding="utf-8") as f:
                st.session_state.categories = json.load(f)
        except Exception:
            st.session_state.categories = DEFAULT_CATEGORIES.copy()
            save_categories()
    else:
        st.session_state.categories = DEFAULT_CATEGORIES.copy()
        save_categories()

if "debits_df" not in st.session_state:
    st.session_state.debits_df = None

if "credits_df" not in st.session_state:
    st.session_state.credits_df = None

if "budgets" not in st.session_state:
    if os.path.exists(BUDGET_FILE):
        try:
            with open(BUDGET_FILE, "r", encoding="utf-8") as f:
                st.session_state.budgets = json.load(f)
        except Exception:
            st.session_state.budgets = {}
    else:
        st.session_state.budgets = {}

if "savings_goal" not in st.session_state:
    if os.path.exists(SAVINGS_GOAL_FILE):
        try:
            with open(SAVINGS_GOAL_FILE, "r", encoding="utf-8") as f:
                st.session_state.savings_goal = json.load(f)
        except Exception:
            st.session_state.savings_goal = {}
    else:
        st.session_state.savings_goal = {}


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
def main():

    # ------------------------------------------------------------------
    # Custom styling
    # ------------------------------------------------------------------
    st.markdown(
        """
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

            html, body, .stApp, [class*="css"] {
                font-family: 'Inter', sans-serif;
            }

            /* Main app background - colorful multi-tone gradient */
            .stApp {
                background: linear-gradient(135deg, #E8F5E9 0%, #E0F7FA 45%, #F1F8E9 100%);
            }

            /* Sidebar - bold dark green gradient */
            section[data-testid="stSidebar"] {
                background: linear-gradient(180deg, #1B5E20 0%, #2E7D32 55%, #1B5E20 100%);
                border-right: none;
            }
            section[data-testid="stSidebar"] > div {
                padding-top: 0.5rem;
            }
            section[data-testid="stSidebar"] * {
                color: #FFFFFF !important;
            }
            section[data-testid="stSidebar"] div[data-testid="stMetric"] {
                background-color: rgba(255, 255, 255, 0.12);
                border: 1px solid rgba(255, 255, 255, 0.25);
            }
            section[data-testid="stSidebar"] div[data-testid="stMetricValue"] {
                color: #FFFFFF !important;
            }
            section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] {
                background-color: rgba(255, 255, 255, 0.1);
                border: 1.5px dashed rgba(255, 255, 255, 0.5);
            }
            section[data-testid="stSidebar"] .stButton > button {
                background-color: rgba(255, 255, 255, 0.15);
                border: 1px solid rgba(255, 255, 255, 0.4);
                color: #FFFFFF;
            }
            section[data-testid="stSidebar"] .stButton > button:hover {
                background-color: rgba(255, 255, 255, 0.25);
            }
            section[data-testid="stSidebar"] div[data-testid="stAlert"] {
                background-color: rgba(255, 255, 255, 0.12);
            }

            /* Default text color across the main content area */
            .stApp, .stApp p, .stApp span, .stApp label {
                color: #1E293B;
            }

            .block-container {
                padding-top: 1.5rem;
                padding-bottom: 3rem;
                max-width: 1200px;
            }

            /* Hero header */
            .hero-banner {
                background: linear-gradient(135deg, #2E7D32 0%, #43A047 100%);
                border-radius: 18px;
                padding: 28px 32px;
                margin-bottom: 24px;
                box-shadow: 0 4px 14px rgba(46, 125, 50, 0.25);
            }
            .hero-banner h1 {
                color: #FFFFFF !important;
                font-size: 2rem;
                font-weight: 800;
                margin: 0 0 6px 0;
            }
            .hero-banner p {
                color: #E8F5E9 !important;
                font-size: 0.95rem;
                margin: 0;
            }

            /* Metric cards */
            div[data-testid="stMetric"] {
                background-color: #FFFFFF;
                border: 1px solid #DCEBDD;
                border-radius: 14px;
                padding: 16px 18px;
                box-shadow: 0 1px 3px rgba(0,0,0,0.04), 0 1px 8px rgba(0,0,0,0.04);
                transition: transform 0.15s ease, box-shadow 0.15s ease;
            }
            div[data-testid="stMetric"]:hover {
                transform: translateY(-2px);
                box-shadow: 0 4px 14px rgba(46, 125, 50, 0.12);
            }
            div[data-testid="stMetricValue"] {
                font-size: 1.55rem;
                font-weight: 800;
                color: #1E3A2B;
            }
            div[data-testid="stMetricLabel"] {
                color: #4B5D52;
                font-weight: 600;
                font-size: 0.85rem;
                text-transform: uppercase;
                letter-spacing: 0.03em;
            }

            /* Section headers */
            h2, h3, h4 {
                font-weight: 700 !important;
                color: #1E3A2B !important;
            }

            /* Tabs - pill style */
            .stTabs [data-baseweb="tab-list"] {
                gap: 6px;
                background-color: transparent;
                border-bottom: none;
            }
            .stTabs [data-baseweb="tab"] {
                border-radius: 999px;
                padding: 8px 20px;
                font-weight: 600;
                background-color: #FFFFFF;
                border: 1px solid #DCEBDD;
                color: #4B5D52;
            }
            .stTabs [aria-selected="true"] {
                background-color: #2E7D32 !important;
                color: #FFFFFF !important;
                border-color: #2E7D32 !important;
            }
            .stTabs [aria-selected="true"] p {
                color: #FFFFFF !important;
            }
            .stTabs [data-baseweb="tab-highlight"] {
                background-color: transparent;
            }

            /* Buttons */
            .stButton > button {
                border-radius: 10px;
                font-weight: 600;
                border: 1px solid #DCEBDD;
                transition: all 0.15s ease;
            }
            .stButton > button[kind="primary"] {
                background: linear-gradient(135deg, #2E7D32 0%, #43A047 100%);
                border: none;
                box-shadow: 0 2px 8px rgba(46, 125, 50, 0.3);
            }
            .stButton > button[kind="primary"]:hover {
                box-shadow: 0 4px 12px rgba(46, 125, 50, 0.4);
                transform: translateY(-1px);
            }

            /* Progress bars */
            .stProgress > div > div > div {
                border-radius: 10px;
                background: linear-gradient(90deg, #2E7D32, #66BB6A);
            }
            .stProgress > div > div {
                background-color: #DCEBDD;
                border-radius: 10px;
            }

            /* Inputs */
            div[data-baseweb="select"] > div, div[data-baseweb="input"] > div,
            .stNumberInput > div > div, .stTextInput > div > div {
                border-radius: 10px !important;
                border-color: #DCEBDD !important;
            }

            /* File uploader */
            [data-testid="stFileUploaderDropzone"] {
                background-color: #FFFFFF;
                border-radius: 12px;
                border: 1.5px dashed #A5D6A7;
            }

            /* Dataframes and tables */
            div[data-testid="stDataFrame"] {
                background-color: #FFFFFF;
                border-radius: 12px;
                overflow: hidden;
                border: 1px solid #DCEBDD;
            }

            /* Info/success/warning/error boxes */
            div[data-testid="stAlert"] {
                border-radius: 12px;
            }

            /* Dividers */
            hr {
                border-color: #DCEBDD !important;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="hero-banner">
            <h1>💰 AI Finance Automation Dashboard</h1>
            <p>Upload a transaction CSV to automatically categorize expenses,
            track spending, set budgets, and follow a savings goal.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ------------------------------------------------------------------
    # Sidebar - branding, AI status, file upload
    # ------------------------------------------------------------------
    st.sidebar.header("💰 Finance Automation")

    model = load_ml_model()

    if model is not None:
        n_classes = len(getattr(model, "classes_", []))
        st.sidebar.success(f"🤖 AI model loaded ({n_classes} categories)")
    else:
        st.sidebar.warning(
            "AI model not found. Train one from the AI Review tab "
            "(keyword rules still work)."
        )

    st.sidebar.divider()

    uploaded_file = st.sidebar.file_uploader(
        "Upload your transaction CSV",
        type=["csv"],
    )

    if (
        uploaded_file is not None
        and st.session_state.debits_df is None
    ):
        df = load_transactions(uploaded_file)

        if df is not None:
            st.session_state.debits_df = (
                df[df["debits/credits"] == "debit"]
                .copy()
                .reset_index(drop=True)
            )

            st.session_state.credits_df = (
                df[df["debits/credits"] == "credit"]
                .copy()
                .reset_index(drop=True)
            )

    if st.session_state.debits_df is None:
        st.info(
            "👋 Upload a CSV from the sidebar to get started. "
            "Columns needed: Date, Details (or 'Transaction Details'), Amount."
        )
        st.write("### Optional: train the categorizer first")
        st.caption(
            "You can train the model on the bundled dataset and keyword "
            "rules before uploading a statement. After upload, uncategorized "
            "rows will be classified automatically."
        )
        if st.button("🧠 Train model now"):
            texts, labels = collect_training_data(include_bundled=True)
            try:
                with st.spinner("Training categorizer..."):
                    metrics = train_category_model(texts, labels)
                msg = (
                    f"Model trained on {metrics['examples']} examples and "
                    f"{len(metrics['categories'])} categories."
                )
                if metrics["test_accuracy"] is not None:
                    msg += f" Held-out accuracy: {metrics['test_accuracy']:.1%}."
                st.success(msg)
                st.rerun()
            except Exception as e:
                st.error(f"Training failed: {e}")
        return

    total_transactions = (
        len(st.session_state.debits_df)
        + len(st.session_state.credits_df)
    )

    st.sidebar.metric("Transactions loaded", total_transactions)

    if st.sidebar.button("🔄 Upload a different file"):
        st.session_state.debits_df = None
        st.session_state.credits_df = None
        st.rerun()

    tab0, tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(
        [
            "🏠 Overview",
            "Expenses (Debits)",
            "Income (Credits)",
            "📊 Charts",
            "📈 Expense Tracking",
            "🎯 Budgets",
            "💰 Savings Goal",
            "🤖 AI Review",
        ]
    )

    # ------------------------------------------------------------------
    # TAB 0 - OVERVIEW
    # ------------------------------------------------------------------
    with tab0:

        st.subheader("Your Financial Snapshot")

        debits_df = st.session_state.debits_df
        credits_df = st.session_state.credits_df

        total_income = credits_df["Amount"].sum()
        total_expense = debits_df["Amount"].sum()
        net = total_income - total_expense

        c1, c2, c3 = st.columns(3)
        c1.metric("Total Income", f"{CURRENCY_SYMBOL}{total_income:,.2f}")
        c2.metric("Total Expenses", f"{CURRENCY_SYMBOL}{total_expense:,.2f}")
        c3.metric(
            "Net",
            f"{CURRENCY_SYMBOL}{net:,.2f}",
            delta=f"{CURRENCY_SYMBOL}{net:,.2f}",
        )

        st.divider()

        col1, col2 = st.columns(2)

        # --- Budget health summary ---
        with col1:
            st.write("#### 🎯 Budget Health")

            budgets = {
                k: v for k, v in st.session_state.budgets.items() if v > 0
            }

            if not budgets:
                st.caption("No budgets set yet — head to the Budgets tab.")
            else:
                actual_by_cat = debits_df.groupby("Category")["Amount"].sum()
                over_count = 0

                for cat, monthly_budget in budgets.items():
                    actual = float(actual_by_cat.get(cat, 0.0))
                    if actual > monthly_budget:
                        over_count += 1

                if over_count == 0:
                    st.success(
                        f"All {len(budgets)} budgeted categories are within budget."
                    )
                else:
                    st.warning(
                        f"{over_count} of {len(budgets)} budgeted categories "
                        "are over budget. See the Budgets tab for details."
                    )

        # --- Savings goal summary ---
        with col2:
            st.write("#### 💰 Savings Goal")

            goal = st.session_state.savings_goal
            target_amount = float(goal.get("target_amount", 0.0))

            if target_amount <= 0:
                st.caption("No savings goal set yet — head to the Savings Goal tab.")
            else:
                current_saved = float(goal.get("current_saved", 0.0))
                progress = min(current_saved / target_amount, 1.0)
                st.progress(progress)
                st.caption(
                    f"{CURRENCY_SYMBOL}{current_saved:,.2f} / "
                    f"{CURRENCY_SYMBOL}{target_amount:,.2f} "
                    f"({progress * 100:.0f}%)"
                )

        st.divider()

        st.write("#### Expenses by Category")

        category_totals = (
            debits_df
            .groupby("Category")["Amount"]
            .sum()
            .reset_index()
            .sort_values("Amount", ascending=False)
        )

        fig_overview_pie = px.pie(
            category_totals,
            values="Amount",
            names="Category",
            hole=0.45,
        )
        fig_overview_pie.update_layout(margin=dict(t=10, b=10, l=10, r=10))
        st.plotly_chart(fig_overview_pie, use_container_width=True)

    # ------------------------------------------------------------------
    # TAB 1 - Expenses
    # ------------------------------------------------------------------
    with tab1:

        st.subheader("Your Expenses")

        st.caption(
            "AI automatically categorizes transactions. "
            "You can correct any category and save the correction."
        )

        display_columns = [
            "Date",
            "Details",
            "Amount",
            "Category",
            "Match Type",
            "AI Confidence",
        ]

        edited_df = st.data_editor(
            st.session_state.debits_df[display_columns],
            column_config={
                "Date": st.column_config.DateColumn(
                    "Date",
                    format="DD/MM/YYYY"
                ),
                "Amount": st.column_config.NumberColumn(
                    "Amount",
                    format=f"{CURRENCY_SYMBOL} %.2f"
                ),
                "Category": st.column_config.SelectboxColumn(
                    "Category",
                    options=list(
                        st.session_state.categories.keys()
                    )
                ),
                "Match Type": st.column_config.TextColumn(
                    "Matched by",
                    disabled=True
                ),
                "AI Confidence": st.column_config.NumberColumn(
                    "AI Confidence",
                    format="%.1f%%",
                    disabled=True
                ),
            },
            hide_index=True,
            use_container_width=True,
            key="category_editor",
        )

        if st.button(
            "Apply Changes",
            type="primary"
        ):

            changes = 0

            for idx, row in edited_df.iterrows():

                old_cat = st.session_state.debits_df.at[
                    idx, "Category"
                ]

                new_cat = row["Category"]

                if new_cat == old_cat:
                    continue

                st.session_state.debits_df.at[
                    idx, "Category"
                ] = new_cat

                st.session_state.debits_df.at[
                    idx, "Match Type"
                ] = "Manual correction"

                st.session_state.debits_df.at[
                    idx, "AI Confidence"
                ] = 1.0

                # Store the full transaction as a keyword so that
                # future identical/similar descriptions can be caught
                # by the transparent rule layer.
                add_keyword_to_category(
                    new_cat,
                    row["Details"]
                )
                add_training_example(row["Details"], new_cat)

                changes += 1

            st.success(
                f"Saved {changes} correction(s). "
                "The rule layer and training set have learned those descriptions."
            )

            st.rerun()

    # ------------------------------------------------------------------
    # TAB 2 - Income
    # ------------------------------------------------------------------
    with tab2:

        st.subheader("Your Income")

        st.dataframe(
            st.session_state.credits_df[
                ["Date", "Details", "Amount"]
            ],
            use_container_width=True,
            hide_index=True,
        )

        st.metric(
            "Total Income",
            f"{CURRENCY_SYMBOL}{st.session_state.credits_df['Amount'].sum():,.2f}"
        )

    # ------------------------------------------------------------------
    # TAB 3 - Charts
    # ------------------------------------------------------------------
    with tab3:

        debits_df = st.session_state.debits_df
        credits_df = st.session_state.credits_df

        total_income = credits_df["Amount"].sum()
        total_expense = debits_df["Amount"].sum()

        c1, c2, c3 = st.columns(3)

        c1.metric(
            "Total Income",
            f"{CURRENCY_SYMBOL}{total_income:,.2f}"
        )

        c2.metric(
            "Total Expenses",
            f"{CURRENCY_SYMBOL}{total_expense:,.2f}"
        )

        c3.metric(
            "Net",
            f"{CURRENCY_SYMBOL}{total_income - total_expense:,.2f}"
        )

        category_totals = (
            debits_df
            .groupby("Category")["Amount"]
            .sum()
            .reset_index()
            .sort_values(
                "Amount",
                ascending=False
            )
        )

        col1, col2 = st.columns([3, 2])

        with col1:
            fig_pie = px.pie(
                category_totals,
                values="Amount",
                names="Category",
                title="Expenses by Category",
                hole=0.4,
            )

            st.plotly_chart(
                fig_pie,
                use_container_width=True
            )

        with col2:
            st.dataframe(
                category_totals,
                column_config={
                    "Amount": st.column_config.NumberColumn(
                        "Amount",
                        format=f"{CURRENCY_SYMBOL} %.2f"
                    )
                },
                use_container_width=True,
                hide_index=True,
            )

        monthly = debits_df.copy()

        monthly["Month"] = (
            monthly["Date"]
            .dt.to_period("M")
            .astype(str)
        )

        monthly_totals = (
            monthly
            .groupby("Month")["Amount"]
            .sum()
            .reset_index()
        )

        fig_bar = px.bar(
            monthly_totals,
            x="Month",
            y="Amount",
            title="Monthly Spending Trend"
        )

        st.plotly_chart(
            fig_bar,
            use_container_width=True
        )

    # ------------------------------------------------------------------
    # TAB 4 - EXPENSE TRACKING
    # ------------------------------------------------------------------
    with tab4:

        st.subheader("📈 Expense Tracking")

        debits_df = st.session_state.debits_df.copy()
        debits_df["Month"] = debits_df["Date"].dt.to_period("M").astype(str)

        st.write("### Monthly Totals")

        monthly_totals = (
            debits_df.groupby("Month")["Amount"]
            .sum()
            .reset_index()
            .sort_values("Month")
        )
        monthly_totals["Change vs prev month"] = monthly_totals["Amount"].diff()

        st.dataframe(
            monthly_totals,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Amount": st.column_config.NumberColumn("Total spent", format=f"{CURRENCY_SYMBOL} %.2f"),
                "Change vs prev month": st.column_config.NumberColumn(format=f"{CURRENCY_SYMBOL} %.2f"),
            },
        )

        st.write("### Category Trends Over Time")

        category_monthly = (
            debits_df.groupby(["Month", "Category"])["Amount"]
            .sum()
            .reset_index()
        )

        fig_trend = px.line(
            category_monthly,
            x="Month",
            y="Amount",
            color="Category",
            markers=True,
            title="Spending by Category Over Time",
        )
        st.plotly_chart(fig_trend, use_container_width=True)

        if len(monthly_totals) >= 2:
            latest_month = monthly_totals.iloc[-1]["Month"]
            prev_month = monthly_totals.iloc[-2]["Month"]

            latest_by_cat = category_monthly[
                category_monthly["Month"] == latest_month
            ].set_index("Category")["Amount"]
            prev_by_cat = category_monthly[
                category_monthly["Month"] == prev_month
            ].set_index("Category")["Amount"]

            diff = (latest_by_cat - prev_by_cat).dropna().sort_values(ascending=False)

            if len(diff) > 0 and diff.iloc[0] > 0:
                st.caption(
                    f"📈 Biggest increase from {prev_month} to {latest_month}: "
                    f"**{diff.index[0]}** (+{diff.iloc[0]:,.2f})"
                )

    # ------------------------------------------------------------------
    # TAB 5 - BUDGETS
    # ------------------------------------------------------------------
    with tab5:

        st.subheader("🎯 Budget vs Actual")

        st.caption(
            "Set a monthly budget per category and see how your actual "
            "spending compares."
        )

        debits_df = st.session_state.debits_df.copy()
        debits_df["Month"] = debits_df["Date"].dt.to_period("M").astype(str)
        months_available = sorted(debits_df["Month"].unique())

        selected_period = st.selectbox(
            "Compare against",
            ["All time"] + months_available,
        )

        if selected_period == "All time":
            filtered = debits_df
            num_months = max(len(months_available), 1)
        else:
            filtered = debits_df[debits_df["Month"] == selected_period]
            num_months = 1

        st.write("### Set monthly budgets")

        categories_list = [
            c for c in st.session_state.categories.keys() if c != "Uncategorized"
        ]

        new_budgets = {}
        budget_cols = st.columns(3)

        for i, cat in enumerate(categories_list):
            with budget_cols[i % 3]:
                current_val = float(st.session_state.budgets.get(cat, 0.0))
                new_budgets[cat] = st.number_input(
                    cat,
                    min_value=0.0,
                    value=current_val,
                    step=50.0,
                    key=f"budget_{cat}",
                )

        if st.button("💾 Save Budgets"):
            st.session_state.budgets.update(new_budgets)
            save_budgets()
            st.success("Budgets saved.")
            st.rerun()

        st.divider()
        st.write(f"### Budget vs Actual ({selected_period})")

        actual_by_cat = filtered.groupby("Category")["Amount"].sum()

        rows = []
        for cat in categories_list:
            monthly_budget = st.session_state.budgets.get(cat, 0.0)
            budget_amt = monthly_budget * num_months
            actual_amt = float(actual_by_cat.get(cat, 0.0))
            rows.append({
                "Category": cat,
                "Budget": budget_amt,
                "Actual": actual_amt,
                "Remaining": budget_amt - actual_amt,
            })

        budget_df = pd.DataFrame(rows)

        if budget_df["Budget"].sum() == 0:
            st.info("Set budgets above to see progress here.")
        else:
            for _, row in budget_df.iterrows():
                if row["Budget"] <= 0:
                    continue

                pct = min(row["Actual"] / row["Budget"], 1.0)
                over = row["Actual"] > row["Budget"]

                label = f"{row['Category']}: {CURRENCY_SYMBOL}{row['Actual']:,.2f} / {CURRENCY_SYMBOL}{row['Budget']:,.2f}"
                if over:
                    label += " ⚠️ Over budget"

                st.write(label)
                st.progress(pct)

            st.dataframe(
                budget_df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Budget": st.column_config.NumberColumn(format=f"{CURRENCY_SYMBOL} %.2f"),
                    "Actual": st.column_config.NumberColumn(format=f"{CURRENCY_SYMBOL} %.2f"),
                    "Remaining": st.column_config.NumberColumn(format=f"{CURRENCY_SYMBOL} %.2f"),
                },
            )

    # ------------------------------------------------------------------
    # TAB 6 - SAVINGS GOAL
    # ------------------------------------------------------------------
    with tab6:

        st.subheader("💰 Savings Goal Tracker")

        goal = st.session_state.savings_goal

        # Seed the widget values ONCE from the saved goal file. After this,
        # the widgets own their own state via their keys, so typing in them
        # and clicking Update works predictably. (Re-seeding on every run
        # would fight the user's typing.)
        if "sg_seeded" not in st.session_state:
            st.session_state.sg_target_amount = float(
                goal.get("target_amount", 0.0)
            )
            st.session_state.sg_current_saved = float(
                goal.get("current_saved", 0.0)
            )
            if goal.get("target_date"):
                st.session_state.sg_target_date = pd.to_datetime(
                    goal["target_date"]
                ).date()
            st.session_state.sg_seeded = True

        col1, col2, col3 = st.columns(3)

        with col1:
            target_amount = st.number_input(
                "Target amount",
                min_value=0.0,
                step=100.0,
                key="sg_target_amount",
            )

        with col2:
            target_date = st.date_input(
                "Target date",
                key="sg_target_date",
            )

        with col3:
            current_saved = st.number_input(
                "Current amount saved",
                min_value=0.0,
                step=50.0,
                key="sg_current_saved",
            )

        if st.button("💾 Update Goal"):
            st.session_state.savings_goal = {
                "target_amount": target_amount,
                "target_date": str(target_date) if target_date else None,
                "current_saved": current_saved,
            }
            save_savings_goal()
            # No st.rerun() here: the widgets already hold these values via
            # their keys, and rerunning would instantly wipe the success
            # message before the user could see it.
            st.success("Savings goal updated.")

        st.divider()

        if target_amount > 0:
            progress = min(current_saved / target_amount, 1.0)
            st.write(
                f"**Progress: {CURRENCY_SYMBOL}{current_saved:,.2f} / {CURRENCY_SYMBOL}{target_amount:,.2f}** "
                f"({progress * 100:.1f}%)"
            )
            st.progress(progress)

            if target_date:
                days_remaining = (
                    pd.Timestamp(target_date) - pd.Timestamp.today().normalize()
                ).days

                if days_remaining > 0:
                    remaining_amount = max(target_amount - current_saved, 0)
                    months_remaining = max(days_remaining / 30.44, 1 / 30.44)
                    monthly_needed = remaining_amount / months_remaining
                    st.info(
                        f"📅 {days_remaining} days remaining. Save about "
                        f"**{CURRENCY_SYMBOL}{monthly_needed:,.2f}/month** to hit your goal."
                    )
                elif days_remaining == 0:
                    st.warning("Target date is today!")
                elif current_saved >= target_amount:
                    st.success("🎉 You reached your goal!")
                else:
                    st.error(
                        "Target date has passed and the goal isn't fully "
                        "funded yet."
                    )

            net_this_load = (
                st.session_state.credits_df["Amount"].sum()
                - st.session_state.debits_df["Amount"].sum()
            )
            st.caption(
                f"Net income minus expenses in the currently loaded file: "
                f"{CURRENCY_SYMBOL}{net_this_load:,.2f}. Add this to 'Current amount saved' "
                "above if it represents new savings."
            )
        else:
            st.info("Set a target amount to start tracking your goal.")

    # ------------------------------------------------------------------
    # TAB 7 - AI REVIEW
    # ------------------------------------------------------------------
    with tab7:

        st.subheader("🤖 AI Categorization Review")

        expenses = st.session_state.debits_df

        ai_count = expenses[
            expenses["Match Type"].str.startswith(
                "AI:",
                na=False
            )
        ].shape[0]

        keyword_count = expenses[
            expenses["Match Type"].str.startswith(
                "Keyword:",
                na=False
            )
        ].shape[0]

        manual_count = expenses[
            expenses["Match Type"] == "Manual correction"
        ].shape[0]

        review_count = expenses[
            expenses["Match Type"] == "Needs review"
        ].shape[0]

        c1, c2, c3, c4 = st.columns(4)

        c1.metric("AI categorized", ai_count)
        c2.metric("Keyword categorized", keyword_count)
        c3.metric("Manual corrections", manual_count)
        c4.metric("Needs review", review_count)

        st.divider()

        st.write("### Train the categorizer")
        st.caption(
            "Keyword rules are a starting point. Train a model on labeled "
            "transactions so similar uncategorized expenses can be classified "
            "automatically. Correct a few rows first (here or on the Expenses "
            "tab), then train."
        )

        include_bundled = st.checkbox(
            "Include bundled training dataset (Food, Travel, Shopping, EMI, Investment)",
            value=os.path.exists(BUNDLED_TRAIN_FILE),
        )
        extra_labeled = st.file_uploader(
            "Optional extra labeled CSV (text + category columns)",
            type=["csv"],
            key="extra_train_csv",
        )

        preview_texts, preview_labels = collect_training_data(
            include_bundled=include_bundled,
            extra_file=extra_labeled,
        )
        if preview_labels:
            st.caption(
                f"{len(preview_texts)} training rows across "
                f"{len(set(preview_labels))} categories "
                f"(includes repeats used to balance rare categories)."
            )
            st.dataframe(
                pd.Series(preview_labels).value_counts().rename("examples"),
                use_container_width=True,
            )
        else:
            st.info(
                "No labeled examples yet. Categorize a few expenses manually, "
                "or enable the bundled dataset."
            )

        train_col, apply_col = st.columns(2)

        with train_col:
            train_clicked = st.button("🧠 Train / retrain model", type="primary")
        with apply_col:
            apply_clicked = st.button("✨ Categorize remaining uncategorized")

        if train_clicked:
            texts, labels = collect_training_data(
                include_bundled=include_bundled,
                extra_file=extra_labeled,
            )
            try:
                with st.spinner("Training categorizer..."):
                    metrics = train_category_model(texts, labels)
                msg = (
                    f"Model trained on {metrics['examples']} examples and "
                    f"{len(metrics['categories'])} categories."
                )
                if metrics["test_accuracy"] is not None:
                    msg += f" Held-out accuracy: {metrics['test_accuracy']:.1%}."
                st.success(msg)
                st.session_state.debits_df, updated = recategorize_uncategorized(
                    st.session_state.debits_df
                )
                st.info(
                    f"Applied the new model to {updated} previously "
                    "uncategorized transaction(s)."
                )
                st.rerun()
            except Exception as e:
                st.error(f"Training failed: {e}")

        if apply_clicked:
            st.session_state.debits_df, updated = recategorize_uncategorized(
                st.session_state.debits_df
            )
            if updated:
                st.success(
                    f"Categorized {updated} previously uncategorized "
                    "transaction(s)."
                )
                st.rerun()
            else:
                st.warning(
                    "No additional uncategorized rows could be labeled "
                    "with the current model confidence threshold. "
                    "Label a few more examples and retrain."
                )

        st.divider()

        st.write("### Needs review")
        st.caption(
            "These are below the AI confidence threshold. Assign a category "
            "and save — those labels become training data for the next retrain."
        )

        review_mask = expenses["Match Type"] == "Needs review"
        needs_review = expenses.loc[
            review_mask,
            ["Date", "Details", "Amount", "Category", "AI Confidence"],
        ]
        review_indices = list(needs_review.index)

        if len(needs_review) > 0:
            edited_review = st.data_editor(
                needs_review.reset_index(drop=True),
                column_config={
                    "Date": st.column_config.DateColumn("Date", format="DD/MM/YYYY"),
                    "Amount": st.column_config.NumberColumn(
                        "Amount",
                        format=f"{CURRENCY_SYMBOL} %.2f",
                    ),
                    "Category": st.column_config.SelectboxColumn(
                        "Category",
                        options=list(st.session_state.categories.keys()),
                    ),
                    "AI Confidence": st.column_config.NumberColumn(
                        "AI Confidence",
                        format="%.1f%%",
                        disabled=True,
                    ),
                    "Details": st.column_config.TextColumn(disabled=True),
                },
                hide_index=True,
                use_container_width=True,
                key="review_editor",
            )

            if st.button("💾 Save review labels for training"):
                labeled = 0
                for i, row in edited_review.iterrows():
                    idx = review_indices[i]
                    new_cat = row["Category"]
                    if new_cat == "Uncategorized":
                        continue
                    st.session_state.debits_df.at[idx, "Category"] = new_cat
                    st.session_state.debits_df.at[idx, "Match Type"] = "Manual correction"
                    st.session_state.debits_df.at[idx, "AI Confidence"] = 1.0
                    add_keyword_to_category(new_cat, row["Details"])
                    add_training_example(row["Details"], new_cat)
                    labeled += 1

                if labeled:
                    st.success(
                        f"Saved {labeled} labeled example(s). "
                        "Click Train / retrain model to teach the AI."
                    )
                    st.rerun()
                else:
                    st.warning("Choose a category other than Uncategorized first.")
        else:
            st.success("No low-confidence transactions need review.")


main()
