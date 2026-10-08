"""Project settings. Constants and small helpers only; no side effects on import."""

from pathlib import Path

# --- Paths ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
SPLITS_DIR = DATA_DIR / "splits"
EMBEDDINGS_DIR = DATA_DIR / "embeddings"

MODELS_DIR = PROJECT_ROOT / "models"
CLASSIFIERS_DIR = MODELS_DIR / "classifiers"
RESULTS_DIR = MODELS_DIR / "results"

SRC_DIR = PROJECT_ROOT / "src"
SAMPLES_DIR = PROJECT_ROOT / "samples"
EXTENSION_DIR = PROJECT_ROOT / "extension"
EXTENSION_MODEL_DIR = EXTENSION_DIR / "model"

DATA_FILE = RAW_DIR / "raw.csv"
ENV_FILE = PROJECT_ROOT / ".env"

# --- Schema ---
ID_COLUMNS = ("inc_number", "ctc_number")  # first non-blank wins (INC over CTC)
TICKET_ID_COLUMN = "ticket_id"             # built by load.py
COMBINED_TEXT_COLUMN = "text"              # built by load.py

TEXT_COLUMNS = ("short_description", "ctc_short_description")  # model input; creation-time info only
START_COLUMNS = ("opened_at", "ctc_opened_at")                 # first non-blank = ticket start
TIME_COLUMN = "start_time"                                     # built by load.py
CLOSED_COLUMN = "resolved_at"

LABEL_COLUMNS = {  # friendly name -> column
    "priority": "priority",
    "category": "category",
    "record_source": "record_source",
}

# Profiling only, never model inputs (known only after the ticket is worked).
# contact_type (always "walk in") and state (always resolved/closed) are omitted.
PROFILE_COLUMNS = ("call_type", "transferred_to")

# --- Targets ---
ENTRY_FIELDS_STRICT = ("priority",)
ENTRY_FIELDS_SOFT = ("category", "record_source")
TARGET = "category"                    # label for similar-ticket checks
TARGET_COLUMN = LABEL_COLUMNS[TARGET]
AT_THE_WINDOW_LABEL = "AT THE WINDOW"  # shown for General Inquiries

# --- Data rules ---
MIN_EXAMPLES_PER_CLASS = 20
RARE_CLASS_POLICY = "merge_to_other"  # or "drop"
OTHER_LABEL = "Other"
MAX_TEXT_CHARS = 2000
INQUIRY_MAX_HOURS = 0.05  # resolved within this = General Inquiry (3 min); CTC-only tickets are always inquiries
INQUIRY_CUTOFF = 0.6  # call a ticket an inquiry only if P(inquiry) >= this; higher = fewer incidents shown AT THE WINDOW

# --- Split (time-based: oldest trains, newest tests) ---
SPLIT_METHOD = "fraction"  # or "date"
TEST_FRACTION = 0.20
VAL_FRACTION = 0.10
SPLIT_DATE = None          # used when SPLIT_METHOD == "date"

# --- Models ---
MODEL_REGISTRY = {  # "folder" is under MODELS_DIR
    "tfidf": {"type": "sparse", "source": None, "folder": "baseline_tfidf", "dim": None},
    "minilm": {"type": "embedding", "source": "sentence-transformers/all-MiniLM-L6-v2", "folder": "minilm", "dim": 384},
    "bge_small": {"type": "embedding", "source": "BAAI/bge-small-en-v1.5", "folder": "bge_small", "dim": 384},
}
DEFAULT_MODEL = "tfidf"

# --- Features ---
TFIDF_WORD_NGRAMS = (1, 2)
TFIDF_CHAR_NGRAMS = (3, 5)
TFIDF_MIN_DF = 2
TFIDF_MAX_FEATURES = 50_000
EMBED_BATCH_SIZE = 32

# --- Training ---
RANDOM_SEED = 42
CLASSIFIER = "logistic_regression"
CLASS_WEIGHT = "balanced"
REGULARIZATION_C = 1.0
REGULARIZATION_C_GRID = (0.1, 1.0, 10.0)
MAX_ITER = 1000

# --- Evaluation ---
TOP_K_VALUES = (1, 3)
SIMILAR_TICKETS_K = 10    # fixed-count neighbors (comparison only)
SIMILARITY_CUTOFF = 0.4    # time estimate uses every past incident at least this similar (0-1)
SIMILAR_MIN = 3            # fewer matches than this = not enough to estimate
SIMILAR_MAX = 50           # at most this many matches are used
BEST_CASE_QUANTILE = 0.1   # best-case time = this share of the matches finished faster (0 = fastest one)
SUGGEST_CUTOFF = 0.5        # close matches used for the suggested description
SUGGEST_MIN = 5             # fewer close matches than this = no suggestion
SUGGEST_MAX = 30            # at most this many matches are compared
SUGGEST_WORD_SHARE = 0.25   # every word must appear in this share of the other matches...
SUGGEST_WORD_MIN_COUNT = 2  # ...and in at least this many of them (drops names and one-off details)
SIMILAR_SHOWN = 3      # similar tickets listed

# --- PII patterns (compiled in validate.py) ---
EID_PATTERN = r"\b[A-Za-z]{2,3}\d{3,5}\b"
EMAIL_PATTERN = r"[\w.+-]+@[\w-]+\.[\w.-]+"
PHONE_PATTERN = r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"
NAME_CONTEXT_PHRASES = ("Hi", "Hello", "Hey", "Thanks", "Thank you", "Regards")

# Constants the Chrome extension also needs (written out by export.py)
EXTENSION_SHARED = (
    "MAX_TEXT_CHARS",
    "TEXT_COLUMNS",
    "TOP_K_VALUES",
    "SIMILAR_TICKETS_K",
    "SIMILARITY_CUTOFF",
    "SIMILAR_MIN",
    "SIMILAR_MAX",
    "BEST_CASE_QUANTILE",
    "SUGGEST_CUTOFF",
    "SUGGEST_MIN",
    "SUGGEST_MAX",
    "SUGGEST_WORD_SHARE",
    "SUGGEST_WORD_MIN_COUNT",
    "SIMILAR_SHOWN",
    "AT_THE_WINDOW_LABEL",
    "INQUIRY_MAX_HOURS",
    "INQUIRY_CUTOFF",
    "TFIDF_WORD_NGRAMS",
    "TFIDF_CHAR_NGRAMS",
)


# --- Helpers ---
def ensure_dirs():
    """Create output folders. Never creates RAW_DIR."""
    for d in (SPLITS_DIR, EMBEDDINGS_DIR, CLASSIFIERS_DIR, RESULTS_DIR, EXTENSION_MODEL_DIR):
        d.mkdir(parents=True, exist_ok=True)


def get_model_spec(name):
    """Registry entry for a model; raises on unknown names."""
    if name not in MODEL_REGISTRY:
        raise KeyError(f"Unknown model '{name}'. Valid: {', '.join(sorted(MODEL_REGISTRY))}")
    return MODEL_REGISTRY[name]


def model_dir(name):
    """Folder of a downloaded model."""
    return MODELS_DIR / get_model_spec(name)["folder"]


def classifier_path(model_name, target=None):
    """Saved classifier path; target is in the filename."""
    get_model_spec(model_name)
    return CLASSIFIERS_DIR / f"{model_name}_{target or TARGET}.joblib"


def embeddings_path(model_name, split):
    """Cached embeddings path for a model and split."""
    get_model_spec(model_name)
    return EMBEDDINGS_DIR / f"{model_name}_{split}.npy"


def results_path(model_name, target=None, suffix="json"):
    """Evaluation output path."""
    get_model_spec(model_name)
    return RESULTS_DIR / f"{model_name}_{target or TARGET}.{suffix}"


def extension_constants():
    """Values in EXTENSION_SHARED, ready for json.dump."""
    g = globals()
    return {name: g[name] for name in EXTENSION_SHARED}


def check_config():
    """Raise on an invalid setting."""
    if TARGET not in LABEL_COLUMNS:
        raise ValueError(f"TARGET '{TARGET}' not in LABEL_COLUMNS {list(LABEL_COLUMNS)}")
    if not (0 < TEST_FRACTION < 1) or not (0 <= VAL_FRACTION < 1):
        raise ValueError("TEST_FRACTION and VAL_FRACTION must be between 0 and 1")
    if TEST_FRACTION + VAL_FRACTION >= 1:
        raise ValueError("TEST_FRACTION + VAL_FRACTION must be below 1")
    if SPLIT_METHOD not in ("fraction", "date"):
        raise ValueError("SPLIT_METHOD must be 'fraction' or 'date'")
    if SPLIT_METHOD == "date" and not SPLIT_DATE:
        raise ValueError("SPLIT_DATE required when SPLIT_METHOD is 'date'")
    if RARE_CLASS_POLICY not in ("merge_to_other", "drop"):
        raise ValueError("RARE_CLASS_POLICY must be 'merge_to_other' or 'drop'")
    for field in (*ENTRY_FIELDS_STRICT, *ENTRY_FIELDS_SOFT):
        if field not in LABEL_COLUMNS:
            raise ValueError(f"Entry field '{field}' not in LABEL_COLUMNS {list(LABEL_COLUMNS)}")
    if DEFAULT_MODEL not in MODEL_REGISTRY:
        raise ValueError(f"DEFAULT_MODEL '{DEFAULT_MODEL}' not in MODEL_REGISTRY")
    for name, spec in MODEL_REGISTRY.items():
        for key in ("type", "source", "folder", "dim"):
            if key not in spec:
                raise ValueError(f"MODEL_REGISTRY['{name}'] missing '{key}'")


def load_secrets():
    """Read .env into a dict (needs python-dotenv)."""
    from dotenv import dotenv_values
    return dict(dotenv_values(ENV_FILE))
