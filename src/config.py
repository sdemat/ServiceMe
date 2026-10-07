from pathlib import Path

# DIRECTORIES
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
EMBEDDINGS_DIR = DATA_DIR / "embeddings"
SPLITS_DIR = DATA_DIR / "splits"

MODELS_DIR = PROJECT_ROOT / "models"
CLASSIFIERS_DIR = MODELS_DIR / "classifiers"
RESULTS_DIR = MODELS_DIR / "results"

SRC_DIR = PROJECT_ROOT / "src"
SAMPLES_DIR = PROJECT_ROOT / "samples"

EXTENSION_DIR = PROJECT_ROOT / "extension"
EXTENSION_MODEL_DIR = EXTENSION_DIR / "model"

DATA_FILE = RAW_DIR / "raw.csv"
ENV_FILE = PROJECT_ROOT / ".env"

# LABELS
ID_COLUMN = ("inc_number", "ctc_number")
TICKET_ID_COLUMN = "ticket_id"
TEXT_COLUMN = ("short_description", "ctc_short_description")
TIME_COLUMN = "opened_at"
CLOSED_COLUMN = "resolved_at"

LABEL_COLUMNS = {
    "category": "category",
    "priority": "priority",
    "record_source" : "record_source"
}

# TARGET COLUMN TO GUESS
TARGET = "category"
TARGET_COLUMN = LABEL_COLUMNS[TARGET]

MIN_EXAMPLES_PER_CLASS = 30
RARE_CLASS_POLICY = "merge_to_other"
OTHER_LABEL = "Other"
MAX_TEXT_CHARS = 2000

# SPLIT SETTINGS
SPLIT_METHOD = "fraction"
TEST_FRACTION = 0.20
VAL_FRACTION = 0.10
SPLIT_DATE = None

MODEL_REGISTRY = {
    "tfidf": {
        "type": "sparse",
        "source": None,
        "folder": "baseline_tfidf",
        "dim": None,
    },
    "minilm": {
        "type": "embedding",
        "source": "sentence-transformers/all-MiniLM-L6-v2",
        "folder": "minilm",
        "dim": 384,
    },
    "bge_small": {
        "type": "embedding",
        "source": "BAAI/bge-small-en-v1.5",
        "folder": "bge_small",
        "dim": 384,
    }
}
DEFAULT_MODEL = "tfidf"

# FEATURE SETTINGS
TFIDF_WORD_NGRAMS = (1, 2)
TFIDF_CHAR_NGRAMS = (3, 5)
TFIDF_MIN_DF = 2
TFIDF_MAX_FEATURES = 50_000
EMBED_BATCH_SIZE = 32

# TRAINING DEFAULTS
RANDOM_SEED = 42
CLASSIFIER = "logistic_regression"
CLASS_WEIGHT = "balanced"
REGULARIZATION_C = 1.0
REGULARIZATION_C_GRID = (0.1, 1.0, 10.0)
MAX_ITER = 1000

# EVALUATION SETTINGS
TOP_K_VALUES = (1, 3)
SIMILAR_TICKETS_K = 5

# VALIDATION AND CLEANING
EID_PATTERN = r"\b[A-Za-z]{2,3}\d{3,5}\b"
EMAIL_PATTERN = r"[\w.+=]+@[\w-]+\.[\w.-]+"
PHONE_PATTERN = r"\b(?:\+?1[-./s]?)?\(?\d{3}\)"

# CHROME EXTENSION CONSTANTS
EXTENSION_SHARED = (
    "MAX_TEXT_CHARS",
    "TEXT_COLUMNS",
    "TOP_K_VALUES",
    "SIMILAR_TICKETS_K",
    "TFIDF_WORD_NGRAMS",
    "TFIDF_CHAR_NGRAMS",
)

# Create the folders scripts write to.
def ensure_dirs():
    for d in (SPLITS_DIR,
              EMBEDDINGS_DIR,
              CLASSIFIERS_DIR,
              RESULTS_DIR,
              EXTENSION_MODEL_DIR):
        d.mkdir(parents=True, exist_ok=True)

# Ensure selected model exists
# Pre: model exists in registry
def get_model_spec(name):
    if name not in MODEL_REGISTRY:
        valid = ", ".join(MODEL_REGISTRY)
        raise KeyError("Unknown model: {name}. Valid options: {valid}")
    return MODEL_REGISTRY[name]

# Folder holding the model
def model_dir(name):
    return MODELS_DIR / get_model_spec(name)["folder"]

# Path of the saved classifier.
def classifier_path(model_name, target=None):
    target = target or TARGET
    get_model_spec(model_name)
    return CLASSIFIERS_DIR / f"{model_name}_{target}.joblib"

# Path of cached embeddings for a model and split ('train', 'val', 'test')
def embeddings_path(model_name, split):
    get_model_spec(model_name)
    return EMBEDDINGS_DIR / f"{model_name}_{split}.npy"

def results_path(model_name, target=None, suffix="json"):
    target = target or TARGET
    get_model_spec(model_name)
    return RESULTS_DIR / f"{model_name}_{target}.{suffix}"

def extension_constants():
    g = globals()
    return {name: g[name] for name in EXTENSION_SHARED}

def check_config():
    if TARGET not in LABEL_COLUMNS:
        raise ValueError("Target must be in LABEL_COLUMNS.")
    if not (0 < TEST_FRACTION < 1.0) or not (0 <= VAL_FRACTION < 1.0):
        raise ValueError("Fraction must be between 0 and 1.")
    if TEST_FRACTION + VAL_FRACTION >= 1.0:
        raise ValueError("Fractions sum must be between 0 and 1.")
    if SPLIT_METHOD not in ("fraction", "date"):
        raise ValueError("Split method must be one of 'fraction' or 'date'.")
    if SPLIT_METHOD == "date" and not SPLIT_DATE:
        raise ValueError("SPLIT_DATE must be set when using SPLIT_METHOD = 'date'.")
    if RARE_CLASS_POLICY not in ("merge_to_other", "drop"):
        raise ValueError("RARE_CLASS_POLICY must be one of 'merge_to_other' or 'drop'.")
    if DEFAULT_MODEL not in MODEL_REGISTRY:
        raise ValueError("Default model must be in MODEL_REGISTRY.")
    for name, spec in MODEL_REGISTRY.items():
        for key in ("type", "source", "folder", "dim"):
            if key not in spec:
                raise ValueError(f"MODEL_REGISTRY['{name}'] is missing '{key}'")

def load_secrets():
    from dotenv import dotenv_values
    return dict(dotenv_values(ENV_FILE))