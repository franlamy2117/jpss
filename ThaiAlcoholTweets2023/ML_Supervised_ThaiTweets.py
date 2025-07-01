import datetime
import re
from pathlib import Path
import pandas as pd
from pythainlp.tokenize import word_tokenize as custom_tokenizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC, SVC, NuSVC
from sklearn.naive_bayes import ComplementNB
from sklearn.preprocessing import FunctionTransformer
from vrac.remove_emoji import remove_emoji


labeled_data = r"D:\CAS tweets\JPSS revision\All_coded_tweets_balanced_dataset.csv"
full_dataset = r"D:\CAS tweets\JPSS revision\PC_only.csv"
MODEL_DIR = Path("../Twitter/models"); MODEL_DIR.mkdir(exist_ok=True)
PREDICT_SUFFIX = "classified"

df = pd.read_csv(labeled_data)
X_raw = df["text"].astype(str).tolist()
y = df["label"].values

def custom_thai_tokeniser(doc_list):
    processed = []
    for row in doc_list:
        row = remove_emoji(row)
        row = re.sub(r'RT @.*: ', '', str(row))
        row = re.sub(r'\b@.*\b', '', str(row))
        row = re.sub(r'555555', '', str(row))
        row = re.sub(r'@\w+\s+\w+', '', str(row))
        row = row.lower()
        row = re.sub(r'\bline:\w+\b', '', str(row))
        row = re.sub(r'\bhttp.*\b', '', str(row))
        row = re.sub(r'\n', ' ', str(row))
        row = re.sub(r'\d{0,2}/\d{0,2}/\d{0,2}', '', str(row))
        row = re.sub(r',|\.|:|-|;|/|…', '', str(row))
        row = " ".join([word for word in row.split() if len(word) > 1])
        row = custom_tokenizer(row)
        processed.append(' '.join(row))
    return processed

TFIDF = TfidfVectorizer(
    tokenizer=lambda x: x.split(),
    preprocessor=lambda x: x,
    token_pattern=None
)

VECT_PIPE = Pipeline([
    ("clean", FunctionTransformer(custom_thai_tokeniser, validate=False)),
    ("tfidf", TFIDF)
])

TO_DENSE = FunctionTransformer(lambda x: x.toarray(), accept_sparse=True)

nb_pipe = Pipeline([
    ("vect", VECT_PIPE),
    ("clf", ComplementNB()),
])
nb_grid = {
    "vect__tfidf__ngram_range": [(1, 1), (1, 2)],
    "vect__tfidf__min_df": [1, 2, 5],
    "clf__alpha": [0.1, 0.5, 1.0],
    "clf__norm": [True, False],
}

svm_pipe = Pipeline([
    ("vect", VECT_PIPE),
    ("clf", SVC()),
])
svm_grid = {
        'vect__tfidf__ngram_range': [(1, 1), (1, 2)],
        'vect__tfidf__min_df': [1, 2, 5],
        'clf__C': [0.1, 1, 10],
        'clf__kernel': ['linear', 'rbf'],
        'clf__gamma': ['scale', 'auto', 0.1]
}

lin_svm_pipe = Pipeline([
    ("vect", VECT_PIPE),
    ("clf",  LinearSVC()),
])
lin_svm_grid = {
    "vect__tfidf__ngram_range": [(1, 1), (1, 2)],
    "vect__tfidf__min_df": [5, 8, 12],
    "clf__C": [0.5, 1, 2, 4],
    "clf__loss": ["hinge", "squared_hinge"],
}

nu_svm_pipe = Pipeline([
    ("vect", VECT_PIPE),
    ("clf",  NuSVC()),
])
nu_svm_grid = {
    "vect__tfidf__ngram_range": [(1, 1), (1, 2)],
    "vect__tfidf__min_df": [5, 8, 12],
    'clf__nu': [0.25, 0.5, 0.75],
    'clf__kernel': ['linear', 'rbf', 'poly'],
    'clf__degree': [2, 3],
    'clf__gamma': ['scale', 'auto'],
    'clf__shrinking': [True, False]
}

rf_pipe = Pipeline([
    ("vect", VECT_PIPE),
    ("to_dense", TO_DENSE),
    ("clf",  RandomForestClassifier(random_state=42)),
])
rf_grid = {
    "vect__tfidf__ngram_range": [(1, 1), (1, 2)],
    "vect__tfidf__min_df": [5, 8, 12],
    "clf__n_estimators": [20, 50, 90],
    "clf__max_depth": [60, 90, 120],
    "clf__min_samples_split": [5, 10, 20],
}

MODELS = {
     # "NB":  (nb_pipe,  nb_grid),
     # "SVM": (svm_pipe, svm_grid),
     # "LSVM": (lin_svm_pipe, lin_svm_grid),
     # "NuSVM": (nu_svm_pipe, nu_svm_grid),
     "RF":  (rf_pipe,  rf_grid),
}

def grid_tune(name, pipe, grid, X, y, scoring="f1_weighted"):
    cv = StratifiedKFold(5, shuffle=True, random_state=42)
    gs = GridSearchCV(pipe, grid, cv=cv, n_jobs=-1, verbose=1, scoring=scoring, refit=True)
    gs.fit(X, y)
    print(f"[{name}] best f1={gs.best_score_:.4f}; params={gs.best_params_}")
    return gs

X_train, X_test, y_train, y_test = train_test_split(X_raw, y, test_size=0.2, random_state=42, stratify=y)

results, cv_logs = {}, {}
for name, (pipe, grid) in MODELS.items():
    gs = grid_tune(name, pipe, grid, X_train, y_train)
    results[name], cv_logs[name] = gs, gs.cv_results_
    y_pred = gs.predict(X_test)
    print(f"\n=== {name} classification report ===")
    print(classification_report(y_test, y_pred, digits=3))

best_name, best_gs = max(results.items(), key=lambda kv: kv[1].best_score_)
STAMP = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
model_path = MODEL_DIR / f"thai_{best_name.lower()}_{STAMP}.joblib"
# joblib.dump(best_gs.best_estimator_, model_path)
print(f"\n>>> Best model: {best_name} (saved → {model_path})")

# with open(MODEL_DIR / f"cv_results_{STAMP}.json", "w", encoding="utf-8") as fp:
#     json.dump(cv_logs, fp, ensure_ascii=False, indent=2)


# if full_dataset.exists():
#     unseen_df = pd.read_csv(full_dataset)
#     unseen_txt = unseen_df["text"].astype(str).tolist()
#     predictions = best_gs.predict(unseen_txt)
#     orig_cols = unseen_df.columns.tolist()
#     unseen_df["predicted"] = predictions
#     unseen_df = unseen_df[orig_cols + ["predicted"]]
#
#     outfile = full_dataset.with_stem(full_dataset.stem + f"_{PREDICT_SUFFIX}")
#     unseen_df.to_csv(outfile, index=False)
#     print(f"Classified {len(unseen_df):,} tweets → {outfile}")
# else:
#     print("No unseen-data file found; skipping final classification step.")
