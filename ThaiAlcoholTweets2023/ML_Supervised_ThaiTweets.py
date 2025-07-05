import re, datetime
import dill
from pathlib import Path
import pandas as pd
from vrac.remove_emoji import remove_emoji
from pythainlp.tokenize import word_tokenize as custom_tokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import FunctionTransformer, Pipeline
from sklearn.svm import SVC
from sklearn.naive_bayes import ComplementNB
from sklearn.neighbors import KNeighborsClassifier
from HeatmapThaiTweets import day_hour_heatmap
from ChiSquare_Heatmaps import chi_square_by_hour

CATEG_TRAINING = Path(r"Training dataset for categoried.csv")
FULL_DATA = Path(r"Full Dataset.csv")
SENTIMENT_TRAINING = Path("Training dataset for sentiment.csv")
MODEL_DIR = Path("models"); MODEL_DIR.mkdir(exist_ok=True)
PREDICT_SUFFIX = "classified"

# Tokenize Thai Tweets
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
        row = re.sub(r'[,.:\-;/…]', '', str(row))
        row = " ".join([word for word in row.split() if len(word) > 1])
        row = custom_tokenizer(row)
        processed.append(' '.join(row))
    return processed

# GridSearch
def grid_tune(name, pipe, param_grid, X, y, scoring="f1_weighted"):
    cv = StratifiedKFold(5, shuffle=True, random_state=42)
    gs = GridSearchCV(pipe, param_grid, cv=cv, n_jobs=-1, verbose=1, scoring=scoring, refit=True)
    gs.fit(X, y)
    print(f"[{name}] Best F1 = {gs.best_score_:.3f}")
    print("Best Params:", gs.best_params_)
    return gs

# TF_IDF Vectorizer
TFIDF = TfidfVectorizer(tokenizer=lambda x: x.split(), preprocessor=lambda x: x, token_pattern=None)
VECT_PIPE = Pipeline([
    ("clean", FunctionTransformer(custom_thai_tokeniser, validate=False)),
    ("tfidf", TFIDF)])

TO_DENSE = FunctionTransformer(lambda x: x.toarray(), accept_sparse=True)

# Model parameters
nb_pipe = Pipeline([
    ("vect", VECT_PIPE),
    ("clf", ComplementNB()),
])
nb_grid = {
    "vect__tfidf__ngram_range": [(1, 1), (1, 2)],
    "vect__tfidf__min_df":      [1, 3, 5],
    "clf__alpha":               [0.1, 0.5, 1.0],
    "clf__norm":                [True, False],
}

svm_pipe = Pipeline([
    ("vect", VECT_PIPE),
    ("clf", SVC()),
])
svm_grid = {
        "vect__tfidf__ngram_range": [(1, 1), (1, 2)],
        "vect__tfidf__min_df":      [1, 3, 5],
        "clf__C":                   [0.1, 1, 4, 10],
        "clf__kernel":              ['linear', 'rbf'],
        "clf__gamma":               ['scale', 'auto', 0.1]
}

knn_pipe = Pipeline([
    ("vect", VECT_PIPE),
    ("clf",  KNeighborsClassifier()),
])
knn_grid = {
    "vect__tfidf__ngram_range": [(1, 1), (1, 2)],
    "vect__tfidf__min_df":      [1, 3, 5],
    "clf__n_neighbors":         [3, 6, 10],
    "clf__weights":             ["uniform", "distance"],
    "clf__metric":              ["euclidean", "cosine"]
}

ML_MODELS = {
     "NB":  (nb_pipe,  nb_grid),
     "SVM": (svm_pipe, svm_grid),
     "KNN": (knn_pipe, knn_grid)
}

# GridSearch for Tweets Category
results, cv_logs = {}, {}
for name, (pipe, grid) in ML_MODELS.items():
    label_df = pd.read_csv(CATEG_TRAINING)
    X = label_df["text"].astype(str).tolist()
    y = label_df["label"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    gs = grid_tune(name, pipe, grid, X_train, y_train)
    results[name], cv_logs[name] = gs, gs.cv_results_
    y_pred = gs.predict(X_test)
    print(f"=== {name} classification report ===")
    print(classification_report(y_test, y_pred, digits=3))
    result_csv = pd.DataFrame(gs.cv_results_)
    result_csv.to_csv(rf"D:\CAS tweets\JPSS revision\Final\cross_validation_score_categories_{name}.csv", index=False)

best_name, best_gs = max(results.items(), key=lambda kv: kv[1].best_score_)
STAMP = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
model_path = MODEL_DIR / f"thai_{best_name.lower()}_{STAMP}.txt"
dill.dumps(best_gs.best_estimator_)
print(f"Best model: {best_name} (saved → {model_path})")

# Category and Sentiment classifications
if FULL_DATA.exists():
    unseen_df = pd.read_csv(FULL_DATA)
    X_type = unseen_df["text"].astype(str).tolist()
    y_dummy = ["label"] * len(X_type)

    # Train tweet categories classifier
    label_df = pd.read_csv(CATEG_TRAINING)
    X = label_df["text"].astype(str).tolist()
    y = label_df["label"]
    preds = best_gs.predict(X_type)
    unseen_df["predicted"] = preds
    classified_path = FULL_DATA.with_stem(FULL_DATA.stem + f"_{PREDICT_SUFFIX}")
    unseen_df.to_csv(classified_path, index=False, encoding='utf-8-sig')
    print(f"Saved classified output → {classified_path}")

    # Create a Personal Communication tweets only DataFrame
    df = unseen_df[unseen_df["predicted"] == "PC"].copy()

    # Remove wrong datetime format rows
    df['created_at'] = pd.to_datetime(df['created_at'], errors='coerce', utc=True)
    df = df.dropna(subset=['created_at'])

    # Load labeled sentiment data
    sent_df = pd.read_csv(SENTIMENT_TRAINING)
    sent_X = sent_df["text"].astype(str).tolist()
    sent_y = sent_df["label"]

    # GridSearch & Train sentiment classifier
    sentiment_results, cv_logs = {}, {}
    for name, (pipe, grid) in ML_MODELS.items():
        X_train, X_test, y_train, y_test = train_test_split(sent_X, sent_y, test_size=0.2, stratify=sent_y,
                                                            random_state=42)
        gs_sent = grid_tune(name, pipe, grid, X_train, y_train)
        sentiment_results[name], cv_logs[name] = gs_sent, gs_sent.cv_results_
        y_pred = gs_sent.predict(X_test)
        print(f"=== {name} classification report ===")
        print(classification_report(y_test, y_pred, digits=3))
        result_csv = pd.DataFrame(gs_sent.cv_results_)
        result_csv.to_csv(rf"D:\CAS tweets\JPSS revision\Final\cross_validation_score_sentiments_{name}.csv",
                          index=False)
    best_sentiment_name, best_sentiment_model = max(sentiment_results.items(), key=lambda kv: kv[1].best_score_)
    STAMP = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = MODEL_DIR / f"thai_sentiment_{best_name.lower()}_{STAMP}.txt"
    dill.dumps(best_sentiment_model.best_estimator_)
    print(f"Best sentiment model: {best_name} (saved → {model_path})")
    df['sentiment'] = best_sentiment_model.predict(df['text'].astype(str))

    # Create DataFrame for PC only and for tweets labelled as Negative and Positive
    pc_all = df.copy()
    pc_all.to_csv(r'D:\CAS tweets\JPSS revision\Final\AllPC.csv', encoding='utf-8-sig')
    pc_pos = df[df['sentiment'] == 'P']
    pc_pos.to_csv(r'D:\CAS tweets\JPSS revision\Final\Positive_sentiments.csv', encoding='utf-8-sig')
    pc_neg = df[df['sentiment'] == 'NG']
    pc_neg.to_csv(r'D:\CAS tweets\JPSS revision\Final\Negative_sentiments.csv', encoding='utf-8-sig')

    # Temporal Heatmaps
    day_hour_heatmap(pc_all, "All PC Tweets")
    day_hour_heatmap(pc_pos, "Positive PC Tweets")
    day_hour_heatmap(pc_neg, "Negative PC Tweets")

    # Chi-square and Cramer's V calculation
    chi_square_by_hour(pc_all, pc_pos)
    chi_square_by_hour(pc_all, pc_neg)
    chi_square_by_hour(pc_pos, pc_neg)
else:
    print("Unseen data or classification result not found.")
