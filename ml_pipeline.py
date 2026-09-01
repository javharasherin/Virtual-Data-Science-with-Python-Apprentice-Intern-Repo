import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, RocCurveDisplay, PrecisionRecallDisplay, classification_report,
)

sns.set_theme(style="whitegrid", font_scale=1.05)
FIG = "/home/claude/figs4/"
os.makedirs(FIG, exist_ok=True)
RANDOM_STATE = 42

df = pd.read_csv("/home/claude/USvideos_cleaned.csv", parse_dates=["trend_date"])


df["title_length"] = df["title"].str.len()
df["title_word_count"] = df["title"].str.split().str.len()
df["title_has_exclaim"] = df["title"].str.contains("!").astype(int)
df["title_has_question"] = df["title"].str.contains(r"\?").astype(int)
df["title_upper_ratio"] = df["title"].apply(
    lambda s: sum(1 for c in s if c.isupper()) / max(len(s.replace(" ", "")), 1)
)
df["tag_count"] = df["tags"].fillna("").apply(lambda s: 0 if s == "" else len(s.split("|")))
df["day_of_week"] = df["trend_date"].dt.day_name()

channel_video_counts = df.groupby("channel_title")["video_id"].nunique()
df["channel_trend_count"] = df["channel_title"].map(channel_video_counts)


threshold = df["views"].quantile(0.75)
df["high_performer"] = (df["views"] >= threshold).astype(int)

print("Target distribution:\n", df["high_performer"].value_counts(normalize=True))
print("View threshold (75th percentile):", threshold)

NUMERIC_FEATURES = [
    "title_length", "title_word_count", "title_has_exclaim", "title_has_question",
    "title_upper_ratio", "tag_count", "channel_trend_count",
]
CATEGORICAL_FEATURES = ["category", "day_of_week"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET = "high_performer"

model_df = df[FEATURES + [TARGET]].dropna().copy()
X = model_df[FEATURES]
y = model_df[TARGET]


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)
print(f"\nTrain size: {len(X_train)}, Test size: {len(X_test)}")
print("Train positive rate:", y_train.mean(), " Test positive rate:", y_test.mean())


preprocess = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), NUMERIC_FEATURES),
        ("cat", OneHotEncoder(handle_unknown="ignore", drop="first"), CATEGORICAL_FEATURES),
    ]
)


log_reg = Pipeline([
    ("prep", preprocess),
    ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE)),
])
tree = Pipeline([
    ("prep", preprocess),
    ("clf", DecisionTreeClassifier(max_depth=5, min_samples_leaf=20,
                                    class_weight="balanced", random_state=RANDOM_STATE)),
])
baseline = Pipeline([
    ("prep", preprocess),
    ("clf", DummyClassifier(strategy="stratified", random_state=RANDOM_STATE)),
])

models = {"Logistic Regression": log_reg, "Decision Tree": tree, "Baseline (stratified random)": baseline}
results = {}

for name, pipe in models.items():
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    y_proba = pipe.predict_proba(X_test)[:, 1]
    train_acc = accuracy_score(y_train, pipe.predict(X_train))
    test_acc = accuracy_score(y_test, y_pred)
    results[name] = dict(
        train_accuracy=train_acc,
        test_accuracy=test_acc,
        precision=precision_score(y_test, y_pred),
        recall=recall_score(y_test, y_pred),
        f1=f1_score(y_test, y_pred),
        roc_auc=roc_auc_score(y_test, y_proba),
    )
    print(f"\n{name}")
    for k, v in results[name].items():
        print(f"  {k}: {v:.4f}")


cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
for name, pipe in [("Logistic Regression", log_reg), ("Decision Tree", tree)]:
    scores = cross_val_score(pipe, X, y, cv=cv, scoring="roc_auc")
    results[name]["cv_auc_mean"] = scores.mean()
    results[name]["cv_auc_std"] = scores.std()
    print(f"{name} 5-fold CV ROC-AUC: {scores.mean():.4f} \u00b1 {scores.std():.4f}")




fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.5))
for ax, (name, pipe) in zip(axes, [("Logistic Regression", log_reg), ("Decision Tree", tree)]):
    cm = confusion_matrix(y_test, pipe.predict(X_test))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["Not top 25%", "Top 25%"], yticklabels=["Not top 25%", "Top 25%"])
    ax.set_title(name)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
plt.tight_layout()
plt.savefig(FIG + "fig1_confusion_matrices.png", dpi=150)
plt.close()

fig, ax = plt.subplots(figsize=(7.5, 6.5))
for name, pipe, color in [("Logistic Regression", log_reg, "#4C72B0"),
                           ("Decision Tree", tree, "#DD8452"),
                           ("Baseline (random)", baseline, "#999999")]:
    RocCurveDisplay.from_estimator(pipe, X_test, y_test, ax=ax, name=name, color=color)
ax.plot([0, 1], [0, 1], linestyle="--", color="black", alpha=0.4, label="Chance")
ax.set_title("ROC Curves \u2014 Top-Quartile View Prediction", fontsize=14)
plt.tight_layout()
plt.savefig(FIG + "fig2_roc_curves.png", dpi=150)
plt.close()

fig, ax = plt.subplots(figsize=(7.5, 6.5))
for name, pipe, color in [("Logistic Regression", log_reg, "#4C72B0"),
                           ("Decision Tree", tree, "#DD8452")]:
    PrecisionRecallDisplay.from_estimator(pipe, X_test, y_test, ax=ax, name=name, color=color)
ax.set_title("Precision-Recall Curves (25% positive class)", fontsize=14)
plt.tight_layout()
plt.savefig(FIG + "fig3_precision_recall.png", dpi=150)
plt.close()


ohe_names = log_reg.named_steps["prep"].named_transformers_["cat"].get_feature_names_out(CATEGORICAL_FEATURES)
all_feature_names = NUMERIC_FEATURES + list(ohe_names)
coefs = log_reg.named_steps["clf"].coef_[0]
coef_df = pd.DataFrame({"feature": all_feature_names, "coefficient": coefs}).sort_values("coefficient")
top_coefs = pd.concat([coef_df.head(8), coef_df.tail(8)])

fig, ax = plt.subplots(figsize=(9, 8))
colors = ["#C44E52" if c < 0 else "#4C72B0" for c in top_coefs["coefficient"]]
ax.barh(top_coefs["feature"], top_coefs["coefficient"], color=colors)
ax.axvline(0, color="black", linewidth=0.8)
ax.set_title("Logistic Regression Coefficients\n(standardized features; blue = pushes toward top-25%)", fontsize=13)
ax.set_xlabel("Coefficient (log-odds)")
plt.tight_layout()
plt.savefig(FIG + "fig4_logreg_coefficients.png", dpi=150)
plt.close()


tree_importances = pd.Series(
    tree.named_steps["clf"].feature_importances_, index=all_feature_names
).sort_values(ascending=False).head(12)
fig, ax = plt.subplots(figsize=(9, 6.5))
tree_importances.iloc[::-1].plot(kind="barh", ax=ax, color="#55A868")
ax.set_title("Decision Tree \u2014 Top Feature Importances", fontsize=14)
ax.set_xlabel("Importance (Gini)")
plt.tight_layout()
plt.savefig(FIG + "fig5_tree_importances.png", dpi=150)
plt.close()


depths = range(1, 16)
train_scores, test_scores = [], []
for d in depths:
    t = Pipeline([("prep", preprocess),
                  ("clf", DecisionTreeClassifier(max_depth=d, min_samples_leaf=20,
                                                  class_weight="balanced", random_state=RANDOM_STATE))])
    t.fit(X_train, y_train)
    train_scores.append(accuracy_score(y_train, t.predict(X_train)))
    test_scores.append(accuracy_score(y_test, t.predict(X_test)))

fig, ax = plt.subplots(figsize=(8.5, 6))
ax.plot(list(depths), train_scores, marker="o", label="Train accuracy", color="#4C72B0")
ax.plot(list(depths), test_scores, marker="o", label="Test accuracy", color="#C44E52")
ax.axvline(5, linestyle="--", color="gray", alpha=0.6, label="Chosen depth (5)")
ax.set_xlabel("Max tree depth")
ax.set_ylabel("Accuracy")
ax.set_title("Overfitting Check: Train vs. Test Accuracy by Tree Depth", fontsize=13.5)
ax.legend()
plt.tight_layout()
plt.savefig(FIG + "fig6_overfitting_curve.png", dpi=150)
plt.close()

print("\nAll figures saved to", FIG)

def clean(o):
    if isinstance(o, dict):
        return {k: clean(v) for k, v in o.items()}
    if isinstance(o, (np.floating, np.integer)):
        return float(o)
    return o

with open("/home/claude/ml_results.json", "w") as f:
    json.dump(clean(results), f, indent=2)


with open("/home/claude/logreg_classification_report.txt", "w") as f:
    f.write(classification_report(y_test, log_reg.predict(X_test), target_names=["Not top 25%", "Top 25%"]))

print(json.dumps(clean(results), indent=2))
print("\nThreshold (75th pct views):", threshold)
print("n rows used:", len(model_df))
