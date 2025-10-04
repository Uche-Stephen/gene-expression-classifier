# Tiny Gene Expression Classifier

A beginner-friendly bioinformatics project that treats a table of **gene expression** values (rows = samples, columns = genes) and builds a tiny classifier to distinguish two classes (e.g., **Normal** vs **Tumor**). Includes **PCA visualization**, **confusion matrix**, **ROC curve**, and a saved **metrics.txt** for easy review.

---

##  Why this matters
Gene expression analysis is core to bioinformatics. Simple pipelines like PCA + classification are used to explore cohort structure, check batch effects, and build baseline predictors before moving to advanced models.

---

##  Features
- Loads a (small) expression table from `data/`
- Splits into **train/test** with stratification
- **Standardizes** features, trains a classifier (Random Forest by default)
- Saves:
  - **Confusion matrix** → `results/confusion_matrix.png`
  - **PCA scatter plot** → `results/pca.png`
  - **ROC curve** → `results/roc_curve.png`
  - **Metrics file** (accuracy, classification report, AUC, CV results) → `results/metrics.txt`

*(You can swap to **Logistic Regression** or other models in one line — see below.)*



##  How to Run

Create and activate a virtual environment (Windows PowerShell):
```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```
Run the analysis:
```bash
python src\train.py
```
## What you’ll get in results/:

pca.png – PCA of the test set, colored by label
confusion_matrix.png – predicted vs true labels
roc_curve.png – ROC curve with AUC
metrics.txt – accuracy, classification report, AUC, and cross-validation results

## Try another model (1-line change)
Open src/train.py and replace the classifier:

# current (Random Forest):
('clf', RandomForestClassifier(n_estimators=300, random_state=42))

# option (Logistic Regression):
('clf', LogisticRegression(max_iter=1000, random_state=42))
Re-run:
```bash
python src\train.py
```
Compare accuracies, AUC, and CV results in results/metrics.txt.

## What I learned
Handling tabular omics data (samples × genes)
Preprocessing with StandardScaler
Basic PCA for structure visualization
Training & evaluating classifiers with confusion matrix, AUC, and CV
Importance of cross-validation for reliable performance estimates

## Next Steps / Future Improvements
Swap the synthetic data for a small real dataset (GEO, TCGA subset, or Kaggle)
Add hyperparameter tuning (GridSearchCV/RandomizedSearchCV)
Add feature importance (e.g., top predictive genes)
Export a clean results/metrics.csv with multiple model comparisons
Package the pipeline as a small CLI script