import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc
from sklearn.ensemble import RandomForestClassifier

# === Helper Functions ===

def plot_confusion(y_true, y_pred, labels):
    """
    Generate and save a confusion matrix plot.
    
    Args:
        y_true: true class labels
        y_pred: predicted class labels
        labels: list of class names (for axis ticks)
    """
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig, ax = plt.subplots()
    im = ax.imshow(cm)  # show matrix as image (color-coded)

    # set axis ticks and labels
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels)

    # write numbers inside each cell
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, cm[i, j], ha='center', va='center')

    ax.set_title('Confusion Matrix')
    plt.tight_layout()

    # ensure results/ folder exists and save plot
    os.makedirs('results', exist_ok=True)
    plt.savefig('results/confusion_matrix.png', dpi=200)


def plot_pca(X, y):
    """
    Perform PCA (2 components) and plot samples colored by their label.
    
    Args:
        X: feature matrix (samples × genes)
        y: class labels
    """
    pca = PCA(n_components=2)          # reduce to 2D for visualization
    X2 = pca.fit_transform(X)

    plt.figure()
    for lab in sorted(set(y)):
        idx = [i for i, yy in enumerate(y) if yy == lab]  # indices of this class
        plt.scatter(X2[idx, 0], X2[idx, 1], label=lab)    # scatter plot for each class

    plt.legend()
    plt.title('PCA of Gene Expression')

    os.makedirs('results', exist_ok=True)
    plt.savefig('results/pca.png', dpi=200)


# === Main script ===
if __name__ == '__main__':
    # Load dataset (synthetic CSV: Label + gene expression columns)
    df = pd.read_csv('data/expression_synthetic.csv')
    y = df['Label'].values               # target labels (Normal/Tumor)
    X = df.drop(columns=['Label']).values  # features (gene expression values)

    # Split into training/testing sets (25% test, stratified to balance classes)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    # Build pipeline: scale features → classify using Random Forest
    pipe = Pipeline([
        ('scaler', StandardScaler()), 
        ('clf', RandomForestClassifier(n_estimators=300, random_state=42))
    ])

    # Train model
    pipe.fit(X_train, y_train)

    # Predict on test set
    y_pred = pipe.predict(X_test)

    # Print accuracy + detailed classification report
    print('Accuracy:', accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    # === ROC Curve + AUC ===
    # Compute probabilities for ROC curve (binary classification: Tumor vs Normal)
    try:
        probs = pipe.predict_proba(X_test)[:, 1]  # probability of 'Tumor'
        fpr, tpr, _ = roc_curve(y_test, probs, pos_label='Tumor')
        roc_auc = auc(fpr, tpr)

        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC = {roc_auc:.2f}')
        plt.plot([0,1], [0,1], color='navy', lw=2, linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend(loc="lower right")

        os.makedirs('results', exist_ok=True)
        plt.savefig('results/roc_curve.png', dpi=200)
        plt.close()
        print(f"AUC = {roc_auc:.2f}")
    except Exception as e:
        print("ROC/AUC not available for this classifier:", e)

    # === Save metrics to a text file (easy to check later / share with supervisors) ===
    os.makedirs('results', exist_ok=True)
    with open('results/metrics.txt', 'w') as f:
        f.write(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}\n\n")
        f.write(classification_report(y_test, y_pred))
        try:
            f.write(f"\nAUC: {roc_auc:.4f}\n")
        except NameError:
            f.write("\nAUC: N/A (no predict_proba)\n")
    print("Saved results -> results/metrics.txt")

    # Save confusion matrix plot + PCA visualization
    plot_confusion(y_test, y_pred, labels=sorted(set(y)))
    plot_pca(X_test, y_test)

    # === Stratified Cross-Validation (5-fold) ===
    # Evaluate pipeline more robustly across 5 different splits
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(pipe, X, y, cv=cv, scoring='accuracy')
    print(f"Cross-validation Accuracy: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

    # Append CV results to metrics file
    with open('results/metrics.txt', 'a') as f:
        f.write(f"Cross-validation Accuracy: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}\n")
