import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

DATA_FILE = 'Iris.csv'
PLOT_DIR = 'iris_plots'

os.makedirs(PLOT_DIR, exist_ok=True)


def load_data(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        print(f"Error: dataset file '{path}' not found in the current folder.")
        sys.exit(1)

    df = pd.read_csv(path)
    expected = {'Id', 'SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm', 'Species'}
    if not expected.issubset(df.columns):
        print('Error: Iris.csv is missing one or more required columns.')
        print('Expected columns:', sorted(expected))
        print('Found columns:', sorted(df.columns.tolist()))
        sys.exit(1)

    return df


def save_confusion_matrix(cm: pd.DataFrame, path: str) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm.values, interpolation='nearest', cmap='Blues')
    ax.figure.colorbar(im, ax=ax)
    ax.set_title('Confusion Matrix')
    ax.set_xlabel('Predicted label')
    ax.set_ylabel('True label')
    ax.set_xticks(range(len(cm.columns)))
    ax.set_yticks(range(len(cm.index)))
    ax.set_xticklabels(cm.columns, rotation=45, ha='right')
    ax.set_yticklabels(cm.index)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm.iat[i, j], ha='center', va='center', color='black')

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_feature_importance(model: RandomForestClassifier, feature_names: list[str], path: str) -> None:
    importance = model.feature_importances_
    indices = importance.argsort()[::-1]
    ordered_names = [feature_names[i] for i in indices]
    ordered_values = importance[indices]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(ordered_names, ordered_values, color='skyblue')
    ax.set_xlabel('Importance')
    ax.set_title('Feature Importances')
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_feature_scatter(df: pd.DataFrame, path: str) -> None:
    features = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']
    ax = scatter_matrix(df[features], figsize=(10, 10), diagonal='hist', alpha=0.6, marker='o', c=pd.factorize(df['Species'])[0], cmap='viridis')
    for axis in ax.flatten():
        if axis is not None:
            axis.tick_params(axis='x', labelrotation=45)
            axis.tick_params(axis='y', labelrotation=0)
    fig = ax[0, 0].figure
    fig.suptitle('Pairwise Iris Feature Scatter Matrix', y=0.92)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    df = load_data(DATA_FILE)
    X = df[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]
    y = df['Species']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy: {accuracy:.4f}')
    print('\nClassification report:')
    print(classification_report(y_test, y_pred, digits=4))

    cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
    cm_df = pd.DataFrame(cm, index=model.classes_, columns=model.classes_)
    print('Confusion matrix:')
    print(cm_df)

    save_confusion_matrix(cm_df, os.path.join(PLOT_DIR, 'iris_confusion_matrix.png'))
    save_feature_importance(model, X.columns.tolist(), os.path.join(PLOT_DIR, 'iris_feature_importance.png'))
    save_feature_scatter(df, os.path.join(PLOT_DIR, 'iris_feature_scatter_matrix.png'))

    print('\nBasic classification insights:')
    print('  - Input features are the sepal and petal measurements for Iris flowers.')
    print('  - The model learns to distinguish the three species by these measurements.')
    print('  - Evaluation on test data gives the model accuracy and class-level performance.')
    print('  - Plots were saved to the iris_plots folder.')


if __name__ == '__main__':
    main()
