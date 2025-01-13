from preprocessing import preprocess_data
from data_loading import load_data
from model_saving import load_model
import matplotlib.pyplot as plt
from model_evaluation import evaluate_model
import numpy as np
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    PrecisionRecallDisplay,
    RocCurveDisplay,
    precision_recall_curve,
    average_precision_score
)
from sklearn.model_selection import learning_curve
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier

articles, tfidf_data = load_data()
X_train_tfidf, X_test_tfidf, y_train, y_test, categories = preprocess_data(articles, tfidf_data)

def plot_confusion_matrix_custom(model, X_test, y_test, class_names):
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred, normalize='true')
    
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Normalized Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    fmt = '.2f'
    thresh = cm.max() / 2.
    for i, j in np.ndindex(cm.shape):
        plt.text(j, i, format(cm[i, j], fmt),
                 ha="center", va="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.show()


def plot_roc_curve_custom(model, X_test, y_test):
    plt.figure(figsize=(8, 6))
    RocCurveDisplay.from_estimator(model, X_test, y_test)
    plt.title('ROC Curve')
    plt.show()

def plot_precision_recall_curve_custom(model, X_test, y_test, class_names):
    y_test_bin = label_binarize(y_test, classes=range(len(class_names)))
    
    if not isinstance(model, OneVsRestClassifier):
        model = OneVsRestClassifier(model)
    model.fit(X_test, y_test_bin)
    
    plt.figure(figsize=(8, 6))
    for i in range(len(class_names)):
        y_score = model.decision_function(X_test)[:, i]
        precision, recall, _ = precision_recall_curve(y_test_bin[:, i], y_score)
        plt.plot(recall, precision, lw=2, label=f'Class {class_names[i]}')

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve (OvR)')
    plt.legend(loc='best')
    plt.show()

def plot_learning_curve(model, X_train, y_train):
    plt.figure(figsize=(8, 6))
    train_sizes, train_scores, test_scores = learning_curve(
        model, X_train, y_train, cv=5, n_jobs=1, train_sizes=np.linspace(0.1, 1.0, 10), scoring='f1_macro'
    )
    train_scores_mean = np.mean(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    
    plt.plot(train_sizes, train_scores_mean, 'o-', color='r', label='Training score')
    plt.plot(train_sizes, test_scores_mean, 'o-', color='g', label='Cross-validation score')
    
    plt.title('Learning Curve')
    plt.xlabel('Training examples')
    plt.ylabel('Score')
    plt.legend(loc='best')
    plt.grid()
    plt.show()

def plot_grid_search_results(grid_search):
    results = grid_search.cv_results_
    mean_test_scores = results['mean_test_score']
    params = results['params']
    
    C_vals = [params[i]['svc__C'] for i in range(len(params))]
    gamma_vals = [params[i]['svc__gamma'] for i in range(len(params))]

    plt.figure(figsize=(8, 6))
    plt.scatter(C_vals, gamma_vals, c=mean_test_scores, cmap='viridis', edgecolor='k')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('C values (log scale)')
    plt.ylabel('Gamma values (log scale)')
    plt.title('Grid Search Results')
    plt.colorbar(label='Mean F1 Score')
    plt.show()

def visualize_svm_model(model, X_train, y_train, X_test, y_test, grid_search=None):
    class_names = np.unique(y_test).astype(str)
    
    plot_confusion_matrix_custom(model, X_test, y_test, class_names)
    
    # evaluate_model(load_model('best_svm_model.joblib'), X_test_tfidf, y_test, categories)
    
    if len(class_names) == 2:
        plot_roc_curve_custom(model, X_test, y_test)
    
    plot_precision_recall_curve_custom(model, X_test, y_test, class_names)
    
    plot_learning_curve(model, X_train, y_train)
    
    if grid_search:
        plot_grid_search_results(grid_search)

visualize_svm_model(load_model('best_svm_model.joblib'), X_train_tfidf, y_train, X_test_tfidf, y_test)
