from data_loading import load_data
from preprocessing import preprocess_data
from model_training import train_svm_model
from model_evaluation import evaluate_model
from model_saving import save_model

def main():
    articles, tfidf_data = load_data()
    X_train_tfidf, X_test_tfidf, y_train, y_test, categories = preprocess_data(articles, tfidf_data)
    best_svm_model = train_svm_model(X_train_tfidf, y_train)
    save_model(best_svm_model, "best_svm_model.joblib")
    evaluate_model(best_svm_model, X_test_tfidf, y_test, categories)

if __name__ == "__main__":
    main()