from sklearn.metrics import accuracy_score, f1_score, classification_report
import mysql.connector
from data_loading import load_data
from preprocessing import preprocess_data
from model_saving import load_model

articles, tfidf_data = load_data()
X_train_tfidf, X_test_tfidf, y_train, y_test, categories = preprocess_data(articles, tfidf_data)

def evaluate_model(model, X_test, y_test, categories):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='macro')
    
    report = classification_report(y_test, y_pred, target_names=categories, output_dict=True)
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 Score (Macro): {f1:.4f}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=categories))


def save_results_to_db(accuracy, f1, report):
    conn = mysql.connector.connect(
        host='localhost',
        user='root',
        password='sumo21213',
        database='mydatabase'
    )
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS svm_model_results (
            id INT AUTO_INCREMENT PRIMARY KEY,
            accuracy FLOAT NOT NULL,
            f1_score FLOAT NOT NULL,
            report TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    import json
    report_json = json.dumps(report)

    cursor.execute("""
        INSERT INTO svm_model_results (accuracy, f1_score, report)
        VALUES (%s, %s, %s)
    """, (accuracy, f1, report_json))

    conn.commit()
    conn.close()

    print(f"Results saved to the database: Accuracy={accuracy}, F1 Score={f1}")

evaluate_model(load_model('best_svm_model.joblib'), X_test_tfidf, y_test, categories)