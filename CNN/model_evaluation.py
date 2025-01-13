from sklearn.metrics import classification_report, accuracy_score, f1_score
from tensorflow.keras.models import load_model
import mysql.connector
import numpy as np
import json
from preprocessing import preprocess_data
from data_loading import load_data

articles = load_data()

max_num_words = 20000
max_sequence_length = 1000
X, y, tokenizer, label_encoder = preprocess_data(articles, max_num_words, max_sequence_length)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)
    
def evaluate_model(model, X_test, y_test, label_encoder):
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)

    y_test_classes = np.argmax(y_test, axis=1) if y_test.ndim > 1 else y_test

    unique_classes_in_test = np.unique(y_test_classes)

    try:
        known_classes = np.isin(unique_classes_in_test, label_encoder.transform(label_encoder.classes_))
        filtered_classes = unique_classes_in_test[known_classes]
    except ValueError:
        print("Warning: Some classes in the test set are not present in the training set.")
        filtered_classes = unique_classes_in_test

    try:
        target_names = label_encoder.inverse_transform(filtered_classes)
    except ValueError:
        target_names = [str(c) for c in filtered_classes] 

    valid_indices = np.isin(y_test_classes, filtered_classes)
    y_test_classes_filtered = y_test_classes[valid_indices]
    y_pred_classes_filtered = y_pred_classes[valid_indices]

    accuracy = accuracy_score(y_test_classes_filtered, y_pred_classes_filtered)
    f1 = f1_score(y_test_classes_filtered, y_pred_classes_filtered, average='macro')

    report = classification_report(
        y_test_classes_filtered,
        y_pred_classes_filtered,
        target_names=target_names,
        labels=filtered_classes, 
        output_dict=True,
        zero_division=0  
    )

    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 Score (Macro): {f1:.4f}")
    print("Classification Report:")
    print(classification_report(
        y_test_classes_filtered,
        y_pred_classes_filtered,
        target_names=target_names,
        labels=filtered_classes, 
        zero_division=0
    ))

    save_results_to_db(accuracy, f1, report)

def save_results_to_db(accuracy, f1, report):
    try:
        conn = mysql.connector.connect(
            host='localhost',
            user='root',
            password='sumo21213',
            database='mydatabase'
        )
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS cnn_model_results (
                id INT AUTO_INCREMENT PRIMARY KEY,
                accuracy FLOAT NOT NULL,
                f1_score FLOAT NOT NULL,
                report TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)

        report_json = json.dumps(report)

        cursor.execute("""
            INSERT INTO cnn_model_results (accuracy, f1_score, report)
            VALUES (%s, %s, %s)
        """, (accuracy, f1, report_json))

        conn.commit()
    except mysql.connector.Error as err:
        print(f"Error: {err}")
    finally:
        conn.close()

    print(f"Results saved to the database: Accuracy={accuracy}, F1 Score={f1}")
    
evaluate_model(load_model('best_cnn_model.keras'), X_test, y_test, label_encoder)