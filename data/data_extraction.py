import mysql.connector
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, accuracy_score, f1_score
from scipy.sparse import lil_matrix, csr_matrix
from collections import defaultdict

# Database connection
db_connection = mysql.connector.connect(
    host='localhost',  # or your DB host
    user='root',
    password='sumo21213',
    database='mydatabase'
)

cursor = db_connection.cursor()

# Load articles data from the database
cursor.execute("SELECT id, content, category FROM articles")
articles = cursor.fetchall()

# Load TF-IDF data from the database
cursor.execute("SELECT doc_id, word, tfidf_value FROM tfidf")
tfidf_data = cursor.fetchall()

# Close the database connection
cursor.close()
db_connection.close()

# Separate articles data into lists
article_ids = []
contents = []
categories = []

for article in articles:
    article_ids.append(article[0])
    contents.append(article[1])
    categories.append(article[2])

# Encode categories into numerical labels
unique_categories = list(set(categories))
category_to_id = {category: index for index, category in enumerate(unique_categories)}
y = [category_to_id[category] for category in categories]

# Split data into training and test sets based on document ids
train_ids, test_ids, y_train, y_test = train_test_split(
    article_ids, y, test_size=0.15, random_state=42
)

# Create a dictionary to map words to columns in the TF-IDF matrix
word_to_index = {}
doc_to_index = {doc_id: idx for idx, doc_id in enumerate(article_ids)}

# Process TF-IDF data to create the TF-IDF matrix
tfidf_dict = defaultdict(list)
for row in tfidf_data:
    doc_id, word, tfidf_value = row
    # Check if the document ID is in the articles list
    if doc_id in doc_to_index:
        if word not in word_to_index:
            word_to_index[word] = len(word_to_index)
        tfidf_dict[doc_id].append((word_to_index[word], tfidf_value))

# Number of documents and words for creating the TF-IDF matrix
num_docs = len(doc_to_index)
num_words = len(word_to_index)

# Initialize a lil_matrix to store TF-IDF values for efficient row-wise insertion
tfidf_matrix_lil = lil_matrix((num_docs, num_words))

# Fill the TF-IDF matrix
for doc_id, word_values in tfidf_dict.items():
    doc_idx = doc_to_index[doc_id]
    for word_idx, tfidf_value in word_values:
        tfidf_matrix_lil[doc_idx, word_idx] = tfidf_value

# Convert lil_matrix to csr_matrix for fast operations and compatibility with scikit-learn
tfidf_matrix = csr_matrix(tfidf_matrix_lil)

# Split the TF-IDF matrix into training and test sets
train_indices = [doc_to_index[doc_id] for doc_id in train_ids]
test_indices = [doc_to_index[doc_id] for doc_id in test_ids]

X_train_tfidf = tfidf_matrix[train_indices, :]
X_test_tfidf = tfidf_matrix[test_indices, :]

print(f"TF-IDF training set shape: {X_train_tfidf.shape}")
print(f"TF-IDF test set shape: {X_test_tfidf.shape}")