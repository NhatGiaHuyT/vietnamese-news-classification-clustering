from scipy.sparse import lil_matrix, csr_matrix
from sklearn.model_selection import train_test_split
from collections import defaultdict

def preprocess_data(articles, tfidf_data):
    article_ids = []
    contents = []
    categories = []

    for article in articles:
        article_ids.append(article[0])
        contents.append(article[1])
        categories.append(article[2])

    unique_categories = list(set(categories))
    category_to_id = {category: index for index, category in enumerate(unique_categories)}
    y = [category_to_id[category] for category in categories]

    train_ids, test_ids, y_train, y_test = train_test_split(
        article_ids, y, test_size=0.15, random_state=42
    )

    word_to_index = {}
    doc_to_index = {doc_id: idx for idx, doc_id in enumerate(article_ids)}

    tfidf_dict = defaultdict(list)
    for row in tfidf_data:
        doc_id, word, tfidf_value = row
        if doc_id in doc_to_index:
            if word not in word_to_index:
                word_to_index[word] = len(word_to_index)
            tfidf_dict[doc_id].append((word_to_index[word], tfidf_value))

    num_docs = len(doc_to_index)
    num_words = len(word_to_index)

    tfidf_matrix_lil = lil_matrix((num_docs, num_words))

    for doc_id, word_values in tfidf_dict.items():
        doc_idx = doc_to_index[doc_id]
        for word_idx, tfidf_value in word_values:
            tfidf_matrix_lil[doc_idx, word_idx] = tfidf_value

    tfidf_matrix = csr_matrix(tfidf_matrix_lil)

    train_indices = [doc_to_index[doc_id] for doc_id in train_ids]
    test_indices = [doc_to_index[doc_id] for doc_id in test_ids]

    X_train_tfidf = tfidf_matrix[train_indices, :]
    X_test_tfidf = tfidf_matrix[test_indices, :]

    return X_train_tfidf, X_test_tfidf, y_train, y_test, unique_categories