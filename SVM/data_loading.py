import mysql.connector

def load_data():
    db_connection = mysql.connector.connect(
        host='localhost',  
        user='root',
        password='sumo21213',
        database='mydatabase'
    )

    cursor = db_connection.cursor()

    cursor.execute("SELECT doc_id, texts, catcategory FROM processed_texts")
    articles = cursor.fetchall()

    cursor.execute("SELECT doc_id, word, tfidf_value FROM tfidf")
    tfidf_data = cursor.fetchall()

    cursor.close()
    db_connection.close()

    return articles, tfidf_data