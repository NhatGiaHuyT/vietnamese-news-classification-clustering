import mysql.connector

def load_data():
    conn = mysql.connector.connect(
        host='localhost',
        user='root',
        password='sumo21213',
        database='mydatabase'
    )
    cursor = conn.cursor()

    cursor.execute("SELECT doc_id, texts, catcategory FROM processed_texts")
    articles = cursor.fetchall()

    conn.close()

    return articles