# Vietnamese News Classification and Clustering Project

## Overview
This project focuses on developing machine learning and deep learning models for **classifying and clustering Vietnamese news articles** into 12 categories. The workflow involves data collection, preprocessing, vectorization, and applying multiple algorithms to achieve accurate classification and meaningful clustering of articles.

## Features
1. **Data Collection**  
   - Articles were collected from Vietnamese news websites: **vnexpress.net**, **vietnamnet.vn**, and **dantri.com.vn**.  
   - Each category includes 500 articles.  

2. **Data Preprocessing**  
   - Extracted metadata: publish time, author, title, snippet, content, category, tags.  
   - Cleaned data to ensure quality:
     - Removed short descriptions and content.  
     - Normalized publish dates to timestamps.  
     - Standardized long author names as "Unknown."  
   - Combined title and content for better feature representation.  
   - Tokenized text, removed special characters, converted to lowercase, and eliminated stopwords.  

3. **TF-IDF Vectorization**  
   - Cleaned text was converted into TF-IDF vectors using `TfidfVectorizer` from `scikit-learn`.  
   - Results stored in a structured database for further processing.  

4. **Machine Learning Models**  
   - **Support Vector Machine (SVM):**  
     - Initial accuracy: ~40%; optimized to achieve better results.  
     - Best parameters: `C=10`, `kernel='rbf'`, `gamma=0.01`.  
   - **Convolutional Neural Network (CNN):**  
     - Achieved ~80% accuracy with embedding, convolutional, pooling, and fully connected layers.  
     - Applied dropout regularization and early stopping to prevent overfitting.  
   - **Long Short-Term Memory (LSTM):**  
     - Compared TF-IDF and SBERT vectorization:  
       - TF-IDF: Accuracy ~39.04%, F1 Macro ~33.61%.  
       - SBERT: Accuracy ~65.77%, F1 Macro ~62.50%.  
     - SBERT proved more effective due to richer representations.  
   - **Naive Bayes:**  
     - Vectorized using TF-IDF and SBERT.  
     - Results: SBERT outperformed TF-IDF but lagged behind LSTM.  

5. **Clustering Algorithms**  
   - Performed using TF-IDF and SBERT vectors.  
   - Techniques:  
     - **KMeans:**  
       - Optimal cluster selection using Silhouette Score, Davies-Bouldin Index, and Calinski-Harabasz Index.  
     - **Hierarchical Agglomerative Clustering (HAC):**  
       - Results compared with KMeans using Adjusted Rand Index (ARI).  
   - Observations: TF-IDF provided better clustering results than SBERT vectors.  

## Results
- **Classification:**  
  - CNN achieved the highest accuracy (~80%), outperforming other models.  
- **Clustering:**  
  - TF-IDF consistently produced better results across clustering metrics.  

## Recommendations
- Address class imbalance to improve model performance.  
- Explore more advanced architectures for handling complex relationships in text data.  
- Further enhance text representation techniques for improved accuracy.  

## Technologies and Tools
- **Languages:** Python  
- **Libraries and Frameworks:**  
  - `scikit-learn`, `TensorFlow`, `PyTorch`, `SentenceTransformers`, `NLTK`  
- **Databases:** MySQL  
- **Other Tools:** FAISS for similarity search  

## Authors
- Võ Luyện - Tôn Đức Thắng University  
- Trần Nhật Gia Huy - Tôn Đức Thắng University  
- Nguyễn Minh Phú - Tôn Đức Thắng University  

## Acknowledgments
This project is part of the team effort by **Team DPLH** at Tôn Đức Thắng University, aiming to advance Vietnamese text classification and clustering methodologies.
