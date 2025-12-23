# src/models.py

import pandas as pd
import numpy as np
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF, TruncatedSVD
from sklearn.metrics.pairwise import cosine_distances
import gensim
import gensim.corpora as corpora

 
# 1. CÁC HÀM TÍNH METRICS (THEO KHOẢNG CÁCH)
 

def compactness_cosine(X, labels):
    compactness_scores = []
    unique_labels = np.unique(labels)
    
    for label in unique_labels:
        cluster_docs = X[labels == label]

        if len(cluster_docs) <= 1:
            continue

        # Tính centroid
        centroid = cluster_docs.mean(axis=0, keepdims=True)
        # Tính khoảng cách từ các điểm đến centroid
        distances = cosine_distances(cluster_docs, centroid)

        compactness_scores.append(distances.mean())

    return np.mean(compactness_scores) if compactness_scores else 0.0

def separation_cosine(X, labels):
    centroids = []
    unique_labels = np.unique(labels)

    for label in unique_labels:
        cluster_docs = X[labels == label]
        if len(cluster_docs) == 0:
            continue
        
        centroid = cluster_docs.mean(axis=0)
        centroids.append(centroid)

    centroids = np.array(centroids)

    if len(centroids) < 2:
        return 0.0

    # Tính khoảng cách giữa các centroids
    distances = cosine_distances(centroids)

    # Lấy trung bình các phần tử tam giác trên (khoảng cách đôi một)
    upper_triangle = distances[np.triu_indices_from(distances, k=1)]

    return upper_triangle.mean()

 
# 2. CÁC HÀM SINH CAPTION (HYBRID)
 

def get_representative_doc_keywords(doc_topic_matrix, df, topic_id, token_col='tokens', top_k_docs=5, top_k_words=3):
    # Lấy index các văn bản thuộc topic
    dominant_topics = np.argmax(doc_topic_matrix, axis=1)
    doc_indices = np.where(dominant_topics == topic_id)[0]
    
    if len(doc_indices) == 0: return []

    # Lấy vector của topic đó để tìm centroid
    X_topic = doc_topic_matrix[doc_indices]
    centroid = X_topic.mean(axis=0, keepdims=True)
    
    # Tìm các văn bản gần centroid nhất
    dists = cosine_distances(X_topic, centroid).flatten()
    closest_indices_local = np.argsort(dists)[:top_k_docs]
    original_indices = doc_indices[closest_indices_local]
    
    # Gom token lại để tìm từ khóa
    all_tokens = []
    for idx in original_indices:
        tokens = df.iloc[idx][token_col]
        if isinstance(tokens, list):
            all_tokens.extend(tokens)
        elif isinstance(tokens, str):
            all_tokens.extend(tokens.split())

    common_words = [w for w, _ in Counter(all_tokens).most_common(top_k_words)]
    return common_words

def generate_hybrid_captions(topic_term_matrix, doc_topic_matrix, vocabulary, df, token_col='tokens'):
    captions = {}
    n_topics = topic_term_matrix.shape[0]
    feature_names = np.array(vocabulary)

    for topic_id in range(n_topics):
        # 1. Keywords từ trọng số Model (Model Weights)
        topic_vector = topic_term_matrix[topic_id]
        top_indices = np.argsort(topic_vector)[::-1][:5]
        model_words = [feature_names[i] for i in top_indices]
        
        # 2. Keywords từ bài viết tiêu biểu (Representative Docs)
        doc_words = get_representative_doc_keywords(
            doc_topic_matrix, df, topic_id, token_col=token_col, top_k_docs=5, top_k_words=3
        )
        
        # 3. Gộp và xóa trùng
        combined = list(dict.fromkeys(model_words + doc_words))
        captions[topic_id] = " | ".join(combined)
        
    return captions

 
# 3. CLASS QUẢN LÝ MODEL
 

class TopicModelManager:
    def __init__(self, df, text_col='clean_text', token_col='tokens'):
        self.df = df
        self.token_col = token_col
        self.documents = df[text_col].tolist()
        
        # Vectorizer cho Sklearn
        self.vectorizer = TfidfVectorizer(max_features=2000, min_df=5, max_df=0.9)
        self.X_sklearn = self.vectorizer.fit_transform(self.documents)
        self.feature_names = self.vectorizer.get_feature_names_out()
        
        # Init cho Gensim
        self.tokenized_docs = df[token_col].tolist()
        self.id2word = corpora.Dictionary(self.tokenized_docs)
        self.corpus = [self.id2word.doc2bow(text) for text in self.tokenized_docs]

    def _calculate_metrics_and_caption(self, topic_term_matrix, doc_topic_matrix, model_type):
        """Hàm tính toán chỉ số và sinh caption"""
        # 1. Xác định nhãn (Dominant Topic) cho từng văn bản
        # doc_topic_matrix shape: (n_docs, n_topics)
        labels = np.argmax(doc_topic_matrix, axis=1)
        
        # 2. Tính Metrics (Theo logic Distance mới)
        comp = compactness_cosine(doc_topic_matrix, labels)
        sep = separation_cosine(doc_topic_matrix, labels)
        
        # Score = Separation / Compactness
        score = sep / comp if comp > 0 else 0.0
        
        # 3. Sinh Caption
        captions_map = generate_hybrid_captions(
            topic_term_matrix, doc_topic_matrix, self.feature_names, self.df, self.token_col
        )
        
        topics_data = []
        n_topics = topic_term_matrix.shape[0]
        
        for topic_id in range(n_topics):
            topics_data.append({
                "Topic_ID": topic_id,
                "Caption": captions_map.get(topic_id, f"Topic {topic_id}"),
                "Dominant_Count": np.sum(labels == topic_id)
            })
            
        return topics_data, comp, sep, score

    # --- LSA ---
    def run_lsa(self, n_topics=5):
        model = TruncatedSVD(n_components=n_topics, random_state=42)
        doc_topic_matrix = model.fit_transform(self.X_sklearn) # Đây chính là X
        topic_term_matrix = model.components_
        return self._calculate_metrics_and_caption(topic_term_matrix, doc_topic_matrix, "LSA")

    # --- NMF ---
    def run_nmf(self, n_topics=5):
        model = NMF(n_components=n_topics, random_state=42, init='nndsvd')
        doc_topic_matrix = model.fit_transform(self.X_sklearn) # Đây chính là X
        topic_term_matrix = model.components_
        return self._calculate_metrics_and_caption(topic_term_matrix, doc_topic_matrix, "NMF")

    # --- LDA ---
    def run_lda(self, n_topics=5):
        lda_model = gensim.models.LdaMulticore(corpus=self.corpus, id2word=self.id2word,
                                               num_topics=n_topics, random_state=42,
                                               workers=2, passes=10)
        
        # Chuyển đổi output của Gensim thành ma trận dense (n_docs, n_topics) để dùng được hàm cosine
        doc_topic_probs = []
        for bow in self.corpus:
            probs = lda_model.get_document_topics(bow, minimum_probability=0.0)
            vec = [0.0] * n_topics
            for topic_id, prob in probs:
                vec[topic_id] = prob
            doc_topic_probs.append(vec)
        doc_topic_matrix = np.array(doc_topic_probs) # Đây chính là X của LDA
        
        # Tạo topic_term_matrix
        topic_term_matrix = np.zeros((n_topics, len(self.feature_names)))
        for topic_id in range(n_topics):
            word_probs = lda_model.get_topic_terms(topic_id, topn=len(self.id2word))
            for word_id, prob in word_probs:
                word = self.id2word[word_id]
                if word in self.vectorizer.vocabulary_:
                    topic_term_matrix[topic_id][self.vectorizer.vocabulary_[word]] = prob
                    
        return self._calculate_metrics_and_caption(topic_term_matrix, doc_topic_matrix, "LDA")

