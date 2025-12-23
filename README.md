# Vietnamese Topic Modeling on Job Reviews

## Giới thiệu
Dự án nhóm 13 tập trung vào việc *Mô hình hóa chủ đề (Topic Modeling)* cho các văn bản Tiếng Việt (cụ thể là về Job Reviews). Mục tiêu là tìm ra các nhóm chủ đề ẩn từ dữ liệu thô, giúp hiểu rõ insight của người dùng/nhân viên.

Dự án áp dụng và so sánh 4 thuật toán phổ biến:
1.  *LSA (Latent Semantic Analysis)*
2.  *NMF (Non-negative Matrix Factorization)*
3.  *LDA (Latent Dirichlet Allocation)*
4.  *BERTopic*

## Điểm nổi bật (Key Features)

* *Xử lý từ Tiếng Việt*
    * Xử lý Teencode (mik, ko, dc...) bằng từ điển tự xây dựng.
    * Dịch thuật Anh-Việt (Google Translate API).
    * *Super Filter:* Bộ lọc rác thông minh loại bỏ spam (seeding, quảng cáo, minigame...).
    * Chuẩn hóa Emoji, Hashtag, Link.
* *Sinh nhãn chủ đề*
    * Kết hợp từ khóa có trọng số cao nhất.
    * Kết hợp từ vựng từ các bài viết tiêu biểu nhất.
    * -> Giúp tên chủ đề dễ hiểu và sát thực tế hơn.
* *Tính toán so sánh các chỉ số*
    * Tự động tính toán *Compactness* (Độ nén) và *Separation* (Độ tách biệt) dựa trên khoảng cách Cosine.
    * Đề xuất số lượng topics (K) tối ưu dựa trên chỉ số tổng hợp *Score = Separation / Compactness*.

## 📂 Cấu trúc thư mục

```text
DoAn_TopicModeling/
│
├── data/
│   ├── raw_dataset.csv
│   ├── processed_dataset.csv
│   └── vietnamese-stopwords.txt
│
├── src/
│   ├── _init_.py
│   ├── preprocessing.py
│   └── models.py
│
├── 1_EDA_Preprocessing.ipynb
├── 2_Modeling.ipynb
├── requirements.txt
└── README.md
