# src/preprocessing.py

import pandas as pd
import numpy as np
import re
import emoji
from pyvi import ViTokenizer
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from deep_translator import GoogleTranslator

# CẤU HÌNH TỪ ĐIỂN (TEENCODE)
TEENCODE_DICT = {
    # Nhóm phủ định
    "cty": "công ty", "cong ty": "công ty", "pv": "phỏng vấn",
    "nv": "nhân viên", "sếp": "quản lý", "sep": "quản lý",
    "ot": "tăng ca", "offer": "mức lương", "deal": "đàm phán",
    "k": "không", "ko": "không", "kh": "không", "hok": "không",
    "hông": "không", "hong" : "không", "khum" : "không", "khôm" : "không",
    "khom" : "không", "hăm": "không", "hôn" : "không", "hem" : "không", "hk" : "không",
    "đc": "được", "dc": "được", "dk": "được", "đk": "được",
    "ntn": "như thế nào", "dt": "điện thoại", "fb": "facebook",
    "mng": "mọi người", "mn": "mọi người", "ng": "người", "sốp" : "mọi người",
    "vs": "với", "wa": "quá", "wá": "quá", "j": "gì",
    "bn": "bao nhiêu", "bh": "bây giờ", "h": "giờ", "bây h" : "bây giờ",
    "t": "tôi", "tao": "tôi", "tui": "tôi", "b": "bạn", "m": "mày",
    "r": "rồi", "z": "vậy", "zậy": "vậy", "dữ": "rất",
    "bjo": "bây giờ", "thik": "thích", "thít" : "thích", "iu": "yêu",
    "chê": "xấu", "ổn": "tốt", "dev": "lập trình viên",
    "mày" : "mọi người", "đh" : "đại học", "dh" : "đại học",
    "sv" : "sinh viên", "cv" : "hồ sơ", "tts" : "thực tập sinh",
    "nd" : "nội dung", "dl" : "deadline", "méo" : "không", "đéo" : "không",
    "éo" : "không", "đ" : "không", "z" : "vậy", "tr" : "trời",
    "xu cà na" : "tệ", "ạh" : "ạ", "1st" : "đầu tiên",
    "mí" : "mấy", "dồi" : "trời", "vl" : "", "cx" : "cũng",
    "vp" : "văn phòng", "tỏi" : "tỉ", "ok" : "ổn", "toi": "tôi",
    "ac" : "anh chị", "ace" : "anh chị em", "a/c" : "anh chị", "ae" : "mọi người",
    "ní" : "mọi người", "tbay" : "mọi người", "cả lò" : "mọi người",
    "dô" : "vô", "rcmd" : "khuyên", "rcm" : "khuyên",
    "ytb" : "youtube", "đko" : "đúng không", "btc" : "ban tổ chức",
    "xiền" : "tiền", "xèng" : "tiền", "sg" : "sài gòn", "hqa" : "hôm qua",
    "hqua" : "hôm qua", "gđ" : "gia đình", "lđ" : "lao động",
    "cũm" : "cũng", "itv" : "interview", "ròi" : "rồi"
}

# 1. DỊCH THUẬT
def translate_en_to_vi(text):
    translator = GoogleTranslator(source='en', target='vi')
    if not isinstance(text, str) or len(text.strip()) < 2: return text
    if "_" in text: text = text.replace("_", " ") 
    try:
        return translator.translate(text)
    except Exception:
        return text

def apply_translation(df, col_name):
    print("--- Bắt đầu dịch thuật ---")
    df[col_name] = df[col_name].apply(translate_en_to_vi)
    return df

# 2. BỘ LỌC NÂNG CAO (SUPER FILTER)
def super_filter(text):
    if not isinstance(text, str): return False
    text = str(text).strip().lower()
    text_check = text.replace("_", " ") # Xử lý cho tokenizer

    # 1. Filter theo số từ
    wc = len(text_check.split())
    if wc < 5 or wc > 1500: # Hạ ngưỡng xuống 5 để tránh lọc nhầm câu ngắn
        return False

    # 2. Filter theo spam keywords
    spam_keywords = [
        # bán hàng
      "giảm giá","flash sale","đặt hàng","ship cod",
      "thanh lý","xả kho","mua ngay","chốt đơn","lấy sỉ",

      # seeding/marketing
      "minigame","giveaway",

      # scam/bot
      "coin","crypto",
      "tiền ảo"

      , "mst" # mã số thuế

      # Giải trí / fandom / showbiz (ngoài ngữ cảnh đi làm)
        "doraemon", "conan", "naruto", "one piece",
        "dragon ball", "pokemon", "thám tử lừng danh",
        "anh trai say hi", "rap việt", "running man",
        "idol", "showbiz", "fan meeting", "concert",
        "kpop", "cp", "otp", "ship couple",

        # Meme / content giải trí rỗng
        "kkk", "lol", "vcl", "vl",
        "xem cho vui", "giải trí là chính",
        "coi cho biết", "cho vui thôi",

        # Seeding / quảng cáo trá hình (không liên quan review)
        "inbox để biết thêm", "ib để biết thêm",
        "link bio", "link dưới comment",
        "đặt hàng tại", "mua tại đây",
        "theo dõi page", "follow page",
        "nhận ưu đãi", "mã giảm",

        # 5. Nội dung lệch hẳn sang giải trí cá nhân
        "phim hay", "xem phim", "review phim",
        "diễn viên", "đạo diễn", "rating phim", "bar",

        # 6. Tên riêng gây nhiễu bạn đã gặp
        "atsh", "congb", "dylan","font","anh hiếu","toà nhà","millenium","karik","du lịch","chụp",
        "pha chế","gv","bar","du học","dương đình nghệ", "em 2k8","hoa thủy tiên",
        "thông tin liên hệ","đường số","đi bão","xe khách","chuyên chở",
        "cầu giấy","em gái mưa", "công viên bờ sông","350 cành", "hà nội","thanh xuân","thanh hóa","đường","địa chỉ","đồ ăn","tầng","tân phú","hcm",
            "bình thạnh","mst","gò vấp","quận","căn","hoàn kiếm", "tòa nhà", "số điện thoại"
    ]
    for kw in spam_keywords:
        if kw in text_check: return False
    return True

 
# 3. DEEP PREPROCESS (THEO CODE CỦA BẠN)
 

def load_stopwords(stopwords_path):
    try:
        with open(stopwords_path, 'r', encoding='utf-8') as f:
            stopwords = set(f.read().splitlines())
        return stopwords
    except FileNotFoundError:
        print(f"LỖI: Không tìm thấy file stopwords tại {stopwords_path}")
        return set()

def deep_preprocess(text, stopwords=set()):
    if not isinstance(text, str): return []

    # 1. Lowercase
    text = text.lower()

    # 2. Map Teencode
    sorted_keys = sorted(TEENCODE_DICT.keys(), key=len, reverse=True)
    pattern = r'\b(' + '|'.join(re.escape(k) for k in sorted_keys) + r')\b'
    text = re.sub(pattern, lambda m: TEENCODE_DICT[m.group(0)], text)

    # 3. Remove Links
    text = re.sub(r'http\S+|www\S+', '', text)
    
    # 4. Remove Mentions (@tag) & Hashtags (#tag)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#([A-Za-z0-9_àáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ]+)', ' ', text)
    text = re.sub(r'(#[A-Za-z0-9_àáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ]+)+', ' ', text)

    # 5. Remove từ kéo dài (aaaa -> a)
    text = re.sub(r'(.)\1{2,}\b', r'\1', text)

    # 6. Remove credit/nguồn
    text = re.sub(r'(cre|credit|nguồn|source)[:\-].*$', '', text)

    # 7. Xóa số
    text = re.sub(r'\d+', '', text)

    # 8. Remove Emoji & Special Char (Giữ lại tiếng Việt và khoảng trắng)
    text = re.sub(r"[^\w\sàáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ]", " ", text)

    # 9. Remove khoảng trắng thừa
    text = re.sub(r'\s+', ' ', text).strip()

    # 10. Tokenize (ViTokenizer)
    tokenized = ViTokenizer.tokenize(text)

    # 11. Remove Stopwords & từ 1 ký tự
    tokens = [
        w for w in tokenized.split()
        if w not in stopwords and len(w) > 1
    ]

    return tokens

 
# 4. PIPELINE CHÍNH
 

def preprocess_pipeline(df, col_name, stopwords_path='data/vietnamese-stopwords.txt'):
    print("--- Bắt đầu tiền xử lý (Tokenization & Normalization) ---")
    initial_len = len(df)
    
    # Load stopwords
    stopwords = load_stopwords(stopwords_path)
    
    # BƯỚC 1: Deep Preprocess -> Tạo cột 'tokens' (List)
    # Truyền stopwords và dictionary vào hàm
    df['tokens'] = df[col_name].apply(lambda x: deep_preprocess(x, stopwords))
    
    # BƯỚC 2: Lọc bỏ những dòng không còn token nào (rỗng)
    df = df[df['tokens'].map(len) > 0].reset_index(drop=True)
    
    # BƯỚC 3: Tạo cột 'clean_text' (String) cho BERTopic
    df['clean_text'] = df['tokens'].apply(lambda x: " ".join(x))
    
    # BƯỚC 4: Chạy Super Filter (Lọc rác ngữ nghĩa) trên clean_text
    print("--- Đang chạy bộ lọc nâng cao (Super Filter) ---")
    df = df[df['clean_text'].apply(super_filter)]
    df = df.reset_index(drop=True)
    
    final_len = len(df)
    print(f"Hoàn tất. Dữ liệu gốc: {initial_len} -> Còn lại: {final_len} dòng.")
    
    return df

 
# 5. EDA
 
def plot_wordcloud(text_series):
    text_combined = " ".join(str(text) for text in text_series if isinstance(text, str))
    if not text_combined: return
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text_combined)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.show()

def plot_doc_length(text_series):
    doc_lengths = text_series.apply(lambda x: len(str(x).split()))
    plt.figure(figsize=(10, 5))
    plt.hist(doc_lengths, bins=50, color='skyblue', edgecolor='black')
    plt.title('Phân bố độ dài văn bản')
    plt.show()