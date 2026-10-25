import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import csv
import json
import subprocess
import sys
import numpy as np
import pandas as pd
from keras.models import load_model
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

MODEL_PATH = 'bo_nao_ai_steam.keras'
FEEDBACK_FILE = 'new_feedback.csv'
GAMES_FILE = 'games.csv'
METADATA_FILE = 'games_metadata.json'

MAX_USERS = 200000
MAX_GAMES = 200000
CURRENT_USER = 15  # Mã ID người dùng hiện tại

# 1. Kiểm tra mô hình AI, nếu chưa có thì khởi tạo qua train.py
if not os.path.exists(MODEL_PATH):
    print("⚠️ Chưa tìm thấy bộ não AI! Đang tự động gọi train.py để khởi tạo...")
    subprocess.run([sys.executable, 'train.py'])

print("1. Đang nạp danh sách game từ games.csv...")
games_df = pd.read_csv(GAMES_FILE, usecols=['app_id', 'title'])
games_df = games_df.dropna().drop_duplicates(subset=['app_id'])

# Nạp thông tin tags/mô tả từ metadata
tags_dict = {}
if os.path.exists(METADATA_FILE):
    print("2. Đang nạp metadata (games_metadata.json)...")
    try:
        with open(METADATA_FILE, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line)
                tags_dict[item.get('app_id')] = " ".join(item.get('tags', []))
    except Exception:
        pass
games_df['tags'] = games_df['app_id'].map(tags_dict).fillna('')
games_df['search_text'] = games_df['title'] + " " + games_df['tags']

print("3. Đang tải mô hình hiểu ngôn ngữ tự nhiên (NLP Sentence Transformer)...")
# Mô hình đa ngôn ngữ cực nhẹ, hiểu tốt cả tiếng Việt và tiếng Anh (~118MB)
nlp_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

print("4. Đang nạp bộ não gợi ý Keras...")
model = load_model(MODEL_PATH)

def tim_game_bang_ngu_nghia(cau_hoi, top_candidates=60):
    """
    Biến câu nói tự nhiên thành vector ngữ nghĩa và tìm các game có mô tả sát ý nhất
    """
    # Lấy mẫu ngẫu nhiên 300 game phổ biến/có tags để so khớp nhanh
    candidates_pool = games_df[games_df['search_text'].str.strip() != ''].sample(
        n=min(300, len(games_df)), random_state=42
    ).copy()

    # Tạo vector ngữ nghĩa cho câu hỏi của người dùng
    vector_cau_hoi = nlp_model.encode([cau_hoi])

    # Tạo vector ngữ nghĩa cho các game trong tập ứng viên
    van_ban_game = candidates_pool['search_text'].tolist()
    vectors_game = nlp_model.encode(van_ban_game, show_progress_bar=False)

    # Tính độ tương đồng ngữ nghĩa bằng Cosine Similarity
    similarities = cosine_similarity(vector_cau_hoi, vectors_game)[0]
    candidates_pool['diem_ngu_nghia'] = similarities

    # Lọc lấy top các game sát ý nhất
    top_candidates_df = candidates_pool.sort_values(by='diem_ngu_nghia', ascending=False).head(top_candidates)
    return top_candidates_df

def ghi_nhan_phan_hoi(user_id, app_id, is_liked):
    """Ghi lại tương tác vào file tạm để train.py học ngầm sau khi thoát"""
    file_exists = os.path.exists(FEEDBACK_FILE)
    with open(FEEDBACK_FILE, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(['user_id', 'app_id', 'is_recommended'])
        writer.writerow([user_id, app_id, is_liked])

print("\n" + "=" * 60)
print(" 🤖 CHATBOT STEAM: HIỂU NGÔN NGỮ TỰ NHIÊN & TƯ VẤN THÔNG MINH ")
print(" (Nhập câu bất kỳ: 'game đáng sợ chơi với bạn', 'game nông trại chill'...)")
print(" (Gõ 'thoat' để dừng và tự động cập nhật bộ não AI)")
print("=" * 60)

so_luong_phan_hoi = 0

while True:
    cau_hoi = input("\nBạn muốn tìm game như thế nào?: ").strip()
    if cau_hoi.lower() in ['thoat', 'exit', 'quit']:
        break
    if not cau_hoi:
        continue

    print("🔍 AI đang phân tích ngữ nghĩa câu nói của bạn...")
    ung_vien = tim_game_bang_ngu_nghia(cau_hoi, top_candidates=40)

    if ung_vien.empty:
        print("🤖 Tư vấn viên AI: Không tìm thấy game phù hợp, bạn thử miêu tả cách khác nhé!")
        continue

    # Đưa danh sách game đã hiểu theo ngữ nghĩa vào mạng nơ-ron để chấm điểm cá nhân hóa
    app_ids_input = (ung_vien['app_id'].values % MAX_GAMES).astype(np.int32)
    users_input = np.full(len(app_ids_input), CURRENT_USER % MAX_USERS, dtype=np.int32)

    # Dự đoán độ hợp gu
    scores = model.predict([users_input, app_ids_input], verbose=0).flatten()
    
    # Điểm tổng hợp = 60% Ngữ nghĩa đúng ý + 40% Độ hợp gu cá nhân
    ung_vien['diem_ca_nhan'] = scores * 100
    ung_vien['diem_tong_hop'] = (ung_vien['diem_ngu_nghia'] * 100 * 0.6) + (ung_vien['diem_ca_nhan'] * 0.4)

    top_3 = ung_vien.sort_values(by='diem_tong_hop', ascending=False).head(3)

    print("\n" + "-" * 55)
    print("🎯 TOP 3 TỰA GAME PHÙ HỢP NHẤT VỚI MÔ TẢ CỦA BẠN:")
    print("-" * 55)
    danh_sach_top = list(top_3.iterrows())
    for idx, (_, row) in enumerate(danh_sach_top, 1):
        print(f" [{idx}] 🎮 {row['title']} (App ID: {row['app_id']})")
        print(f"      • Khớp với mong muốn của bạn: {row['diem_ngu_nghia']*100:.1f}%")
        print(f"      • Điểm AI đánh giá hợp gu: {row['diem_ca_nhan']:.1f}%")
    print("-" * 55)

    # Thu thập phản hồi để AI học tiếp
    for idx, (_, row) in enumerate(danh_sach_top, 1):
        fb = input(f"❓ Bạn thấy tựa game [{idx}] '{row['title']}' có đúng ý không? (1: Hợp / 0: Không / Enter: Bỏ qua): ").strip()
        if fb in ['1', '0']:
            ghi_nhan_phan_hoi(CURRENT_USER, row['app_id'], int(fb))
            so_luong_phan_hoi += 1
            print("   📝 Đã lưu nhận xét!")

print("\n============================================================")
print("👋 Đã kết thúc buổi trò chuyện!")
if so_luong_phan_hoi > 0:
    print(f"🚀 Có {so_luong_phan_hoi} phản hồi mới! Đang tự động gọi train.py để huấn luyện lại...")
    subprocess.run([sys.executable, 'train.py'])
else:
    print("ℹ️ Không có đánh giá mới nào để cập nhật.")