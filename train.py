import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import pandas as pd
import numpy as np
import tensorflow as tf
from keras.layers import Embedding, Flatten, Dense, Input, Concatenate
from keras.models import Model, load_model

MODEL_PATH = 'bo_nao_ai_steam.keras'
FEEDBACK_FILE = 'new_feedback.csv'
RECOMMENDATIONS_FILE = 'recommendations.csv'

# Định nghĩa giới hạn ID tối đa để Embedding không bị lỗi vượt quá giới hạn khi học thêm ID mới
MAX_USERS = 200000
MAX_GAMES = 200000

def build_new_model():
    """Khởi tạo kiến trúc mạng nơ-ron mới hoàn toàn"""
    user_input = Input(shape=(1,), name="User_Input")
    user_emb = Embedding(MAX_USERS, 32, name="User_Embedding")(user_input)
    user_vec = Flatten()(user_emb)

    game_input = Input(shape=(1,), name="Game_Input")
    game_emb = Embedding(MAX_GAMES, 32, name="Game_Embedding")(game_input)
    game_vec = Flatten()(game_emb)

    concat = Concatenate()([user_vec, game_vec])
    dense = Dense(64, activation='relu')(concat)
    output = Dense(1, activation='sigmoid', name="Du_Doan_Thich")(dense)

    model = Model(inputs=[user_input, game_input], outputs=output)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def train_initial():
    """Huấn luyện lần đầu tiên từ tập dữ liệu lớn"""
    print("==================================================")
    print("🚀 BẮT ĐẦU HUẤN LUYỆN MÔ HÌNH TỪ ĐẦU (INITIAL TRAIN)")
    print("==================================================")
    if not os.path.exists(RECOMMENDATIONS_FILE):
        print(f"❌ Không tìm thấy file {RECOMMENDATIONS_FILE}! Đang khởi tạo mô hình rỗng để sẵn sàng học...")
        model = build_new_model()
        model.save(MODEL_PATH)
        print(f"✅ Đã tạo và lưu bộ não cơ bản tại: {MODEL_PATH}")
        return

    print("1. Đang đọc dữ liệu từ recommendations.csv (lấy 100,000 dòng đầu để train nhanh)...")
    df = pd.read_csv(RECOMMENDATIONS_FILE, nrows=100000)

    # Giới hạn ID trong phạm vi MAX_USERS và MAX_GAMES
    user_ids = (df['user_id'].values % MAX_USERS).astype(np.int32)
    game_ids = (df['app_id'].values % MAX_GAMES).astype(np.int32)
    labels = df['is_recommended'].astype(int).values

    model = build_new_model()
    print("2. Đang huấn luyện AI...")
    model.fit(
        x=[user_ids, game_ids],
        y=labels,
        epochs=3,
        batch_size=512,
        validation_split=0.2
    )
    model.save(MODEL_PATH)
    print(f"✅ Đã lưu bộ não AI thành công tại: {MODEL_PATH}")

def update_from_feedback():
    """Tự động học hỏi lại từ lịch sử tương tác ở file test.py"""
    print("\n==================================================")
    print("🔄 [TRAIN.PY] ĐANG TỰ ĐỘNG HỌC LẠI TỪ PHẢN HỒI MỚI...")
    print("==================================================")

    if not os.path.exists(MODEL_PATH):
        print("⚠️ Chưa có bộ não AI, tiến hành khởi tạo...")
        train_initial()

    if not os.path.exists(FEEDBACK_FILE):
        print("ℹ️ Chưa có phản hồi mới nào từ người dùng để học thêm.")
        return

    fb_df = pd.read_csv(FEEDBACK_FILE)
    if fb_df.empty:
        print("ℹ️ File phản hồi rỗng, không cần cập nhật.")
        return

    print(f"📚 Phát hiện {len(fb_df)} đánh giá mới từ buổi tư vấn! Đang nạp vào bộ não AI...")
    model = load_model(MODEL_PATH)

    user_ids = (fb_df['user_id'].values % MAX_USERS).astype(np.int32)
    game_ids = (fb_df['app_id'].values % MAX_GAMES).astype(np.int32)
    labels = fb_df['is_recommended'].astype(int).values

    # Cho AI học kỹ lại trên tập dữ liệu phản hồi mới (5 vòng lặp để ghi nhớ sâu)
    history = model.fit(
        x=[user_ids, game_ids],
        y=labels,
        epochs=5,
        batch_size=max(1, len(fb_df)),
        verbose=1
    )

    # Lưu đè lại bộ não đã thông minh hơn
    model.save(MODEL_PATH)
    print("✅ Bộ não AI đã cập nhật kiến thức mới thành công!")

    # Ghi nối phản hồi mới vào file recommendations.csv gốc để lưu trữ lâu dài
    if os.path.exists(RECOMMENDATIONS_FILE):
        with open(RECOMMENDATIONS_FILE, mode='a', newline='', encoding='utf-8') as f:
            for _, row in fb_df.iterrows():
                f.write(f"{int(row['app_id'])},0,0,2026-10-21,{bool(row['is_recommended'])},1.0,{int(row['user_id'])},feedback\n")

    # Xóa file tạm sau khi đã học xong để lần sau không học trùng lặp
    os.remove(FEEDBACK_FILE)
    print("🧹 Đã dọn dẹp bộ nhớ đệm phản hồi. Quá trình tự học hoàn tất!")

if __name__ == '__main__':
    # Nếu có file phản hồi mới từ test.py gửi sang -> Tự động học thêm
    # Nếu chưa có mô hình -> Train mới từ đầu
    if os.path.exists(FEEDBACK_FILE):
        update_from_feedback()
    elif not os.path.exists(MODEL_PATH):
        train_initial()
    else:
        print("Bộ não AI đã sẵn sàng. Nếu muốn ép train lại từ đầu, hãy xóa file bo_nao_ai_steam.keras rồi chạy lại.")
        chon = input("Bạn có muốn train lại từ đầu ngay bây giờ không? (y/n): ").strip().lower()
        if chon == 'y':
            train_initial()
