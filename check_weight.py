import torch

# Khám nghiệm file tạ
print("🔍 ĐANG NỘI SOI FILE CHECKPOINT...")
ckpt = torch.load("model_old/best_model_tumor.pt", map_location="cpu")

# 1. Xem ba-lô có mấy ngăn
print(f"👉 Các ngăn trong ba-lô: {ckpt.keys()}")

# 2. Móc ruột ra xem
if "model" in ckpt:
    state = ckpt["model"]
    print("👉 Đã tìm thấy ngăn 'model'!")
elif "model_state_dict" in ckpt:
    state = ckpt["model_state_dict"]
    print("👉 Đã tìm thấy ngăn 'model_state_dict'!")
else:
    state = ckpt
    print("👉 Khả năng lưu thẳng state_dict!")

# 3. In thử 5 cái tên cốt lõi trong ruột xem nó tên gì
print("\n🔑 5 CHÌA KHÓA ĐẦU TIÊN TRONG FILE:")
for key in list(state.keys())[:5]:
    print(" -", key)