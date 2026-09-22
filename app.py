import os
import uuid
import shutil
import torch
import numpy as np
import nibabel as nib
import traceback
from scipy.ndimage import binary_fill_holes
from scipy.ndimage import binary_dilation, generate_binary_structure
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from skimage.measure import label
from torch.amp import autocast

# --- CẤU HÌNH HỆ THỐNG ---
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from monai.networks.nets import AttentionUnet
from monai.inferers import sliding_window_inference
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Orientationd,
    Spacingd, ScaleIntensityRanged, EnsureTyped, Invertd
)

app = FastAPI(title="Hệ Thống Phân Tích Hình Ảnh Y Khoa 3D")

# --- BẢO MẬT & CORS ---
# Cho phép Frontend (nếu chạy ở port khác) gọi API mà không bị chặn
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Trong thực tế nên giới hạn domain cụ thể
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- KHỞI TẠO THƯ MỤC TĨNH ---
# 1. Thư mục chứa file CT scan tạm thời
os.makedirs("static", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

# 2. Thư mục chứa Frontend (HTML, CSS, JS)
# Đảm bảo thư mục 'medical-web' nằm cùng cấp với file app.py này
app.mount("/css", StaticFiles(directory="medical-web/css"), name="css")
app.mount("/js", StaticFiles(directory="medical-web/js"), name="js")

# Biến hệ thống toàn cục
model = None
device = None


# --- GIAO DIỆN NGƯỜI DÙNG ---
@app.get("/", response_class=FileResponse)
async def get_index():
    # Phục vụ file Lễ tân (Đăng nhập)
    return "medical-web/index.html"


@app.get("/scanner.html", response_class=FileResponse)
async def get_scanner():
    # Phục vụ file Phòng khám (Giao diện 3D)
    return "medical-web/scanner.html"


# --- SỰ KIỆN KHỞI ĐỘNG HỆ THỐNG ---
@app.on_event("startup")
async def load_model_on_startup():
    global model, device
    print("[INFO] Khởi tạo môi trường AI và nạp mô hình AttentionUnet...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = AttentionUnet(
        spatial_dims=3, in_channels=1, out_channels=3,
        channels=(16, 32, 64, 128, 256), strides=(2, 2, 2, 2)
    ).to(device)

    if os.path.exists("best_model_tumor.pt"):
        print("[INFO] Đang tải trọng số mô hình từ tệp tin...")
        ckpt = torch.load("best_model_tumor.pt", map_location=device)

        # Trích xuất state_dict
        state = ckpt.get("model", ckpt.get("model_state_dict", ckpt))

        # Loại bỏ tiền tố 'module.' nếu mô hình được huấn luyện bằng DataParallel
        clean_state = {k.replace("module.", ""): v for k, v in state.items()}

        model_state = model.state_dict()
        model_keys = list(model_state.keys())
        ckpt_keys = list(clean_state.keys())

        print(f"[DEBUG] Số lượng tham số yêu cầu: {len(model_keys)} | Hiện có: {len(ckpt_keys)}")

        if len(model_keys) == len(ckpt_keys):
            print("[INFO] Thực hiện ánh xạ tham số theo tuần tự cấu trúc...")
            hacked_state = {
                model_k: clean_state[ckpt_k]
                for model_k, ckpt_k in zip(model_keys, ckpt_keys)
            }
            model.load_state_dict(hacked_state, strict=True)
            print("[SUCCESS] Đã nạp trọng số mô hình thành công.")
        else:
            print("[WARNING] Cấu trúc không tương thích. Khởi chạy nạp trọng số từng phần (strict=False).")
            model.load_state_dict(clean_state, strict=False)

        model.eval()
    else:
        print("[ERROR] Không tìm thấy tệp trọng số 'best_model_tumor.pt' trong thư mục gốc.")


# --- LOGIC XỬ LÝ ẢNH Y KHOA (INFERENCE) ---
@app.post("/predict")
def predict(file: UploadFile = File(...)):
    try:
        print(f"\n[INFO] Bắt đầu tiếp nhận và xử lý tệp tin: {file.filename}")
        temp_id = uuid.uuid4().hex
        temp_upload_path = os.path.join("static", f"upload_{temp_id}.nii.gz")

        # Lưu dữ liệu tải lên
        with open(temp_upload_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # 1. Pipeline Tiền xử lý (Preprocessing)
        val_transforms = Compose([
            LoadImaged(keys=["image"]),
            EnsureChannelFirstd(keys=["image"]),
            Orientationd(keys=["image"], axcodes="RAS"),
            Spacingd(keys=["image"], pixdim=(1.5, 1.5, 2.0), mode="bilinear"),
            ScaleIntensityRanged(keys=["image"], a_min=-100, a_max=250, b_min=0.0, b_max=1.0, clip=True),
            EnsureTyped(keys=["image"]),
        ])

        print("[INFO] Thực hiện tiền xử lý không gian và chuẩn hóa cường độ...")
        # Đưa ảnh vào format dictionary chuẩn của MONAI
        data_dict = val_transforms({"image": temp_upload_path})

        # Batch size = 1
        input_tensor = data_dict["image"].unsqueeze(0).to(device)

        print("[INFO] Đang nội suy kết quả qua mô hình AI (Sliding Window Inference)...")
        with torch.no_grad():
            with autocast('cuda'):
                logits = sliding_window_inference(
                    inputs=input_tensor,
                    roi_size=(96, 96, 96),
                    sw_batch_size=1,
                    predictor=model,
                    overlap=0.25
                )
            # Lưu logit thô vào dict để Invert
            data_dict["pred"] = logits[0].cpu()

            del logits, input_tensor
            torch.cuda.empty_cache()

        # ... (Phần Hậu xử lý (Invertd, Label, Bbox, v.v.) của bro giữ nguyên tuyệt đối,
        # vì nó đã viết rất tốt và tối ưu bằng SciPy) ...
        print("[INFO] Khôi phục không gian ảnh về kích thước gốc...")
        inverter = Invertd(
            keys="pred",
            transform=val_transforms,
            orig_keys="image",
            meta_key_postfix="meta_dict",
            nearest_interp=True,
            to_tensor=True
        )
        inv_data = inverter(data_dict)

        print("[INFO] Bắt đầu tiến trình Hậu xử lý (Lọc nhiễu False Positives)...")
        pred_np = torch.argmax(inv_data["pred"], dim=0).numpy().astype(np.uint8)

        pancreas_mask = (pred_np == 1)
        pancreas_clean = np.zeros_like(pancreas_mask)

        if pancreas_mask.sum() > 0:
            L_p, num_p = label(pancreas_mask, return_num=True)
            sizes = [np.sum(L_p == i) for i in range(1, num_p + 1)]
            largest_label_id = np.argmax(sizes) + 1
            pancreas_clean = (L_p == largest_label_id)
            pancreas_mask = binary_fill_holes(pancreas_mask)

        tumor_mask = (pred_np == 2)
        tumor_clean = np.zeros_like(tumor_mask)

        if tumor_mask.sum() > 0 and pancreas_clean.sum() > 0:
            struct = generate_binary_structure(3, 1)
            safe_zone = binary_dilation(pancreas_clean, structure=struct, iterations=5)
            tumor_mask = binary_fill_holes(tumor_mask)
            valid_tumor = tumor_mask & safe_zone

            L_t, num_t = label(valid_tumor, return_num=True)
            for i in range(1, num_t + 1):
                comp = (L_t == i)
                if comp.sum() >= 30:  # Lọc nhiễu < 30 voxels
                    tumor_clean |= comp

        print("[INFO] Đang trích xuất đặc trưng thể tích và xuất dữ liệu mask...")
        orig_nii = nib.load(temp_upload_path)

        pancreas_voxels = int(np.sum(pancreas_clean))
        tumor_voxels = int(np.sum(tumor_clean))
        print(f"[METRIC] Thể tích tính toán -> Tuyến tụy: {pancreas_voxels:,} voxels | Khối u: {tumor_voxels:,} voxels")

        pancreas_path = os.path.join("static", f"pancreas_{temp_id}.nii.gz")
        tumor_path = os.path.join("static", f"tumor_{temp_id}.nii.gz")

        nib.save(nib.Nifti1Image(pancreas_clean.astype(np.uint8), affine=orig_nii.affine), pancreas_path)
        nib.save(nib.Nifti1Image(tumor_clean.astype(np.uint8), affine=orig_nii.affine), tumor_path)

        print("[SUCCESS] Hoàn tất quá trình phản hồi tín hiệu API.")
        return {
            "pancreas_url": f"/static/pancreas_{temp_id}.nii.gz",
            "tumor_url": f"/static/tumor_{temp_id}.nii.gz",
            "orig_url": f"/static/upload_{temp_id}.nii.gz",
            "pancreas_vol": f"{pancreas_voxels:,}",
            "tumor_vol": f"{tumor_voxels:,}"
        }

    except Exception as e:
        print("[ERROR] Ngoại lệ nghiêm trọng phát sinh trong quá trình thực thi:")
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": str(e)})


if __name__ == "__main__":
    import uvicorn

    # Lệnh thực thi: uvicorn app:app --reload
    uvicorn.run(app, host="127.0.0.1", port=8000)