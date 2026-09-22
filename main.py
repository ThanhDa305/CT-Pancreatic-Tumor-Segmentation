import os
import uuid
import shutil
import torch
import numpy as np
import nibabel as nib
import traceback
import time  # <--- CHÈN THÊM ĐỂ ĐO HIỆU NĂNG
from scipy.ndimage import binary_fill_holes
from scipy.ndimage import binary_dilation, generate_binary_structure
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from skimage.measure import label
from torch.amp import autocast

# --- MONAI IMPORTS ---
from monai.networks.nets import AttentionUnet
from monai.inferers import sliding_window_inference
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Orientationd,
    Spacingd, ScaleIntensityRanged, EnsureTyped, Invertd
)

# --- CẤU HÌNH HỆ THỐNG ---
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

MODEL_WEIGHTS = "best_model_tumor.pt"
STATIC_DIR = "static"

app = FastAPI(title="Pancreas & Tumor Segmentation API", version="1.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

os.makedirs(STATIC_DIR, exist_ok=True)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

model = None
device = None


@app.on_event("startup")
async def load_model_on_startup():
    global model, device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AttentionUnet(
        spatial_dims=3, in_channels=1, out_channels=3,
        channels=(16, 32, 64, 128, 256), strides=(2, 2, 2, 2)
    ).to(device)

    if os.path.exists(MODEL_WEIGHTS):
        ckpt = torch.load(MODEL_WEIGHTS, map_location=device)
        state = ckpt.get("model", ckpt.get("model_state_dict", ckpt))
        clean_state = {k.replace("module.", ""): v for k, v in state.items()}
        model.load_state_dict(clean_state, strict=False)
        model.eval()
        print("[INFO] Hệ thống đã sẵn sàng.")


@app.post("/predict")
def predict(file: UploadFile = File(...)):
    start_total = time.perf_counter()
    try:
        print(f"\n{'=' * 60}")
        print(f"[INFO] Nhận yêu cầu dự đoán: {file.filename}")

        session_id = uuid.uuid4().hex
        temp_upload_path = os.path.join(STATIC_DIR, f"upload_{session_id}.nii.gz")

        # --- GIAI ĐOẠN 0: LƯU FILE ---
        start_load = time.perf_counter()
        with open(temp_upload_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        t_load = time.perf_counter() - start_load

        # --- GIAI ĐOẠN 1: TIỀN XỬ LÝ (O(N) ---
        start_pre = time.perf_counter()
        print("[INFO] Đang thực hiện Resampling & Normalization...")
        val_transforms = Compose([
            LoadImaged(keys=["image"]),
            EnsureChannelFirstd(keys=["image"]),
            Orientationd(keys=["image"], axcodes="RAS"),
            Spacingd(keys=["image"], pixdim=(1.5, 1.5, 2.0), mode="bilinear"),
            ScaleIntensityRanged(keys=["image"], a_min=-100, a_max=240, b_min=0.0, b_max=1.0, clip=True),
            EnsureTyped(keys=["image"]),
        ])

        data_dict = val_transforms({"image": temp_upload_path})
        input_tensor = data_dict["image"].unsqueeze(0).to(device)
        t_pre = time.perf_counter() - start_pre

        # --- GIAI ĐOẠN 2: NỘI SUY (MODEL INFERENCE) ---
        start_inf = time.perf_counter()
        print("[INFO] Đang chạy Model Inference trên GPU...")
        with torch.no_grad():
            with autocast('cuda'):
                logits = sliding_window_inference(
                    inputs=input_tensor,
                    roi_size=(128, 128, 96),
                    sw_batch_size=1,
                    predictor=model,
                    overlap=0.5
                )
            data_dict["pred"] = logits[0].cpu()

        del logits, input_tensor
        torch.cuda.empty_cache()
        t_inf = time.perf_counter() - start_inf

        # --- GIAI ĐOẠN 3: HẬU XỬ LÝ & LƯU KẾT QUẢ ---
        start_post = time.perf_counter()
        print("[INFO] Đang thực hiện Inverse Transform & Morphological Post-processing...")
        inverter = Invertd(
            keys="pred", transform=val_transforms, orig_keys="image",
            meta_key_postfix="meta_dict", nearest_interp=True, to_tensor=True
        )
        inv_data = inverter(data_dict)

        probs_np = torch.softmax(inv_data["pred"], dim=0).numpy()
        pred_np = np.argmax(probs_np, axis=0).astype(np.uint8)

        # Ngưỡng và lọc nhiễu
        TUMOR_CONFIDENCE_THRESHOLD = 0.75
        MIN_TUMOR_VOLUME = 100

        tumor_mask = (pred_np == 2) & (probs_np[2] >= TUMOR_CONFIDENCE_THRESHOLD)
        pancreas_mask = (pred_np == 1) | ((pred_np == 2) & (probs_np[2] < TUMOR_CONFIDENCE_THRESHOLD))

        # Hậu xử lý hình thái
        pancreas_clean = np.zeros_like(pancreas_mask)
        if pancreas_mask.sum() > 0:
            L_p, num_p = label(pancreas_mask, return_num=True)
            largest_label_id = np.argmax([np.sum(L_p == i) for i in range(1, num_p + 1)]) + 1
            pancreas_clean = (L_p == largest_label_id)
            pancreas_clean = binary_fill_holes(pancreas_clean)

        tumor_clean = np.zeros_like(tumor_mask)
        if tumor_mask.sum() > 0 and pancreas_clean.sum() > 0:
            struct = generate_binary_structure(3, 1)
            safe_zone = binary_dilation(pancreas_clean, structure=struct, iterations=5)
            valid_tumor = (binary_fill_holes(tumor_mask)) & safe_zone
            L_t, num_t = label(valid_tumor, return_num=True)
            for i in range(1, num_t + 1):
                comp = (L_t == i)
                if comp.sum() >= MIN_TUMOR_VOLUME:
                    tumor_clean |= comp

        # Lưu file NIfTI
        orig_nii = nib.load(temp_upload_path)
        pancreas_path = os.path.join(STATIC_DIR, f"pancreas_{session_id}.nii.gz")
        tumor_path = os.path.join(STATIC_DIR, f"tumor_{session_id}.nii.gz")
        nib.save(nib.Nifti1Image(pancreas_clean.astype(np.uint8), affine=orig_nii.affine), pancreas_path)
        nib.save(nib.Nifti1Image(tumor_clean.astype(np.uint8), affine=orig_nii.affine), tumor_path)

        t_post = time.perf_counter() - start_post
        total_time = time.perf_counter() - start_total

        # --- IN BÁO CÁO CHI TIẾT RA CONSOLE ---
        num_slices = orig_nii.shape[-1]
        print(f"\n--- BÁO CÁO HIỆU NĂNG ---")
        print(f"Số lát cắt (Slices): {num_slices}")
        print(f"T_load : {t_load:.2f}s")
        print(f"T_pre  : {t_pre:.2f}s (Độ phức tạp O(N))")
        print(f"T_inf  : {t_inf:.2f}s")
        print(f"T_post : {t_post:.2f}s")
        print(f"T_TOTAL: {total_time:.2f}s")
        print(f"Kết quả -> Tụy: {np.sum(pancreas_clean):,} voxels | U: {np.sum(tumor_clean):,} voxels")
        print(f"{'=' * 60}")

        return {
            "pancreas_url": f"/{STATIC_DIR}/pancreas_{session_id}.nii.gz",
            "tumor_url": f"/{STATIC_DIR}/tumor_{session_id}.nii.gz",
            "orig_url": f"/{STATIC_DIR}/upload_{session_id}.nii.gz",
            "pancreas_vol": f"{int(np.sum(pancreas_clean)):,}",
            "tumor_vol": f"{int(np.sum(tumor_clean)):,}",
            "performance": {
                "total_time": round(total_time, 2),
                "pre_processing": round(t_pre, 2),
                "inference": round(t_inf, 2),
                "num_slices": num_slices
            }
        }

    except Exception as e:
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": str(e)})


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)