<div align="center">

# Phân Vùng Khối U Tuyến Tụy Trên Ảnh CT Bằng Học Sâu
### (3D Attention U-Net & Active Learning)

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![MONAI](https://img.shields.io/badge/MONAI-Medical_AI-5C2D91?style=for-the-badge)](https://monai.io/)
[![FastAPI](https://img.shields.io/badge/FastAPI-Framework-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)

</div>

---

## 📌 Giới thiệu nghiên cứu
Hệ thống tự động phân đoạn nhu mô tuyến tụy và khối u bệnh lý từ ảnh chụp cắt lớp vi tính (CT) 3D vùng bụng. Dự án giải quyết các thách thức lớn trong ảnh y tế như ranh giới mờ, độ tương phản mô mềm thấp và mất cân bằng lớp dữ liệu cực đoan. 

Bằng cách tích hợp chiến lược **Học chủ động (Active Learning)** dựa trên đo lường độ bất định, pipeline giúp giảm thiểu tối đa thời gian và chi phí gán nhãn thủ công từ bác sĩ chuyên khoa.

---

## ✨ Tính năng nổi bật
*  **Mô hình 3D Attention U-Net**: Sử dụng cơ chế Cổng chú ý (Attention Gates - AGs) giúp mạng nơ-ron tự động tập trung vào khu vực tổn thương và loại bỏ nhiễu từ các cơ quan lân cận trong khoang bụng.
*  **Xử lý mất cân bằng lớp**: Tích hợp hàm mất mát **Tversky Loss** để phạt nặng các lỗi bỏ sót khối u (False Negative).
*  **Pipeline Active Learning**: Định lượng độ bất định thông qua **TTA (Test-Time Augmentation)** và chỉ số **Shannon Entropy**, tự động trích xuất Top-K mẫu dữ liệu khó nhất để ưu tiên gán nhãn.
*  **Ứng dụng End-to-End**: Hệ thống tích hợp máy chủ API (**FastAPI**) và giao diện Web tích hợp bộ công cụ **NiiVue**, cho phép bác sĩ tải file `.nii.gz` và tương tác với khối dựng 3D/2D trực tiếp trên trình duyệt.

---

## 🏗️ Kiến trúc & Luồng xử lý (Pipeline)
1. **Tiền xử lý (Pre-processing)**: Cắt dải Windowing cản quang mô mềm (Soft-Tissue Windowing), chuẩn hóa chỉ số Hounsfield Unit (HU), và tái lấy mẫu không gian 3D.
2. **Huấn luyện mô hình cơ sở**: Huấn luyện 3D Attention U-Net trên tập dữ liệu gán nhãn ban đầu (*Seed set*).
3. **Vòng lặp Active Learning**:
   * Chạy suy luận TTA đa hướng trên tập ẩn nhãn (`Unlabeled Pool`).
   * Tính toán bản đồ độ bất định Shannon Entropy tại vùng quan tâm.
   * Sắp xếp và trích chọn Top-K mẫu có độ bất định cao nhất để cập nhật vào vòng huấn luyện tiếp theo.
4. **Triển khai Web Service**: Trích xuất trọng số tối ưu (*Best Checkpoint*) sang FastAPI Backend để phục vụ chẩn đoán thời gian thực.

---

## 📊 Kết quả thực nghiệm
Đánh giá định lượng trên bộ dữ liệu chuẩn quốc tế **Medical Segmentation Decathlon (MSD) Task 07 Pancreas**:

| Đối tượng phân đoạn | Hệ số tương đồng Dice (DSC) |
| :--- | :---: |
|  **Nhu mô Tuyến tụy (Pancreas)** | **0.8133** |
|  **Khối u Tuyến tụy (Tumor)** | **0.6297** |

> 💡 **Hiệu quả Active Learning**: Mô hình đạt ngưỡng hội tụ tối ưu mà không cần gán nhãn toàn bộ tập dữ liệu, giảm thiểu đáng kể khối lượng công việc chú thích y khoa từ chuyên gia.

---

## 🛠️ Công nghệ sử dụng (Tech Stack)
* **Core Deep Learning**: PyTorch, MONAI framework.
* **Xử lý ảnh y tế**: SimpleITK, Nibabel, NiiVue Engine, ITK-Snap.
* **Backend & Service**: FastAPI, Uvicorn, NumPy, SciPy.
* **Frontend**: HTML5/CSS3, JavaScript (NiiVue WebGL Integration).

---

## 📁 Cấu trúc thư mục dự án
```text
Pancreas_Web/
├── .venv/                        # Môi trường ảo Python
├── backend/                      # Các mô đun xử lý backend API
├── medical-web/                  # Giao diện ứng dụng Web
│   ├── css/                      # File định kiểu trang web
│   ├── js/                       # Mã nguồn JavaScript & bộ dựng 3D NiiVue
│   ├── index.html                # Trang chủ hệ thống
│   └── scanner.html              # Giao diện xem và phân tích ảnh CT
├── static/                       # Tài nguyên tĩnh của hệ thống
├── templates/                    # File giao diện HTML render từ backend
├── app.py                        # Khởi chạy ứng dụng Web Server
├── best_model_tumor.pt          # Trọng số mô hình dự đoán khối u tối ưu
├── check_weight.py              # Script kiểm tra và load trọng số mô hình
├── Kythuattcdl.png              # Sơ đồ kiến trúc / kỹ thuật dự án
└── main.py                      # Entrypoint chính xử lý suy luận
```
## 🖥️ Giao diện phân đoạn
<img width="1919" height="897" alt="Segmentation-interface" src="https://github.com/user-attachments/assets/ad69b7b9-2def-4ec9-b922-5d69388cd782" />

<img width="1919" height="1079" alt="Segmentation-interface-tumor" src="https://github.com/user-attachments/assets/ed502bf2-6196-44b1-afa8-af4cb4f9dfbd" />

---

## 🚀 Hướng dẫn cài đặt & Chạy ứng dụng

### 1. Khởi tạo môi trường
```bash
# Clone dự án từ GitHub
git clone [https://github.com/your-username/Pancreas_Web.git](https://github.com/your-username/Pancreas_Web.git)
cd Pancreas_Web

# Tạo và kích hoạt môi trường ảo
python -m venv .venv

# Trên Windows:
.venv\Scripts\activate

# Trên macOS/Linux:
source .venv/bin/activate

# Cài đặt các thư viện phụ thuộc
pip install -r requirements.txt
```

### 2. Khởi chạy Backend Server
```bash
python main.py
# Hoặc khởi chạy qua app.py
python app.py
```

### 3. Trải nghiệm giao diện Web
* Mở trình duyệt web và truy cập vào đường dẫn local server (mặc định: `http://127.0.0.1:8000`).
* Hoặc mở trực tiếp file `medical-web/scanner.html` để chọn file NIfTI (`.nii.gz`) và trải nghiệm trình tương tác ảnh CT 3D.

---

<div align="center">

**Tác giả**: Nguyễn Thanh Đa (MSSV: B2203436)  
*Khoa Hệ thống Thông tin — Trường Công nghệ Thông tin & Truyền thông, Đại học Cần Thơ.*

</div>
