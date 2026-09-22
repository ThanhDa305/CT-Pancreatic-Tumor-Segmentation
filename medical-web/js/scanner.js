import { Niivue } from "https://esm.sh/@niivue/niivue";

// Kiểm tra xác thực phiên làm việc (Session Authentication)
const token = localStorage.getItem('session_token');
if (!token) {
    alert("Vui lòng xác thực phiên làm việc (Đăng nhập) để tiếp tục thao tác.");
    window.location.href = 'index.html';
} else {
    initApp();
}

function initApp() {
    // 1. Khởi tạo Engine render Niivue
    const nv = new Niivue({
        show3Dcrosshair: true,
        crosshairColor: [0.5, 0.5, 0.5, 0.8], // Đổi thành màu Xám nhạt, hơi trong suốt
        backColor: [0, 0, 0, 1],
        multiplanarForceRender: true
    });
    nv.attachTo('canvas');
    nv.setSliceType(nv.sliceTypeMultiplanar);

    // ==============================================================================
    // 2. CẤU HÌNH BỘ ĐIỀU HƯỚNG LÁT CẮT (SLICE NAVIGATOR)
    // ==============================================================================
    const sliceSlider = document.getElementById('sliceSlider');
    const sliceInput = document.getElementById('sliceInput');
    const maxSliceLabel = document.getElementById('maxSlice');

    let totalSlices = 1; // Tổng số lát cắt trục Z

    // Hàm đồng bộ vị trí Z từ UI vào không gian 3D của Niivue
    const updateCrosshairZ = (sliceValue) => {
        if (nv.volumes.length > 0 && totalSlices > 1) {
            const zFrac = (sliceValue - 1) / (totalSlices - 1);
            const currentPos = nv.scene.crosshairPos;
            nv.scene.crosshairPos = [currentPos[0], currentPos[1], zFrac];
            nv.drawScene();
        }
    };

    sliceSlider.addEventListener('input', (e) => {
        const val = parseInt(e.target.value, 10);
        sliceInput.value = val;
        updateCrosshairZ(val);
    });

    sliceInput.addEventListener('change', (e) => {
        let val = parseInt(e.target.value, 10);
        if (isNaN(val) || val < 1) val = 1;
        if (val > totalSlices) val = totalSlices;
        e.target.value = val;
        sliceSlider.value = val;
        updateCrosshairZ(val);
    });

    // Đồng bộ ngược: Khi người dùng lăn chuột trên vùng Canvas -> Cập nhật UI
    nv.onLocationChange = function(data) {
        if (data && data.vox && totalSlices > 1) {
            const currentSlice = data.vox[2] + 1;
            sliceSlider.value = currentSlice;
            sliceInput.value = currentSlice;
        }
    };

    // ==============================================================================
    // 3. CẤU HÌNH GIAO DIỆN LỚP HIỂN THỊ (LAYER TOGGLES)
    // ==============================================================================
    document.getElementById('togglePancreas').addEventListener('change', (e) => {
        if (nv.volumes.length > 1) nv.setOpacity(1, e.target.checked ? 0.4 : 0.0);
    });

    document.getElementById('toggleTumor').addEventListener('change', (e) => {
        if (nv.volumes.length > 2) nv.setOpacity(2, e.target.checked ? 0.9 : 0.0);
    });

    document.getElementById('toggleCT').addEventListener('change', (e) => {
        if (nv.volumes.length > 0) nv.setOpacity(0, e.target.checked ? 1.0 : 0.0);
    });

    // ==============================================================================
    // 4. KIỂM SOÁT LUỒNG DỮ LIỆU & GỌI API (INFERENCE TRIGGER)
    // ==============================================================================
    document.getElementById('btnModeMPR').addEventListener('click', () => nv.setSliceType(nv.sliceTypeMultiplanar));
    document.getElementById('btnModeVR').addEventListener('click', () => nv.setSliceType(nv.sliceTypeRender));

    document.getElementById('btnLogout').addEventListener('click', () => {
        localStorage.removeItem('session_token');
        window.location.href = 'http://127.0.0.1:8000';
    });

    document.getElementById('btnRunAI').addEventListener('click', async () => {
        const fileInput = document.getElementById('fileInput');
        const status = document.getElementById('status');
        const btn = document.getElementById('btnRunAI');
        const volPanel = document.getElementById('volumeInfo');

        if (!fileInput.files[0]) {
            // Lấy ngôn ngữ hiện tại từ biến global bên HTML (nếu có), fallback về 'vi'
            const isEn = window.currentLang === 'en';
            alert(isEn ? "Please upload a NIfTI (.nii.gz) file to proceed." : "Vui lòng nạp tệp dữ liệu chuẩn NIfTI (.nii.gz) để tiếp tục.");
            return;
        }

        // --- CẬP NHẬT TRẠNG THÁI (ĐA NGÔN NGỮ) ---
        status.style.borderLeftColor = "#ffaa00";
        status.style.color = "#ffaa00";
        status.setAttribute('data-vi', "Đang tiến hành nội suy AI. Vui lòng giữ kết nối...");
        status.setAttribute('data-en', "Processing AI inference. Please remain connected...");
        if(window.updateLanguage) window.updateLanguage(); // Kích hoạt render ngôn ngữ

        btn.disabled = true;
        volPanel.style.display = "none";

        const formData = new FormData();
        formData.append('file', fileInput.files[0]);

        try {
            const res = await fetch('http://127.0.0.1:8000/predict', { method: 'POST', body: formData });
            const data = await res.json();

            if (res.ok && data.pancreas_url) {
                // --- THÀNH CÔNG (ĐA NGÔN NGỮ) ---
                status.style.borderLeftColor = "#06d6a0";
                status.style.color = "#06d6a0";
                status.setAttribute('data-vi', "Hoàn tất giải mã và phân vùng hình thái.");
                status.setAttribute('data-en', "Decoding and morphological segmentation completed.");
                if(window.updateLanguage) window.updateLanguage();

                document.getElementById('volPancreas').innerText = data.pancreas_vol;
                document.getElementById('volTumor').innerText = data.tumor_vol;
                volPanel.style.display = 'flex';

                if (data.performance) {
                    document.getElementById('performanceInfo').style.display = 'flex';
                    document.getElementById('perfTotal').innerText = data.performance.total_time + "s";
                    document.getElementById('perfPre').innerText = data.performance.pre_processing + "s";
                    document.getElementById('perfInf').innerText = data.performance.inference + "s";
                    document.getElementById('perfSlices').innerText = data.performance.num_slices;
                }

                document.getElementById('placeholderGrid').style.display = 'none';
                document.getElementById('layerToggles').style.display = 'flex';

                const volumeList = [
                    { url: 'http://127.0.0.1:8000' + data.orig_url, opacity: 1, colormap: "gray" },
                    { url: 'http://127.0.0.1:8000' + data.pancreas_url, opacity: 0.4, colormap: "red", cal_min: 0, cal_max: 1 },
                    { url: 'http://127.0.0.1:8000' + data.tumor_url, opacity: 0.9, colormap: "green", cal_min: 0, cal_max: 1 }
                ];

                await nv.loadVolumes(volumeList);
                nv.setSliceType(nv.sliceTypeMultiplanar);

                if (nv.volumes.length > 0) {
                    totalSlices = nv.volumes[0].hdr.dims[3];

                    sliceSlider.max = totalSlices;
                    sliceInput.max = totalSlices;
                    maxSliceLabel.textContent = totalSlices;

                    const midSlice = Math.floor(totalSlices / 2);
                    sliceSlider.value = midSlice;
                    sliceInput.value = midSlice;
                    updateCrosshairZ(midSlice);
                }

            } else {
                throw new Error(data.error || "Phát sinh lỗi ngoại lệ từ Backend System.");
            }
        } catch (e) {
            // --- THẤT BẠI (ĐA NGÔN NGỮ) ---
            status.style.borderLeftColor = "#ef233c";
            status.style.color = "#ef233c";
            status.setAttribute('data-vi', "Mất kết nối API: " + e.message);
            status.setAttribute('data-en', "API Connection Lost: " + e.message);
            if(window.updateLanguage) window.updateLanguage();
        } finally {
            btn.disabled = false;
        }
    });
}