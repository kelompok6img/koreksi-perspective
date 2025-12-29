import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io
import matplotlib.pyplot as plt
from scipy import ndimage
import time

# Konfigurasi halaman
st.set_page_config(
    page_title="Koreksi Perspektif Dokumen & Plat Kendaraan",
    page_icon="📄",
    layout="wide"
)

# Fungsi preprocessing
def preprocess_image(image, resize_factor=0.5, denoise_strength=10, normalize=True):
    """Preprocess gambar: resize, denoise, normalization"""
    # Konversi ke grayscale jika perlu
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # Resize
    height, width = gray.shape
    new_width = int(width * resize_factor)
    new_height = int(height * resize_factor)
    resized = cv2.resize(gray, (new_width, new_height))
    

    # Denoise
    kernel_size = max(3, min(denoise_strength * 2 + 1, 101))
    if kernel_size % 2 == 0:
        kernel_size += 1
    denoised = cv2.medianBlur(resized, kernel_size)
    
    if normalize:
        clip_limit = max(1.0, denoise_strength / 5.0)
        clahe = cv2.createCLAHE(
            clipLimit=clip_limit,
            tileGridSize=(8,8)
        )
        normalized = clahe.apply(denoised)
        
        # **TAMBAHKAN SHARPENING untuk kontras lebih**
        kernel_sharpen = np.array([[-1,-1,-1],
                                   [-1, 9,-1],
                                   [-1,-1,-1]])
        normalized = cv2.filter2D(normalized, -1, kernel_sharpen)
    else:
        normalized = denoised
    
    return normalized, new_width, new_height

def detect_document_contour(image, method="canny", canny_thresh1=50, canny_thresh2=150):
    """Deteksi kontur dokumen dengan berbagai metode"""
    if method == "canny":
        edges = cv2.Canny(image, canny_thresh1, canny_thresh2)
        edges = cv2.dilate(edges, None, iterations=1)
        edges = cv2.erode(edges, None, iterations=1)
    elif method == "adaptive":
        thresh = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                      cv2.THRESH_BINARY, 11, 2)
        edges = cv2.bitwise_not(thresh)
    else:  # threshold
        _, thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        edges = thresh
    
    # Temukan kontur
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Urutkan kontur berdasarkan area (dari terbesar)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    
    return edges, contours

def detect_license_plate(image):
    """Deteksi khusus untuk plat kendaraan"""
    # Konversi ke grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image.copy()
    
    # Apply bilateral filter untuk mengurangi noise tapi menjaga edges
    filtered = cv2.bilateralFilter(gray, 11, 17, 17)
    
    # Edge detection
    edged = cv2.Canny(filtered, 30, 200)
    
    # Temukan kontur
    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    # Urutkan kontur berdasarkan area
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
    
    return edged, contours

def find_corners_from_contour(contour, epsilon_factor=0.02):
    """Temukan 4 sudut dari kontur dokumen"""
    epsilon = epsilon_factor * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)
    
    # Jika tidak mendapatkan 4 titik, coba pendekatan lain
    if len(approx) != 4:
        # Gunakan convex hull
        hull = cv2.convexHull(contour)
        epsilon = 0.1 * cv2.arcLength(hull, True)
        approx = cv2.approxPolyDP(hull, epsilon, True)
        
        # Jika masih tidak 4 titik, gunakan bounding rectangle
        if len(approx) != 4:
            rect = cv2.minAreaRect(contour)
            box = cv2.boxPoints(rect)
            approx = box.astype(np.int32)  # PERBAIKAN DI SINI: ganti np.int0 dengan astype(np.int32)
    
    # Pastikan ada tepat 4 titik
    if len(approx) > 4:
        # Pilih 4 titik terluar
        approx = approx[np.argsort(approx[:, 0, 0] + approx[:, 0, 1])[:4]]
    
    return approx.reshape(4, 2)

def order_corners(pts):
    """Urutkan sudut: kiri atas, kanan atas, kanan bawah, kiri bawah"""
    # Susun berdasarkan sumbu x
    x_sorted = pts[np.argsort(pts[:, 0]), :]
    
    # Pisahkan kiri dan kanan
    left_most = x_sorted[:2, :]
    right_most = x_sorted[2:, :]
    
    # Urutkan kiri (y ascending)
    left_most = left_most[np.argsort(left_most[:, 1]), :]
    tl, bl = left_most[0], left_most[1]
    
    # Urutkan kanan (y ascending)
    right_most = right_most[np.argsort(right_most[:, 1]), :]
    tr, br = right_most[0], right_most[1]
    
    return np.array([tl, tr, br, bl], dtype="float32")

def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")

    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]      # top-left
    rect[2] = pts[np.argmax(s)]      # bottom-right

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]   # top-right
    rect[3] = pts[np.argmax(diff)]   # bottom-left

    return rect


def perspective_transform(image, corners):
    rect = order_points(corners)
    (tl, tr, br, bl) = rect

    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxWidth = int(max(widthA, widthB))

    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxHeight = int(max(heightA, heightB))

    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]
    ], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    return warped, M


def calculate_quality_score(image):
    """Hitung skor kualitas gambar (0-100)"""
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # Hitung kontras (standar deviasi)
    contrast = np.std(gray)
    
    # Hitung sharpness (variance of Laplacian)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    sharpness = laplacian.var()
    
    # Hitung brightness (rata-rata)
    brightness = np.mean(gray)
    
    # Normalisasi skor
    contrast_score = min(contrast / 50 * 100, 100)
    sharpness_score = min(sharpness / 100 * 100, 100)
    brightness_score = 100 - abs(brightness - 128) / 128 * 100
    
    # Skor total (bobot)
    total_score = 0.4 * contrast_score + 0.4 * sharpness_score + 0.2 * brightness_score
    
    return int(total_score), {
        "contrast": int(contrast_score),
        "sharpness": int(sharpness_score),
        "brightness": int(brightness_score)
    }

def find_best_contour(contours, min_area=1000, max_aspect_ratio=10, min_aspect_ratio=0.1):
    """Cari kontur terbaik berdasarkan kriteria tertentu"""
    best_contour = None
    best_score = 0
    
    for contour in contours:
        # Hitung area
        area = cv2.contourArea(contour)
        
        # Skip jika area terlalu kecil
        if area < min_area:
            continue
            
        # Dapatkan bounding rectangle
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        box = box.astype(np.int32)
        
        # Hitung aspect ratio
        width = np.linalg.norm(box[1] - box[0])
        height = np.linalg.norm(box[2] - box[1])
        
        if width == 0 or height == 0:
            continue
            
        aspect_ratio = max(width, height) / min(width, height)
        
        # Kriteria untuk plat kendaraan atau dokumen
        # Plat kendaraan biasanya memiliki aspect ratio 2:1 sampai 4:1
        # Dokumen biasanya memiliki aspect ratio mendekati A4 (1.414)
        if min_aspect_ratio < aspect_ratio < max_aspect_ratio:
            # Hitung skor berdasarkan kekompakan
            perimeter = cv2.arcLength(contour, True)
            if perimeter > 0:
                compactness = 4 * np.pi * area / (perimeter * perimeter)
                score = area * (1 - abs(compactness - 1))  # Semakin dekat 1 semakin baik
                
                if score > best_score:
                    best_score = score
                    best_contour = contour
    
    return best_contour

# UI Aplikasi
st.title("Koreksi Perspektif Dokumen & Plat Kendaraan")
st.markdown("Aplikasi ini mengoreksi perspektif gambar dokumen, kuitansi, dan plat kendaraan yang miring menggunakan computer vision.")

# --- CSS untuk memperbesar area upload ---
st.markdown(
    """
    <style>
    .stFileUploader {
        border: 2px dashed #666 !important;
        padding: 50px !important;
        border-radius: 15px !important;
        text-align: center !important;
        background-color: #2b2b2b !important;
        transition: 0.3s ease-in-out;
    }
    .stFileUploader:hover {
        border-color: #888 !important;
        background-color: #3b3b3b !important;
    }
    .stFileUploader label {
        font-size: 1.1rem !important;
        color: #ccc !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# --- Upload Gambar (Drag & Drop di Tengah) ---
uploaded_file = st.file_uploader(
    "⬆️ Drag & Drop gambar Anda di sini (JPG / PNG)",
    type=["jpg", "jpeg", "png"],
    key="file_uploader"
)

    
# Konten utama
if uploaded_file is not None:
    # Baca gambar
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    original_image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    original_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    
    
    # Sidebar
    with st.sidebar:
        st.header("Pengaturan")
    
        st.subheader("Jenis Objek")
        object_type = st.radio("Pilih jenis objek:", ["Dokumen/Kuitansi", "Plat Kendaraan"])
    
        st.subheader("Preprocessing")
        resize_factor = st.slider("Faktor Resize", 0.1, 1.0, 0.5, 0.05)
        denoise_strength = st.slider("Kekuatan Denoise", 1, 20, 10, 1)
        normalize = st.checkbox("Normalisasi Kontras", value=True)
    
        st.subheader("Deteksi Kontur")
        if object_type == "Dokumen/Kuitansi":
            contour_method = st.selectbox("Metode Deteksi", ["canny", "adaptive", "threshold"])
            
            if contour_method == "canny":
                canny_thresh1 = st.slider("Canny Threshold 1", 1, 100, 50, 1)
                canny_thresh2 = st.slider("Canny Threshold 2", 50, 200, 150, 5)
            else:
               canny_thresh1 = 50
               canny_thresh2 = 150
        else:
            contour_method = "license_plate"
    
        epsilon_factor = st.slider("Faktor Epsilon (deteksi sudut)", 0.01, 0.1, 0.02, 0.01)
    
        st.subheader("Filter Kontur")
        min_area = st.slider("Area Minimum Kontur", 100, 50000, 5000, 100)
        if object_type == "Plat Kendaraan":
            min_aspect_ratio = st.slider("Rasio Aspek Minimum", 0.1, 5.0, 1.5, 0.1)
            max_aspect_ratio = st.slider("Rasio Aspek Maksimum", 1.0, 10.0, 5.0, 0.1)
        else:
            min_aspect_ratio = 0.5
            max_aspect_ratio = 3.0
    
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Gambar Asli")
        st.image(original_rgb, use_container_width=True)
        st.caption(f"Ukuran: {original_image.shape[1]} x {original_image.shape[0]} piksel")
    
    if uploaded_file is not None:
        with st.spinner("Memproses gambar..."):
            # Progress bar
            progress_bar = st.progress(0)
            
            # 1. Preprocessing
            progress_bar.progress(10)
            time.sleep(0.1)
            
            preprocessed, new_width, new_height = preprocess_image(
                original_image, resize_factor, denoise_strength, normalize
            )
            
            # 2. Deteksi kontur berdasarkan jenis objek
            progress_bar.progress(30)
            time.sleep(0.1)
            
            if object_type == "Plat Kendaraan":
                edges, contours = detect_license_plate(original_image)
            else:
                edges, contours = detect_document_contour(
                    preprocessed, contour_method, canny_thresh1, canny_thresh2
                )
            
            if len(contours) == 0:
                st.error("Tidak ada kontur yang terdeteksi! Coba ubah parameter deteksi.")
                st.stop()
            
            # 3. Cari kontur terbaik
            progress_bar.progress(50)
            time.sleep(0.1)
            
            best_contour = find_best_contour(contours, min_area, max_aspect_ratio, min_aspect_ratio)
            
            if best_contour is None:
                st.warning("Tidak ditemukan kontur yang memenuhi kriteria. Menggunakan kontur terbesar.")
                best_contour = contours[0]
            
            # Skala ulang koordinat ke ukuran asli
            scale_x = original_image.shape[1] / new_width
            scale_y = original_image.shape[0] / new_height
            
            # Temukan sudut dari kontur terbaik
            corners = find_corners_from_contour(best_contour, epsilon_factor)
            
            # Skala koordinat ke ukuran asli
            corners_scaled = corners.copy()
            corners_scaled[:, 0] = corners_scaled[:, 0] * scale_x
            corners_scaled[:, 1] = corners_scaled[:, 1] * scale_y
            
            # 4. Transformasi perspektif
            progress_bar.progress(70)
            time.sleep(0.1)
            
            warped, transform_matrix = perspective_transform(original_image, corners_scaled)
            warped_rgb = cv2.cvtColor(warped, cv2.COLOR_BGR2RGB)
            
            # 5. Hitung skor kualitas
            progress_bar.progress(90)
            time.sleep(0.1)
            
            quality_score, score_details = calculate_quality_score(warped)
            
            progress_bar.progress(100)
            time.sleep(0.2)
        
        with col2:
            st.subheader("Hasil Koreksi")
            st.image(warped_rgb, use_container_width=True)
            st.caption(f"Ukuran: {warped.shape[1]} x {warped.shape[0]} piksel")
        
        # Visualisasi proses
        st.divider()
        st.subheader("📊 Visualisasi Proses")
        
        col3, col4, col5 = st.columns(3)
        
        with col3:
            # Gambar dengan kontur
            contour_overlay = original_rgb.copy()
            
            # Gambar semua kontur
            cv2.drawContours(contour_overlay, contours, -1, (255, 0, 0), 2)
            
            # Gambar kontur terbaik
            cv2.drawContours(contour_overlay, [corners_scaled.astype(int)], -1, (0, 255, 0), 3)
            
            # Gambar dengan sudut
            for i, corner in enumerate(corners_scaled):
                x, y = corner.astype(int)
                cv2.circle(contour_overlay, (x, y), 10, (255, 0, 0), -1)
                cv2.putText(contour_overlay, str(i+1), (x-10, y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            st.image(contour_overlay, caption="Deteksi Kontur & Sudut", use_container_width=True)
        
        with col4:
            # Edge detection result
            if object_type == "Plat Kendaraan":
                edges_display = edges
            else:
                edges_display = cv2.resize(edges, (original_rgb.shape[1] // 2, original_rgb.shape[0] // 2))
            st.image(edges_display, caption="Hasil Deteksi Tepi", use_container_width=True, clamp=True)
        
        with col5:
            # Grafik skor kualitas
            fig, ax = plt.subplots(figsize=(6, 4))
            categories = ['Kontras', 'Ketajaman', 'Kecerahan']
            scores = [score_details['contrast'], score_details['sharpness'], score_details['brightness']]
            
            colors = ['#4CAF50', '#2196F3', '#FF9800']
            bars = ax.bar(categories, scores, color=colors)
            ax.set_ylim([0, 100])
            ax.set_ylabel('Skor (0-100)')
            ax.set_title('Analisis Kualitas Gambar')
            
            # Tambahkan nilai di atas bar
            for bar, score in zip(bars, scores):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 2,
                       f'{score}', ha='center', va='bottom')
            
            # Tambahkan indikator warna untuk total skor
            if quality_score >= 80:
                total_color = '#4CAF50'
            elif quality_score >= 60:
                total_color = '#FFC107'
            else:
                total_color = '#F44336'
            
            st.pyplot(fig)
            st.metric("Skor Kualitas Total", f"{quality_score}/100", 
                     delta="Baik" if quality_score >= 70 else "Perlu Perbaikan",
                     delta_color="normal" if quality_score >= 70 else "inverse")
        
        # Informasi kontur
        st.divider()
        st.subheader("ℹ️ Informasi Deteksi")
        
        info_col1, info_col2 = st.columns(2)
        
        with info_col1:
            # Hitung properti kontur terbaik
            area = cv2.contourArea(best_contour)
            perimeter = cv2.arcLength(best_contour, True)
            
            # Hitung aspect ratio
            rect = cv2.minAreaRect(best_contour)
            box = cv2.boxPoints(rect)
            box = box.astype(np.int32)
            width = np.linalg.norm(box[1] - box[0])
            height = np.linalg.norm(box[2] - box[1])
            aspect_ratio = max(width, height) / min(width, height) if min(width, height) > 0 else 0
            
            st.markdown(f"""
            **Properti Kontur:**
            - **Area**: {int(area)} piksel²
            - **Perimeter**: {int(perimeter)} piksel
            - **Rasio Aspek**: {aspect_ratio:.2f}
            - **Jumlah Sudut**: {len(corners)}
            - **Jumlah Kontur Ditemukan**: {len(contours)}
            """)
        
        with info_col2:
            st.markdown(f"""
            **Status Deteksi:**
            - **Jenis Objek**: {object_type}
            - **Metode Deteksi**: {contour_method.upper() if object_type == "Dokumen/Kuitansi" else "LICENSE PLATE"}
            - **Status Sudut**: {'✅ Lengkap' if len(corners) == 4 else '⚠️ Tidak lengkap'}
            - **Transformasi**: {'✅ Berhasil' if warped is not None else '❌ Gagal'}
            - **Waktu Proses**: {time.strftime('%H:%M:%S')}
            """)
        
        # Penjelasan proses
        st.divider()
        st.subheader("📝 Penjelasan Proses")
        
        exp_col1, exp_col2 = st.columns(2)
        
        with exp_col1:
            st.markdown("""
            **Penjelasan Sidebar:**
            1. **Faktor Resize**: Semakin besar nilainya semakin akurat deteksi tepinya
            2. **Kekuatan Denoise**: Menghilangkan noise/butiran pada gambar. Noise dapat mengganggu deteksi tepian.
            3. **Normalisasi Kontras**: Menyeimbangkan kecerahan dan kontras gambar secara otomatis.
            4. **Metode Canny**: Deteksi berdasarkan perubahan intensitas warna yang tajam.
            5. **Metode Adaptive** : Otomatis yang menyesuaikan dengan area sekitar.
            6. **Metode Threshold** : Otomatis yang mencari perbedaan terbaik antara objek dan background.
            4. **Canny Threshold 1**: Ambang batas minimum untuk mendeteksi tepian.
            5. **Canny Threshold 2**: Ambang batas maksimum untuk mendeteksi tepian.
            6. **Faktor Epsilon**: Menentukan seberapa presisi sudut-sudut dokumen dideteksi.
            7. **Area Minimum Kontur**: Filter berdasarkan ukuran kontur.
            """)
        
        with exp_col2:
            if object_type == "Plat Kendaraan":
                st.markdown("""
                **Tips untuk Plat Kendaraan:**
                - Pastikan plat terlihat jelas dalam gambar
                - Background kontras dengan warna plat
                - Hindari refleksi cahaya berlebihan
                - Posisi kamera sejajar dengan plat
                - Gunakan pencahayaan yang cukup
                """)
            else:
                st.markdown("""
                **Tips untuk Dokumen:**
                - Letakkan dokumen pada permukaan datar
                - Hindari bayangan pada dokumen
                - Ambil gambar dari atas (bird's eye view)
                - Pastikan seluruh dokumen terlihat
                - Gunakan background yang kontras
                """)
        
        # Tombol download
        st.divider()
        st.subheader("📥 Download Hasil")
        
        # Konversi ke format yang bisa didownload
        is_success, buffer = cv2.imencode(".jpg", warped)
        if is_success:
            img_bytes = buffer.tobytes()
            
            col_dl1, col_dl2, col_dl3 = st.columns(3)
            
            with col_dl1:
                st.download_button(
                    label="Download Hasil (JPG)",
                    data=img_bytes,
                    file_name="hasil_koreksi_perspektif.jpg",
                    mime="image/jpeg",
                    use_container_width=True
                )
            
            with col_dl2:
                # Juga simpan sebagai PNG
                is_success_png, buffer_png = cv2.imencode(".png", warped)
                if is_success_png:
                    png_bytes = buffer_png.tobytes()
                    st.download_button(
                        label="Download Hasil (PNG)",
                        data=png_bytes,
                        file_name="hasil_koreksi_perspektif.png",
                        mime="image/png",
                        use_container_width=True
                    )
            
            with col_dl3:
                # Download transform matrix
                matrix_str = f"Transform Matrix untuk {object_type}:\n" + str(transform_matrix)
                st.download_button(
                    label="Download Matriks Transformasi",
                    data=matrix_str,
                    file_name="transform_matrix.txt",
                    mime="text/plain",
                    use_container_width=True
                )

else:
    # Tampilan awal jika belum upload gambar
    st.info("Silakan upload gambar untuk memulai.")
    
    col_intro1, col_intro2 = st.columns(2)
    
    with col_intro1:
        st.markdown("""
        **Fitur Utama:**
        - Upload gambar dokumen/kuitansi/plat kendaraan
        - Deteksi khusus untuk plat kendaraan
        - Preprocessing: resize, denoise, normalization
        - Deteksi kontur dan ekstraksi sudut
        - Transformasi perspektif 4 titik
        - Visualisasi before-after
        - Overlay kontur dan sudut
        - Download hasil
        """)
    
    with col_intro2:
        st.markdown("""
        **Contoh gambar yang cocok:**
        - Plat nomor kendaraan (motor/mobil)
        - Dokumen (KTP, SIM, paspor)
        - Kuitansi dan struk belanja
        - Kartu nama
        - Surat resmi
        
        **Tips untuk hasil terbaik:**
        1. Pilih jenis objek yang sesuai
        2. Atur parameter deteksi sesuai kebutuhan
        3. Pastikan objek terlihat jelas
        4. Background kontras dengan objek
        5. Pencahayaan cukup dan merata
        """)

# Footer
st.divider()
st.caption("Aplikasi Koreksi Perspektif Dokumen dan Plat Kendaraan | Dibuat dengan Streamlit & OpenCV")