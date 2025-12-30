kelompok 6 : koreksi perspective dokumen dan plat kendaraan
nama kelompok :
1.Ferdi satrio (23010068)
2.Radithya Alhaq (23010007)
3.Dicky Triharyadi (23010021)
4.Haifan Muhammad Izza (24019004)

penjelasan singkat mengenai project:
KOREKSI PERPECTIVE DOKUMEN DAN PLAT KENDARAAN
Aplikasi Computer Vision berbasis Streamlit dan OpenCV untuk mengoreksi perspektif gambar dokumen dan plat kendaraan yang miring atau terdistorsi. 
Aplikasi ini secara otomatis mendeteksi kontur objek utama, mengekstrak empat sudut terluar, lalu menerapkan four-point perspective transform agar objek terlihat lurus dan proporsional.
Project ini cocok untuk digitalisasi dokumen, preprocessing OCR, serta analisis citra berbasis visi komputer.

✨ Fitur Utama
⦁	Upload gambar (JPG / PNG)
⦁	Deteksi objek:
  •  Dokumen / kuitansi
  •  Plat kendaraan
• Preprocessing gambar:
  1. Resize
  2. Denoise
  3. Normalisasi kontras (CLAHE)
  4. Sharpening
• Deteksi kontur dengan berbagai metode:
  1. Canny Edge Detection
  2. Adaptive Threshold
  3. Otsu Threshold
• Deteksi dan pengurutan 4 sudut otomatis
• Koreksi perspektif (warp transform)
• Visualisasi proses:
  1. Kontur & sudut
  2. Hasil edge detection
  3. Perbandingan before-after
• Analisis kualitas hasil:
  1. Kontras
  2. Ketajaman
  3. Kecerahan
  4. Skor total (0–100)
• Download hasil:
  1. JPG
  2. PNG
  3. Matriks transformasi perspektive
     
ALUR APLIKASI:
1. Upload gambar
2. Preprocessing
  • Konversi grayscale
  • Resize untuk efisiensi
  • Denoise dan peningkatan kontras
3. Deteksi tepi & kontur
  • Metode disesuaikan dengan jenis objek
4. Seleksi kontur terbaik
  • Berdasarkan area, rasio aspek, dan kekompakan
5. Ekstraksi 4 sudut
6. Transformasi perspektif
7. Evaluasi kualitas hasil
8. Download output

TECH STACK YANG DIGUNAKAN :
• python - bahasa pemrograman. lalu untuk library python:
  1. numpy - operasi numerik
  2. streamlit - antarmuka web interaktif
  3. pillow - pemrosesan gambar
  4. opencv - pengolahan citra & computer vision
  5. matplotlib - visualisasi grafik
  6. scipy - operasi pendukung image processing
