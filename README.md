# Phát Hiện Cạnh Dựa Trên Logic Mờ Kết Hợp Tăng Tốc Với GPU

Thuật toán **phát hiện cạnh bằng logic mờ**, tăng tốc với **CUDA** cho ảnh **Full HD** (1920x1080). Bài tập lớn môn Logic Mờ, Trường ĐH Công nghệ, ĐHQGHN, cải tiến so với nghiên cứu gốc (CPU, ảnh SD).

## Tính năng
- Dùng logic mờ với mặt nạ 3x3 cho ảnh mịn và nhiễu.
- Tăng tốc GPU, xử lý nhanh ảnh Full HD.
- Kết quả sắc nét hơn ảnh SD.

## Yêu cầu
- Ubuntu, CUDA 11.0+, OpenCV 4.5.0+, CMake.

## Cài đặt
1. Tải mã nguồn:
   ```bash
   git clone https://github.com/Kienk66-it/GPU_Fuzzy_Logic_Edge_Detection.git
   cd <GPU_Fuzzy_Logic_Edge_Detection>
   ```
2. Cài công cụ:
   ```bash
   sudo apt-get install nvidia-cuda-toolkit libopencv-dev cmake
   ```
3. Biên dịch:
   ```bash
   mkdir build && cd build
   cmake ..
   make
   ```

## Sử dụng
1. Đặt ảnh vào `images/`.
2. Chạy:
   ```bash
   ./Fuzzy_Edge_Detection images/<tên-ảnh>.jpg
   ```
   Ví dụ:
   ```bash
   ./Fuzzy_Edge_Detection images/sample.jpg
   ```

## Kết quả
- Xử lý Full HD trong chưa đầy 1 giây, nhanh hơn CPU.
- Precision 0.4064, ít cạnh giả hơn Canny.
- Cạnh sắc nét, tốt với ảnh nhiễu.
