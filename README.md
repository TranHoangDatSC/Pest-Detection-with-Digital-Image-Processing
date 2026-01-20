cat <<EOF > README.md

# PEST DETECTION WITH DIGITAL IMAGE PROCESSING (OpenCV)

Dự án sử dụng các kỹ thuật xử lý ảnh cổ điển để phát hiện và đếm số lượng bọ phấn trắng (Whiteflies) trên lá cây.

## 1. QUY TRÌNH XỬ LÝ (PIPELINE)

Dự án triển khai chuỗi xử lý tuần tự nhằm tối ưu khả năng nhận diện:

1. **Tiền xử lý:** Chuyển sang Grayscale và áp dụng bộ lọc Gaussian Blur (3x3), Median Filter (21x21) để khử nhiễu bề mặt lá.
2. **Phân đoạn (Segmentation):** Sử dụng **Adaptive Thresholding (Gaussian C)** để xử lý vấn đề ánh sáng không đồng đều và gradient bóng đổ.
3. **Hình thái học (Morphology):** Áp dụng phép đóng (Closing) với kernel 5x5 để lấp đầy lỗ hổng bên trong thân sâu hại.
4. **Phân tích thành phần liên thông:** Sử dụng 'connectedComponentsWithStats' để xác định tọa độ và diện tích từng cá thể.

## 2. KẾT QUẢ & THẢO LUẬN

- **Độ chính xác:** Hệ thống nhận diện tốt các cá thể trưởng thành, khoanh vùng bằng Bounding Box xanh lục.
- **Hạn chế:** Thuật toán nhạy cảm với tham số diện tích tối thiểu (MIN_PEST_AREA). Kết quả thực nghiệm cho thấy sự phân kỳ giữa diện tích 900px và 1500px tùy thuộc vào độ phân giải ảnh đầu vào (thử input1 và input2)
- **Kết luận:** Phương pháp này hiệu quả cho các bài toán kiểm soát môi trường cố định nhưng cần nâng cấp lên Deep Learning để tăng tính vững chãi (Robustness).

Tác giả: Trần Hoàng Đạt & Nông Thị Nhật Lệ.
EOF
