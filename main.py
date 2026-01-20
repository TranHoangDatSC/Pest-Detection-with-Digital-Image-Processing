import os
import cv2
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import pygame 

INPUT_FOLDER = "../Final Project DIP - Pest Detection/input"
BASE_OUTPUT_FOLDER = "processed_images"

PARAMS_ADULT = {
    "NAME": "Adults",
    "COLOR": (0, 255, 0),        # Green BBox

    # --- LỌC NHIỄU VÀ LÀM MỜ ---
    "MEDIAN_KSIZE": 21, 

    # --- NHỊ PHÂN HÓA THÍCH ỨNG ---
    "ADAPTIVE_BLOCK_SIZE": 31, 
    "ADAPTIVE_C_VALUE": 5, 

    # --- XỬ LÝ HÌNH THÁI HỌC (MORPHOLOGY) ---
    "MORPH_KERNEL_SIZE": 5,        
    "MORPH_ITERATIONS": 1,         
    "MORPH_OP": cv2.MORPH_OPEN,    

    # --- LẤP ĐẦY VÀ ĐÓNG LỖ (FILL) (Không sử dụng trong hàm fill_and_smooth_outlines) ---
    "FILL_KSIZE": (8, 8), 
    "FILL_ITERATIONS": 5, 

    # --- NGƯỠNG ĐỐI TƯỢNG ---
    "MIN_PEST_AREA": 1500,   # 1500 cho Input1, 900 cho Input2
    "MAX_PEST_AREA_PERCENT": 0.99 # Ngưỡng tối đa 99% diện tích ảnh
}

# Tham số chung
GAUSSIAN_KSIZE = (3, 3) 
GAUSSIAN_SIGMA = 0


# CÁC HÀM XỬ LÝ ẢNH CƠ BẢN (TRUYỀN TRỰC TIẾP ẢNH ARRAY)
def conversion(image_name, input_folder=INPUT_FOLDER):
    """Đọc ảnh màu và chuyển sang Grayscale."""
    input_path = os.path.join(input_folder, image_name)
    image = cv2.imread(input_path)
    if image is None: 
        print(f"[ERROR] Không tìm thấy hoặc không đọc được file: {input_path}")
        return None, None
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return gray_image, image

def gaussian(image, ksize=GAUSSIAN_KSIZE, sigma=GAUSSIAN_SIGMA):
    """Làm mịn ảnh bằng Gaussian Blur."""
    if image is None: return None
    blur = cv2.GaussianBlur(image, ksize, sigma)
    return blur

def median_filter(image, kernel_size):
    """Lọc nhiễu bằng Median Filter."""
    if image is None: return None
    if kernel_size % 2 == 0 or kernel_size < 3:
        kernel_size = 5
    median = cv2.medianBlur(image, kernel_size)
    return median

def fill_and_smooth_outlines(image, morph_params):
    """Lấp đầy viền bằng Morphology Closing."""
    if image is None: return None
    
    kernel_fill = np.ones((morph_params["MORPH_KERNEL_SIZE"], morph_params["MORPH_KERNEL_SIZE"]), np.uint8)
    filled_image = cv2.morphologyEx(
        image, 
        cv2.MORPH_CLOSE, 
        kernel_fill, 
        iterations=morph_params["MORPH_ITERATIONS"]
    )
    return filled_image

# HÀM XỬ LÝ CHÍNH (Segmentation + Counting + Bounding Box)
def segmentation_and_count(processed_image, detection_params, original_color_image):
    """Thực hiện Segmentation, Connected Components và đếm/vẽ Bounding Box.
    Đã bao gồm cơ chế lọc BBox dẹt, chiếm hết chiều rộng."""
    
    # Kiểm tra ảnh đầu vào
    if processed_image is None or processed_image.size == 0: 
        return 0, original_color_image, None, None
    
    # TÍNH TOÁN NGƯỠNG TỐI ĐA (GUARDRAIL)
    h, w = processed_image.shape[:2]
    total_area = h * w
    max_area_threshold = total_area * detection_params["MAX_PEST_AREA_PERCENT"]

    # --- THAM SỐ LỌC BBOX DẸT, RỘNG ---
    MAX_VALID_ASPECT_RATIO = 20.0 # Tỷ lệ khung hình tối đa chấp nhận được (rộng/cao)
    MAX_HEIGHT_PERCENT = 0.05      # Chiều cao BBox tối đa cho đối tượng dẹt (5% chiều cao ảnh)
    max_valid_height = h * MAX_HEIGHT_PERCENT
    
    # 1. Adaptive Thresholding
    thresh = cv2.adaptiveThreshold(
        processed_image, 
        255, 
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, 
        detection_params["ADAPTIVE_BLOCK_SIZE"], 
        detection_params["ADAPTIVE_C_VALUE"]
    )

    # 2. Morphology Closing
    closing = fill_and_smooth_outlines(thresh, detection_params)
    if closing is None: 
        return 0, original_color_image, thresh, None

    # 3. Connected Components (Đếm)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        closing, 5, cv2.CV_32S
    )
    
    valid_pests = 0
    
    # Lặp qua tất cả đối tượng tìm được (bỏ qua nhãn 0 - nền)
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w_box = stats[i, cv2.CC_STAT_WIDTH]
        h_box = stats[i, cv2.CC_STAT_HEIGHT]

        # LỌC 1: LỌC DIỆN TÍCH TỐI ĐA (GUARDRAIL 99%)
        if area >= max_area_threshold:
            print(f"[DEBUG] OBJECT REJECTED (Area): Area {area} exceeds max threshold ({max_area_threshold:.0f})!")
            continue 
        
        # LỌC 2: LỌC BBOX DẸT, RỘNG (GIẢI QUYẾT LỖI MÉP DƯỚI)
        if w_box > 0 and h_box > 0:
            aspect_ratio = w_box / h_box
            # Nếu tỷ lệ rộng/cao quá lớn (dẹt) VÀ chiều cao quá nhỏ (<= 5% ảnh) VÀ chiếm gần hết chiều rộng
            if aspect_ratio >= MAX_VALID_ASPECT_RATIO and h_box <= max_valid_height and w_box >= 0.95 * w:
                print(f"[DEBUG] OBJECT REJECTED (Aspect Ratio/Height): AR={aspect_ratio:.1f}, H={h_box} (w={w_box}). Likely border noise.")
                continue 

        # LỌC TỐI THIỂU: Chỉ giữ lại đối tượng lớn (Sâu trưởng thành)
        if area >= detection_params["MIN_PEST_AREA"]:
            valid_pests += 1
            
            # Vẽ Bounding Box
            cv2.rectangle(
                original_color_image, 
                (x, y), 
                (x + w_box, y + h_box), 
                detection_params["COLOR"], 
                2
            )

    print(f"[INFO] {detection_params['NAME']} detected: {valid_pests}")
    return valid_pests, original_color_image, thresh, closing 

# HÀM XỬ LÝ TỪNG ẢNH ĐƠN LẺ (SINGLE PASS)
def process_single_image(image_file_name, input_base_folder, output_base_folder):
    """Xử lý từng ảnh, hiển thị và lưu kết quả."""
    file_name_without_ext = os.path.splitext(image_file_name)[0]
    DYNAMIC_OUTPUT_FOLDER = os.path.join(output_base_folder, f"processed_{file_name_without_ext}")
    os.makedirs(DYNAMIC_OUTPUT_FOLDER, exist_ok=True)
    print(f"\n[SETUP] Bắt đầu xử lý: {image_file_name}")

    # 1. Grayscale
    gray_image, original_image = conversion(image_file_name, input_folder=input_base_folder)
    if gray_image is None:
        return 0, DYNAMIC_OUTPUT_FOLDER 
    
    # >>> LƯU ẢNH TRUNG GIAN 1: Grayscale
    gray_output_path = os.path.join(DYNAMIC_OUTPUT_FOLDER, f"01_Grayscale_{file_name_without_ext}.png")
    cv2.imwrite(gray_output_path, gray_image)
    print(f"[SAVE] Ảnh Grayscale đã lưu tại: {gray_output_path}")

    detection_image = original_image.copy() 

    # --- PASS: Đếm Sâu Trưởng Thành (ADULTS) ---
    # Chuỗi xử lý ảnh: Grayscale -> Gaussian -> Median Filter
    blur_image = gaussian(gray_image) 
    
    # >>> LƯU ẢNH TRUNG GIAN 2: Gaussian Blur
    blur_output_path = os.path.join(DYNAMIC_OUTPUT_FOLDER, f"02_Gaussian_{file_name_without_ext}.png")
    cv2.imwrite(blur_output_path, blur_image)
    print(f"[SAVE] Ảnh Gaussian đã lưu tại: {blur_output_path}")
    
    median_image = median_filter(blur_image, PARAMS_ADULT["MEDIAN_KSIZE"])
    
    # >>> LƯU ẢNH TRUNG GIAN 3: Median Filter
    median_output_path = os.path.join(DYNAMIC_OUTPUT_FOLDER, f"03_Median_{file_name_without_ext}.png")
    cv2.imwrite(median_output_path, median_image)
    print(f"[SAVE] Ảnh Median Filter đã lưu tại: {median_output_path}")

    # Segmentation và đếm (giờ trả về 4 giá trị)
    adult_count, final_image, thresh_image, closing_image = segmentation_and_count(
        median_image, 
        PARAMS_ADULT, 
        detection_image
    )

    # >>> LƯU ẢNH TRUNG GIAN 4: Adaptive Threshold
    if thresh_image is not None:
        thresh_output_path = os.path.join(DYNAMIC_OUTPUT_FOLDER, f"04_Thresh_{file_name_without_ext}.png")
        cv2.imwrite(thresh_output_path, thresh_image)
        print(f"[SAVE] Ảnh Threshold đã lưu tại: {thresh_output_path}")
    
    # >>> LƯU ẢNH TRUNG GIAN 5: Morphology Closing
    if closing_image is not None:
        closing_output_path = os.path.join(DYNAMIC_OUTPUT_FOLDER, f"05_Closing_{file_name_without_ext}.png")
        cv2.imwrite(closing_output_path, closing_image)
        print(f"[SAVE] Ảnh Morphology Closing đã lưu tại: {closing_output_path}")
    
    # --- Hiển thị kết quả ---
    plt.figure(figsize=(10, 8))
    
    final_image_rgb = cv2.cvtColor(final_image, cv2.COLOR_BGR2RGB) 
    
    plt.imshow(final_image_rgb)
    plt.title(f"Pest Detected: {adult_count} (Green BBox)")
    plt.axis('off')
    
    plt.show()

    # Lưu ảnh kết quả cuối cùng
    output_path = os.path.join(DYNAMIC_OUTPUT_FOLDER, f"06_Result_{file_name_without_ext}_{adult_count}.png")
    cv2.imwrite(output_path, final_image)
    print(f"[INFO] Ảnh kết quả cuối cùng đã được lưu tại: {output_path}")
    
    return adult_count, DYNAMIC_OUTPUT_FOLDER

# Main Execution
if __name__ == "__main__":
    image_file_name_test = "input1.jpg" 
    process_single_image(image_file_name_test, INPUT_FOLDER, BASE_OUTPUT_FOLDER)