# -*- coding: utf-8 -*-

"""
Evaluation & Metrics.

Tính toán các chỉ số đánh giá cho toàn bộ hệ thống dự đoán nguy cơ té ngã, 
đáp ứng chính xác yêu cầu mục 10 trong báo cáo ý tưởng:
  - Classification/Risk Metrics: Accuracy, Precision, Recall, F1, AUC
  - Early Warning Metrics: Time-to-warning, False alarm rate
  - Regression/Depth Metrics: MAE, RMSE (nếu có ground-truth)
  
Đặc biệt lưu ý: Trong bài toán an toàn, Recall quan trọng hơn Precision (bỏ sót nguy hiểm nghiêm trọng hơn báo động nhầm).
"""

import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("Evaluator")

def evaluate_classification(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray = None):
    """
    Đánh giá mô hình dự đoán phân loại nguy cơ (SNN, KAN, MLP).
    y_true, y_pred: 1D array chứa nhãn lớp (0=safe, 1=warning, 2=danger)
    y_prob: Xác suất dự đoán để tính AUC (tùy chọn)
    """
    # 1. Các chỉ số tổng quát
    acc = accuracy_score(y_true, y_pred)
    # Tính average='macro' để đánh giá đều trên các class
    precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
    recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    
    # 2. Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # 3. Tính AUC nếu có xác suất
    auc = None
    if y_prob is not None:
        try:
            # y_prob phải là ma trận [n_samples, n_classes]
            auc = roc_auc_score(y_true, y_prob, multi_class='ovr')
        except ValueError:
            pass # Bỏ qua nếu dữ liệu không đủ class để tính ROC
            
    # Đặc biệt báo cáo Recall của lớp nguy hiểm nhất (ví dụ class 2 - danger)
    danger_recall = None
    if cm.shape == (3, 3):
        # Recall class 2 = TP / (TP + FN)
        danger_recall = cm[2, 2] / (cm[2, 0] + cm[2, 1] + cm[2, 2]) if sum(cm[2]) > 0 else 0
        
    logger.info("=== KẾT QUẢ ĐÁNH GIÁ PHÂN LOẠI ===")
    logger.info(f"Accuracy : {acc:.4f}")
    logger.info(f"Precision: {precision:.4f}")
    logger.info(f"Recall   : {recall:.4f} (Đặc biệt quan trọng)")
    logger.info(f"F1-Score : {f1:.4f}")
    if auc is not None:
        logger.info(f"ROC AUC  : {auc:.4f}")
    if danger_recall is not None:
        logger.info(f"Recall riêng cho class Danger (Mức 2): {danger_recall:.4f}")
    logger.info("Confusion Matrix:")
    logger.info(f"\n{cm}")
    
    return {
        "accuracy": acc, "precision": precision, "recall": recall, 
        "f1": f1, "auc": auc, "danger_recall": danger_recall
    }


def evaluate_early_warning(predictions_over_time: list, ground_truth_events: list, fps: float = 30.0):
    """
    Đánh giá hệ thống theo chỉ số cảnh báo sớm (Time-to-warning) và Tỷ lệ báo động giả (False alarm rate).
    
    Args:
        predictions_over_time: List nhãn dự đoán cho từng frame liên tiếp
        ground_truth_events: Vị trí (frame index) mà sự cố thực sự xảy ra
        fps: Số khung hình trên giây
    """
    if not ground_truth_events:
        logger.warning("Không có event ground truth nào để đánh giá early warning.")
        return None
        
    ttw_list = []
    false_alarms = 0
    total_alarms = 0
    
    # Tìm kiếm các cảnh báo trong dự đoán (giả sử nhãn > 0 là có cảnh báo)
    preds = np.array(predictions_over_time)
    alarm_indices = np.where(preds > 0)[0]
    total_alarms = len(alarm_indices)
    
    # Gom cụm các cảnh báo liên tiếp thành 1 event báo động
    alarm_events = []
    if total_alarms > 0:
        current_event_start = alarm_indices[0]
        for i in range(1, len(alarm_indices)):
            if alarm_indices[i] - alarm_indices[i-1] > fps: # Nếu cách nhau quá 1s thì tính là cảnh báo mới
                alarm_events.append(current_event_start)
                current_event_start = alarm_indices[i]
        alarm_events.append(current_event_start)
    
    # So sánh các alarm event với ground_truth_events
    for gt_frame in ground_truth_events:
        # Tìm cảnh báo xảy ra trước sự kiện thực tế trong khoảng thời gian hợp lý (ví dụ: max 5 giây trước đó)
        valid_alarms = [a for a in alarm_events if 0 < (gt_frame - a) <= 5 * fps]
        
        if valid_alarms:
            # Lấy cảnh báo sớm nhất hợp lệ
            earliest_alarm = min(valid_alarms)
            time_to_warning = (gt_frame - earliest_alarm) / fps
            ttw_list.append(time_to_warning)
            # Loại bỏ alarm này khỏi danh sách để xét báo động giả
            alarm_events.remove(earliest_alarm)
        else:
            ttw_list.append(0.0) # Không có cảnh báo sớm
            
    # Số alarm còn lại không gắn với event nào -> Báo động giả
    false_alarms = len(alarm_events)
    false_alarm_rate = false_alarms / len(preds) * 100 # Tỷ lệ trên tổng số frames
    
    avg_ttw = np.mean(ttw_list)
    
    logger.info("=== KẾT QUẢ ĐÁNH GIÁ CẢNH BÁO SỚM ===")
    logger.info(f"Average Time-to-Warning (TtW): {avg_ttw:.2f} giây")
    logger.info(f"False Alarm Rate (FAR): {false_alarm_rate:.4f}% (trên tổng số frame)")
    
    return {"avg_time_to_warning": avg_ttw, "false_alarms": false_alarms, "false_alarm_rate": false_alarm_rate}


def evaluate_depth(depth_pred: np.ndarray, depth_gt: np.ndarray):
    """Đánh giá chất lượng của Depth Estimation (MA, RMSE) nếu có ground-truth từ mô phỏng."""
    mask = depth_gt > 0 # Bỏ qua các điểm không hợp lệ
    if np.sum(mask) == 0:
        return None
        
    dp = depth_pred[mask]
    dg = depth_gt[mask]
    
    mae = np.mean(np.abs(dp - dg))
    rmse = np.sqrt(np.mean((dp - dg) ** 2))
    abs_rel = np.mean(np.abs(dp - dg) / dg)
    
    logger.info("=== KẾT QUẢ ĐÁNH GIÁ ĐỘ SÂU (DEPTH) ===")
    logger.info(f"MAE     : {mae:.4f}")
    logger.info(f"RMSE    : {rmse:.4f}")
    logger.info(f"Abs-Rel : {abs_rel:.4f}")
    
    return {"mae": mae, "rmse": rmse, "abs_rel": abs_rel}


# Ví dụ gọi thử nghiệm
if __name__ == "__main__":
    logger.info("Chạy bài test mô phỏng dữ liệu...")
    
    # 1. Test Phân loại
    y_true = np.random.choice([0, 1, 2], size=100, p=[0.7, 0.2, 0.1])
    y_pred = np.copy(y_true)
    # Tạo lỗi cố ý để test
    y_pred[10:20] = 0 # Bỏ sót nguy hiểm (rất tệ với Recall)
    
    evaluate_classification(y_true, y_pred)
    
    # 2. Test Cảnh báo sớm
    preds_timeline = np.zeros(300)
    preds_timeline[80:100] = 1 # Cảnh báo đúng lúc
    preds_timeline[200:210] = 2 # Báo động giả
    gt_events = [110] # Sự cố xảy ra ở frame 110
    
    evaluate_early_warning(preds_timeline, gt_events, fps=30)
