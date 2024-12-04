import cv2
import numpy as np

def detect_qr_code(image_path):
    image = cv2.imread(image_path)
    qr_detector = cv2.QRCodeDetector()
    data, bbox, _ = qr_detector.detectAndDecode(image)
    return data, bbox