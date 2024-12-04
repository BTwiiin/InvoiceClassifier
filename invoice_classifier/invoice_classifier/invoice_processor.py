from .qr_code_detector import detect_qr_code
from .bbox_text_extractor import extract_bounding_boxes_text
from .classifier import InvoiceClassifier
import numpy as np
import json
import cv2
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
from joblib import load

class InvoiceProcessor:
    def __init__(self, classifier_model_path, scaler_path, vectorizer_path):
        self.classifier = InvoiceClassifier(classifier_model_path)
        self.scaler = load(scaler_path)
        self.vectorizer = load(vectorizer_path)

    def process_invoice(self, image_path, draw_bboxes=0):
        qr_data, qr_bbox = detect_qr_code(image_path)

        if not qr_data:
            print("No QR code detected.")
        else:
            print(f"QR Code Data: {qr_data}")

        bboxes, texts = extract_bounding_boxes_text(image_path)
        print(f"Extracted Text: {texts}")

        if draw_bboxes == 1:
            self.__draw_bbox(image_path, bboxes, texts)

        features = self.prepare_features(texts, bboxes)

        prediction = self.classifier.predict(features)

        return prediction

    def prepare_features(self, texts, bboxes):
        bbox_features = self.__extract_bbox_features(bboxes)

        normalized_bbox_features = self.scaler.transform(bbox_features)

        text_embeddings = self.vectorizer.transform(texts).toarray()

        text_embeddings_sparse = csr_matrix(text_embeddings)
        bbox_features_sparse = csr_matrix(normalized_bbox_features)

        combined_embeddings_sparse = csr_matrix(np.hstack([text_embeddings_sparse.toarray(), bbox_features_sparse.toarray()]))

        return combined_embeddings_sparse
    
    def __draw_bbox(self, image_path, bboxes, texts):
        image = cv2.imread(image_path)

        # Iterate through bounding boxes and corresponding texts
        for bbox_str, text in zip(bboxes, texts):
            try:
                # Parse the bbox string to convert it to a list of coordinates
                bbox = json.loads(bbox_str)  # Convert the JSON-like string into a list

                # Convert the bounding box points to integer tuples
                pts = [tuple(map(int, point)) for point in bbox]

                # Draw the bounding box and the corresponding text on the image
                cv2.polylines(image, [np.array(pts)], isClosed=True, color=(0, 255, 0), thickness=2)
                cv2.putText(image, text[:30], (pts[0][0], pts[0][1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            
            except json.JSONDecodeError:
                print(f"Error parsing bbox: {bbox_str}")

        # Display the result
        plt.figure(figsize=(12, 10))
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.show() 


    def __extract_bbox_features(self, bboxes):
        features = []
        for bbox in bboxes:
            if isinstance(bbox, str):
                try:
                    coords = json.loads(bbox)
                    x_min, y_min = coords[0]
                    x_max, y_max = coords[1]
                    width = x_max - x_min
                    height = y_max - y_min
                    area = width * height
                    aspect_ratio = height / width if width > 0 else 0
                    features.append([x_min, y_min, x_max, y_max, width, height, area, aspect_ratio])
                except (json.JSONDecodeError, ValueError):
                    features.append([0, 0, 0, 0, 0, 0, 0, 0])
            else:
                features.append([0, 0, 0, 0, 0, 0, 0, 0])
        return np.array(features)
    