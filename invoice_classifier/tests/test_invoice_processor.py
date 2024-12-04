import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../invoice_classifier')))
from invoice_classifier.invoice_processor import InvoiceProcessor

processor = InvoiceProcessor(classifier_model_path=r'invoice_classifier/models/xgb_model.pkl', 
                             scaler_path=r'invoice_classifier/models/scaler.pkl',
                             vectorizer_path=r'invoice_classifier/models/tfidf_vectorizer.pkl')

prediction = processor.process_invoice(r'invoice_classifier/tests/Template39_Instance41.jpg', draw_bboxes=1)
print(f"Invoice Classification Result: {prediction}")