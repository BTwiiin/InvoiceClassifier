import json
import easyocr

def extract_bounding_boxes_text(image_path):
    reader = easyocr.Reader(['en'])
    results = reader.readtext(
        image_path,
        paragraph=True,
        decoder='wordbeamsearch',
        x_ths=1,
        y_ths=0.5 
    )

    bboxes = [json.dumps(result[0]) for result in results]
    texts = [result[1] for result in results]

    print(f'{bboxes}')

    return bboxes, texts
