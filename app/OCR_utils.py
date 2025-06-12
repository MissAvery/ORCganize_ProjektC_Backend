import numpy as np
from PIL import Image
import easyocr

reader = easyocr.Reader(['de'], gpu=False)

def recognize_text_from_image(image_array: np.ndarray) -> str:

    image = Image.fromarray(image_array).convert("RGB")
    result = reader.readtext(np.array(image), detail=0)
    return " ".join(result)