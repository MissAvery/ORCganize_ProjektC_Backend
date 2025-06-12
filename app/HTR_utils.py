from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
import numpy as np
import torch

# Modell und Prozessor laden
processor = TrOCRProcessor.from_pretrained("fhswf/TrOCR_german_handwritten")
model = VisionEncoderDecoderModel.from_pretrained("fhswf/TrOCR_german_handwritten")
model.eval()

# Funktion zur Texterkennung
def recognize_text_from_image(image: np.ndarray) -> str:
    # Konvertiere np.ndarray in PIL-Bild (RGB-Modus erwartet)
    pil_image = Image.fromarray(image).convert("RGB")
    pixel_values = processor(images=pil_image, return_tensors="pt").pixel_values

    with torch.no_grad():
        generated_ids = model.generate(pixel_values)

    text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    return text

