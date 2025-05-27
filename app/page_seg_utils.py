import kraken
from kraken import binarization, pageseg
from PIL import Image
import cv2
import numpy as np
from io import BytesIO

import zipfile

# Kraken only works on Linux -------------------------------------------------------------------------------------------

def segment_image(image_contents):
    pil_img = Image.open(BytesIO(image_contents)).convert('L') #Grayscale
    bin_img = binarization.nlbin(pil_img, threshold=0.5)                      

    segmentation = pageseg.segment(bin_img) #Default no_hlines=False (doesnt work anyway)

    # Load original (non-binarized) for cropping
    np_img = np.frombuffer(image_contents, np.uint8)
    original = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

    zip_buffer = BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
        for i, line in enumerate(segmentation.lines):
            x0, y0, x1, y1 = map(int, line.bbox)

            # Ensure coordinates are within image bounds
            x0 = max(x0 - 10, 0)
            y0 = max(y0 - 10, 0)
            x1 = min(x1 + 10, original.shape[1])
            y1 = min(y1 + 10, original.shape[0])

            # Crop line image
            cropped = original[y0:y1, x0:x1]

            # Save line image
            is_success, buffer = cv2.imencode(".png", cropped)
            if is_success:
                zip_file.writestr(f"line_{i+1:03d}.png", buffer.tobytes())

    zip_buffer.seek(0)
    return zip_buffer

