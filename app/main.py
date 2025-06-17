from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse
from ner_utils import extract_tokens
from ics_utils import create_file_content
from page_seg_utils import segment_image
from HTR_utils import recognize_text_from_image as recognize_text_from_image_handwritten
from OCR_utils import recognize_text_from_image as recognize_text_from_image_printed
from io import BytesIO
from pydantic import BaseModel
import numpy as np
import cv2


app = FastAPI(
    title = "Event Extraction API",
    description = "Extrahiert Event-Informationen aus Bildern und erstellt .ics-Dateien.",
    version = "1.0.0",
)

class TextInput(BaseModel): #Text Input Files
    content: str  


def mocked_OCR_function_handwritten(image: np.ndarray):
    return "handwritten"

def mocked_OCR_function_printed(image: np.ndarray):
    return "printed"

@app.get("/")
def root():
    return {"Hello": "World"}

@app.post("/image_upload")
async def image_upload(handwritten: bool, file: UploadFile = File(...)):    
    contents = await file.read()
    extracted_text = ""

    #Handwritten Segmentation & OCR
    if(handwritten):
        img_list = segment_image(contents)  #Segment Image
        for image in img_list:
            extracted_text += recognize_text_from_image_handwritten(image) + " "

    #Printed OCR
    else:
        np_img = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
        extracted_text = recognize_text_from_image_printed(image)
    
    #NER & Output File Creation
    name_tokens, date_tokens, time_tokens, location_tokens, duration_tokens, link_tokens = extract_tokens(extracted_text)
    content = create_file_content(name_tokens, date_tokens, time_tokens, location_tokens, duration_tokens, link_tokens)

    #For Testing purpose-----
    content += '\n'
    content += extracted_text
    #------------------------

    file_like = BytesIO()
    file_like.write(content.encode("utf-8"))
    file_like.seek(0)  # Cursor zurück zum Anfang setzen

    return StreamingResponse(
        file_like,
        media_type="text/plain",
        headers={"Content-Disposition": "attachment; filename=entry.txt"}
    )
   

@app.post("/upload")
def upload(data: TextInput):# -> str:
    #picture -> page segmentation -> ocr -> sentence
    sentence = data.content
    name_tokens, date_tokens, time_tokens, location_tokens, duration_tokens, link_tokens = extract_tokens(sentence)
    content = create_file_content(name_tokens, date_tokens, time_tokens, location_tokens, duration_tokens, link_tokens)

    file_like = BytesIO()
    file_like.write(content.encode("utf-8"))
    file_like.seek(0)  # Cursor zurück zum Anfang setzen

    return StreamingResponse(
        file_like,
        media_type="text/plain",
        headers={"Content-Disposition": "attachment; filename=entry.txt"}
    )

