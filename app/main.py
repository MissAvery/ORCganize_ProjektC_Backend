from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse
from ner_utils import extract_tokens
from ics_utils import create_file_content
from page_seg_utils import segment_image
from io import BytesIO
from pydantic import BaseModel


app = FastAPI(
    title = "Event Extraction API",
    description = "Extrahiert Event-Informationen aus Bildern und erstellt .ics-Dateien.",
    version = "1.0.0",
)

class TextInput(BaseModel): #Text Input Files
    content: str  

@app.get("/")
def root():
    return {"Hello": "World"}

@app.post("/image_upload")
async def image_upload(file: UploadFile = File(...)):       #Boolean zu Handwritten?
    contents = await file.read()
    zip_buffer = segment_image(contents)
    zip_buffer.seek(0)
    return StreamingResponse(zip_buffer, media_type="application/zip", headers={
        "Content-Disposition": "attachment; filename=lines.zip"
    })
   

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
