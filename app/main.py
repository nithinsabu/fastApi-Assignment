from fastapi import FastAPI, UploadFile, HTTPException
from fastapi.responses import HTMLResponse
from ultralytics import YOLO
import numpy as np
import cv2
import os
app = FastAPI()
model = YOLO("yolo11n.pt")


@app.get("/")
async def root():
    # results = model.predict("./bus.jpg", save=True)
    # return_object = results[0].to_json()
    with open(os.path.join(os.path.dirname(__file__),"static", 'index.html'), 'r') as f:
        html_content = f.read()
    return HTMLResponse(content=html_content, status_code=200)


@app.post("/upload/")
async def create_upload_file(file: UploadFile):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Corrupted Image file")
    result = model.predict(img)[0]
    return_object = {}
    count = {}
    try:
        for i in result.boxes:
            obj = result.names[int(i.cls)]
            prob = float(i.conf)
            if obj not in count.keys():
                count[obj] = 1
            else:
                count[obj]+=1
            return_object[obj+" "+str(count[obj])] = prob
        print(return_object)
    except:
        raise HTTPException(status_code=500, detail="Error with server")
    return return_object
