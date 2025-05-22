from fastapi.testclient import TestClient
from .main import app
import requests
import numpy as np
from PIL import Image
import io 
client = TestClient(app)

def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert "<html" in response.text.lower()

def test_upload_invalid_file():
    file = ("test.txt", b"this is not an image", "multipart/form-data")
    response = client.post("/upload/", files={"file": file})
    assert response.status_code == 400
    assert response.json() == {"detail": "Corrupted Image file"}

def test_upload_valid_file():
    response = requests.get("https://ultralytics.com/images/bus.jpg", stream=True)
    byte_stream = response.raw.read()
    file = ("test.txt", byte_stream, "multipart/form-data")
    response = client.post('/upload/', files ={"file": file})
    assert response.status_code == 200
    assert response.json()

def test_upload_empty_file():
    file = ("test.txt", b"", "multipart/form-data")
    response = client.post('/upload/', files ={"file": file})
    assert response.status_code == 400
    assert response.json()

def test_upload_large_file():
    image_data = np.random.randint(0, 256, (5000, 5000, 3), dtype=np.uint8)
    image = Image.fromarray(image_data, 'RGB')
    buffer = io.BytesIO()
    image.save(buffer, format="PNG", compress_level=0)  
    content = buffer.getvalue()
    
    file = ("test.txt", content, "multipart/form-data")
    response = client.post('/upload/', files ={"file": file})
    assert response.status_code == 400
    assert response.json()

