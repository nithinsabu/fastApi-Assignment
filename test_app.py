from fastapi.testclient import TestClient
from app import app
import requests
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
