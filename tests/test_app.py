from fastapi.testclient import TestClient
from app import app

# Create a test client (like a fake browser)
client = TestClient(app)

def test_home():
    # We didn't make a home page, so checking /docs or just ensuring app loads
    assert app is not None

def test_prediction():
    # Send a fake spam message
    response = client.post("/predict", json={"text": "Win a free iPhone now"})
    
    # Check if the server replies "OK"
    assert response.status_code == 200
    
    # Check if the reply contains the "prediction" field
    assert "prediction" in response.json()