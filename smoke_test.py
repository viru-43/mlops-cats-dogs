import requests
import time

def test_health():
    time.sleep(5)  # give container time to start
    response = requests.get("http://localhost:8000/health")
    assert response.status_code == 200