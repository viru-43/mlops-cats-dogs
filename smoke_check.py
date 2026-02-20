import requests
import time

def test_health():
    url = "http://localhost:8000/health"

    for _ in range(30):  # retry up to 30 seconds
        try:
            response = requests.get(url)
            if response.status_code == 200:
                return
        except requests.exceptions.ConnectionError:
            pass

        time.sleep(1)

    raise Exception("Smoke test failed: Service did not start in time.")