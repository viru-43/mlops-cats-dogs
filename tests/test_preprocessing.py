from PIL import Image
from src.data_preprocessing import IMG_SIZE

def test_image_resize():
    img = Image.new("RGB", (500, 500))
    resized = img.resize(IMG_SIZE)
    assert resized.size == IMG_SIZE