
import requests

from io import BytesIO
from PIL import Image
from pathlib import Path

MEDIUM_ARTICLE_EXAMPLE_URL = "https://images.rawpixel.com/image_800/cHJpdmF0ZS9sci9pbWFnZXMvd2Vic2l0ZS8yMDIyLTA1L3Vwd2s2MTc3Nzk0MS13aWtpbWVkaWEtaW1hZ2Uta293YnN1MHYuanBn.jpg"



def download_image(save_path: Path, url: str = MEDIUM_ARTICLE_EXAMPLE_URL):
    r = requests.get(url, timeout=4.0)
    if r.status_code != requests.codes.ok:
        assert False, 'Status code error: {}.'.format(r.status_code)

    with Image.open(BytesIO(r.content)) as im:
        im.save(save_path)
    print('Image downloaded from url: {} and saved to: {}.'.format(url, save_path))