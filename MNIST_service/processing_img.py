import re
import base64
from io import BytesIO
from PIL import Image
import numpy as np
from PIL.Image import Resampling


def processing(data):
    img = base64_to_image(data)
    img = crop(img)
    img = resize(img)
    x = np.array(img)
    x = np.delete(x, [0, 1, 2], axis=2)
    x = x.astype('float32')
    x /= 255
    return x


def crop(img):
    box = img.getbbox()
    print(box[0], box[1], box[2], box[3])
    width = box[2] - box[0]
    height = box[3] - box[1]
    s = (width - height)/2
    if s > 0:
        box = (box[0], box[1] - s, box[2], box[3] + s)
    else:
        box = (box[0] + s, box[1], box[2] - s, box[3])
    img = img.crop(box)
    return img

def base64_to_image(data):
    b64_img = re.sub('^data:image/png;base64,', '', data)
    decoded_img = base64.b64decode(b64_img)
    pil_img = Image.open(BytesIO(decoded_img))
    return pil_img


def resize(img):
    img = img.resize((22, 22), Resampling.LANCZOS)
    new_img = Image.new('RGBA', (28, 28), (0, 0, 0, 0))
    new_img.paste(img, (3, 3, 25, 25))
    return new_img


