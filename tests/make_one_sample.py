from io import BytesIO
from PIL import Image
import base64
import json

dic = {
    "image": "/data/hanxiao36/2383.202541811132.2.156.14702.30.1000.3.3202504180618585904.20250418061858.110132363711.0788.jpg",
    "question": "根据胸部X光报告，以下哪些影像学表现支持慢性支气管炎伴肺气肿的诊断？",
    "choices": {
        "A": "气胸",
        "B": "胸腔积液",
        "C": "肺气肿",
        "D": "肺不张"
    },
    "type": "single_choice",
    "system": "你是一名资深放射科医生，正在阅读一个病人的胸部X光片（可能包括正位片和/或侧位片）。",
}


def image_to_base64(img):
    output_buffer = BytesIO()
    img.save(output_buffer, format='png')
    byte_data = output_buffer.getvalue()
    base64_str = base64.b64encode(byte_data)
    return "data:image/png;base64," + base64_str.decode()


image = dic["image"]
image = Image.open(image).convert("RGB")

dic["image"] = image_to_base64(image)

with open("demo.json", "w", encoding="utf-8") as f:
    json.dump([dic] * 10, f, indent=4)
