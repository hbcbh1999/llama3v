from llama3v.image_processing import process_image
from PIL import Image
from transformers import AutoTokenizer

def test():
    tokenizer = AutoTokenizer.from_pretrained('openbmb/MiniCPM-Llama3-V-2_5', trust_remote_code=True)
    image = Image.open("house.jpeg").convert("RGB")

    images, tgt_sizes, _, _ = process_image(image, tokenizer)
    print(tgt_sizes)

test()