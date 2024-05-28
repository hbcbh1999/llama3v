from transformers import AutoTokenizer, AutoModel
from PIL import Image

model = AutoModel.from_pretrained(
    "mustafaaljadery/llama3v", from_pt=True).cuda()
tokenizer = AutoTokenizer.from_pretrained("mustafaaljadery/llama3v")

image = Image.open("test_image.png")

answer = model.generate(
    image=image, message="What is this image?", temperature=0.1, tokenizer=tokenizer)

print(answer)
