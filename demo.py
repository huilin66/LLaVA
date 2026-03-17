from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path
from llava.eval.run_llava import eval_model

# model_path = "ckpt/llava-v1.5-7b"
model_path = "checkpoints/llava-v1.5-7b-lora-defect1-merge"
# tokenizer, model, image_processor, context_len = load_pretrained_model(
#     model_path=model_path,
#     model_base=None,
#     model_name=get_model_name_from_path(model_path)
# )

model_name = 'llava-v1.5-7b'

# prompt = "What are the things I should be cautious about when I visit here?"
# image_file = "https://llava-vl.github.io/static/images/view.jpg"

prompt = "please describe this image in table format."
image_file = "/nfsv4/23039356r/data/defect_caption/defect_demo/2.jpg"


args = type('Args', (), {
    "model_path": model_path,
    "model_base": None,
    "model_name": model_name,
    "query": prompt,
    "conv_mode": None,
    "image_file": image_file,
    "sep": ",",
    "temperature": 0,
    "top_p": None,
    "num_beams": 1,
    "max_new_tokens": 512
})()
print(args)
eval_model(args)





# import gradio as gr
# from transformers import pipeline
# import time
#
# def greet(name):
#     return "Hello " + name + "!"
#
# def process_data(text, image):
#     # 假设这里有数据处理逻辑
#     processed_text = text.upper()
#     return processed_text, image
#
#
# def classify_image(model, img):
#     return {i['label']: i['score'] for i in model(img)}
#
# def process_image(img, filter_type):
#     if filter_type == "Black and White":
#         img = img.convert("L")
#     return img
#
# def slow_echo(message, history):
#     for i in range(len(message)):
#         time.sleep(0.05)
#         yield "机器人回复: " + message[: i+1]
#
# def function1(input1):
#     return f"处理结果: {input1}"
#
# def function2(input2):
#     return f"分析结果: {input2}"



# iface = gr.Interface(
#     fn=process_image,
#     inputs=[gr.Image(type="pil"), gr.Radio(["None", "Black and White"])],
#     outputs="image"
# )
#
# iface.launch()

# data = [
#     ['USER',
#      ('What is unusual about this image?\n<image>',
#       # <PIL.Image.Image image mode=RGB size=570x380 at 0x7F9CADAEA290>,
#       'Default')
#      ],
#     [
#         'ASSISTANT',
#      "The unusual aspect of this image is that a man is hanging out of the back of a moving yellow taxi. This is not a common sight, as people usually sit inside the taxi to travel. The man's actions might be a part of a promotional event or a stunt, but it is still an unconventional and potentially dangerous situation."
#      ]
#     ]
# from pathlib import Path
# import gradio as gr
#
# def upload_file(filepath):
#     name = Path(filepath).name
#     return [gr.UploadButton(visible=True), gr.DownloadButton(label=f"Download {name}", value=filepath, visible=True)]
#
# def download_file():
#     return [gr.UploadButton(visible=True), gr.DownloadButton(value='/nfsv4/23039356r/data/defect_caption/defect2k.json',visible=True)]
#
# with gr.Blocks() as demo:
#     gr.Markdown("First upload a file and and then you'll be able download it (but only once!)")
#     with gr.Row():
#         u = gr.UploadButton("Upload a file", file_count="single")
#         d = gr.DownloadButton("Download the file", visible=True)
#
#     u.upload(upload_file, u, [u, d])
#     d.click(download_file, None, [u, d])


# if __name__ == "__main__":
#     demo.launch()

