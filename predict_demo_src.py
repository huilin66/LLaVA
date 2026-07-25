from llava.eval.run_llava import eval_model

model_path = "ckpt/llava-v1.5-7b"
# model_path = "checkpoints/llava-v1.5-7b-lora-defect1-merge"

model_name = 'llava-v1.5-7b'

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
