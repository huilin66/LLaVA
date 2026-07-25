follow readme.md

pip install -e ".[train]"
pip install flash_attn-2.6.3+cu123torch2.1cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
pip install transformers==4.37.2 accelerate==0.27.2 peft==0.4.0
pip install reportlab
pip install gradio==4.31.4
pip install gradio==4.35.0 fastapi==0.112.2 pydantic==2.8.2


python -m llava.serve.controller --host 0.0.0.0 --port 10000
python -m llava.serve.model_worker --host 0.0.0.0 --controller http://localhost:10000 --port 40000 --worker http://localhost:40000 --model-path /localnvme/project/LLaVA-main/checkpoints/llava-v1.5-7b-lora-defect1-merge
python -m llava.serve.gradio_web_server --controller http://localhost:10000 --model-list-mode reload
