import os.path
import shutil

from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path
Model_name = 'llava-v1.5-7b-lora'
Model_base = '/nfsv4/23039356r/repository/LLaVA-main/ckpt/llava-v1.5-7b'

def merge_lora(model_path, save_model_path):
    # model_name = get_model_name_from_path(model_path)
    # print(model_name)
    # cfg_file = os.path.join(model_path, 'config.json')
    # if not os.path.exists(cfg_file):
    #     src_cfg = os.path.join(os.path.dirname(model_path), 'config.json')
    #     assert os.path.exists(src_cfg), ValueError(src_cfg, 'not exists')
    #     shutil.copyfile(src_cfg, cfg_file)

    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, Model_base, Model_name, device_map='cpu')

    model.save_pretrained(save_model_path)
    tokenizer.save_pretrained(save_model_path)

if __name__ == '__main__':

    # model_path = r'checkpoints/llava-v1.5-7b-lora-signboard1'
    # merge_lora(model_path = model_path, save_model_path=model_path+'-merge')
    # model_path = r'checkpoints/llava-v1.5-7b-lora-signboard1keep'
    # merge_lora(model_path = model_path, save_model_path=model_path+'-merge')
    #
    # model_path = r'checkpoints/llava-v1.5-7b-lora-signboard2'
    # merge_lora(model_path = model_path, save_model_path=model_path+'-merge')
    # model_path = r'checkpoints/llava-v1.5-7b-lora-signboard2keep'
    # merge_lora(model_path = model_path, save_model_path=model_path+'-merge')
    #
    # model_path = r'checkpoints/llava-v1.5-7b-lora-signboard3'
    # merge_lora(model_path = model_path, save_model_path=model_path+'-merge')
    # model_path = r'checkpoints/llava-v1.5-7b-lora-signboard3keep'
    # merge_lora(model_path = model_path, save_model_path=model_path+'-merge')
    # model_path = r'checkpoints/llava-v1.5-7b-lora-signboard3det'
    # merge_lora(model_path = model_path, save_model_path=model_path+'-merge')


    model_path = r'checkpoints/llava-v1.5-7b-lora-signboard4crop'
    merge_lora(model_path = model_path, save_model_path=model_path+'-merge')
    model_path = r'checkpoints/llava-v1.5-7b-lora-signboard4keep'
    merge_lora(model_path = model_path, save_model_path=model_path+'-merge')
    model_path = r'checkpoints/llava-v1.5-7b-lora-signboard4det'
    merge_lora(model_path = model_path, save_model_path=model_path+'-merge')

    model_path = r'checkpoints/llava-v1.5-7b-lora-signboard5crop'
    merge_lora(model_path = model_path, save_model_path=model_path+'-merge')
    model_path = r'checkpoints/llava-v1.5-7b-lora-signboard5keep'
    merge_lora(model_path = model_path, save_model_path=model_path+'-merge')
    model_path = r'checkpoints/llava-v1.5-7b-lora-signboard5det'
    merge_lora(model_path = model_path, save_model_path=model_path+'-merge')
