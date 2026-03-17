import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # 只显示 GPU 0 设备
import time
import json
import warnings
from tqdm import tqdm
from datasets import load_dataset
from llava.eval.run_llava import eval_model_batch1, eval_model_batch3, load_pretrained_model
from copy import deepcopy

warnings.filterwarnings("ignore")

Model_name = 'llava-v1.5-7b'
model_path = 'ckpt/llava-v1.5-7b'

# prompt0 = "please describe this image"
prompt0 = "Task: please answer the question based on this image.\nQuestion:"
prompt1 = ('Task: Analyze this image and answer the following question with cultural sensitivity. Follow these steps:\n'
           '1. **Visual Description**: Describe key objects, actions, and scene context in the image.)\n'
           '2. **Cultural Context**: Identify potential cultural elements (e.g., clothing, rituals, symbols) and their possible origin (country/region). If uncertain, state "I observe [elements] which may relate to [cultures], but need more context."\n'
           '3. **Question Answering**: Answer the question based on visual and cultural analysis. If the cultural context is unclear, say "I cannot determine the cultural context precisely, but generally [answer]."\n'
           'Question:"')
Prompt = prompt1


def llava_inference_cultureVQA(model_path, prompt, output_path):
    args1 = type('Args', (), {
        "model_path": model_path,
        "model_base": None,
        "model_name": Model_name,
        "query": prompt,
        "conv_mode": None,
        "sep": ",",
        "temperature": 0,
        "top_p": None,
        "num_beams": 1,
        "max_new_tokens": 512
    })()
    # model, _, tokenizer, image_processor = eval_model_batch1(args1)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        args1.model_path, args1.model_base, args1.model_name
    )

    culturalvqa_dataset = load_dataset('mair-lab/CulturalVQA')

    t1 = time.time()
    results = []
    for data in tqdm(culturalvqa_dataset['test']):
        image, question, uid = data['image'], data['question'], data['u_id']
        args3 = type('Args', (), {
            "model_path": model_path,
            "model_base": None,
            "model_name": Model_name,
            "query": prompt+question+'"',
            "conv_mode": None,
            "sep": ",",
            "temperature": 0,
            "top_p": None,
            "num_beams": 1,
            "max_new_tokens": 512
        })()
        print(prompt+question+'"')
        outputs = eval_model_batch3(args3, model, prompt, tokenizer, image_processor, [image])
        result = {"id": uid, "pred": outputs}
        results.append(result)
        print(result)
        break

    # write result into a json file
    with open(output_path, 'w') as f:
        json.dump(results, f)
    t2 = time.time()
    print(f'total time: {t2 - t1}\n'
          f'total image: {len(culturalvqa_dataset["test"])}\n'
          f'FPS: {len(culturalvqa_dataset["test"])/(t2 - t1)}\n')

if __name__ == '__main__':
    output_path1 = f'cultural_vqa/{Model_name}_prompt0.json'
    llava_inference_cultureVQA(model_path, Prompt, output_path1)

