import os
import pandas as pd

import argparse
from llava.eval.run_llava import eval_model

os.environ['CUDA_VISIBLE_DEVICES'] = '0'




def simple_predict(args_model):
    print(args_model)
    outputs = eval_model(args_model)
    print(outputs)
    return outputs

def format_predict(args_model, save_path=None):
    outputs = simple_predict(args_model)
    assert outputs.startswith('| Defect property | Defect value |'), print('output error')
    df = pd.DataFrame(None, columns=['property', 'value'])
    data_list = outputs.split('\n')
    for data in data_list[2:]:
        values = data.split('|')[1:3]
        values = [value[1:] if value[0] == ' ' else value for value in values]
        values = [value[:-1] if value[-1] == ' ' else value for value in values]
        df.loc[len(df)] = values
    print(df)

    if save_path is not None:
        df.to_csv('defect_predict.csv')

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--image_file", type=str, required=True)
    parser.add_argument("--model_name", type=str, default='llava-v1.5-7b')
    parser.add_argument("--prompt", type=str, default=r'Please detect all defects in this image and output the results in a Markdown table format with columns.')
    args = parser.parse_args()

    args_model = type('Args', (), {
        "model_path": args.model_path,
        "model_base": None,
        "model_name":args.model_name,
        "query": args.prompt,
        "conv_mode": None,
        "image_file": args.image_file,
        "sep": ",",
        "temperature": 0,
        "top_p": None,
        "num_beams": 1,
        "do_sample": False,
        "max_new_tokens": 512
    })()

    format_predict(args_model)
