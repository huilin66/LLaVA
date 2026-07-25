import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # 只显示 GPU 0 设备
import pandas as pd
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")
import time
from llava.eval.run_llava import eval_model, eval_model_batch1, eval_model_batch2
attributes = ['deformation', 'broken', 'abandonment', 'corrosion']
attributes_name = ['deformation', 'broken', 'abandonment', 'corrosion']
# model_path = "ckpt/llava-v1.5-7b"
model5crop_path = "checkpoints/llava-v1.5-7b-lora-signboard5crop-merge"

Model_name = 'llava-v1.5-7b'

prompt5 = "Please check if the entered signboard has any of the following defects:'deformation' 'broken' 'abandonment' 'corrosion' "

image_file_crop = "/nfsv4/23039356r/data/signboard_llava/demo/FLIR0732_0.jpg"
image_file_keep = "/nfsv4/23039356r/data/signboard_llava/demo/FLIR0732_0_keep.jpg"
image_file_det = "/nfsv4/23039356r/data/signboard_llava/demo/FLIR0732_0_det.jpg"


image_dir_crop_infer5 = "/nfsv4/23039356r/data/signboard_llava/images_infer/images_infer_crop_result5"


def llava_inference(model_path, image_file, prompt):
    args = type('Args', (), {
        "model_path": model_path,
        "model_base": None,
        "model_name": Model_name,
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
    outputs = eval_model(args)
    # assert outputs.startswith('| property | value |'), print('output error')
    # df = pd.DataFrame(None, columns=['property', 'value'])
    # data_list = outputs.split('\n')
    # for data in data_list[2:]:
    #     values = data.split('|')[1:3]
    #     values = [value[1:] if value[0] == ' ' else value for value in values]
    #     values = [value[:-1] if value[-1] == ' ' else value for value in values]
    #     # print(values)
    #     df.loc[len(df)] = values
    # print(df)
    # df.to_csv('signboard_predict.csv', header=True, index=False)

def llava_inference_dir(model_path, img_dir, prompt, output_dir, postprocess=1):
    os.makedirs(output_dir, exist_ok=True)
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
    model, prompt, tokenizer, image_processor = eval_model_batch1(args1)
    img_names = os.listdir(img_dir)[:10]
    t1 = time.time()
    for img_name in tqdm(img_names):
        img_path = os.path.join(img_dir, img_name)
        save_path = os.path.join(output_dir, img_name.replace('.jpg', '.txt'))
        args2 = type('Args', (), {
            "model_path": model_path,
            "model_base": None,
            "model_name": Model_name,
            "query": prompt,
            "conv_mode": None,
            "image_file": img_path,
            "sep": ",",
            "temperature": 0,
            "top_p": None,
            "num_beams": 1,
            "max_new_tokens": 512
        })()
        outputs = eval_model_batch2(args2, model, prompt, tokenizer, image_processor)
        # print(outputs)
        if postprocess == 1:
            if outputs.startswith('| property | value |'):
                df = pd.DataFrame(None, columns=['property', 'value'])
                data_list = outputs.split('\n')
                for data in data_list[2:]:
                    values = data.split('|')[1:3]
                    values = [value[1:] if value[0] == ' ' else value for value in values]
                    values = [value[:-1] if value[-1] == ' ' else value for value in values]
                    # print(values)
                    df.loc[len(df)] = values
                df.to_csv(save_path, header=True, index=False)
            else:
                with open(save_path, 'w') as file:
                    pass  # 不写入任何内容，文件将保持为空
        elif postprocess == 2:
            if outputs.startswith('| defect property |'):
                df = pd.DataFrame(None, columns=['property', 'value'])
                data_list = outputs.split('\n')
                for data in data_list[2:]:
                    values = data.split('|')[1:3]
                    values = [value[1:] if value[0] == ' ' else value for value in values]
                    values = [value[:-1] if value[-1] == ' ' else value for value in values]
                    # print(values)
                    df.loc[len(df)] = values
                df.to_csv(save_path, header=True, index=False)
            else:
                with open(save_path, 'w') as file:
                    pass  # 不写入任何内容，文件将保持为空
        elif postprocess == 3:
            if outputs.startswith('The entered signboard has'):
                df = pd.DataFrame(None, columns=['property', 'value'])
                for i in range(len(attributes)):
                    attribute_name = attributes_name[i]
                    attribute = attributes[i]
                    if attribute_name in outputs:
                        df.loc[i] = [attribute_name, True]
                    else:
                        df.loc[i] = [attribute_name, False]
                df.to_csv(save_path, header=True, index=False)
            else:
                with open(save_path, 'w') as file:
                    pass  # 不写入任何内容，文件将保持为空
    t2 = time.time()
    print('FPS:', len(img_names)/(t2-t1) )
if __name__ == '__main__':
    pass

    llava_inference_dir(model5crop_path, image_dir_crop, prompt5, image_dir_crop_infer5, postprocess=3)
    # llava_inference_dir(model5keep_path, image_dir_keep, prompt5, image_dir_keep_infer5, postprocess=3)
    # llava_inference_dir(model5det_path, image_dir_det, prompt55, image_dir_det_infer5, postprocess=3)

