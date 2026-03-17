import json

import pandas as pd

if __name__ == '__main__':
    pass
    # json_root=r'/nfsv4/23039356r/data/defect_caption/llava_v1_5_mix665k.json'
    # with open(json_root, 'r') as f:
    #     json_info = json.load(f)
    # save_info=json_info[620000:620320]
    # with open('/nfsv4/23039356r/data/defect_caption/llava_v1_5_vqa320.json', 'w') as fp:
    #     json.dump(save_info, fp, indent=4)

    # json_root=r'/nfsv4/23039356r/data/defect_caption/llava_v1_5_vqa320.json'
    # with open(json_root, 'r') as f:
    #     json_info = json.load(f)
    #
    #     print(len(json_info))


    # json_root=r'/nfsv4/23039356r/data/defect_caption/defect4k.json'
    # with open(json_root, 'r') as f:
    #     json_info = json.load(f)
    #
    #     print(len(json_info))
    #     print(json_info[0])


    data_str = '| properity | value |\n| --- | --- |\n| background | road |\n| defect types | background |\n| defect numbers | 1 |\n| defect level | serious |\n| possible causes of the defects | heavy traffic, temperature changes, and other factors. |\n| required actions of the defects| Contact the highway department to repair the defect. |'
    print(data_str)
    print()
    df = pd.DataFrame(None, columns=['property', 'value'])
    data_list = data_str.split('\n')
    for data in data_list[2:]:
        values = data.split('|')[1:3]
        values = [value[1:] if value[0] == ' ' else value for value in values]
        values = [value[:-1] if value[-1] == ' ' else value for value in values]
        # print(values)
        df.loc[len(df)] = values
    print(df)
    df.to_csv('test.csv')
