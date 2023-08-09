import json
import pandas as pd

# 读取JSON文件
fd = "xiezhi.v1.json"
jsonfile = open(fd, encoding='utf-8')

# 初始化DataFrame
df = pd.DataFrame(columns=['id', 'question', 'A', 'B', 'C', 'D', 'answer', 'explanation'])

# 解析JSON数据，逐行添加到DataFrame中
idx = 0
for jsonline in jsonfile:
    jsonline = json.loads(jsonline)
    options = jsonline['options'].split('\n')
    df.loc[idx] = [
        idx+1,  # id自动递增编号
        jsonline['question'],
        options[0],
        options[1],
        options[2],
        options[3],
        ord(jsonline['answer']) - 65,  # 将答案转换为选项号，并存储在answer列中
        ' '  # 将标签以逗号分隔，并存储在explanation列中
    ]

# 打印DataFrame
print(df)