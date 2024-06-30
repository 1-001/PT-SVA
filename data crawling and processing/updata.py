import pandas as pd
from tqdm import tqdm

# 读取两个CSV文件
output_df = pd.read_csv('output.csv')
output2file_df = pd.read_csv('C:\Dateset\data_crawl\merged_allvul_with_description.csv')

# 初始化tqdm进度条
total_iterations = len(output2file_df)
with tqdm(total=total_iterations) as pbar:
    # 用于存储更新后的数据
    updated_data = []

    for index, row in output2file_df.iterrows():
        cve_id = row['cve_id']
        # 在output.csv中查找对应的cve_id
        corresponding_row = output_df[output_df['cve_id'] == cve_id]

        if not corresponding_row.empty:
            # 如果找到了对应的cve_id，则替换相应的数据
            row['cvss_vector'] = corresponding_row.iloc[0]['cvss_vector']
            row['cvss_base_score'] = corresponding_row.iloc[0]['cvss_base_score']
            row['cvss_base_severity'] = corresponding_row.iloc[0]['cvss_base_severity']
            updated_data.append(row)
        else:
            # 如果未找到对应的cve_id，则输出该cve_id
            print("未找到cve_id为", cve_id)
        pbar.update(1)

# 将更新后的数据转换为DataFrame
output2file_updated = pd.DataFrame(updated_data)

# 将更新后的DataFrame写入到新的CSV文件中
output2file_updated.to_csv('output3file_updated.csv', index=False)

print("替换完成，并将结果保存到output2file_updated.csv文件中。")
