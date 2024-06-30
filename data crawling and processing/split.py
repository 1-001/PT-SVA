import pandas as pd
from sklearn.model_selection import train_test_split

# 读取数据集
data = pd.read_excel("C:\Dateset\SOTitlePlus\SOTitlePlus\data_c++\megavul_simple_cpp_processed.xlsx")

# 按照 cve_id 划分数据集
unique_cve_ids = data['cve_id'].unique()
train_cve_ids, test_val_cve_ids = train_test_split(unique_cve_ids, test_size=0.2, random_state=42)
test_cve_ids, val_cve_ids = train_test_split(test_val_cve_ids, test_size=0.5, random_state=42)

# 按照 cve_id 划分数据
train_set = data[data['cve_id'].isin(train_cve_ids)]
test_set = data[data['cve_id'].isin(test_cve_ids)]
val_set = data[data['cve_id'].isin(val_cve_ids)]

# 打印数据集大小
print("Train set size:", len(train_set))
print("Test set size:", len(test_set))
print("Validation set size:", len(val_set))

# 保存数据集
train_set.to_excel("C:\Dateset\SOTitlePlus\SOTitlePlus\data_c++\\train.xlsx", index=False)
test_set.to_excel("C:\Dateset\SOTitlePlus\SOTitlePlus\data_c++\\test.xlsx", index=False)
val_set.to_excel("C:\Dateset\SOTitlePlus\SOTitlePlus\data_c++\\valid.xlsx", index=False)
