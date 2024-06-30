import pandas as pd
import numpy as np

# 读取Excel文件
file_path = 'E:\wjy\SOTitlePlus\SOTitlePlus\data_c++\megavul_simple_cpp_processed.xlsx'  # 将此处替换为你的Excel文件路径
df = pd.read_excel(file_path)

# 计算每行中单词的数量
df['description_word_count'] = df['description'].apply(lambda x: len(str(x).split()))
df['abstract_func_before_word_count'] = df['abstract_func_before'].apply(lambda x: len(str(x).split()))

# 计算统计信息
stats = {
    'description': {
        'mean': df['description_word_count'].mean(),
        '40%': np.percentile(df['description_word_count'], 40),
        '60%': np.percentile(df['description_word_count'], 60),
        '80%': np.percentile(df['description_word_count'], 80),
        '90%': np.percentile(df['description_word_count'], 90),
        '100%': np.percentile(df['description_word_count'], 100)
    },
    'abstract_func_before': {
        'mean': df['abstract_func_before_word_count'].mean(),
        '40%': np.percentile(df['abstract_func_before_word_count'], 40),
        '60%': np.percentile(df['abstract_func_before_word_count'], 60),
        '80%': np.percentile(df['abstract_func_before_word_count'], 80),
        '90%': np.percentile(df['abstract_func_before_word_count'], 90),
        '100%': np.percentile(df['abstract_func_before_word_count'], 100)
    }
}

# 打印统计信息
print("Description Column Stats:")
for key, value in stats['description'].items():
    print(f"{key}: {value}")

print("\nAbstract Func Before Column Stats:")
for key, value in stats['abstract_func_before'].items():
    print(f"{key}: {value}")
