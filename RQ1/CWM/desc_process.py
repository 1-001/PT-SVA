import pandas as pd
import string
from nltk.corpus import stopwords  # 这里可能是一个笔误，应为 nltk.corpus
from nltk.stem import PorterStemmer  # 这里可能是一个笔误，应为 nltk.stem
from nltk.tokenize import word_tokenize  # 这里可能是一个笔误，应为 nltk.tokenize
import nltk  # 这里可能是一个笔误，应为 nltk 的正确导入方式 nltk.download

# 下载 NLTK 数据（如果尚未下载）
# 注意：这里应该使用 nltk.download() 而不是 nltk.download
# nltk.download('punkt')
# nltk.download('stopwords')

# 加载数据
file_path = './Data3/train.xlsx'
df = pd.read_excel(file_path)

# 定义停用词和词干提取器
stop_words = set(stopwords.words('english'))
custom_stop_words = set(['aka'])  # 在这里添加任何额外的停用词
all_stop_words = stop_words.union(custom_stop_words)
stemmer = PorterStemmer()


# 文本预处理函数
def preprocess_text(text):
    # 转换为小写
    text = text.lower()

    # 对文本进行分词
    words = word_tokenize(text)

    # 移除停用词和标点符号
    words = [word for word in words if word not in all_stop_words and word not in string.punctuation]

    # 应用词干提取
    words = [stemmer.stem(word) for word in words]

    # 将单词重新组合成一个字符串
    processed_text = ' '.join(words)

    return processed_text


# 对 'description' 列应用预处理
df['processed_description'] = df['description'].apply(preprocess_text)

# 将处理后的数据保存到新的 Excel 文件中
output_file_path = './Data3/processed_train.xlsx'
df.to_excel(output_file_path, index=False)