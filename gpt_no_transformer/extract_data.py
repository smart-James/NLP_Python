import lzma
from tqdm import tqdm
import os


def xz_files_in_dir(dir_path):
    """Return a list of all xz files in a directory."""
    files = []
    for filename in os.listdir(dir_path):
        if filename.endswith(".xz") and os.path.isfile(os.path.join(dir_path, filename)):
            files.append(filename)
    return files

folder_path = 'your_folder_path'
output_file_train = 'output_train.txt'
output_file_val = 'output_val.txt'
vocab_file = 'vocab.txt'

# split_files = int(input('How many files would you like to split into? '))

files = xz_files_in_dir(folder_path)
total_files = len(files)

split_index = int(total_files * 0.9)
files_train = files[:split_index]
files_val = files[split_index:]

# //: 整除并向下取整
# max_count = total_files // split_files if split_files != 0 else total_files

vocab = set()

with open(output_file_train,'w',encoding='utf-8') as outfile:
    for filename in tqdm(files_train,total = len(files_train)):
        file_path = os.path.join(folder_path, filename)
        with lzma.open(file_path, 'rt', encoding='utf-8') as infile:
            text = infile.read()
            outfile.write(text)
            characters = set(text)
            vocab.update(characters)
            
with open(output_file_val,'w',encoding='utf-8') as outfile:
    for filename in tqdm(files_val,total = len(files_val)):
        file_path = os.path.join(folder_path, filename)
        with lzma.open(file_path, 'rt', encoding='utf-8') as infile:
            text = infile.read()
            outfile.write(text)
            characters = set(text)
            vocab.update(characters)


# 遍历分割文件的数量
# for i in range(split_files):
#     # 打开输出文件进行写操作
#     with open(output_file.format(i), 'w', encoding='utf-8') as out_file:
#         # 遍历文件列表，使用tqdm显示进度条
#         for count, filename in enumerate(tqdm(files[:max_count], total=max_count)):
#             # 如果计数超过最大计数，则跳出循环
#             if count >= max_count:
#                 break
#             # 构建文件路径
#             file_path = os.path.join(folder_path, filename)
#             # 打开压缩文件进行读操作
#             with lzma.open(file_path, 'rt', encoding='utf-8') as in_file:
#                 text = in_file.read()
#                 out_file.write(text)
#                 characters = set(text)
#                 vocab.update(characters)
#         # 更新文件列表，移除已处理的文件
#         files = files[max_count:]

with open(vocab_file, 'w', encoding='utf-8') as vfile:
    for char in vocab:
        vfile.write(char + '\n')