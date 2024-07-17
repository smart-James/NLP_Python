
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import os
import mmap
import random
import pickle
import argparse


parser = argparse.ArgumentParser(description='This is a sample program to demonstrate argparse module in Python.')
parser.add_argument('-llms', type = str, help='Please provide an llm')
parser.add_argument('-batch_size', type = str, required=True, help='Please set batch size')

args = parser.parse_args()



# 引用gpu，这两种写法在功能上是完全相同的，第二种写法可读性更胜一筹
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

# batch_size：一次训练的样本数量（分多少批次，或者说并行处理最大个数）
# 64
batch_size = args.batch_size

# block_size：context的最大长度
# 128
# 这里必须和trainingz.py中的block_size一致，否则会报错
block_size = 16

# dropout: 随机失活
dropout = 0.2

n_embd = 384

n_head = 4

# 4 个 encoder layer
n_layer = 4

chars = ''
with open('../data/vocab.txt', 'r', encoding='utf-8') as f:
    text = f.read()
    chars = sorted(list(set(text)))
    
vocab_size = len(chars)

# 将字符串中的每个字符映射到一个整数,并将这些整数按照字符在字符串中的顺序排列。
string_to_int = {ch:i for i,ch in enumerate(chars)}

# 创建整数到字符的映射字典
int_to_string = {i:ch for i,ch in enumerate(chars)}

# 编码函数：将字符串转换为整数列表
encode = lambda s: [string_to_int[c] for c in s]

# 解码函数：将整数列表转换为字符串
decode = lambda l: ''.join([int_to_string[i] for i in l])

class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()  # 初始化父类
        self.key = nn.Linear(n_embd, head_size, bias=False)  # 线性变换得到 key
        self.query = nn.Linear(n_embd, head_size, bias=False)  # 线性变换得到 query
        self.value = nn.Linear(n_embd, head_size, bias=False)  # 线性变换得到 value
        # 注册 tril, 降低计算量，防止做这个计算的一些额外开销
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size))) 
        self.dropout = nn.Dropout(dropout)  # Dropout 层

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)  # 计算 key
        q = self.query(x)  # 计算 query
        wei = q @ k.transpose(-2, -1) * k.shape[-1]**-0.5  # 计算注意力权重
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))  # 掩码处理
        wei = F.softmax(wei, dim=-1)  # Softmax 得到注意力权重
        wei = self.dropout(wei)  # Dropout
        
        v = self.value(x)  # 计算 value
        out = wei @ v  # 加权求和得到输出
        return out

class MultiheadAttention(nn.Module):
    def __init__(self, num_heads,head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(num_heads*head_size, n_embd)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        out = torch.cat([head(x) for head in self.heads], dim=-1) # (B, T, F) -> (B, T, [h1, h2, h3...])
        out = self.proj(out)
        out = self.dropout(out)
        return out

class FeedForward(nn.Module):
    # 初始化方法，接受一个参数 n_embd
    def __init__(self, n_embd):
        super().__init__()
        # 定义神经网络结构
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4*n_embd),  # 线性层，输入维度 n_embd，输出维度 4*n_embd
            nn.ReLU(),  # ReLU 激活函数
            nn.Linear(4*n_embd, n_embd),  # 线性层，输入维度 4*n_embd，输出维度 n_embd
            nn.Dropout(dropout)  # dropout 层，使用 dropout 参数
            )
    # 前向传播方法，接受输入 x
    def forward(self, x):
        return self.net(x)  # 返回神经网络的前向传播结果

class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_szie = n_embd // n_head  # 计算每个头的大小
        self.sa = MultiheadAttention(n_embd, head_szie)  # 多头注意力机制
        self.ffwd = FeedForward(n_embd)  # 前馈神经网络
        self.ln1 = nn.LayerNorm(n_embd)  # 第一个 Layer Norm
        self.ln2 = nn.LayerNorm(n_embd)  # 第二个 Layer Norm
    
    def forward(self, x):
        y = self.sa(x)
        x = self.ln1(x + y)
        x = self.ffwd(x)
        x = self.ln2(x + y)
        return x

# 定义GPT语言模型类
class GPTLanguageModel(nn.Module):
    # 初始化函数
    def __init__(self, vocab_size):
        # 调用父类初始化函数
        super().__init__()
        # 定义词嵌入表
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head = n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)
        
        self.apply(self._init_weights) # 初始化权重
        
    # 初始化权重函数
    def _init_weights(self, module):
        # 初始化神经网络模块的权重
        if isinstance(module, nn.Linear):  # 如果是线性层
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)  # 使用正态分布初始化权重
            if module.bias is not None:  # 如果存在偏置项
                torch.nn.init.zeros_(module.bias)  # 将偏置项初始化为零
        elif isinstance(module, nn.Embedding):  # 如果是嵌入层
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)  # 使用正态分布初始化权重
        
        
    # 前向传播函数
    def forward(self, index, targets=None):
        
        # 词嵌入表
        # logits = self.token_embedding_table(index)
        B,T = index.shape
        '''
            B表示batch size，即一次训练或测试的样本数量。
            T表示时间步数（或序列长度），表示模型处理的时间序列长度。
            C表示类别数，表示模型输出的类别数量。
        '''
        tok_emb = self.token_embedding_table(index) # (B, T, C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T, C)
        # pos_emb = pos_emb.unsqueeze(0).expand(index.shape[0], -1, -1) # (B, T, C)
        # 将词嵌入和位置嵌入相加得到最终的嵌入
        emb = tok_emb + pos_emb # (B, T, C)
        
        emb = self.blocks(emb) # (B, T, C)
        # 层归一化
        emb = self.ln_f(emb)
        
        # 输出层
        logits = self.lm_head(emb) # (B, T, C)
        
        # 如果有标签，计算损失函数
             
        if targets is not None:
            B,T,C = logits.shape
            # 创建一个形状为(B,C)的零张量
            # 其中B是批处理大小，C是词汇大小
            # 该张量将用于存储预测的下一个词
            # logits.view(B*T, C)，将其重塑为(B*T, C)
            logits = logits.view(B*T, C)
            # targets.view(B*T)，将其重塑为(B*T)
            target = targets.view(B*T)
            # 交叉熵损失函数（CrossEntropy Loss）来计算模型预测结果（logits）与实际标签（target）之间的损失。
            loss = F.cross_entropy(logits, target)
        else:
            loss = None

        return logits, loss
    
    # 生成函数
    # index 表示输入的索引或上下文
    def generate(self, index, max_new_tokens):
        # index is (B,T) array of indices in the current context
        # 生成函数的输入是当前上下文（context）的索引（index）
        for _ in range(max_new_tokens):    
            
            if index.size(1) > block_size:
                index_cond = index[:, -block_size:]
            else:
                index_cond = index

            # 计算logits和loss
            logits, loss = self.forward(index_cond)
            # 获取logits的最后一个时间步
            # 如果targets为None，则logits.shape = (B, T, C)
            # 因此这里的 -1 是为了上面结果带来的影响，让其shape保持为（B, C）
            
            logits = logits[:, -1, :] # logits.shape = (B, C)
            
            # 将logits转换为概率分布
            probs = F.softmax(logits, dim=-1) #  (B, C)
            # 随机选择一个下一个词
            # 这里使用torch.multinomial来从概率分布中随机选择一个索引
            # 参数num_samples表示选择的样本数量，这里为1
            index_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # 将当前的索引和选择的下一个词拼接在一起
            index = torch.cat((index, index_next), dim=1) #  (B, T+1)
            
        return index

# 创建模型实例
model = GPTLanguageModel(vocab_size)

# context = torch.zeros((1,1),dtype=torch.long, device=device)
# generated_chars = decode(m.generate(context, max_new_tokens=500)[0].tolist())

# print(generated_chars)
if os.path.exists('model/model-01.pkl'):
    print('loading model parameters...')
    with open('model/model-01.pkl', 'rb') as f:
        model =pickle.load(f)
        
    print('model loaded successfully.')
else:
    print('model not found.')

# 移动模型到GPU上
m = model.to(device)


while True:
    prompt = input('Enter prompt: ')
    context = torch.tensor(encode(prompt), dtype = torch.long, device=device)
    
    # max_new_tokens=150 会报错，因为我们的block_size=128，而生成的字符的长度超过了128，所以需要截断
    # 在上方generate函数中添加 index_cond = index[:, block_size:] 解决了这个问题
    generated_chars = decode(m.generate(context.unsqueeze(0), max_new_tokens=128)[0].tolist())
    print(f'Completions: {generated_chars}')
    
    
# context = torch.zeros((1,1),dtype=torch.long, device=device)
# generated_chars = decode(m.generate(context, max_new_tokens=100)[0].tolist())

# print(generated_chars)