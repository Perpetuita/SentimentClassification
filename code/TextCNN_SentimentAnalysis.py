#coding: utf-8
import os
import sys
import torch
import random
import collections
from torch import nn
from tqdm import tqdm
import d2lzh_pytorch as d2l
import torchtext.vocab as Vocab
import torch.utils.data as Data
import torch.nn.functional as F

sys.path.append("..")
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DATA_ROOT = "DATA"  #当前文件目录下存放数据的文件名称

'''================预处理函数================'''
#从目录下读取训练数据
def read_imdb(folder= 'train', data_root= 'DATA/aclimdb'):
    data = []
    for label in ['pos', 'neg']:
        folder_name = os.path.join(data_root, folder, label)
        for file in tqdm(os.listdir(folder_name)):  #利用tqdm显示读取进度
            with open(os.path.join(folder_name, file), 'rb') as f:
                review = f.read().decode('utf-8').replace('\n', '').lower()
                data.append([review, 1 if label == 'pos' else 0])
    random.shuffle(data)  #将数据随机打乱
    return data

#对每一条评论做分词处理
def get_tokenized_imdb(data):
    '''
    :param data: [review, label] where label is 0 or 1, pos is 1 and neg is 0
    :return: [review]
    '''
    def tokenizer(text):
        return [tok.lower() for tok in text.split(' ')]
    return [tokenizer(review) for review, _ in data]

#统计语料库中的词语个数并建立词汇表
def get_vocab_imdb(data):
    '''
    :param data: all reviews and  each review is a list of words
    :return: vocab.vocab(counter, min_freq)
    '''
    tokenized_data = get_tokenized_imdb(data)
    counter = collections.Counter([tk for st in tokenized_data for tk in st])
    return Vocab.Vocab(counter, min_freq=5)

#对每一条评论进行分词，通过词典转换成索引，然后通过截断或补齐使语句长度一致
def preprocess_imdb(data, vocab):
    max_l = 500  #通过截断或者补0使每一条语句的长度固定为500

    def pad(x):
        return x[ : max_l] if len(x) > max_l else x + [0] * (max_l - len(x))

    tokenized_data = get_tokenized_imdb(data)
    features = torch.tensor([pad([vocab.stoi[word] for word in words]) for words in tokenized_data])
    labels = torch.tensor([score for _, score in data])
    return features, labels

#根据预训练词向量将词转化为嵌入矩阵
def load_pretrained_embedding(words, pretrained_vocab):
    '''
    :param words: 根据语料库得到的词
    :param pretrained_vocab: 预训练的词向量词典
    :return: embedding matrix
    '''
    embed = torch.zeros(len(words), pretrained_vocab.vectors[0].shape[0])
    oov_count = 0  #out of vocabulary
    for i, word in enumerate(words):  #将一个可遍历对象组合为一个索引序列
        try:
            idx = pretrained_vocab.stoi[word]
            embed[i, : ] = pretrained_vocab.vectors[idx]
        except KeyError:
            oov_count += 1
    if oov_count > 0:
        print("There are %d out of vocabulary words." % oov_count)   #====打印输出====
    return embed

'''================预处理函数END================'''

#一维卷积
def corrld(X, K):
    '''
    :param X: 输入数组
    :param K: 卷积核数组
    :return: 返回数组Y
    '''
    w = K.shape[0]
    Y = torch.zeros((X.shape[0] - w + 1))
    for i in range(Y.shape[0]):
        Y[i] = (X[i : i + w] * K).sum()
    return Y

#多通道卷积计算
def corrld_multi_in(X, K):
    """
    :param X: 多通道的输入
    :param K: 多通道卷积核
    :return: 卷积后的结果
    """
    return torch.stack([corrld(x, k) for x, k in zip(X, K)]).sum(dim=0)

# X = torch.tensor([[0, 1, 2, 3, 4, 5, 6],
#                   [1, 2, 3, 4, 5, 6, 7],
#                   [2, 3, 4, 5, 6, 7, 8]])
# K = torch.tensor([[1, 2],
#             [3, 4],
#             [-1, -3]])
# print(corrld_multi_in(X, K))

#时序最大池化层
class GlobalMaxPool1d(nn.Module):
    def __init__(self):
        super(GlobalMaxPool1d, self).__init__()

    def forward(self, x):
        #x shape: (batch_size, channel, seq_len)
        #return shape: (batch_size, channel, 1)
        return F.max_pool1d(x, kernel_size= x.shape[2])

#TextCNN 模型
class TextCNN(nn.Module):
    def __init__(self, vocab, embed_size, kernel_sizes, num_channels):
        super(TextCNN, self).__init__()
        self.embedding = nn.Embedding(len(vocab), embed_size)
        self.constant_embedding = nn.Embedding(len(vocab), embed_size)  #不参与训练的嵌入层
        self.dropout = nn.Dropout(0.5)
        self.decoder = nn.Linear(sum(num_channels), 2)
        self.pool = GlobalMaxPool1d()  #全局最大池化层
        self.convs = nn.ModuleList()  #创建一个一维卷积
        #根据传入的参数设置不同的滤波尺寸
        for c, k in zip(num_channels, kernel_sizes):
            self.convs.append(nn.Conv1d(in_channels= 2 * embed_size,
                                        out_channels= c,
                                        kernel_size= k))

    def forward(self, inputs):
        embeddings = torch.cat((self.embedding(inputs),
                                self.constant_embedding(inputs)), dim= 2)
        embeddings = embeddings.permute(0, 2, 1)  #将张量的维度换位 (1, 2, 3) -> (1, 3, 2)由一个2*3的变成了3个1*2的
        encoding = torch.cat([self.pool(F.relu(conv(embeddings))).squeeze(-1) for conv in self.convs], dim= 1)
        outputs = self.decoder(self.dropout(encoding))
        return outputs

#定义预测函数
def predict_sentiment(net, vocab, sentence):
    """sentence是词语的列表"""
    device = list(net.parameters())[0].device
    sentence = torch.tensor([vocab.stoi[word] for word in sentence], device=device)
    label = torch.argmax(net(sentence.view((1, -1))), dim=1)
    return 'positive' if label.item() == 1 else 'negative'

#读取和预处理IMDB数据集
batch_size = 64
train_data, test_data = read_imdb('train'), read_imdb('test')  #读出训练数据和测试数
vocab = get_vocab_imdb(train_data)
train_set = Data.TensorDataset(*preprocess_imdb(train_data, vocab))
test_set = Data.TensorDataset(*preprocess_imdb(test_data, vocab))
train_iter = Data.DataLoader(train_set, batch_size, shuffle= True)
test_iter = Data.DataLoader(test_set, batch_size)

'''==============================建立模型并训练数据啦 ٩(๑❛ᴗ❛๑)۶ =============================='''
#创建一个TextCNN实例，三个卷积层，核尺寸为3*4*5， 输出通道数均为100
embed_size, kernel_sizes, nums_channels = 100, [3, 4, 5], [100, 100, 100]  #embed_size要和GloVe词向量维度保持一致
net = TextCNN(vocab, embed_size, kernel_sizes, nums_channels)

#加载预训练词向量
glov_vocab = Vocab.GloVe(name= '6B', dim= 100, cache= os.path.join(DATA_ROOT, 'glove'))
net.embedding.weight.data.copy_(load_pretrained_embedding(vocab.itos, glov_vocab))
net.constant_embedding.weight.data.copy_(load_pretrained_embedding(vocab.itos, glov_vocab))
net.constant_embedding.weight.requires_garde = False

#训练并评估模型
lr, epochs = 0.001, 2
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr= lr)
loss = nn.CrossEntropyLoss()
d2l.train(train_iter, test_iter, net, loss, optimizer, device, epochs)

#预测英文评论的情感倾向
sentence = "users' comments"
while sentence.lower() != "end judge":
    sentence = input("输入预测句子：")
    tokenized = sentence.split(' ')
    print(predict_sentiment(net, vocab, tokenized))

''''===============================END Training (〃'▽'〃) ==============================='''