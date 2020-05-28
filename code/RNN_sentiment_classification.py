#coding: utf-8
import os
import sys
import torch
import random
import tarfile
import collections
from tqdm import tqdm
from torch import nn
import d2lzh_pytorch as d2l
import torchtext.vocab as Vocab
import torch.utils.data as Data

sys.path.append("..")

os.environ["CUDA_VISIBLE_DEVICES"] = '0'  #设置当前使用的GPU编号
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  #设置计算工具为GPU为CPU
DATA_ROOT = "DATA"  #当前文件目录下存放数据的文件名称

#从目录下读取训练数据
def read_imdb(folder= 'train', data_root= 'DATA/aclimdb'):
    data = []
    for label in ['pos', 'neg']:
        folder_name = os.path.join(data_root, folder, label)
        for file in tqdm(os.listdir(folder_name)):  #利用tqdm显示读取进度
            with open(os.path.join(folder_name, file), 'rb') as f:
                review = f.read().decode('utf-8').replace('\n', '').lower()  #读取数据时将换行符去掉
                data.append([review, 1 if label == 'pos' else 0])
    random.shuffle(data)  #将数据随机打乱
    return data  #将读出的带有极性标签的评论信息以列表的方式返回

#对每一条评论中的单词转化为小写
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
    tokenized_data = get_tokenized_imdb(data)  #将评论单词转化为lower_case
    counter = collections.Counter([tk for st in tokenized_data for tk in st])  #记录每个所有句子中出现的单词的次数[word, cnt]
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
    embed = torch.zeros(len(words), pretrained_vocab.vectors[0].shape[0])  #初始化嵌入层
    oov_count = 0  #out of vocabulary
    for i, word in enumerate(words):  #将一个可遍历对象组合为一个索引序列
        try:
            idx = pretrained_vocab.stoi[word]  #获取单词在词汇表中的位置
            embed[i, : ] = pretrained_vocab.vectors[idx]  #设置嵌入层对应位置的词向量
        except KeyError:
            oov_count += 1
    if oov_count > 0:
        print("There are %d out of vocabulary words." % oov_count)   #====打印输出====
    return embed

#finally: 定义预测函数
def predict_sentiment(net, vocab, sentence):
    '''
    :param net: 神经网络模型
    :param vocab: 记录单词的此汇表，包括频率和向量表示
    :param sentence: 列表存储的句子形式以单词为单位
    :return:
    '''
    device = list(net.parameters())[0].device
    sentence = torch.tensor([vocab.stoi[word] for word in sentence], device= device)
    label = torch.argmax(net(sentence.view((1, -1))), dim= 1)
    return 'positive' if label.item() == 1 else 'negative'

#使用循环神经网络模型
class BiRNN(nn.Module):
    def __init__(self, vocab, embed_size, num_hiddens, num_layers):
        super(BiRNN, self).__init__()
        self.embedding = nn.Embedding(len(vocab), embed_size)
        self.encoder = nn.LSTM(input_size= embed_size,
                               hidden_size= num_hiddens,
                               num_layers= num_layers,
                               bidirectional= True)

        # self.encoder = nn.RNN(input_size= embed_size,   #对照组，RNN和LSTM网络的对比
        #                        hidden_size= num_hiddens,
        #                        num_layers= num_layers,
        #                        bidirectional= True)

        #初始时间步和最终时间步的隐藏状态作为全连接层输入
        self.decoder = nn.Linear(4 * num_hiddens, 2)

    def forward(self, inputs):
        embeddings = self.embedding(inputs.permute(1, 0))
        outputs, _ = self.encoder(embeddings)
        #连接初始时间步和最终时间步的隐藏状态作为全连接层的输入，它的形状为(批量大小， 4 * 隐层单元的个数)
        encoding = torch.cat((outputs[0], outputs[-1]), -1)
        outs = self.decoder(encoding)
        return outs

'''=================================读取数据并预处理================================='''
train_data, test_data = read_imdb('train'), read_imdb('test')  #读出训练数据和测试数据
vocab = get_vocab_imdb(train_data)
print("#words in vacab: ", len(vocab))

#创建数据迭代器，每次返回一个数据的mini_batch
batch_size = 64
train_set = Data.TensorDataset(*preprocess_imdb(train_data, vocab))
test_set = Data.TensorDataset(*preprocess_imdb(test_data, vocab))
train_iter = Data.DataLoader(train_set, batch_size, shuffle= True)
test_iter = Data.DataLoader(test_set, batch_size)

#打印第一个mini_batch数据的形状和训练集中mini_batch的数量
for X, y in train_iter:
    print('X', X.shape, 'y', y.shape)
    break
print('#batches: ', len(train_iter))
'''==============================读取数据并预处理 END =============================='''

'''==============================建立模型并训练数据啦 ٩(๑❛ᴗ❛๑)۶ =============================='''
#创建一个含有两个隐藏层的双向循环神经网络
embed_size, num_hiddens, num_layers = 100, 100, 2  #嵌入层维度 隐层特征数量 网络层数
net = BiRNN(vocab, embed_size, num_hiddens, num_layers)  #建立一个双向的循环神经网络


#加载预训练的词向量
glove_vocab = Vocab.GloVe(name= '6B', dim= 100, cache= os.path.join(DATA_ROOT, "glove"))

#设置嵌入层词向量
net.embedding.weight.data.copy_(load_pretrained_embedding(vocab.itos, glove_vocab))  #将训练好的词向量传入到网络中
net.embedding.weight.requires_grad = False  #反向传播的过程中不更新词向量


#设置模型参数并训练模型
lr, num_epochs = 0.001, 20  #lr is learning_rate; num_epochs is #iterations
loss = nn.CrossEntropyLoss()  #使用交叉熵损失函数
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr= lr)  #设置Adam优化算法的参数; Betas的参数默认为0.9和0.999
d2l.train(train_iter, test_iter, net, loss, optimizer, device, num_epochs)  #开始训练模型

#预测英文评论的情感倾向
sentence = "users' comments"
while sentence.lower() != "end judge":
    sentence = input("输入预测句子：")
    tokenized = sentence.split(' ')
    print(predict_sentiment(net, vocab, tokenized))

'''===============================END Training (〃'▽'〃) ==============================='''
