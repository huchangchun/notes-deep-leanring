#encoding=utf-8
import time
from collections import namedtuple
import numpy as np
import tensorflow as tf

def get_batches(arr, n_seqs,n_steps):
    """
    对已有的数组进行mini-batch分割
    arr:待分割的数组
    n_seq:一个batch中序列个数
    n_steps:单个序列包含的字符数
    """
    batch_size = n_seqs * n_steps
    n_batches = int(len(arr) /batch_size)
    arr  = arr[:batch_size * n_batches]
    arr = arr.reshape((n_seqs, -1))
    for n in range(0,arr.shape[1], n_steps):
        x = arr[:,n:n+n_steps]
        y = np.zeros_like(x)
        y[:,:-1],y[:,-1] = x[:,1:],x[:,0]
        yield x,y

#模型构建：
"""
包括输入层，LSTM层，输出层，loss ，optimizer
"""
def build_inputs(batch_size, num_steps):
    """
    构建输入层
    num_steps:每个batch中的序列个数
    num_steps:每个序列包含的字符数
    """
    inputs = tf.placeholder(tf.int32, shape=(batch_size,num_steps),name = 'inputs')
    targets = tf.placeholder(tf.int32, shape=(batch_size, num_steps), name ='targets')
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')
    return inputs,targets,keep_prob

def build_lstm(lstm_size, num_layers, batch_size, keep_prob):
    """
    构建lstm层
    keep_prob 
    lstm_size:lstm隐层节点个数
    num_layers:lstm的隐层数目
    batch_size:batch size
    """
    def lstm_cell(hidder_size):
        return tf.contrib.rnn.BasicLSTMCell(num_units=hidder_size,state_is_tuple=True)
    def dropout(hidder_size):
        cell = lstm_cell(hidder_size)
        return tf.contrib.rnn.DropoutWrapper(cell,output_keep_prob=keep_prob)
    def multi_lstm(hidder_size,num_layers):
        cells = [dropout(hidder_size) for _ in range(num_layers)]
        MultiRNN_cell = tf.contrib.rnn.MultiRNNCell(cells,state_is_tuple=True)
        return MultiRNN_cell
    
    #堆叠
    cell = multi_lstm(lstm_size, num_layers)

    initial_state = cell.zero_state(batch_size, tf.float32)#
    
    return cell, initial_state
def build_output(lstm_output, in_size, out_size):
    """
    构造输出层
    lstm_output:lstm层的输出结果
    in_size:lstm输出层重塑后的size
    out_size :softmax层的size
    """
    #将lstm的输出按照concate,例如[[1,2,3],[7,8,9]],concate为[1,2,3,7,8,9]
    seq_output = tf.concat(lstm_output,1) #<tf.Tensor 'concat:0' shape=(100, 100, 512) dtype=float32>
 
    #reshape  三维变两维
    x = tf.reshape(seq_output, [-1,in_size])#<tf.Tensor 'Reshape:0' shape=(10000, 512) dtype=float32>
    
    #将lstm层与softmax层全连接
    with tf.variable_scope('softmax'):
        softmax_w = tf.Variable(tf.truncated_normal([in_size, out_size],stddev=0.1)) #（512,83）
        softmax_b = tf.Variable(tf.zeros(out_size))
    #计算logits
    logits = tf.matmul(x, softmax_w) + softmax_b
    #softmax层返回概率分布
    out = tf.nn.softmax(logits,name='predictions')
    return out, logits
def build_loss(logits, targets,lstm_size,num_class):
    """
    根据logits和target计算损失
    logits：全连接层的输出结果（不经过softmax)
    target:targets
    lstm_size
    num_classes :vocab_size
    """
    
    #one_hot
    y_one_hot = tf.one_hot(targets, num_class)
    y_reshaped = tf.reshape(y_one_hot,logits.get_shape())
    
    loss = tf.nn.softmax_cross_entropy_with_logits(logits=logits,labels=y_reshaped)
    loss = tf.reduce_mean(loss)

"""
我们知道RNN会遇到梯度爆炸（gradients exploding）和梯度弥散（gradients disappearing)的问题。
LSTM解决了梯度弥散的问题，但是gradient仍然可能会爆炸，因此我们采用gradient clippling的方式来防止梯度爆炸。
即通过设置一个阈值，当gradients超过这个阈值时，就将它重置为阈值大小，这就保证了梯度不会变得很大。
"""
def build_optimizer(loss, learning_rate, grad_clip):
    """
    构造 Optimizer
    loss:损失
    learning_rate:学习率
    """
    #使用clipping gradients
    tvars = tf.trainable_variables()
    grads, _ = tf.clip_by_global_norm(tf.gradients(loss,tvars),grad_clip)
    train_op = tf.train.AdamOptimizer(learning_rate)
    optimizer = train_op.apply_gradients(zip(grads,tvars))
    return optimizer

class CharRNN:
    def __init__(self, num_classes, batch_size=64, num_steps=50, 
                       lstm_size=128, num_layers=2, learning_rate=0.001, 
                       grad_clip=5, sampling=False):
        #如果sampling是True则采用SGD
        if sampling == True:
            batch_size ,num_steps = 1,1
        else:
            batch_size, num_steps = batch_size,num_steps
        tf.reset_default_graph()
        # 输入层
        self.inputs, self.targets, self.keep_prob = build_inputs(batch_size, num_steps)
    
        # LSTM层 # shape=(100, 512)
        cell, self.initial_state = build_lstm(lstm_size, num_layers, batch_size, self.keep_prob)
    
        # 对输入进行one-hot编码
        x_one_hot = tf.one_hot(self.inputs, num_classes)#<tf.Tensor 'one_hot:0' shape=(100, 100, 83) dtype=float32>
     #shape=(100, 100, 83) 
        # 运行RNN
        outputs, state = tf.nn.dynamic_rnn(cell, x_one_hot, initial_state=self.initial_state)
        self.final_state = state
    
        # 预测结果
        self.prediction, self.logits = build_output(outputs, lstm_size, num_classes)
    
        # Loss 和 optimizer (with gradient clipping)
        self.loss = build_loss(self.logits, self.targets, lstm_size, num_classes)
        self.optimizer = build_optimizer(self.loss, learning_rate, grad_clip)
        


with open("./data/anna.txt",'r') as f:
    text = f.read()
vocab  = set(text)
vocab_to_int = {c:i for i ,c in enumerate(vocab)}
int_to_vocab = dict(enumerate(vocab))
print(text[:100])
encoded = np.array([vocab_to_int[c] for c in text],dtype =np.int32)
print(encoded[:100])
batches = get_batches(encoded, 10, 50)
batch_size = 100
num_steps = 100
lstm_size = 512
num_layers = 2
learning_rate =0.001
keep_prob = 0.5
epochs = 20

#每n轮进行一次变量保存
save_every_n = 200
model = CharRNN(len(vocab), batch_size=batch_size, num_steps=num_steps,
                lstm_size=lstm_size, num_layers=num_layers, 
                learning_rate=learning_rate)
saver = tf.train.Saver(max_to_keep=100)
tensorboard_dir = "./log_anna"
if not os.path.exists(tensorboard_dir):
    os.mkdir(tensorboard_dir)

with tf.Session() as sess:
    writer = tf.summary.FileWriter(tensorboard_dir,sess.graph) 
    sess.run(tf.global_variables_initializer())
    counter = 0
    for e in range(epochs):
        #train network
        new_state = sess.run(model.initial_state)
        loss = 0
        for x,y in get_batches(encoded,batch_size,num_steps):
            counter += 1
            start = time.time()
            feed = {model.inputs:x,
                    model.targets:y,
                    model.keep_prob:keep_prob,
                    model.initial_state:new_state}
            batch_loss,new_state,_= sess.run([model.loss,
                                              model.final_state,
                                              model.optimizer],
                                              feed_dict =feed)
            end = time.time()
            if counter % 100 == 0:
                print("epoch:{}/{}...".format(e+1,epochs),
                      "train steps:{} ...".format(counter),
                      "train error:{:.4f}...".format(batch_loss),
                      "{:.4f} sec/batch".format((end-start))
                      )
            if(counter % save_every_n == 0):
                saver.save(sess,"checkpoints/i{}_l{}.ckpt".format(counter,lstm_size))
        saver.save(sess,"checkpoints/i{}_l{}.ckpt".format(counter,lstm_size))
                
                
tf.train.get_checkpoint_state('checkpoints')

def pick_top_n(preds,vocab_size, top_n = 5):
    """
    从预测结果中选取前top_n个最可能的字符
    preds:预测结果
    vocab_size
    top_n
    """
    p = np.squeeze(preds)
    p[np.argsort(p)[:-top_n]] =0
    #归一化概率
    p = p/np.sum(p)
    c = np.random.choice(vocab_size, 1, p=p)[0]
    return c
def sample(checkpoint, n_samples,lstm_size,vocab_size,prime="The "):
    """
    生成新文本
    checkpoint:某轮迭代的参数文件
    n_sample:新文本的字符产股
    lstm_size:隐层节点
    vocab_size
    prime:起始文本
    
    """
    #输入单词转换为单个字符
    samples = [c for c in prime]
    model = CharRNN(len(vocab),lstm_size,sampling=True)
    saver= tf.train.Saver()
    with tf.Session() as sess:
        #加载训练好的模型参数，恢复训练
        saver.restore(sess,checkpoint)
        new_state = sess.run(model.initial_state)
        for c in prime:
            x = np.zeros((1,1))
            x[0,0] = vocab_to_int[c]
            feed = {model.inputs:x,
                    model.keep_prob:1,
                    model.initial_state:new_state}
            preds,new_state = sess.run([model.prediction,model.final_state],feed_dict=feed)
        c = pick_top_n(preds,len(vocab))
        #不断生成字符，知道达到指定数目
        for i in range(n_samples):
            x[0,0]=c
            feed = {model.inputs:x,
                    model.keep_prob:1,
                    model.initial_state:new_state}
            preds,new_state = sess.run([model.prediction,model.final_state],
                                       feed_dict=feed)
            c = pick_top_n(preds,len(vocab))
            samples.append(int_to_vocab[c])
            
        return ''.join(samples)
tf.train.latest_checkpoint('checkpoint')

#选用最终的训练参数作为输入进行文本生成
checkpoint =tf.train.latest_checkpoint('checkpoing')
samp = sample(checkpoint,2000, lstm_size, len(vocab))