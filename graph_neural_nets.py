#Python script showing the implementation of graph neural networks for classification purposes. 
#Dataset Used: The Cora  Corpus Dataset
#Objective: Using Graph Neural Networks for Classification purposes
#Motivation: Graph Neural Networks have been known to perform extremely well in structured and unstructured data. Here, their performance is compared against that of feed forward recurrent neural networks
#Code adopted from: https://github.com/imayachita/Graph_Convolutional_Networks_Node_Classification/blob/master/Node_Classification_GCN_Semi-Sup_CORA.ipynb
#GCN Paper: T. Kipf and M. Welling, Semi-Supervised Classification with Graph Convolutional Networks (2017). arXiv preprint arXiv:1609.02907. ICLR 2017 
#Spektral paper: D. Grattarola and C. Alippi, Graph Neural Networks in TensorFlow and Keras with Spektral (2020). arXiv:2006.12138. ICML 2020 - GRL+ Workshop
import numpy as np
import os
import networkx as nx
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from sklearn.utils import shuffle
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from spektral.layers import GCNConv
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dropout, Dense
from tensorflow.keras import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping
from tensorflow.keras.regularizers import l2
from collections import Counter
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

#The CORA Dataset
#cora.content ==> Has papers(nodes) plus their features(node_features)
#cora.cites ==> our edges
#The graph network created here is undirected
#Loading the data
all_data = []
all_edges = []

for root, dirs, files, in os.walk('/home/dev-works/Desktop/tensorflow/GCN/cora'):
    for file in files:
        if '.content' in file:
            with open(os.path.join(root, file), 'r') as f:
                all_data.extend(f.read().splitlines())
        elif 'cites' in file:
            with open(os.path.join(root, file), 'r') as f:
                all_edges.extend(f.read().splitlines())

#the data is then shuffled to remove the ordering in the data
random_state = 77
all_data = shuffle(all_data, random_state = random_state)
#Recap of the CORA DATASET
#the core.content file has the first element indicating the node(paper) name. 
#the second to the second last are node features of the paper. 
#the last element represents the category(label of node) of the paper. 
#The cora.cites file contains a tuple indicating the tuple of connected nodes
#PARSING THE DATA
labels = []
nodes = []

X = []

for i, data in enumerate(all_data):
    elements = data.split('\t')
    labels.append(elements[-1])
    X.append(elements[1:-1])
    nodes.append(elements[0])
    #the above lines instantiate the nodes, the node features(X) and the labels
X = np.array(X, dtype=int)
N = X.shape[0] #number of nodes
F = X.shape[1] #the size of node features
print ('X shape:' , X.shape)

#PARSING THE EDGEA
edge_list = []
for edge in all_edges:
    e = edge.split('\t')
    edge_list.append((e[0], e[1]))
print('\nNumber of nodes (N): ', N)
print('\nNumber of features (F) of each node: ', F)
print('\nCategories: ', set(labels))

num_classes = len(set(labels))
print('\nNumber of paper categories: ', num_classes)

#WE THEN SELECT EXAMPLES FOR TRAINING, VALIDATION AND TEST SETS:
#THEREAFTER WE APPLY MASKING TO THE DATA FOR THE TRAINING


def limit_data(labels,limit=20,val_num=500,test_num=1000):
    '''
    Get the index of train, validation, and test data
    '''
    label_counter = dict((l, 0) for l in labels)
    train_idx = []

    for i in range(len(labels)):
        label = labels[i]
        if label_counter[label]<limit:
            #add the example to the training data
            train_idx.append(i)
            label_counter[label]+=1
        
        #exit the loop once we found 20 examples for each class
        if all(count == limit for count in label_counter.values()):
            break
    
    #get the indices that do not go to traning data
    rest_idx = [x for x in range(len(labels)) if x not in train_idx]
    #get the first val_num
    val_idx = rest_idx[:val_num]
    test_idx = rest_idx[val_num:(val_num+test_num)]
    return train_idx, val_idx,test_idx


train_idx,val_idx,test_idx = limit_data(labels)


#Applying the mask
train_mask = np.zeros((N,), dtype=bool)
train_mask[train_idx] = True

val_mask = np.zeros((N,), dtype=bool)
val_mask[val_idx] = True

test_mask = np.zeros((N, ),dtype=bool)
test_mask[test_idx] = True

#Onehot Encoding the labels
def encode_label(labels):
    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(labels)
    labels = to_categorical(labels)
    return labels, label_encoder.classes_
labels_encoded, classes = encode_label(labels)
#we then build a graph using NetworkX(Adjacency Matrix) usingt he obtained nodes and edges list
G = nx.Graph()
G.add_nodes_from(nodes)
G.add_edges_from(edge_list)
A = nx.adjacency_matrix(G)
print('Graph info: ', nx.info(G))

#We now build and train the Graph Convolution Networks
#initializing the parameters
channels = 16 #number of channels in the first layer
dropout = 0.5 
l2_reg = 5e-4
learning_rate = 1e-2 
epochs = 200
es_patience = 10 
A = GCNConv.preprocess(A).astype('f4')

#defining the model
X_in = Input(shape = (F, ))
fltr_in = Input((N, ), sparse=True)
dropout_1= Dropout(dropout)(X_in)
graph_conv_1 = GCNConv(channels, activation='relu', 
        kernel_regularizer = l2(l2_reg), 
        use_bias=False)([dropout_1, fltr_in])
dropout_2 = Dropout(dropout)(graph_conv_1)
graph_conv_2 = GCNConv(num_classes, activation='softmax', 
        use_bias=False)([dropout_2, fltr_in])
#we then build the mode as follows:
model = Model(inputs=[X_in, fltr_in], outputs = graph_conv_2)
optimizer = Adam(lr=learning_rate)
model.compile(optimizer=optimizer, loss = 'categorical_crossentropy', weighted_metrics=['acc'])
model.summary()

'''
tbCallBack_GCN = tf.keras.callbacks.Tensorboard(
        log_dir = './Tensorboard_GCN_cora', 
        )
callback_GCN = tbCallBack_GCN
'''
validation_data = ([X, A], labels_encoded, val_mask)
model.fit([X, A], 
        labels_encoded, 
        sample_weight = train_mask, 
        epochs = epochs, 
        batch_size = N, 
        validation_data = validation_data, 
        shuffle = False, 
        callbacks = [
            EarlyStopping(patience = es_patience, restore_best_weights=True)])
        #The classification report of the model can then be obtained as follows:
'''
X_te  = X[test_mask]
A_te = A[test_mask,:][:,test_mask]
y_te = labels_encoded[test_mask]

y_pred = model.predict([X_te, A_te], batch_size=N)
report = classification_report(np.argmax(y_te,axis=1), np.argmax(y_pred,axis=1), target_names=classes)
print('GCN Classification Report: \n {}'.format(report))

#The Hidden layer representation of the Graph conv net can be obtained as follows:
layer_outputs = [layer.output for layer in model.layers]
activation_model = Model(inputs=model.input, outputs=layer_outputs)
activations = activation_model.predict([X,A],batch_size=N)

#Get t-SNE Representation
#get the hidden layer representation after the first GCN layer
x_tsne = TSNE(n_components=2).fit_transform(activations[3])
def plot_tSNE(labels_encoded, x_tsne):
    color_map = np.argmax(labels_encoded, axis=1)
    plt.figure(figsize=(10,10))
    for cl in range(num_classes):
        indices = np.where(color_map == cl)
        indices = indices[0]
        plt.scatter(x_tsne[indices, 0], x_tsne[indices, 1], label=cl)
    plt.legend()
    plt.title('tSNE Distribution of Different Paper Categories using Graph Neural Nets')
    plt.show()
plot_tSNE(labels_encoded, x_tsne)
'''
#A feed forward Neural Net is then Compiled as follows to compare the results
es_patience = 10
optimizer = Adam(lr=1e-2)
l2_reg = 5e-4
epochs = 200

#Compare with FNN
#Construct the model
model_fnn = Sequential()
model_fnn.add(Dense(
                    128,
                    input_dim=X.shape[1],
                    activation=tf.nn.relu,
                    kernel_regularizer=tf.keras.regularizers.l2(l2_reg))
             )
model_fnn.add(Dropout(0.5))
model_fnn.add(Dense(256, activation=tf.nn.relu))
model_fnn.add(Dropout(0.5))
model_fnn.add(Dense(num_classes, activation=tf.keras.activations.softmax))


model_fnn.compile(optimizer=optimizer,
              loss='categorical_crossentropy',
              weighted_metrics=['acc'])


#define TensorBoard
tbCallBack_FNN = TensorBoard(
    log_dir='./Tensorboard_FNN_cora',
)

#Train model
validation_data_fnn = (X, labels_encoded, val_mask)
model_fnn.fit(
                X,labels_encoded,
                sample_weight=train_mask,
                epochs=epochs,
                batch_size=N,
                validation_data=validation_data_fnn,
                shuffle=False,
                callbacks=[
                  EarlyStopping(patience=es_patience,  restore_best_weights=True),
                  tbCallBack_FNN
          ])
#The feed forward NN has an accuracy of 55%. Compared to GCN's accuracy of 76%, this is pretty low. 
#The difference in accuracy shows the effectiveness of Graph neural networks in solving problems involving unstructured data. 
#Common application areas of GNN include social media reccomendations, studying molecular structure of materials and many more. 
