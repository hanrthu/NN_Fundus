import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pickle

def base_reduction(graph):
    # Simple version of image reduction
    # print(graph.shape)
    stride_x = int(graph.shape[1] / 299)
    stride_y = int(graph.shape[2] / 299)
    x_sample = np.linspace(0,graph.shape[1]-1,299).astype(int)
    y_sample = np.linspace(0,graph.shape[2]-1,299).astype(int)
    # print(x_sample.shape)
    # print(y_sample.shape)
    # graph = graph[:,::stride_x,::stride_y]
    graph = graph[:,x_sample,:]
    graph = graph[:,:,y_sample]
    # print(graph.shape)
    return graph
    # print(stride_x)
def load_fundus(data_dir,label):
    if os.path.exists('../data.pkl'):
        f = open('../data.pkl','rb')
        data = pickle.load(f)
        print('Successfully load data!')
        return data['trX'],data['teX'],data['trY'],data['teY']
    images = []
    labels = []
    with open(label) as f:
        for line in f:
            content = line.split(',')
            if content[0] == 'image_path':
                continue
            images.append(content[0])
            labels.append(int(content[2].strip('\n')))
    print(images[0])
    print(labels[0])
    labels = np.array(labels)
    data = []
    count = 0
    print(len(images))
    for image in images:
        # print(image)
        count+=1
        print(count,end='\r')
        path = os.path.join(data_dir,image)
        # print(path)
        graph = cv2.imread(path)
        # print(graph)
        graph = graph.transpose(2,1,0)
        graph = base_reduction(graph)
        # cv2.imshow('example',graph)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        data.append(graph)
    print('Done!')
    print(len(data))
    X = np.reshape(np.concatenate(data,axis=0),(2000,3,299,299))
    print(X.shape)
    trX = X[0:1800,:,:,:]
    teX = X[1800:,:,:,:]
    trY = labels[0:1800]
    teY = labels[1800:]
    trX = ((trX - 128.0) / 255.0).astype(np.float32)
    teX = ((teX - 128.0) / 255.0).astype(np.float32)
    data = {}
    data['trX'] = trX
    data['teX'] = teX
    data['trY'] = trY
    data['teY'] = teY
    pfile = open('../data.pkl','wb')
    pickle.dump(data,pfile)
    return trX,teX,trY,teY
