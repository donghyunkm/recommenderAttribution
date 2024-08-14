import torch
import torch.nn as nn
import numpy as np
import copy
import argparse
import os
from util import *
from lstm import *
from copy import deepcopy

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path',type=str,default='data/movie.tsv',help='path of the dataset (it should be tsv format; see data/movie.tsv')
    parser.add_argument('--gpu',default='0',type=str,help='GPU number that will be used')
    parser.add_argument('--output',type=str, help = "path of the model stability result")
    parser.add_argument('--epochs', default=50, type = int, help='number of training epochs')
    parser.add_argument('--max_seq_len', default=50, type = int, help='maximum sequence length')
    parser.add_argument('--test_data_ratio', default=0.1, type = float, help='last K% of interactions of each user will be used as test data')
    parser.add_argument('--batch_size', default=1024, type = int, help='mini-batch size for training')
    parser.add_argument('--learning_rate', default=0.001, type = float, help='learning rate for training')

    args = parser.parse_args()
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu

    output_path = args.output
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    available_gpus = [torch.cuda.device(i) for i in range(torch.cuda.device_count())]
    
    #Reading & cleaning the dataset
    raw_data = np.loadtxt(args.data_path, delimiter="\t")
    look_back = args.max_seq_len
    processed_data = data_cleaning(raw_data, args.test_data_ratio)

    f = open(output_path,'w')

    (train,test) = train_test_generator(processed_data,look_back)
    train2 = deepcopy(train)
    test2 = deepcopy(test)

    testAppear = {}
    
    for i in range(len(test)):
        testAppear[test[i][1]] = testAppear.get(test[i][1], 0) + 1

    sortedtestAppear = sorted(testAppear.items(), key=lambda x:x[1], reverse=True)

    sortedItem = []
    for elem in sortedtestAppear:
        sortedItem.append(elem[0])

    model = LSTM(data = processed_data,input_size=128, output_size=len(np.unique(processed_data[:,1])) + 1, hidden_dim=64, n_layers=1, device=device,args=args).to(device)
    model2 = LSTM(data = processed_data,input_size=128, output_size=len(np.unique(processed_data[:,1])) + 1, hidden_dim=64, n_layers=1, device=device,args=args).to(device)

    model.LSTM.flatten_parameters()
    model2.LSTM.flatten_parameters()

    # Train LSTM model on original dataset and obtain list of opponents with TracIn method
    top_opponents = model.traintest(train=train,test=test, testItems = sortedItem, epochs = args.epochs, original_probs=-1)

    # Train LSTM model after removing training data with adverse influence
    trainFinal = []
    for i in range(len(train2)):
        appendIfTrue = True
        for toRemove in top_opponents:
            if i == toRemove:
                appendIfTrue = False
        
        if appendIfTrue:
            trainFinal.append(train2[i])

    model2.calculateMMR(train=trainFinal, test=test2, testItems=sortedItem, epochs = args.epochs, original_probs=-1)
    

if __name__ == "__main__":
    main()
