import math
import os
import random
import time
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.nn import functional as F

from recommenderAttribution.util import *


class LSTM(nn.Module):
    def __init__(self, data, input_size, output_size, hidden_dim, args, n_layers=1, device="cpu"):
        super(LSTM, self).__init__()

        self.num_items = output_size
        self.device = device
        self.emb_length = input_size
        self.batch_size = args.batch_size
        self.item_emb = nn.Embedding(self.num_items, self.emb_length, padding_idx=0)
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.learning_rate = args.learning_rate

        self.LSTM = nn.LSTM(input_size, hidden_dim, n_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_size)

    def forward(self, x):
        batch_size = x.size(0)
        hidden = self.init_hidden(batch_size)
        out, hidden = self.LSTM(x, hidden)
        inp = out[:, -1, :].contiguous().view(-1, self.hidden_dim)  # only last time step
        out = self.fc(inp)

        return out, hidden

    def init_hidden(self, batch_size):
        hidden = (
            torch.zeros(self.n_layers, batch_size, self.hidden_dim).to(self.device).detach(),
            torch.zeros(self.n_layers, batch_size, self.hidden_dim).to(self.device).detach(),
        )
        return hidden

    def traintest(self, train, test, testItems, epochs, original_probs):
        idx_to_item = items_dic()
        total_train_num = len(train)

        current_labels = []

        for i in range(total_train_num):
            train[i][0] = self.item_emb(torch.LongTensor(train[i][0]).to(self.device))
            current_labels.append(train[i][1])
        train_out = torch.LongTensor(current_labels).to(self.device)
        total_test_num = len(test)
        current_labels = []
        for i in range(total_test_num):
            test[i][0] = self.item_emb(torch.LongTensor(test[i][0]).to(self.device))
            current_labels.append(test[i][1])

        test_out = torch.LongTensor(current_labels).to(self.device)

        weights = []

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=1e-5)
        start_time = time.time()

        out_file = open("log.txt", "a")

        probs = [0 for i in range(total_test_num)]
        MRR, HITS = 0, 0
        for epoch in range(epochs):
            train_loss = 0
            for iteration in range(int(total_train_num / self.batch_size) + 1):
                st_idx, ed_idx = iteration * self.batch_size, (iteration + 1) * self.batch_size
                if ed_idx > total_train_num:
                    ed_idx = total_train_num

                optimizer.zero_grad()
                output, hidden = self.forward(torch.stack([train[i][0] for i in range(st_idx, ed_idx)], dim=0).detach())
                loss = criterion(output, train_out[st_idx:ed_idx])
                loss.backward()
                train_loss += loss.item()
                optimizer.step()

            if epoch % 30 == 15:
                torch.save(self.state_dict(), f"model_v1_epoch_{epoch}")
                weights.append(f"model_v1_epoch_{epoch}")

            text = "Epoch {}\tTrain Loss: {}\tElapsed time: {} \n".format(
                epoch, train_loss / total_train_num, time.time() - start_time
            )
            out_file.write(text)

            start_time = time.time()

        for iteration in range(int(total_test_num / self.batch_size) + 1):
            st_idx, ed_idx = iteration * self.batch_size, (iteration + 1) * self.batch_size
            if ed_idx > total_test_num:
                ed_idx = total_test_num
            output, hidden = self.forward(torch.stack([test[i][0] for i in range(st_idx, ed_idx)], dim=0).detach())
            test_loss = criterion(output, test_out[st_idx:ed_idx])

            output = output.view(-1, self.num_items)
            prob = nn.functional.softmax(output, dim=1).data.cpu()
            np_prob = prob.numpy()
            current_val = np.zeros((np_prob.shape[0], 1))
            for i in range(st_idx, ed_idx):
                current_test_label = test[i][1]
                current_val[i - st_idx, 0] = np_prob[i - st_idx, current_test_label]

            new_prob = np_prob - current_val
            ranks = np.count_nonzero(new_prob > 0, axis=1)

            for i in range(st_idx, ed_idx):
                rank = ranks[i - st_idx] + 1
                MRR += 1 / rank
                HITS += 1 if rank <= 10 else 0
                probs[i] = np_prob[i - st_idx, :]

        MRR /= total_test_num
        HITS /= total_test_num
        text = "\n\n Test MRR = {}\tTest Recall@10 = {}\n\n".format(MRR, HITS)
        out_file.write(text)

        out_file.close()

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=1e-5)
        LR = 0.001
        score_matrix = np.zeros(
            (int(total_train_num))
        )  # score_matrix aggregates influence of each training point on model's prediction
        train_grad_saved = [[] for _ in range(total_train_num)]
        start_time = time.time()

        for train_id in range(total_train_num):
            if train_id % 5000 == 0:
                print("Train_id {}\tElapsed time: {}".format(train_id, time.time() - start_time))
                start_time = time.time()

                total_memory, used_memory, free_memory = map(int, os.popen("free -t -m").readlines()[-1].split()[1:])
                print("RAM memory % used:", round((used_memory / total_memory) * 100, 2))

            for i, w in enumerate(weights):
                optimizer.zero_grad()
                self.load_state_dict(torch.load(w))
                self.train()
                output, hidden = self.forward(train[train_id][0].unsqueeze(0).detach())
                loss = criterion(output, train_out[train_id].unsqueeze(0))
                loss.backward()
                train_grad = torch.cat([
                    param.grad.reshape(-1).cpu() for param in self.parameters() if param.grad is not None
                ])
                train_grad_saved[train_id].append(train_grad)

        start_time = time.time()

        self.batch_size = 128

        for iteration in range(int(total_test_num / self.batch_size) + 1):
            if iteration % 2 == 0:
                print("iteration {}\tElapsed time: {}".format(iteration, time.time() - start_time))
                start_time = time.time()
                total_memory, used_memory, free_memory = map(int, os.popen("free -t -m").readlines()[-1].split()[1:])
                print("RAM memory % used:", round((used_memory / total_memory) * 100, 2))

            for i, w in enumerate(weights):
                optimizer.zero_grad()
                self.load_state_dict(torch.load(w))
                self.train()

                st_idx, ed_idx = iteration * self.batch_size, (iteration + 1) * self.batch_size
                if ed_idx > total_test_num:
                    ed_idx = total_test_num
                output, hidden = self.forward(torch.stack([test[i][0] for i in range(st_idx, ed_idx)], dim=0).detach())
                loss = criterion(output, test_out[st_idx:ed_idx])

                loss.backward()
                test_grad = torch.cat([param.grad.reshape(-1) for param in self.parameters() if param.grad is not None])

                for j in range(total_train_num):
                    score_matrix[j] += LR * torch.dot(
                        train_grad_saved[j][i].to(self.device), test_grad
                    )  # TracIn formula

        top_opponents = np.argsort(score_matrix)[:500]

        return top_opponents

    def calculate_mmr(self, train, test, testItems, epochs, original_probs):
        idx_to_item = items_dic()
        total_train_num = len(train)

        current_labels = []

        for i in range(total_train_num):
            train[i][0] = self.item_emb(torch.LongTensor(train[i][0]).to(self.device))
            current_labels.append(train[i][1])
        train_out = torch.LongTensor(current_labels).to(self.device)

        total_test_num = len(test)
        current_labels = []
        for i in range(total_test_num):
            test[i][0] = self.item_emb(torch.LongTensor(test[i][0]).to(self.device))
            current_labels.append(test[i][1])

        test_out = torch.LongTensor(current_labels).to(self.device)

        weights = []

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=1e-5)
        start_time = time.time()

        out_file = open("log.txt", "a")

        probs = [0 for i in range(total_test_num)]
        MRR, HITS = 0, 0
        for epoch in range(epochs):
            train_loss = 0

            for iteration in range(int(total_train_num / self.batch_size) + 1):
                st_idx, ed_idx = iteration * self.batch_size, (iteration + 1) * self.batch_size
                if ed_idx > total_train_num:
                    ed_idx = total_train_num

                optimizer.zero_grad()
                output, hidden = self.forward(torch.stack([train[i][0] for i in range(st_idx, ed_idx)], dim=0).detach())
                loss = criterion(output, train_out[st_idx:ed_idx])
                loss.backward()
                train_loss += loss.item()
                optimizer.step()

            text = "Epoch {}\tTrain Loss: {}\tElapsed time: {} \n".format(
                epoch, train_loss / total_train_num, time.time() - start_time
            )
            out_file.write(text)
            start_time = time.time()

        for iteration in range(int(total_test_num / self.batch_size) + 1):
            st_idx, ed_idx = iteration * self.batch_size, (iteration + 1) * self.batch_size
            if ed_idx > total_test_num:
                ed_idx = total_test_num
            output, hidden = self.forward(torch.stack([test[i][0] for i in range(st_idx, ed_idx)], dim=0).detach())
            test_loss = criterion(output, test_out[st_idx:ed_idx])

            output = output.view(-1, self.num_items)
            prob = nn.functional.softmax(output, dim=1).data.cpu()
            np_prob = prob.numpy()
            current_val = np.zeros((np_prob.shape[0], 1))
            for i in range(st_idx, ed_idx):
                current_test_label = test[i][1]
                current_val[i - st_idx, 0] = np_prob[i - st_idx, current_test_label]

            new_prob = np_prob - current_val
            ranks = np.count_nonzero(new_prob > 0, axis=1)

            for i in range(st_idx, ed_idx):
                rank = ranks[i - st_idx] + 1
                MRR += 1 / rank
                HITS += 1 if rank <= 10 else 0
                probs[i] = np_prob[i - st_idx, :]

        MRR /= total_test_num
        HITS /= total_test_num
        text = "\n\n Test MRR = {}\tTest Recall@10 = {}\n\n".format(MRR, HITS)
        out_file.write(text)
        out_file.close()
