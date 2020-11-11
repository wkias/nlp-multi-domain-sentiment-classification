import sys
import math
import time
import torch
import numpy as np
import json
import os
import util
import loadData
import NNManager


class HistInfo():
    def __init__(self, config, tar, epochs, max_steps) -> None:
        self.config = config
        self.target_domain = tar
        self.epochs = epochs
        self.max_steps = max_steps
        self.index = []
        self.start_time = []
        self.end_time = []
        self.gap = []
        self.loss = []
        self.task_loss = []
        self.valid_acc = []
        self.test_acc = []
        self.max_val_acc = 0
        self.max_test_acc = 0
        self.iter = iter(range(epochs))

    def append(self, s, e, l, tl, v, t, mv, mt):
        self.index.append(self.iter.__next__())
        self.start_time.append(s)
        self.end_time.append(e)
        self.gap.append(e - s)
        self.loss.append(l)
        self.task_loss.append(tl)
        self.valid_acc.append(v)
        self.test_acc.append(t)
        self.max_valid_acc = mv
        self.max_test_acc = mt

    def to_csv(self):
        rst = {'loss':self.loss, 'task_loss':self.task_loss, 'valid_acc':self.valid_acc, 'test_acc':self.test_acc}
        json.dump(rst, open('results/' + self.target_domain + '_' + str(time.time())  + '.json', 'w'))
    
    def to_csv(self):
        tm = str(time.time())
        os.mkdir('results/' + tm)
        open('results/' + tm + '/' + self.target_domain, 'w').write(str(self.config))
        csv = ', start_time, end_time, gap, loss, task_loss, valid_acc, test_acc\n'
        for i in range(len(self.loss)):
            csv += str(i) + ', '
            csv += str(self.start_time[i]) + ', '
            csv += str(self.end_time[i]) + ', '
            csv += str(self.gap[i]) + ', '
            csv += str(self.loss[i]) + ', '
            csv += str(self.task_loss[i]) + ', '
            csv += str(self.valid_acc[i]) + ', '
            csv += str(self.test_acc[i]) + ', '
            csv += '\n'
        open('results/' + tm + '/' + self.target_domain + '.csv', 'w').write(csv)


class Main():
    def __init__(self):
        self.maxValidAcc = 0.0
        self.maxTestAcc = 0.0
        self.epochs = 0
        self.config = util.get_args()
        self.config.lr_decay = self.config.lr_decay * \
            (1200 // self.config.batch_size)
        self.config.lr_decay_begin = self.config.lr_decay_begin * \
            (1200 // self.config.batch_size)
        self.config.maxSteps = self.config.epochs * 1200 // self.config.batch_size
        self.texti = loadData.TextIterator(self.config)
        self.config.text_vocab_size = len(self.texti.word2id)
        embed_weight = np.load("vector_" + self.config.wordemb_suffix+".npy")
        embed_weight = np.insert(embed_weight, embed_weight.shape[0], values=np.zeros(
            [1, embed_weight.shape[1]]), axis=0)
        self.model = NNManager.Model(self.config, self.config.model_name)
        self.model.emb.emb.weight.data.copy_(torch.from_numpy(embed_weight))
        if self.config.dual_gpu:
            self.model = torch.nn.DataParallel(self.model)
            self.model.module.emb.emb.weight.data.copy_(
                torch.from_numpy(embed_weight))
        if self.config.pretrain == 1:
            self.model.load_state_dict(torch.load(
                self.config.pretrain_path, map_location='cpu'))
        # self.model.cuda()
        self.optimizer = torch.optim.Adam(self.model.parameters(
        ), lr=self.config.learning_rate, weight_decay=self.config.weight_decay)
        self.lossfunc = torch.nn.CrossEntropyLoss()
        self.histInfo = HistInfo(
            self.config, self.config.pred_domain, self.config.epochs, self.config.maxSteps)
        self.start = time.time()
        self.end = time.time()
        self.valid_acc = 0
        self.test_acc = 0
        self.loss = 0
        self.task_loss = 0
        print('\n\n--------------------------------------')
        print('target domain\t', self.config.pred_domain)
        print(self.config.maxSteps, " max steps\n\n")

    def valid(self):
        print('epoch:', self.epochs)
        self.epochs += 1
        tot = [0 for i in range(self.config.task)]
        err = [0 for i in range(self.config.task)]
        while True:
            validX_c, validY_c, validDomain_c, validLength_c, flag = self.texti.getValid()

            if not flag:
                break

            validX = [torch.autograd.Variable(torch.from_numpy(
                # validX_c[i])).cuda() for i in range(self.config.task)]
                validX_c[i])) for i in range(self.config.task)]
            validY = [torch.autograd.Variable(torch.from_numpy(
                # validY_c[i]).long()).cuda() for i in range(self.config.task)]
                validY_c[i]).long()) for i in range(self.config.task)]
            validDomain = [torch.autograd.Variable(torch.from_numpy(
                # validDomain_c[i]).long()).cuda() for i in range(self.config.task)]
                validDomain_c[i]).long()) for i in range(self.config.task)]
            validLength = [torch.autograd.Variable(torch.from_numpy(
                # validLength_c[i]).long()).cuda() for i in range(self.config.task)]
                validLength_c[i]).long()) for i in range(self.config.task)]

            taskLogit, advLogit, weightLogit, tmpShareOutput, enableVector = self.model(
                validX, validY, validLength)

            for i in range(self.config.task):
                taskOutput = np.argmax(taskLogit[i].cpu().data.numpy(), axis=1)
                err[i] += sum(taskOutput == validY_c[i])
                tot[i] += validX_c[i].shape[0]
        if self.config.cross is None:
            self.valid_acc = sum(err) / sum(tot)
            print("valid acc: " + str(self.valid_acc))
            if self.valid_acc > self.maxValidAcc:
                self.maxValidAcc = self.valid_acc
        else:
            self.valid_acc = err[self.config.cross] / tot[self.config.cross]
            print("valid acc: " + str(self.valid_acc))
            if self.valid_acc > self.maxValidAcc:
                self.maxValidAcc = self.valid_acc

        tot = [0 for i in range(self.config.task)]
        err = [0 for i in range(self.config.task)]
        while True:
            testX_c, testY_c, testDomain_c, testLength_c, flag = self.texti.getTest()
            if flag == False:
                break
            testX = [torch.autograd.Variable(torch.from_numpy(
                # testX_c[i])).cuda() for i in range(self.config.task)]
                testX_c[i])) for i in range(self.config.task)]
            testY = [torch.autograd.Variable(torch.from_numpy(
                # testY_c[i]).long()).cuda() for i in range(self.config.task)]
                testY_c[i]).long()) for i in range(self.config.task)]
            testDomain = [torch.autograd.Variable(torch.from_numpy(
                # testDomain_c[i]).long()).cuda() for i in range(self.config.task)]
                testDomain_c[i]).long()) for i in range(self.config.task)]
            testLength = [torch.autograd.Variable(torch.from_numpy(
                # testLength_c[i]).long()).cuda() for i in range(self.config.task)]
                testLength_c[i]).long()) for i in range(self.config.task)]

            taskLogit, advLogit, weightLogit, tmpShareOutput, enableVector = self.model(
                testX, testY, testLength)

            for i in range(self.config.task):
                taskOutput = np.argmax(taskLogit[i].cpu().data.numpy(), axis=1)
                err[i] += sum(taskOutput == testY_c[i])
                tot[i] += testX_c[i].shape[0]
        if self.config.cross is None:
            self.test_acc = sum(err) / sum(tot)
            print("test acc: " + str(self.test_acc))
            if self.test_acc > self.maxTestAcc:
                self.maxTestAcc = self.test_acc
        else:
            self.test_acc = err[self.config.cross] / tot[self.config.cross]
            print("test acc: " + str(self.test_acc))
            if self.test_acc > self.maxTestAcc:
                self.maxTestAcc = self.test_acc

        print('max valid acc: ', self.maxValidAcc)
        print('max test acc: ', self.maxTestAcc, '\n')

    def display(self, loss, lossT):
        self.end = time.time()
        self.histInfo.append(self.start, self.end, loss, lossT,
                             self.valid_acc, self.test_acc, self.maxValidAcc, self.test_acc)
        print("loss: {0:.5f}, lossTask: {1:.5f}, time: {2:.5f}".format(
            loss, lossT, self.end-self.start))
        self.start = self.end

    def trainingProcess(self):
        avgLoss = 0.0
        avgLossTask = 0.0
        avgLossAdv = 0.0
        self.model.train()
        step = 1
        while step <= self.config.maxSteps:
            batchX, batchY, batchDomain, batchLength, batchDomainName = self.texti.nextBatch()

            self.optimizer.zero_grad()

            batchX = [torch.autograd.Variable(torch.from_numpy(
                # batchX[i])).cuda() for i in range(self.config.task)]
                batchX[i])) for i in range(self.config.task)]
            batchY = [torch.autograd.Variable(torch.from_numpy(
                # batchY[i]).long()).cuda() for i in range(self.config.task)]
                batchY[i]).long()) for i in range(self.config.task)]
            batchDomain = [torch.autograd.Variable(torch.from_numpy(
                # batchDomain[i]).long()).cuda() for i in range(self.config.task)]
                batchDomain[i]).long()) for i in range(self.config.task)]
            batchLength = [torch.autograd.Variable(torch.from_numpy(
                # batchLength[i]).long()).cuda() for i in range(self.config.task)]
                batchLength[i]).long()) for i in range(self.config.task)]

            lossTask = 0.0
            lossAdv = 0.0
            lossWeight = 0.0
            lossDomain = 0.0

            taskLogit, advLogit, weightLogit, tmpShareOutput, enableVector = self.model(
                batchX, batchY, batchLength, batchDomainName, training=True)

            batchY = torch.cat(batchY, dim=0)
            batchDomain = torch.cat(batchDomain, dim=0)
            lossTask = self.lossfunc(taskLogit, batchY)
            lossDomain = self.lossfunc(advLogit, batchDomain)

            if step > 10 * (1400 // self.config.batch_size):
                loss = lossTask
            else:
                loss = lossTask + self.config.lamb*(lossDomain)

            if step == 10 * (1400 // self.config.batch_size):
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = self.config.learning_rate

            avgLoss += float(loss)
            avgLossTask += float(lossTask)
            loss.backward()

            # clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.config.maxClip)

            self.optimizer.step()

            if step*self.config.batch_size % 1200 == 0:
                print("step: ", step, "/", self.config.maxSteps)
                self.display(avgLoss/(self.config.maxSteps/self.config.epochs),
                             avgLossTask/(self.config.maxSteps/self.config.epochs))
                avgLoss = 0.0
                avgLossTask = 0.0
                avgLossAdv = 0.0

            if step*self.config.batch_size % 1200 == 0:
                self.model.eval()
                self.valid()
                self.model.train()

            step += 1

            if self.config.decay_method == "exp":
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = self.config.learning_rate * \
                        (self.config.lr_decay_rate ** (step/self.config.lr_decay))
            elif self.config.decay_method == "linear":
                if step > self.config.lr_decay_begin and (step - self.config.lr_decay_begin) % self.config.lr_decay == 0:
                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] = param_group['lr'] * \
                            self.config.lr_decay_rate
            elif self.config.decay_method == "cosine":
                gstep = min(step, self.config.lr_decay)
                cosine_decay = 0.5 * \
                    (1 + math.cos(math.pi * gstep / self.config.lr_decay))
                decayed = (1 - self.config.min_lr) * \
                    cosine_decay + self.config.min_lr
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = self.config.learning_rate * decayed

        print("last eval:")
        self.model.eval()
        self.valid()
        self.histInfo.to_csv()


if __name__ == "__main__":
    m = Main()
    m.trainingProcess()
