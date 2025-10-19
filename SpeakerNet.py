#!/usr/bin/python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy, sys, random
import time, itertools, importlib

from DatasetLoader import test_dataset_loader
from torch.cuda.amp import autocast, GradScaler


class WrappedModel(nn.Module):

    ## The purpose of this wrapper is to make the model structure consistent between single and multi-GPU

    def __init__(self, model):
        super(WrappedModel, self).__init__()
        self.module = model

    def forward(self, x, label=None):
        return self.module(x, label)


class SpeakerNet(nn.Module):
    def __init__(self, model, optimizer, trainfunc, nPerSpeaker, **kwargs):
        super(SpeakerNet, self).__init__()

        SpeakerNetModel = importlib.import_module("models." + model).__getattribute__("MainModel")
        self.__S__ = SpeakerNetModel(**kwargs)

        LossFunction = importlib.import_module("loss." + trainfunc).__getattribute__("LossFunction")
        self.__L__ = LossFunction(**kwargs)

        self.nPerSpeaker = nPerSpeaker
        self.outp = None

    def forward(self, data, label=None):

        data = data.reshape(-1, data.size()[-1]).cuda()
        self.outp = self.__S__.forward(data)

        if label == None:
            return self.outp

        else:

            self.outp = self.outp.reshape(self.nPerSpeaker, -1, self.outp.size()[-1]).transpose(1, 0).squeeze(1)

            nloss, prec1 = self.__L__.forward(self.outp, label)

            return nloss, prec1


class ModelTrainer(object):
    def __init__(self, speaker_model, optimizer, scheduler, gpu, mixedprec, **kwargs):
        self.__model__ = speaker_model
        self.use_center_loss = hasattr(self.__model__.module.__L__, 'center_loss')

        Optimizer = importlib.import_module("optimizer." + optimizer).__getattribute__("Optimizer")
        Scheduler = importlib.import_module("scheduler." + scheduler).__getattribute__("Scheduler")

        if self.use_center_loss: # Если центральный лосс, через него обновляются только параметры самой сети
            self.__optimizer__ = Optimizer(self.__model__.module.__S__.parameters(), **kwargs)
            self.optimizer_W = Optimizer(self.__model__.module.__L__.fc.parameters(), **kwargs) # Для W - отдельный оптимизатор
            self.scheduler_W, self.lr_step_W = Scheduler(self.optimizer_W, **kwargs)
            self.scaler_W = GradScaler()
        else:
            self.__optimizer__ = Optimizer(self.__model__.module.parameters(), **kwargs)

        self.__scheduler__, self.lr_step = Scheduler(self.__optimizer__, **kwargs)

        self.scaler = GradScaler()

        self.gpu = gpu

        self.mixedprec = mixedprec

        assert self.lr_step in ["epoch", "iteration"]

    # ## ===== ===== ===== ===== ===== ===== ===== =====
    # ## Train network
    # ## ===== ===== ===== ===== ===== ===== ===== =====

    def train_network(self, loader, verbose):
        self.__model__.module.train()

        stepsize = loader.batch_size

        counter = 0
        index = 0
        loss = 0
        top1 = 0
        # EER or accuracy

        tstart = time.time()

        for data, data_label in loader:
            data = data.transpose(1, 0)
            self.__model__.module.zero_grad()
            label = torch.LongTensor(data_label).cuda()

            # ЕСЛИ ЦЕНТРАЛЬНЫЙ ЛОСС
            if self.use_center_loss:
                torch.autograd.set_detect_anomaly(True)
                if self.mixedprec:
                    # НЕ ДОДЕЛАНО!!!
                    with autocast():
                        losses, prec1 = self.__model__(data, label)
                        nloss, softmax_loss = losses[0], losses[1]
                    self.scaler.scale(nloss).backward(retain_graph=True)
                    self.scaler.step(self.__optimizer__)
                    self.scaler.update()
                    self.__model__.module.__L__.zero_grad()
                    self.scaler_W.scale(softmax_loss).backward()
                    self.scaler_W.step(self.optimizer_W)
                    self.scaler_W.update()
                else: # NON-MIXEDPREC ВЕРСИЯ
                    #print("NON-MIXEDPREC")
                    # Вычисляем градиенты по центральному лоссу: обновляются только параметры самой сети
                    losses, prec1 = self.__model__(data, label)
                    nloss, softmax_loss = losses[0], losses[1]
                    nloss.backward(retain_graph=True) # Если не сохранить граф вычислений, при повторном бэкворде будет ошибка
                    self.__optimizer__.step()

                    # Вычисляем градиенты по софтмакс-лоссу: обновляются только параметры последнего слоя (как убрать подсчет градиентов для сетки?)
                    self.__model__.module.__L__.zero_grad() # Обнуляем градиенты полносвязного слоя: они обновятся только по софтмакс-лоссу

                    # Тут происходит конфликт версий: из-за того, что параметры уже были изменены, backward возвращает ошибку
                    # НО БЭКВОРД НЕ ПОЗВОЛЯЕТ ВЫЧИСЛИТЬ ГРАДИЕНТ ТОЛЬКО ДЛЯ ОПРЕДЕЛЕННОГО СЛОЯ
                    #softmax_loss.backward()

                    # Вычисляем градиенты вручную
                    # Только параметры, которые требуют градиентов
                    trainable_params = []
                    for name, param in self.__model__.module.__L__.named_parameters():
                        if param.requires_grad:
                            trainable_params.append(param)
                    grads = torch.autograd.grad(softmax_loss, trainable_params)
                    # Ручное присвоение градиентов
                    for param, grad in zip(trainable_params, grads):
                        param.grad = grad
                    self.optimizer_W.step()
                # Обновляем эмбеддинги центров классов
                embeddings = self.__model__.module.outp # Эмбеддинги аудиозаписей в батче, сохраненные после forward pass
                self.__model__.module.__L__.center_loss.update_centers(embeddings, label)

            # ЕСЛИ НЕ ЦЕНТРАЛЬНЫЙ ЛОСС, СТАНДАРТНАЯ ЛОГИКА
            else:
                if self.mixedprec:
                    with autocast():
                        nloss, prec1 = self.__model__(data, label)
                    self.scaler.scale(nloss).backward()
                    self.scaler.step(self.__optimizer__)
                    self.scaler.update()
                else:
                    nloss, prec1 = self.__model__(data, label)
                    nloss.backward()
                    self.__optimizer__.step()

            loss += nloss.detach().cpu().item()
            top1 += prec1.detach().cpu().item()
            counter += 1
            index += stepsize

            telapsed = time.time() - tstart
            tstart = time.time()

            if verbose:
                sys.stdout.write("\rProcessing {:d} of {:d}:".format(index, loader.__len__() * loader.batch_size))
                sys.stdout.write("Loss {:f} TEER/TAcc {:2.3f}% - {:.2f} Hz ".format(loss / counter, top1 / counter, stepsize / telapsed))
                sys.stdout.flush()

            if self.lr_step == "iteration":
                self.__scheduler__.step()
                if self.use_center_loss:
                    self.scheduler_W.step()

        if self.lr_step == "epoch":
            self.__scheduler__.step()
            if self.use_center_loss:
                self.scheduler_W.step()

        return (loss / counter, top1 / counter)

    ## ===== ===== ===== ===== ===== ===== ===== =====
    ## Evaluate from list
    ## ===== ===== ===== ===== ===== ===== ===== =====

    def evaluateFromList(self, test_list, test_path, nDataLoaderThread, distributed, print_interval=100, num_eval=10, **kwargs):

        if distributed:
            rank = torch.distributed.get_rank()
        else:
            rank = 0

        self.__model__.module.eval()

        lines = []
        files = []
        feats = {}
        tstart = time.time()

        ## Read all lines
        with open(test_list) as f:
            lines = f.readlines()

        ## Get a list of unique file names
        files = list(itertools.chain(*[x.strip().split()[-2:] for x in lines]))
        setfiles = list(set(files))
        setfiles.sort()

        ## Define test data loader
        test_dataset = test_dataset_loader(setfiles, test_path, num_eval=num_eval, **kwargs)

        if distributed:
            sampler = torch.utils.data.distributed.DistributedSampler(test_dataset, shuffle=False)
        else:
            sampler = None

        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=nDataLoaderThread, drop_last=False, sampler=sampler)

        ## Extract features for every image
        for idx, data in enumerate(test_loader):
            inp1 = data[0][0].cuda()
            with torch.no_grad():
                ref_feat = self.__model__(inp1).detach().cpu()
            feats[data[1][0]] = ref_feat
            telapsed = time.time() - tstart

            if idx % print_interval == 0 and rank == 0:
                sys.stdout.write(
                    "\rReading {:d} of {:d}: {:.2f} Hz, embedding size {:d}".format(idx, test_loader.__len__(), idx / telapsed, ref_feat.size()[1])
                )

        all_scores = []
        all_labels = []
        all_trials = []

        if distributed:
            ## Gather features from all GPUs
            feats_all = [None for _ in range(0, torch.distributed.get_world_size())]
            torch.distributed.all_gather_object(feats_all, feats)

        if rank == 0:

            tstart = time.time()
            print("")

            ## Combine gathered features
            if distributed:
                feats = feats_all[0]
                for feats_batch in feats_all[1:]:
                    feats.update(feats_batch)

            ## Read files and compute all scores
            for idx, line in enumerate(lines):

                data = line.split()

                ## Append random label if missing
                if len(data) == 2:
                    data = [random.randint(0, 1)] + data

                ref_feat = feats[data[1]].cuda()
                com_feat = feats[data[2]].cuda()

                if self.__model__.module.__L__.test_normalize:
                    ref_feat = F.normalize(ref_feat, p=2, dim=1)
                    com_feat = F.normalize(com_feat, p=2, dim=1)

                dist = torch.cdist(ref_feat.reshape(num_eval, -1), com_feat.reshape(num_eval, -1)).detach().cpu().numpy()

                score = -1 * numpy.mean(dist)

                all_scores.append(score)
                all_labels.append(int(data[0]))
                all_trials.append(data[1] + " " + data[2])

                if idx % print_interval == 0:
                    telapsed = time.time() - tstart
                    sys.stdout.write("\rComputing {:d} of {:d}: {:.2f} Hz".format(idx, len(lines), idx / telapsed))
                    sys.stdout.flush()

        return (all_scores, all_labels, all_trials)

    ## ===== ===== ===== ===== ===== ===== ===== =====
    ## Save parameters
    ## ===== ===== ===== ===== ===== ===== ===== =====

    def saveParameters(self, path):

        torch.save(self.__model__.module.state_dict(), path)

    ## ===== ===== ===== ===== ===== ===== ===== =====
    ## Load parameters
    ## ===== ===== ===== ===== ===== ===== ===== =====

    def loadParameters(self, path, model_name):

        self_state = self.__model__.module.state_dict()
        loaded_state = torch.load(path, map_location="cuda:%d" % self.gpu)

        if model_name == "ECAPA":
                print(model_name)
                new_dict = {}
                for key, value in loaded_state.items():
                    # Пропускаем ВСЕ параметры лосса
                    if 'speaker_loss' in key:
                        print(f"Skipping classifier parameter: {key}")
                        continue

                    # Преобразуем имена энкодера
                    if 'speaker_encoder' in key:
                        new_key = key.replace('speaker_encoder.', '__S__.')  # Заменяем префикс
                        new_dict[new_key] = value
                    else:
                        new_dict[key] = value
                loaded_state = new_dict

        elif "redimnet" in model_name:
            print(model_name)
            new_dict = {}
            for key, value in loaded_state.items():
                new_key = '__S__.' + key
                new_dict[new_key] = value
            loaded_state = new_dict

        if len(loaded_state.keys()) == 1 and "model" in loaded_state:
            loaded_state = loaded_state["model"]
            newdict = {}
            delete_list = []
            for name, param in loaded_state.items():
                new_name = "__S__."+name
                newdict[new_name] = param
                delete_list.append(name)
            loaded_state.update(newdict)
            for name in delete_list:
                del loaded_state[name]
        for name, param in loaded_state.items():
            origname = name
            if name not in self_state:
                name = name.replace("module.", "")

                if name not in self_state:
                    print("{} is not in the model.".format(origname))
                    continue

            if self_state[name].size() != loaded_state[origname].size():
                print("Wrong parameter length: {}, model: {}, loaded: {}".format(origname, self_state[name].size(), loaded_state[origname].size()))
                continue

            self_state[name].copy_(param)


        if model_name == "ECAPA":
            # ЗАМОРОЗКА ВСЕХ СЛОЕВ КРОМЕ ФИНАЛЬНЫХ
            for name, param in self.__model__.module.named_parameters():
                if '__L__.center_loss' in name:
                    param.requires_grad = False
                '''
                # Замораживаем ВСЁ, кроме fc6 и bn6
                if 'fc6' in name or 'bn6' in name or '__L__.fc' in name:
                    param.requires_grad = True  # Размораживаем
                    print(f"Trainable: {name}")
                else:
                    param.requires_grad = False  # Замораживаем
                    print(f"Frozen: {name}")
                '''
            #print("Only fc6 and bn6 layers are trainable")

        elif "redimnet" in model_name:
            # ЗАМОРОЗКА ВСЕХ СЛОЕВ КРОМЕ ФИНАЛЬНЫХ
            # ТОЧНЫЕ полные имена финальных слоёв
            trainable_layers = {
                '__S__.pool.linear2.weight',
                '__S__.pool.linear2.bias',
                '__S__.bn2.weight',
                '__S__.bn2.bias',
                '__S__.bn2.running_mean',
                '__S__.bn2.running_var',
                '__S__.bn2.num_batches_tracked',
                '__S__.linear.weight',
                '__S__.linear.bias'
            }
            for name, param in self.__model__.module.named_parameters():
                if name in trainable_layers:
                    param.requires_grad = True
                    print(f"Trainable: {name}")
                else:
                    param.requires_grad = False
