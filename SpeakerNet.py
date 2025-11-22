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
        self.trainfunc = trainfunc
        self.model_name = model
        self.nPerSpeaker = nPerSpeaker
        self.outp = None

        if self.model_name == 'ECAPA' and trainfunc == 'cos_contrast_loss': # Если наша реализация лосса, берем софтмакс-лосс из ECAPA, но добавляем свою обработку
            LossFunction = importlib.import_module("loss." + 'aam_softmax_ecapa').__getattribute__("LossFunction")
            self.loss_function = importlib.import_module("loss." + trainfunc).__getattribute__("LossFunction")(**kwargs)
        else:
            LossFunction = importlib.import_module("loss." + trainfunc).__getattribute__("LossFunction")
        self.__L__ = LossFunction(**kwargs)

        if self.model_name == 'ECAPA':
            self.adapter_params = ['__L__.weight']
            self.first_params = ['__S__.fc6', '__S__.bn6']

    def forward(self, data, label=None):
        data = data.reshape(-1, data.size()[-1]).cuda()
        self.outp = self.__S__.forward(data)

        if self.trainfunc == 'cos_contrast_loss':
            with torch.no_grad():
                nostat_embs = self.__S__.forward(data, mode='get_stat_embs')
                framed_embs = nostat_embs.unfold(dimension=2, size=10, step=5)
                cos_sim = []
                for i in range(framed_embs.shape[2]):
                    cur_res_embs = self.__S__.forward(framed_embs[:, :, i], mode='get_res_embs')
                    cur_cos_sim = self.loss_function.cos(cur_res_embs, self.outp)
                    cos_sim.append(cur_cos_sim)

        if label == None:
            return self.outp
        else:
            label = label.repeat_interleave(self.nPerSpeaker) # Повторяем метки nPerSpeaker раз
            if self.trainfunc != 'cos_contrast_loss':
                if 'softmax' not in self.trainfunc: # Для чисто косинусного/контрастного лосса добавляем ось nPerSpeaker
                    self.outp = self.outp.reshape(self.nPerSpeaker, -1, self.outp.size()[-1]).transpose(1, 0).squeeze(1)
                nloss, prec1 = self.__L__.forward(self.outp, label)
                return nloss, prec1
            else:
                linear_embs = self.__L__.forward(self.outp, label)
                loss, prec1 = self.loss_function(cos_sim, linear_embs, label)
                return loss, prec1



class ModelTrainer(object):
    def __init__(self, speaker_model, optimizer, scheduler, gpu, mixedprec, max_no_improve_steps, lr, lr_param, lr_decay, weight_decay, valid_steps, **kwargs):
        self.__model__ = speaker_model
        self.model_name = self.__model__.module.model_name
        self.trainfunc = self.__model__.module.trainfunc
        self.use_center_loss = self.trainfunc == 'center_loss'
        self.use_cos_contrast_loss = self.trainfunc == 'cos_contrast_loss'
        self.steps_before_validation = 0
        self.valid_steps = valid_steps

        self.lr_param = lr_param # Режим обучения: адаптивный, обучаем всё или только хвост
        self.is_adaptive = self.lr_param == 'adaptive'
        if self.is_adaptive:
            optimizer = 'adam' # Для адаптивного пока только Адам

        Optimizer = importlib.import_module("optimizer." + optimizer).__getattribute__("Optimizer")
        Scheduler = importlib.import_module("scheduler." + scheduler).__getattribute__("Scheduler")

        self.lr_decay = lr_decay # На что домножаем lr для уменьшения
        self.lr = lr # Первоначальный lr
        self.unfreeze_params = [] # Параметры, которые нужно размораживать, идут по порядку (шаблоны названий)
        if self.is_adaptive and self.use_cos_contrast_loss:  # Если этот лосс, сперва замораживаем всё, кроме адаптера (если в адаптере вообще есть веса)
            self.unfreeze_params = [self.__model__.module.adapter_params, self.__model__.module.first_params]
        elif self.lr_param == 'last' or (self.lr_param == 'adaptive' and not self.use_cos_contrast_loss): # Если режим "хвост" или адаптера нет, сразу первые параметры
            self.unfreeze_params = [self.__model__.module.adapter_params + self.__model__.module.first_params]

        # Список самих параметров по уровням разморозки; если last, у нас будет разморожен только первый слой, иначе добавляем еще слой остальных параметров
        self.unfreeze_params_params = [[] for _ in range(len(self.unfreeze_params) + int(not(lr_param == 'last')))]
        for name, param in self.__model__.module.named_parameters():
            is_on_first_levels = False # Находится ли параметр на уровне адаптера/последних слоев для разморозки
            for i, name_tuple in enumerate(self.unfreeze_params):  # name_tuple - коллекция названий параметров на уровне i
                if i == 0 and len(self.unfreeze_params) > 1:
                    level_name = "Adapter_level"
                else:
                    level_name = "First level"
                if any(name_pattern in name for name_pattern in name_tuple):  # Если имя параметра совпадает с каким-либо из списка для разморозки
                    self.unfreeze_params_params[i].append(name)  # Вставляем на нужный уровень ссылки на параметры, соответствующие именам
                    print(f"{level_name}: {name}")
                    is_on_first_levels = True
                    break
            if not is_on_first_levels and lr_param != 'last':
                self.unfreeze_params_params[-1].append(name) # Последняя коллекция - список параметров, которые размораживаются в последнюю очередь
                print(f"Last level: {name}")

        cur_params = []
        for name, param in self.__model__.module.named_parameters():
            if name in self.unfreeze_params_params[0]:
                cur_params.append(param)

        if self.use_center_loss: # Если центральный лосс, через него обновляются только параметры самой сети
            # Здесь адаптивное обучение пока не реализовано
            self.__optimizer__ = Optimizer(self.__model__.module.__S__.parameters(), **kwargs)
            self.optimizer_W = Optimizer(self.__model__.module.__L__.fc.parameters(), **kwargs) # Для W - отдельный оптимизатор
            self.scheduler_W, self.lr_step_W = Scheduler(self.optimizer_W, **kwargs)
            self.scaler_W = GradScaler()
        else:
            # Пока только первые параметры для разморозки
            self.__optimizer__ = Optimizer(cur_params, lr, weight_decay, **kwargs)

        self.__scheduler__, self.lr_step = Scheduler(self.__optimizer__, lr_decay=lr_decay, **kwargs)

        self.best_nloss = numpy.inf
        self.max_no_improve_steps = max_no_improve_steps
        self.no_improve_steps = 0
        if self.is_adaptive: # Если у нас меняющийся lr...
            self.global_best_nloss = numpy.inf # Если lr уже менялся, будем сравнивать с глобальным лучшим лоссом
            self.freeze_part = 0 # Части разморозки: адаптер, последний слой, вся модель

        self.scaler = GradScaler()

        self.gpu = gpu

        self.mixedprec = mixedprec

        assert self.lr_step in ["epoch", "iteration"]



    def validate(self, valid_loader):
        self.__model__.module.eval()
        total_loss = 0
        total_top1 = 0
        counter = 0

        with torch.no_grad():
            for data, data_label in valid_loader:
                data = data.transpose(1, 0)
                label = torch.LongTensor(data_label).cuda()

                if self.use_center_loss:
                    losses, prec1 = self.__model__(data, label)
                    nloss = losses[0]  # берем основной лосс
                else:
                    nloss, prec1 = self.__model__(data, label)

                total_loss += nloss.detach().cpu().item()
                total_top1 += prec1.detach().cpu().item()
                counter += 1

        self.__model__.module.train()
        return total_loss / counter, total_top1 / counter



    # ## ===== ===== ===== ===== ===== ===== ===== =====
    # ## Train network
    # ## ===== ===== ===== ===== ===== ===== ===== =====

    def train_network(self, loader, valid_loader, verbose):
        self.__model__.module.train()

        stepsize = loader.batch_size

        counter = 0
        index = 0
        loss = 0
        top1 = 0
        # EER or accuracy

        if self.is_adaptive: # Если у нас меняющийся lr...
            verbose = True # Будем логгировать каждый шаг

        tstart = time.time()
        train_lr = self.lr

        for data, data_label in loader:
            data = data.transpose(1, 0)
            #print(data)
            #print(data.shape)
            self.__model__.module.zero_grad()
            label = torch.LongTensor(data_label).cuda()

            # ЕСЛИ ЦЕНТРАЛЬНЫЙ ЛОСС (адаптивный lr для него не сделан)
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

            #if verbose:
                #sys.stdout.write("\rProcessing {:d} of {:d}:".format(index, loader.__len__() * loader.batch_size))
                #sys.stdout.write("Loss {:f} TEER/TAcc {:2.3f}% - {:.2f} Hz ".format(loss / counter, top1 / counter, stepsize / telapsed))
                #sys.stdout.flush()

            self.steps_before_validation += 1
            if self.steps_before_validation == self.valid_steps: # Каждые valid_steps шагов проверяем валидационный лосс
                self.steps_before_validation = 0
                # Валидация
                nloss, acc = self.validate(valid_loader)
                print(f"Step: {index}, valid loss: {nloss}")

                if nloss < self.best_nloss:
                    self.best_nloss = nloss
                    self.no_improve_steps = 0
                else: # Считаем число шагов, когда лосс не улучшается
                    self.no_improve_steps += 1
                    print(f"No improve, step: {index}, best loss: {self.best_nloss}, our loss: {nloss}")

                    if self.no_improve_steps >= self.max_no_improve_steps:  # Если лосс не уменьшался дольше определенного числа шагов...
                        if not self.is_adaptive: # Если не адаптивный lr, просто прекращаем обучение
                            return (self.best_nloss, top1 / counter, True)
                        else:
                            print(f"Global best loss: {self.global_best_nloss}, best loss: {self.best_nloss}")
                            if self.best_nloss < self.global_best_nloss: # Если адаптивный и при этом достигли лучшего лосса по сравнению с предыдущим lr, уменьшаем lr
                                train_lr *= self.lr_decay
                                self.global_best_nloss = self.best_nloss
                                self.no_improve_steps = 0
                            else: # Но если при этом при новом lr совершенно никакого улучшения лосса не последовало (остались на плато):
                                # Размораживаем следующую часть модели либо заканчиваем обучение
                                self.freeze_part += 1
                                if self.freeze_part >= len(self.unfreeze_params_params): # Если параметры закончились, выход из функции
                                    return (self.best_nloss, top1 / counter, True)
                                # Переходим к разморозке следующей части
                                print("РАЗМОРОЗКА")
                                cur_params = []
                                for name, param in self.__model__.module.named_parameters():
                                    if name in self.unfreeze_params_params[self.freeze_part]:
                                        param.requires_grad = True
                                        cur_params.append(param)

                                self.__optimizer__.add_param_group({
                                    'params': cur_params
                                    })
                                '''        
                                for param in self.unfreeze_params_params[self.freeze_part]:
                                    param.requires_grad = True
                                
                                self.__optimizer__.add_param_group({
                                    'params': self.unfreeze_params_params[self.freeze_part]
                                    })
                                '''
                                # Возвращаем lr
                                train_lr = self.lr
                                self.best_nloss = numpy.inf # Возвращаемся к первоначальным показателям lr
                                self.no_improve_steps = 0
                            # Обновляем lr для оптимизатора
                            for param_group in self.__optimizer__.param_groups:
                                param_group['lr'] = train_lr
                            print("NEW LEARNING RATE: ", train_lr)

            # Изменение lr (для не адаптивного варианта)
            if not self.is_adaptive and self.lr_step == "iteration":
                self.__scheduler__.step()
                if self.use_center_loss:
                    self.scheduler_W.step()

        if not self.is_adaptive and self.lr_step == "epoch":
            self.__scheduler__.step()
            if self.use_center_loss:
                self.scheduler_W.step()

        #return (loss / counter, top1 / counter, False)
        return (self.best_nloss, top1 / counter, False)

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

    def loadParameters(self, path):

        self_state = self.__model__.module.state_dict()
        loaded_state = torch.load(path, map_location="cuda:%d" % self.gpu)

        '''
        Убираются лишние слои, остальные приводятся к корректному виду
        '''
        if self.model_name == "ECAPA":
                new_dict = {}
                for key, value in loaded_state.items():
                    # Пропускаем параметры лосса
                    if 'speaker_loss' in key:
                        if self.trainfunc == 'cos_contrast_loss': # Если наш синтетический лосс, берем L из ECAPA, а не заменяем на свой
                            new_key = key.replace('speaker_loss.', '__L__.')  # Заменяем префикс
                            new_dict[new_key] = value
                        else:
                            print(f"Skipping classifier parameter: {key}")
                            continue
                    # Преобразуем имена энкодера
                    elif 'speaker_encoder' in key:
                        new_key = key.replace('speaker_encoder.', '__S__.')  # Заменяем префикс
                        new_dict[new_key] = value
                    else:
                        new_dict[key] = value
                loaded_state = new_dict

        elif self.model_name == "redimnet":
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


        '''
        Заморозка слоев
        '''
        '''
        print(len(self.unfreeze_params_params))
        print(len(self.unfreeze_params_params[0]))
        for i, param_list in enumerate(self.unfreeze_params_params):
                if i == 0:
                    for param in param_list:
                        param.requires_grad = True # Размораживаем только параметры в первом слое разморозки
                else:
                    for param in param_list:
                        param.requires_grad = False
        '''
        print(self.unfreeze_params_params[0])

        for name, param in self.__model__.module.named_parameters():
            if name in self.unfreeze_params_params[0]:
                param.requires_grad = True
            else:
                param.requires_grad = False

        for name, param in self.__model__.module.named_parameters():
            print("NAME OF PARAM", name)
            print(param.requires_grad == True)

        if self.model_name == "redimnet":
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
