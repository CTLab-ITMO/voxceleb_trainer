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
    def __init__(self, model, optimizer, trainfunc, nPerSpeaker, disable_adapter=False, freeze_level=0, **kwargs):
        super(SpeakerNet, self).__init__()

        SpeakerNetModel = importlib.import_module("models." + model).__getattribute__("MainModel")
        self.__S__ = SpeakerNetModel(**kwargs)

        self.disable_adapter = disable_adapter
        self.freeze_level = freeze_level
        
        if not disable_adapter:
            LossFunction = importlib.import_module("loss." + trainfunc).__getattribute__("LossFunction")
            self.__L__ = LossFunction(**kwargs)
            print(f"Adapter (Loss Function) enabled: {trainfunc}")
        else:
            LossFunction = importlib.import_module("loss.cosine").__getattribute__("LossFunction")
            self.__L__ = LossFunction(**kwargs)
            print("Adapter (Loss Function) disabled - using cosine similarity loss")

        self.nPerSpeaker = nPerSpeaker
        
        if freeze_level > 0:
            print(f"\nApplying initial freeze level: {freeze_level}")
            self.apply_freeze_level(freeze_level)

    def forward(self, data, label=None):

        data = data.reshape(-1, data.size()[-1]).cuda()
        outp = self.__S__.forward(data)

        if label == None:
            return outp

        else:
            outp = outp.reshape(self.nPerSpeaker, -1, outp.size()[-1]).transpose(1, 0).squeeze(1)
            nloss, prec1 = self.__L__.forward(outp, label)
            return nloss, prec1
        
    
    def get_layer_groups(self):
        model = self.__S__
        
        conv_params = []
        early_params = []
        late_params = []
        final_params = []
        adapter_params = []
        
        if not self.disable_adapter and hasattr(self, '__L__'):
            for param in self.__L__.parameters():
                adapter_params.append(param)
        
        for name, param in model.named_parameters():
            name_lower = name.lower()
            
            if any(p is param for p in adapter_params):
                continue
            
            if any(x in name_lower for x in ['conv1d', 'conv2d', 'conv.', 'stem.0', 'torchfb', 'spec', 'preemph', 'melbank']):
                conv_params.append(param)
                
            elif any(x in name_lower for x in ['pool', 'linear', 'fc', 'proj', 'bn.weight', 'bn.bias', 'emb_bn']):
                final_params.append(param)
                
            elif any(x in name_lower for x in ['attention', 'k_proj', 'v_proj', 'q_proj', 'out_proj', 'feed_forward']):
                late_params.append(param)
                
            elif any(x in name_lower for x in ['layer_norm', 'final_layer_norm', 'norm.weight', 'norm.bias']):
                late_params.append(param)
                
            else:
                early_params.append(param)
        
        groups = [
            ("conv", conv_params),
            ("early", early_params), 
            ("late", late_params),
            ("final", final_params),
            ("adapter", adapter_params)
        ]
        
        return groups

    def apply_freeze_level(self, level):
        """
        Apply freezing based on level:
        0: No freezing
        1: Freeze all except adapter  
        2: Unfreeze final layers (pooling, projection, final BN)
        3: Unfreeze late layers (attention, layer norms)
        4: Unfreeze early layers (backbone, but keep conv frozen)
        5+: Unfreeze everything including conv
        """
        groups = self.get_layer_groups()
        
        freeze_status = {
            "conv": level < 5,  
            "early": level < 4, 
            "late": level < 3,  
            "final": level < 2,  
            "adapter": False  
        }
        
        total_params = 0
        frozen_params = 0
        active_params = 0
        
        for group_name, params in groups:
            should_freeze = freeze_status.get(group_name, False)
            
            for param in params:
                param.requires_grad_(not should_freeze)
                total_params += param.numel()
                if should_freeze:
                    frozen_params += param.numel()
                else:
                    active_params += param.numel()
        
        print(f"Freeze level {level}: {frozen_params}/{total_params} parameters frozen ({100*frozen_params/total_params:.1f}%)")
        print(f"Active parameters: {active_params} ({100*active_params/total_params:.1f}%)")
        
        for group_name, params in groups:
            if params:
                status = "FROZEN" if freeze_status.get(group_name, False) else "ACTIVE"
                group_size = sum(p.numel() for p in params)
                print(f"  {group_name}: {group_size} params, {status}")
        
        if active_params == 0:
            print("WARNING: No active parameters! Unfreezing final layer as fallback...")
            for group_name, params in groups:
                if group_name == "final":
                    for param in params:
                        param.requires_grad_(True)
                        active_params += param.numel()
                    print(f"Fallback: Activated {sum(p.numel() for p in params)} final layer parameters")
                    break

class ModelTrainer(object):
    def __init__(self, speaker_model, optimizer, scheduler, gpu, mixedprec, **kwargs):

        self.__model__ = speaker_model

        Optimizer = importlib.import_module("optimizer." + optimizer).__getattribute__("Optimizer")
        self.__optimizer__ = Optimizer(self.__model__.parameters(), **kwargs)

        Scheduler = importlib.import_module("scheduler." + scheduler).__getattribute__("Scheduler")
        self.__scheduler__, self.lr_step = Scheduler(self.__optimizer__, **kwargs)

        self.scaler = GradScaler()

        self.gpu = gpu

        self.mixedprec = mixedprec

        assert self.lr_step in ["epoch", "iteration"]

    # ## ===== ===== ===== ===== ===== ===== ===== =====
    # ## Train network
    # ## ===== ===== ===== ===== ===== ===== ===== =====

    def train_network(self, loader, verbose):

        self.__model__.train()

        stepsize = loader.batch_size

        counter = 0
        index = 0
        loss = 0
        top1 = 0
        # EER or accuracy

        tstart = time.time()

        for data, data_label in loader:

            data = data.transpose(1, 0)

            self.__model__.zero_grad()

            label = torch.LongTensor(data_label).cuda()

            # print(f"Processing batch {index // stepsize + 1}/{loader.__len__()}: {data_label}")
            
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

        if self.lr_step == "epoch":
            self.__scheduler__.step()

        return (loss / counter, top1 / counter)

    ## ===== ===== ===== ===== ===== ===== ===== =====
    ## Evaluate from list
    ## ===== ===== ===== ===== ===== ===== ===== =====

    def evaluateFromList(self, test_list, test_path, nDataLoaderThread, distributed, print_interval=100, num_eval=10, **kwargs):

        if distributed:
            rank = torch.distributed.get_rank()
        else:
            rank = 0

        self.__model__.eval()

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

                if self.__model__.module.__L__ is None or self.__model__.module.__L__.test_normalize:
                    ref_feat = F.normalize(ref_feat, p=2, dim=1)
                    com_feat = F.normalize(com_feat, p=2, dim=1)

                dist = torch.cdist(ref_feat.reshape(num_eval, -1), com_feat.reshape(num_eval, -1)).detach().cpu().numpy()

                score = -1 * numpy.mean(dist)

                all_scores.append(score)
                all_labels.append(int(data[0]))
                all_trials.append(data[1] + " " + data[2])

                if idx % print_interval == 0:
                    telapsed = time.time() - tstart
                    hz = idx / telapsed if telapsed > 1e-6 else 0.0
                    sys.stdout.write("\rComputing {:d} of {:d}: {:.2f} Hz".format(idx, len(lines), hz))
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
