#!/usr/bin/python
#-*- coding: utf-8 -*-

import sys, time, os, argparse
import yaml
import numpy
import torch
import glob
import zipfile
import warnings
import datetime
from embedding_visualizer import EmbeddingVisualizer
from tuneThreshold import *
from SpeakerNet import *
from DatasetLoader import *
import torch.distributed as dist
import torch.multiprocessing as mp
warnings.simplefilter("ignore")

## ===== ===== ===== ===== ===== ===== ===== =====
## Parse arguments
## ===== ===== ===== ===== ===== ===== ===== =====

parser = argparse.ArgumentParser(description = "SpeakerNet")

parser.add_argument('--config',         type=str,   default=None,   help='Config YAML file')

## Data loader
parser.add_argument('--max_frames',     type=int,   default=200,    help='Input length to the network for training')
parser.add_argument('--eval_frames',    type=int,   default=300,    help='Input length to the network for testing 0 uses the whole files')
parser.add_argument('--batch_size',     type=int,   default=200,    help='Batch size, number of speakers per batch')
parser.add_argument('--max_seg_per_spk', type=int,  default=500,    help='Maximum number of utterances per speaker per epoch')
parser.add_argument('--nDataLoaderThread', type=int, default=5,     help='Number of loader threads')
parser.add_argument('--augment',        type=bool,  default=False,  help='Augment input')
parser.add_argument('--seed',           type=int,   default=10,     help='Seed for the random number generator')

## Training details
parser.add_argument('--test_interval',  type=int,   default=5,     help='Test and save every [test_interval] epochs')
parser.add_argument('--max_epoch',      type=int,   default=500,    help='Maximum number of epochs')
parser.add_argument('--trainfunc',      type=str,   default="amsoftmax",     help='Loss function')

## Optimizer
parser.add_argument('--optimizer',      type=str,   default="adam", help='sgd or adam')
parser.add_argument('--scheduler',      type=str,   default="steplr", help='Learning rate scheduler')
parser.add_argument('--lr',             type=float, default=0.001,  help='Learning rate')
parser.add_argument("--lr_decay",       type=float, default=0.95,   help='Learning rate decay every [test_interval] epochs')
parser.add_argument('--weight_decay',   type=float, default=0,      help='Weight decay in the optimizer')

## Loss functions
parser.add_argument("--hard_prob",      type=float, default=0.5,    help='Hard negative mining probability, otherwise random, only for some loss functions')
parser.add_argument("--hard_rank",      type=int,   default=10,     help='Hard negative mining rank in the batch, only for some loss functions')
parser.add_argument('--margin',         type=float, default=0.1,    help='Loss margin, only for some loss functions')
parser.add_argument('--scale',          type=float, default=30,     help='Loss scale, only for some loss functions')
parser.add_argument('--nPerSpeaker',    type=int,   default=1,      help='Number of utterances per speaker per batch, only for metric learning based losses')
parser.add_argument('--nClasses',       type=int,   default=5994,   help='Number of speakers in the softmax layer, only for softmax-based losses')
parser.add_argument('--eachOther',      dest='eachOther', action='store_true', help='Ensure every speaker is compared with every other speaker in exhaustive pairwise batches')
parser.add_argument('--range_alpha',    type=float, default=1e-5,   help='Weight for the intra-class term in range loss')
parser.add_argument('--range_beta',     type=float, default=1e-4,   help='Weight for the inter-class term in range loss')
parser.add_argument('--range_lambda',   type=float, default=1.0,    help='Overall multiplier for range loss contribution')
parser.add_argument('--range_margin',   type=float, default=0.5,    help='Margin for inter-class center distances in range loss')
parser.add_argument('--range_k',        type=int,   default=2,      help='Number of intra-class distances considered in range loss')

## Evaluation parameters
parser.add_argument('--dcf_p_target',   type=float, default=0.05,   help='A priori probability of the specified target speaker')
parser.add_argument('--dcf_c_miss',     type=float, default=1,      help='Cost of a missed detection')
parser.add_argument('--dcf_c_fa',       type=float, default=1,      help='Cost of a spurious detection')
parser.add_argument('--num_eval',       type=int,   default=10,     help='Number of evaluation segments per utterance')

## Load and save
parser.add_argument('--initial_model',  type=str,   default="",     help='Initial model weights')
parser.add_argument('--save_path',      type=str,   default="exps/exp1", help='Path for model and logs')

## Training and test data
parser.add_argument('--train_list',     type=str,   default="data/train_list.txt",  help='Train list')
parser.add_argument('--test_list',      type=str,   default="data/test_list.txt",   help='Evaluation list')
parser.add_argument('--train_path',     type=str,   default="data/voxceleb2", help='Absolute path to the train set')
parser.add_argument('--test_path',      type=str,   default="data/voxceleb1", help='Absolute path to the test set')
parser.add_argument('--musan_path',     type=str,   default="data/musan_split", help='Absolute path to the test set')
parser.add_argument('--rir_path',       type=str,   default="data/RIRS_NOISES/simulated_rirs", help='Absolute path to the test set')

## Model definition
parser.add_argument('--n_mels',         type=int,   default=40,     help='Number of mel filterbanks')
parser.add_argument('--log_input',      type=bool,  default=False,  help='Log input features')
parser.add_argument('--model',          type=str,   default="",     help='Name of model definition')
parser.add_argument('--encoder_type',   type=str,   default="SAP",  help='Type of encoder')
parser.add_argument('--nOut',           type=int,   default=512,    help='Embedding size in the last FC layer')
parser.add_argument('--sinc_stride',    type=int,   default=10,    help='Stride size of the first analytic filterbank layer of RawNet3')
parser.add_argument('--disable_adapter', dest='disable_adapter', action='store_true', help='Disable adapter in model')
# ReDimNet (IDRnD) optional configuration, used only when --model ReDimNetIDRND
parser.add_argument('--redim_model_name', type=str, default='b0',  help='ReDimNet size preset: b0,b1,b2,b3,b5,b6,M,...')
parser.add_argument('--redim_train_type', type=str, default='ptn', help='ReDimNet train type: ptn, ft_lm, ft_mix')
parser.add_argument('--redim_dataset',    type=str, default='vox2',help='ReDimNet upstream dataset: vox2, vb2+vox2+cnc')

## Layer freezing parameters
parser.add_argument('--freeze_level',    type=int,   default=0,     help='Freeze level: 0=no freeze, 1=freeze all except adapter, 2=unfreeze last FC+BN, 3=unfreeze more layers')
parser.add_argument('--unfreeze_epoch',  type=int,   default=0,    help='Epoch to increase unfreeze level (0 to disable)')
parser.add_argument('--max_unfreeze_level', type=int, default=4,    help='Maximum unfreeze level to reach')

## For test only
parser.add_argument('--eval',           dest='eval', action='store_true', help='Eval only')

## Distributed and mixed precision training
parser.add_argument('--port',           type=str,   default="8888", help='Port for distributed training, input as text')
parser.add_argument('--distributed',    dest='distributed', action='store_true', help='Enable distributed training')
parser.add_argument('--mixedprec',      dest='mixedprec',   action='store_true', help='Enable mixed precision training')
parser.add_argument('--visualize_embeddings', dest='visualize_embeddings', action='store_true', help='Visualize embeddings during evaluation')
parser.add_argument('--pca', dest='pca', action='store_true', help='Visualize embeddings with PCA')
parser.add_argument('--tsne', dest='tsne', action='store_true', help='Visualize embeddings with t-SNE')
parser.add_argument('--stats', dest='stats', action='store_true', help='Visualize embedding statistics')

args = parser.parse_args()

## Parse YAML
def find_option_type(key, parser):
    for opt in parser._get_optional_actions():
        if ('--' + key) in opt.option_strings:
           return opt.type
    raise ValueError

if args.config is not None:
    with open(args.config, "r") as f:
        yml_config = yaml.load(f, Loader=yaml.FullLoader)
    for k, v in yml_config.items():
        if k in args.__dict__:
            typ = find_option_type(k, parser)
            args.__dict__[k] = typ(v)
        else:
            sys.stderr.write("Ignored unknown parameter {} in yaml.\n".format(k))


## ===== ===== ===== ===== ===== ===== ===== =====
## Trainer script
## ===== ===== ===== ===== ===== ===== ===== =====

def visualize_embeddings_if_needed(args, s, loader, save_path, suffix):
    if args.visualize_embeddings or args.pca or args.tsne:
        visualizer = EmbeddingVisualizer(save_path, max_samples_per_speaker=20)
        methods = []
        if args.pca:
            methods.append('pca')
        if args.tsne:
            methods.append('tsne')
        if args.stats:
            methods.append('stats')
        if not methods and args.visualize_embeddings:
            methods = ['all']
        visualizer.visualize_embeddings(s, loader, num_speakers=10, title_suffix=suffix, methods=methods)
        print(f"Visualization plots saved to: {save_path}")

def main_worker(gpu, ngpus_per_node, args):

    args.gpu = gpu

    ## Load models
    s = SpeakerNet(**vars(args))

    if args.distributed:
        os.environ['MASTER_ADDR']='localhost'
        os.environ['MASTER_PORT']=args.port

        dist.init_process_group(backend='nccl', world_size=ngpus_per_node, rank=args.gpu)

        torch.cuda.set_device(args.gpu)
        s.cuda(args.gpu)

        s = torch.nn.parallel.DistributedDataParallel(s, device_ids=[args.gpu], find_unused_parameters=True)

        print('Loaded the model on GPU {:d}'.format(args.gpu))

    else:
        s = WrappedModel(s).cuda(args.gpu)

    if args.freeze_level > 0 and args.gpu == 0:
        print(f"\nInitial freeze level: {args.freeze_level}")
        
        speaker_net = s.module if hasattr(s, 'module') else s
        if hasattr(speaker_net, 'apply_freeze_level'):
            speaker_net.apply_freeze_level(args.freeze_level)

    it = 1
    eers = [100]

    if args.gpu == 0:
        ## Write args to scorefile
        scorefile   = open(args.result_save_path+"/scores.txt", "a+")

    ## Initialise trainer and data loader
    train_dataset = train_dataset_loader(**vars(args))

    train_sampler = train_dataset_sampler(train_dataset, **vars(args))
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.nDataLoaderThread,
        sampler=train_sampler,
        pin_memory=False,
        worker_init_fn=worker_init_fn,
        drop_last=True,
    )

    trainer     = ModelTrainer(s, **vars(args))

    ## Load model weights
    modelfiles = glob.glob('%s/model0*.model'%args.model_save_path)
    modelfiles.sort()

    if(args.initial_model != ""):
        trainer.loadParameters(args.initial_model)
        print("Model {} loaded!".format(args.initial_model))
    elif len(modelfiles) >= 1:
        trainer.loadParameters(modelfiles[-1])
        print("Model {} loaded from previous state!".format(modelfiles[-1]))
        it = int(os.path.splitext(os.path.basename(modelfiles[-1]))[0][5:]) + 1

    visualize_embeddings = (args.visualize_embeddings or args.pca or args.tsne or args.stats)

    for ii in range(1,it):
        trainer.__scheduler__.step()

    ## Evaluation code - must run on single GPU
    if args.eval == True:

        pytorch_total_params = sum(p.numel() for p in s.module.__S__.parameters())

        print('Total parameters: ',pytorch_total_params)
        print('Test list',args.test_list)
        
        if visualize_embeddings and args.gpu == 0:
            print("Starting embedding visualization...")
            
            with open(args.test_list) as f:
                lines = f.readlines()
            
            files = []
            for line in lines:
                parts = line.strip().split()
                if len(parts) >= 3:
                    files.extend([parts[1], parts[2]])
            
            files = list(set(files))  
            
            vis_dataset = test_dataset_loader(files, args.test_path, eval_frames=args.eval_frames, num_eval=args.num_eval)
            vis_loader = torch.utils.data.DataLoader(vis_dataset, batch_size=1, shuffle=False, num_workers=args.nDataLoaderThread)
            
            viz_save_path = os.path.join(args.result_save_path, "embeddings_visualization")
            adapter_suffix = " (without adapter)" if args.disable_adapter else " (with adapter)"
            
            visualize_embeddings_if_needed(args, s, vis_loader, viz_save_path, adapter_suffix)
            
        sc, lab, _ = trainer.evaluateFromList(**vars(args))

        if args.gpu == 0:

            result = tuneThresholdfromScore(sc, lab, [1, 0.1])

            fnrs, fprs, thresholds = ComputeErrorRates(sc, lab, thresholds, args.dcf_p_target, args.dcf_c_miss, args.dcf_c_fa)

            print('\n',time.strftime("%Y-%m-%d %H:%M:%S"), "VEER {:2.4f}".format(result[1]), "MinDCF {:2.5f}".format(mindcf))

        return

    ## Save training code and params
    if args.gpu == 0:
        pyfiles = glob.glob('./*.py')
        strtime = datetime.datetime.now().strftime("%Y%m%d%H%M%S")

        zipf = zipfile.ZipFile(args.result_save_path+ '/run%s.zip'%strtime, 'w', zipfile.ZIP_DEFLATED)
        for file in pyfiles:
            zipf.write(file)
        zipf.close()

        with open(args.result_save_path + '/run%s.cmd'%strtime, 'w') as f:
            f.write('%s'%args)

    ## Core training script
    for it in range(it,args.max_epoch+1):

        train_sampler.set_epoch(it)
        


        clr = [x['lr'] for x in trainer.__optimizer__.param_groups]

        loss, traineer = trainer.train_network(train_loader, verbose=(args.gpu == 0))

        if args.gpu == 0:
            print('\n',time.strftime("%Y-%m-%d %H:%M:%S"), "Epoch {:d}, TEER/TAcc {:2.2f}, TLOSS {:f}, LR {:f}".format(it, traineer, loss, max(clr)))
            scorefile.write("Epoch {:d}, TEER/TAcc {:2.2f}, TLOSS {:f}, LR {:f} \n".format(it, traineer, loss, max(clr)))

        if it % args.test_interval == 0:

            sc, lab, _ = trainer.evaluateFromList(**vars(args))

            if args.gpu == 0:
                
                result = tuneThresholdfromScore(sc, lab, [1, 0.1])

                fnrs, fprs, thresholds = ComputeErrorRates(sc, lab)
                mindcf, threshold = ComputeMinDcf(fnrs, fprs, thresholds, args.dcf_p_target, args.dcf_c_miss, args.dcf_c_fa)

                eers.append(result[1])

                print('\n',time.strftime("%Y-%m-%d %H:%M:%S"), "Epoch {:d}, VEER {:2.4f}, MinDCF {:2.5f}".format(it, result[1], mindcf))
                scorefile.write("Epoch {:d}, VEER {:2.4f}, MinDCF {:2.5f}\n".format(it, result[1], mindcf))

                trainer.saveParameters(args.model_save_path+"/model%09d.model"%it)

                with open(args.model_save_path+"/model%09d.eer"%it, 'w') as eerfile:
                    eerfile.write('{:2.4f}'.format(result[1]))

                scorefile.flush()

                if visualize_embeddings:
                    # Визуализация на тестовой выборке после тестирования
                    test_files = []
                    with open(args.test_list) as f:
                        for line in f:
                            parts = line.strip().split()
                            if len(parts) >= 3:
                                test_files.extend([parts[1], parts[2]])
                    test_files = list(set(test_files))
                    vis_dataset = test_dataset_loader(test_files, args.test_path, eval_frames=args.eval_frames, num_eval=args.num_eval)
                    vis_loader = torch.utils.data.DataLoader(vis_dataset, batch_size=1, shuffle=False, num_workers=args.nDataLoaderThread)
                    viz_save_path = os.path.join(args.result_save_path, f"embeddings_visualization_epoch_{it}")
                    adapter_suffix = f" (epoch {it})"
                    visualize_embeddings_if_needed(args, s, vis_loader, viz_save_path, adapter_suffix)
        
        if args.unfreeze_epoch > 0 and it % args.unfreeze_epoch == 0 and it > 0:
            new_freeze_level = max(0, args.freeze_level + (it // args.unfreeze_epoch))
            new_freeze_level = min(new_freeze_level, args.max_unfreeze_level)
            
            speaker_net = s.module if hasattr(s, 'module') else s
            
            if hasattr(speaker_net, 'apply_freeze_level') and new_freeze_level != speaker_net.freeze_level:
                print(f"\nEpoch {it}: Updating freeze level from {speaker_net.freeze_level} to {new_freeze_level}")
                speaker_net.freeze_level = new_freeze_level
                speaker_net.apply_freeze_level(new_freeze_level)
                
                print("Recreating optimizer for newly unfrozen parameters...")
                active_params = [p for p in speaker_net.parameters() if p.requires_grad]
                print(f"Active parameters: {sum(p.numel() for p in active_params)}")
                
                trainer.__optimizer__.param_groups[0]['params'] = active_params

    if args.gpu == 0:
        scorefile.close()


## ===== ===== ===== ===== ===== ===== ===== =====
## Main function
## ===== ===== ===== ===== ===== ===== ===== =====


def main():
    args.model_save_path     = args.save_path+"/model"
    args.result_save_path    = args.save_path+"/result"
    args.feat_save_path      = ""

    os.makedirs(args.model_save_path, exist_ok=True)
    os.makedirs(args.result_save_path, exist_ok=True)

    n_gpus = torch.cuda.device_count()

    print('Python Version:', sys.version)
    print('PyTorch Version:', torch.__version__)
    print('Number of GPUs:', torch.cuda.device_count())
    print('Save path:',args.save_path)

    if args.distributed:
        mp.spawn(main_worker, nprocs=n_gpus, args=(n_gpus, args))
    else:
        main_worker(0, None, args)


if __name__ == '__main__':
    main()
