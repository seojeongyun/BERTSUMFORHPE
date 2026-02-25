#!/usr/bin/env python
"""
    Main training workflow
"""
from __future__ import division

import argparse
import glob
import os
import random
import signal
import time

import torch

import distributed
from models import data_loader, model_builder
from models.data_loader import load_dataset
from models.model_builder import ExtSummarizer
from models.trainer_ext import build_trainer
from others.logging import logger, init_logger
from torch.utils.data.dataloader import DataLoader
model_flags = ['hidden_size', 'ff_size', 'heads', 'inter_layers', 'encoder', 'ff_actv', 'use_interval', 'rnn_size']


def train_multi_ext(args):
    """ Spawns 1 process per GPU """
    init_logger()

    nb_gpu = args.world_size
    mp = torch.multiprocessing.get_context('spawn')

    # Create a thread to listen for errors in the child processes.
    error_queue = mp.SimpleQueue()
    error_handler = ErrorHandler(error_queue)

    # Train with multiprocessing.
    procs = []
    for i in range(nb_gpu):
        device_id = i
        procs.append(mp.Process(target=run, args=(args,
                                                  device_id, error_queue,), daemon=True))
        procs[i].start()
        logger.info(" Starting process pid: %d  " % procs[i].pid)
        error_handler.add_child(procs[i].pid)
    for p in procs:
        p.join()


def run(args, device_id, error_queue):
    """ run process """
    setattr(args, 'gpu_ranks', [int(i) for i in args.gpu_ranks])

    try:
        gpu_rank = distributed.multi_init(device_id, args.world_size, args.gpu_ranks)
        print('gpu_rank %d' % gpu_rank)
        if gpu_rank != args.gpu_ranks[device_id]:
            raise AssertionError("An error occurred in \
                  Distributed initialization")

        train_single_ext(args, device_id)
    except KeyboardInterrupt:
        pass  # killed by parent, do nothing
    except Exception:
        # propagate exception to parent process, keeping original traceback
        import traceback
        error_queue.put((args.gpu_ranks[device_id], traceback.format_exc()))


class ErrorHandler(object):
    """A class that listens for exceptions in children processes and propagates
    the tracebacks to the parent process."""

    def __init__(self, error_queue):
        """ init error handler """
        import signal
        import threading
        self.error_queue = error_queue
        self.children_pids = []
        self.error_thread = threading.Thread(
            target=self.error_listener, daemon=True)
        self.error_thread.start()
        signal.signal(signal.SIGUSR1, self.signal_handler)

    def add_child(self, pid):
        """ error handler """
        self.children_pids.append(pid)

    def error_listener(self):
        """ error listener """
        (rank, original_trace) = self.error_queue.get()
        self.error_queue.put((rank, original_trace))
        os.kill(os.getpid(), signal.SIGUSR1)

    def signal_handler(self, signalnum, stackframe):
        """ signal handler """
        for pid in self.children_pids:
            os.kill(pid, signal.SIGINT)  # kill children processes
        (rank, original_trace) = self.error_queue.get()
        msg = """\n\n-- Tracebacks above this line can probably
                 be ignored --\n\n"""
        msg += original_trace
        raise Exception(msg)


def validate_ext(args, config, device_id):
    timestep = 0
    if (args.test_all):
        cp_files = sorted(glob.glob(os.path.join(args.model_path, 'model_epoch_*.pt')))
        cp_files.sort(key=os.path.getmtime)
        xent_lst = []
        for i, cp in enumerate(cp_files):
            step = int(cp.split('.')[-2].split('_')[-1])
            xent = validate(args, config, device_id, cp, step)
            xent_lst.append((xent, cp))
            max_step = xent_lst.index(min(xent_lst))
            if (i - max_step > 10):
                break
        xent_lst = sorted(xent_lst, key=lambda x: x[0])[:3]
        logger.info('PPL %s' % str(xent_lst))
        for xent, cp in xent_lst:
            step = int(cp.split('.')[-2].split('_')[-1])
            # test_ext(args, device_id, cp, step)
    else:
        while (True):
            # cp_files = sorted(glob.glob(os.path.join('/home/hjchoi/PycharmProjects/BERTSUMFORHPE(bk_251210)', args.model_path, 'RELATIVE_BASIS/relative_FalseNUM_LAYER:4/*.pt')))
            cp_files = sorted(glob.glob(os.path.join()))
            cp_files.sort(key=os.path.getmtime)
            if (cp_files):
                cp = cp_files[-1]
                time_of_cp = os.path.getmtime(cp)
                if (not os.path.getsize(cp) > 0):
                    time.sleep(60)
                    continue
                if (time_of_cp > timestep):
                    timestep = time_of_cp
                    step = int(cp.split('.')[-2].split('_')[-1])
                    validate(args, config, device_id, cp, step)
                    # test_ext(args, device_id, cp, step)

            cp_files = sorted(glob.glob(os.path.join(args.model_path, 'model_epoch_*.pt')))
            cp_files.sort(key=os.path.getmtime)
            if (cp_files):
                cp = cp_files[-1]
                time_of_cp = os.path.getmtime(cp)
                if (time_of_cp > timestep):
                    continue
            else:
                time.sleep(300)


def validate(args, config, device_id, pt, step):
    device = "cpu" if args.visible_gpus == '-1' else "cuda"
    if (pt != ''):
        test_from = pt
    else:
        test_from = args.test_from
    logger.info('Loading checkpoint from %s' % test_from)
    checkpoint = torch.load(test_from, map_location='cpu')
    new_state_dict = {}
    for key, value in checkpoint['model'].items():
        if key.startswith('bert.encoder.'):
            continue
        elif key.startswith('bert.model.'):
            new_state_dict[key] = value
        elif key.startswith('ext_layer.'):
            new_state_dict[key] = value
        elif key.startswith('embeddings.') or key.startswith('encoder.') or key.startswith('pooler.'):
            new_key = 'bert.model.' + key
            new_state_dict[new_key] = value
        else:
            new_state_dict[key] = value

    modified_checkpoint = checkpoint.copy()
    modified_checkpoint['model'] = new_state_dict

    opt = vars(checkpoint['opt'])
    for k in opt.keys():
        if (k in model_flags):
            setattr(args, k, opt[k])
    print(args)

    model = ExtSummarizer(args, device, modified_checkpoint)
    model.eval()
    # from torch.utils.data.dataloader import DataLoader
    # from Embedder.data_loader import Video_Loader
    # from Embedder.Embedder_config import config
    # video_dataset = Video_Loader(config=config, data_path='/storage/hjchoi/BERTSUMFORHPE/embedder_valid.json')
    # video_loader = torch.utils.data.DataLoader(
    #     video_dataset,
    #     batch_size=config.BATCH_SIZE,
    #     shuffle=True,
    #     num_workers=config.WORKERS,
    #     pin_memory=True,
    #     collate_fn=video_dataset.collate_fn)
    from torch.utils.data.dataloader import DataLoader
    from Embedder.data_loader import Video_Loader
    from Embedder.Embedder_API import Embedder
    video_dataset = Video_Loader(config=config, mode=args.mode)

    video_loader = torch.utils.data.DataLoader(
        video_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.WORKERS,
        pin_memory=True,
        collate_fn=video_dataset.collate_fn)

    trainer = build_trainer(args, config, device_id, model, None)
    stats = trainer.validate(video_loader, video_dataset, step)
    return stats.xent()


def test_ext(args, device_id, pt, step):
    device = "cpu" if args.visible_gpus == '-1' else "cuda"
    if (pt != ''):
        test_from = pt
    else:
        test_from = args.test_from
    logger.info('Loading checkpoint from %s' % test_from)
    checkpoint = torch.load(test_from, map_location=lambda storage, loc: storage)
    new_state_dict = {}
    for key, value in checkpoint['model'].items():
        if key.startswith('bert.encoder.'):
            continue
        elif key.startswith('bert.model.'):
            new_state_dict[key] = value
        elif key.startswith('ext_layer.'):
            new_state_dict[key] = value
        elif key.startswith('embeddings.') or key.startswith('encoder.') or key.startswith('pooler.'):
            new_key = 'bert.model.' + key
            new_state_dict[new_key] = value
        else:
            new_state_dict[key] = value

    modified_checkpoint = checkpoint.copy()
    modified_checkpoint['model'] = new_state_dict
    opt = vars(checkpoint['opt'])
    for k in opt.keys():
        if (k in model_flags):
            setattr(args, k, opt[k])
    print(args)

    model = ExtSummarizer(args, device, modified_checkpoint)
    model.eval()

    test_iter = data_loader.Dataloader(args, load_dataset(args, 'test', shuffle=False),
                                       args.test_batch_size, device,
                                       shuffle=False, is_test=True)
    trainer = build_trainer(args, device_id, model, None)
    trainer.test(test_iter, step)

def train_ext(args, config, device_id):
    if (args.world_size > 1):
        train_multi_ext(args)
    else:
        train_single_ext(args, config, args.device_id)


def train_single_ext(args, config, device_id):
    if args.log_file is not None:
        init_logger(args.log_file)

    if args.device_id == '-1':
        device = torch.device('cpu')
    else:
        device = torch.device('cuda:{}'.format(args.device_id))
        torch.cuda.set_device(args.device_id)

    logger.info('Device ID %d' % device_id)
    logger.info('Device %s' % device)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True

    if device_id >= 0:
        torch.cuda.set_device(device_id)
        torch.cuda.manual_seed(args.seed)
    #
    # torch.manual_seed(args.seed)
    # random.seed(args.seed)
    # torch.backends.cudnn.deterministic = True

    if args.train_from != '':
        logger.info('Loading checkpoint from %s' % args.train_from)
        checkpoint = torch.load(args.train_from,
                                map_location=lambda storage, loc: storage)
        new_state_dict = {}
        for key, value in checkpoint['model'].items():
            if key.startswith('bert.encoder.'):
                continue
            elif key.startswith('bert.model.'):
                new_state_dict[key] = value
            elif key.startswith('ext_layer.'):
                new_state_dict[key] = value
            elif key.startswith('embeddings.') or key.startswith('encoder.') or key.startswith('pooler.'):
                new_key = 'bert.model.' + key
                new_state_dict[new_key] = value
            else:
                new_state_dict[key] = value

        modified_checkpoint = checkpoint.copy()
        modified_checkpoint['model'] = new_state_dict

        if args.bert_random_init:
            sample_keys = [k for k in modified_checkpoint['model'].keys()
                           if k.startswith('bert.model.encoder.layer.0.attention.self.query')]
            before_stats = {}
            for key in sample_keys:
                before_stats[key] = {
                    'mean': modified_checkpoint['model'][key].mean().item(),
                    'std': modified_checkpoint['model'][key].std().item(),
                    'sum': modified_checkpoint['model'][key].sum().item()
                }

            init_count = 0
            for key,value in modified_checkpoint['model'].items():
                if key.startswith('bert.model.encoder.'):
                    if 'weight' in key:
                        if 'LayerNorm' in key:
                            torch.nn.init.ones_(modified_checkpoint['model'][key])
                        else:
                            torch.nn.init.xavier_uniform_(modified_checkpoint['model'][key])
                        init_count += 1
                    elif 'bias' in key:
                        torch.nn.init.zeros_(modified_checkpoint['model'][key])
                        init_count += 1
            for key in sample_keys:
                after_mean = modified_checkpoint['model'][key].mean().item()
                after_std = modified_checkpoint['model'][key].std().item()
                after_sum = modified_checkpoint['model'][key].sum().item()

                print(f'Key: {key}')
                print(
                    f'  Before - mean: {before_stats[key]["mean"]:.6f}, std: {before_stats[key]["std"]:.6f}, sum: {before_stats[key]["sum"]:.4f}')
                print(f'  After  - mean: {after_mean:.6f}, std: {after_std:.6f}, sum: {after_sum:.4f}')

                changed = abs(before_stats[key]["sum"] - after_sum) > 0.0001
                print(f'  Changed: {changed}')

            print(f'Total reinitialized BERT parameters: {init_count}')
        if args.embedder_random_init:
            sample_keys = [k for k in modified_checkpoint['model'].keys()
                           if k.startswith('bert.model.embeddings.word_embeddings.weight')]

            # Store statistics before initialization
            before_stats = {}
            for key in sample_keys:
                before_stats[key] = {
                    'mean': modified_checkpoint['model'][key].mean().item(),
                    'std': modified_checkpoint['model'][key].std().item(),
                    'sum': modified_checkpoint['model'][key].sum().item()
                }

            init_count = 0
            for key, value in modified_checkpoint['model'].items():
                # Initialize embeddings layers (word, position, token_type embeddings)
                if key.startswith('bert.model.embeddings.'):
                    if 'weight' in key:
                        if 'LayerNorm' in key:
                            torch.nn.init.ones_(modified_checkpoint['model'][key])
                        else:
                            torch.nn.init.xavier_uniform_(modified_checkpoint['model'][key])
                        init_count += 1
                    elif 'bias' in key:
                        torch.nn.init.zeros_(modified_checkpoint['model'][key])
                        init_count += 1

            # Print verification statistics
            for key in sample_keys:
                after_mean = modified_checkpoint['model'][key].mean().item()
                after_std = modified_checkpoint['model'][key].std().item()
                after_sum = modified_checkpoint['model'][key].sum().item()
                print(f'Key: {key}')
                print(
                    f'  Before - mean: {before_stats[key]["mean"]:.6f}, std: {before_stats[key]["std"]:.6f}, sum: {before_stats[key]["sum"]:.4f}')
                print(f'  After  - mean: {after_mean:.6f}, std: {after_std:.6f}, sum: {after_sum:.4f}')
                changed = abs(before_stats[key]["sum"] - after_sum) > 0.0001
                print(f'  Changed: {changed}')
            print(f'Total reinitialized embedder parameters: {init_count}')
        opt = vars(checkpoint['opt'])
        for k in opt.keys():
            if k in model_flags:
                setattr(args, k, opt[k])
    else:
        modified_checkpoint = None
        checkpoint = None

    model = ExtSummarizer(args, device, modified_checkpoint)
    optim = model_builder.build_optim(args, model, modified_checkpoint) # BERTModel weight

    logger.info(model)

    trainer = build_trainer(args, config, device_id, model, optim)
    trainer.train(device, args.train_steps)

