import os
import torch.nn as nn
import numpy as np
import time
import torch
import math
from einops import rearrange,repeat
from tqdm import tqdm
import distributed
from models.reporter_ext import ReportMgr, Statistics
from others.logging import logger
from others.utils import test_rouge, rouge_results_to_str
from torch.utils.data.dataloader import DataLoader
from Embedder.data_loader import Video_Loader
from Embedder.Embedder_API import Embedder
from models.data_loader_joint import joint_dataset
from models.encoder import PositionalEncoding

def _tally_parameters(model):
    n_params = sum([p.nelement() for p in model.parameters()])
    return n_params


def build_trainer(args, config, device_id, model, optim):
    """
    Simplify `Trainer` creation based on user `opt`s*
    Args:
        opt (:obj:`Namespace`): user options (usually from argument parsing)
        model (:obj:`onmt.models.NMTModel`): the model to train
        fields (dict): dict of fields
        optim (:obj:`onmt.utils.Optimizer`): optimizer used during training
        data_type (str): string describing the type of data
            e.g. "text", "img", "audio"
        model_saver(:obj:`onmt.models.ModelSaverBase`): the utility object
            used to save the model
    """

    grad_accum_count = args.accum_count
    n_gpu = args.world_size

    if device_id >= 0:
        gpu_rank = int(args.device_id)
    else:
        gpu_rank = 0
        n_gpu = 0

    print("gpu_rank %d" % gpu_rank)

    #tensorboard_log_dir = args.model_path
    #
    #writer = SummaryWriter(tensorboard_log_dir, comment="Unmt")
    #
    #report_manager = ReportMgr(args.report_every, start_time=-1, tensorboard_writer=writer)

    trainer = Trainer(args, config, model, optim, grad_accum_count, n_gpu, gpu_rank)

    # print(tr)
    if model:
        n_params = _tally_parameters(model)
        logger.info("* number of parameters: %d" % n_params)

    return trainer

class Trainer(object):
    """
    Class that controls the training process.

    Args:
            model(:py:class:`onmt.models.model.NMTModel`): translation model
                to train
            train_loss(:obj:`onmt.utils.loss.LossComputeBase`):
               training loss computation
            valid_loss(:obj:`onmt.utils.loss.LossComputeBase`):
               training loss computation
            optim(:obj:`onmt.utils.optimizers.Optimizer`):
               the optimizer responsible for update
            trunc_size(int): length of truncated back propagation through time
            shard_size(int): compute loss in shards of this size for efficiency
            data_type(string): type of the source input: [text|img|audio]
            norm_method(string): normalization methods: [sents|tokens]
            grad_accum_count(int): accumulate gradients this many times.
            report_manager(:obj:`onmt.utils.ReportMgrBase`):
                the object that creates reports, or None
            model_saver(:obj:`onmt.models.ModelSaverBase`): the saver is
                used to save a checkpoint.
                Thus nothing will be saved if this parameter is None
    """

    def __init__(
            self, args, config, model, optim, grad_accum_count=1, n_gpu=1, gpu_rank=1, report_manager=None,
    ):
        # Basic attributes.
        self.args = args
        self.config = config
        self.save_checkpoint_steps = args.save_checkpoint_steps
        self.save_checkpoint_epoch = args.save_checkpoint_epoch
        self.model = model
        self.train_epoch = args.train_epoch
        self.optim = optim
        self.grad_accum_count = grad_accum_count
        self.n_gpu = n_gpu
        self.gpu_rank = gpu_rank
        self.report_manager = report_manager
        self.device = torch.device('cuda:{}'.format(args.device_id))
        #
        self.video_dataset = Video_Loader(config=self.config, mode=self.args.mode)
        self.data_loader = self.get_dataloader(self.video_dataset)
        if self.args.mode == 'train-valid':
            self.video_dataset = Video_Loader(config=self.config, mode='validate')
            self.valid_data_loader = self.get_dataloader(self.video_dataset)
        self.embedder = Embedder(self.config, mode=self.args.mode).to(self.device)
        #
        self.loss = torch.nn.CrossEntropyLoss(reduction="mean")
        #
        if self.args.emb_mode == 'RELATIVE_BASIS' or self.args.emb_mode == 'BASIS':
            self.preweights = torch.load(self.config.PRETRAINED_EMB_PATH)
            self.pre_weights = nn.Embedding.from_pretrained(self.preweights['weight'], freeze=True).to(self.device)
        elif self.args.emb_mode == 'RELATIVE':
            self.pre_weights = nn.Embedding(22, 768).to(self.device)
        self.segment_emb = nn.Embedding(2, 768).to(self.device)
        self.pos_enc = PositionalEncoding(dropout=0.2, dim=768).to(self.device)
        assert grad_accum_count > 0
        # Set model in training mode.
        if model:
            self.model.train()

    def get_dataloader(self, dataset):
        video_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.config.BATCH_SIZE,
            shuffle=True,
            num_workers=self.config.WORKERS,
            pin_memory=True,
            collate_fn=self.video_dataset.collate_fn)

        return video_loader

    # def emb_(self, data):
    #     pre_weights = nn.Embedding.from_pretrained(self.preweights['weight'], freeze=True)
    #     pad_emb = pre_weights(torch.tensor(self.args.pad_id).to(self.device))
    #     sep_emb = pre_weights(torch.tensor(self.args.sep_id).to(self.device))
    #     #
    #     segment_emb = nn.Embedding(2, 768).to(self.device)
    #     cls_emb = torch.zeros([768]).to(self.device)
    #     pos_enc = PositionalEncoding(dropout=0.2, dim=768)
    #     #
    #     batch_videos = []
    #     batch_segs = []
    #     batch_pos = []
    #     for video_idx in range(data.shape[0]):
    #         video = data[video_idx]
    #         tensor_list = [cls_emb]
    #         seg_list = [0]
    #         for frame_idx in range(self.config.MAX_FRAMES):
    #             if frame_idx % 2 == 0:
    #                 seg_id = 0
    #             else:
    #                 seg_id = 1
    #             tensor_list.append(sep_emb)
    #             seg_list.append(seg_id)
    #             frame = video[frame_idx]
    #             for joint in range(len(frame)):
    #                 tensor_list.append(frame[joint].to(self.device))
    #                 seg_list.append(seg_id)
    #         A_video = torch.stack(tensor_list)
    #         batch_videos.append(A_video)
    #         A_video_segs = torch.tensor(seg_list)
    #         batch_segs.append(A_video_segs)
    #         A_video_pos = pos_enc(A_video).squeeze(0)
    #         batch_pos.append(A_video_pos)
    #     inputs_embeds = torch.stack(batch_videos)
    #     seg_tensor = torch.stack(batch_segs).to(self.device)
    #     seg_emb = segment_emb(seg_tensor)
    #     pos_emb = torch.stack(batch_pos).to(self.device)
    #     #
    #     embeddings = inputs_embeds.to(self.device) + seg_emb.to(self.device) + pos_emb.to(self.device)
    #
    #     mask_src = (1 - (torch.all(inputs_embeds == pad_emb, dim=-1) * 1)).to(self.device)
    #     mask = rearrange(mask_src, 'bs seq_len -> bs 1 1 seq_len')
    #     padding_mask = repeat(mask, 'bs 1 1 seq_len -> bs 1 new seq_len', new=mask_src.shape[-1])
    #
    #     return embeddings, seg_emb, padding_mask

    def emb_(self, data):

        # data: [Batch, Frames, Joints, Dim]
        B, F, J, D = data.shape
        device = data.device

        # -----------------------------------------------------------
        # 1. Inputs Embedding ���� (Vectorization)
        # -----------------------------------------------------------

        # [SEP] ���� ����: [1, 1, 1, D] -> [B, F, 1, D]
        sep_vec = self.pre_weights(self.model.sep_id)
        sep_expanded = sep_vec.view(1, 1, 1, D).expand(B, F, 1, D)

        # [SEP] + [Joints] ����: [B, F, 1+J, D]
        frames_combined = torch.cat([sep_expanded, data], dim=2)

        # ������ ������ (Flatten): [B, F*(1+J), D]
        seq_flattened = frames_combined.reshape(B, -1, D)

        # [CLS] ���� ���� (���� ���� ���� ���� ���������� ����)
        if hasattr(self.model, 'cls_emb'):
            cls_vec = self.model.cls_emb.to(device)
        else:
            # cls_emb�� ���� ����(���� ������ ����) 0���� ����
            cls_vec = torch.zeros(D, device=device)

        cls_expanded = cls_vec.view(1, 1, D).expand(B, 1, D)

        # ���� ���� ����: [B, 1 + F*(1+J), D]
        inputs_embeds = torch.cat([cls_expanded, seq_flattened], dim=1)

        # -----------------------------------------------------------
        # 2. Segment IDs ���� (Vectorization)
        # -----------------------------------------------------------

        # Frame �� 0, 1 ���� ����: [0, 1, 0, 1...]
        frame_pattern = torch.arange(F, device=device) % 2

        # �� ������ �� ���� ����(1+J��)�� ���� ID ����
        # [F] -> [F, 1+J] -> [F*(1+J)]
        seg_ids_flat = frame_pattern.unsqueeze(-1).expand(F, 1 + J).reshape(-1)

        # �� �� CLS(ID: 0) ���� �� ���� ����
        cls_seg_id = torch.zeros(1, dtype=torch.long, device=device)
        full_seg_ids = torch.cat([cls_seg_id, seg_ids_flat]).unsqueeze(0).expand(B, -1)

        # Segment Embedding ����
        seg_emb = self.segment_emb(full_seg_ids)

        # -----------------------------------------------------------
        # 3. Positional Encoding (���� ���� ����)
        # -----------------------------------------------------------

        # ���� PE �������� ���� ������ �������� ���� �� ��������(unsqueeze ���� ��),
        # ������ PE�� ����(pe)�� ���� �������� '������ ������ ����'�� ����������.
        # ���� ����: inputs + seg + pos_enc(inputs)

        # 3-1. Scale & Add PE (inputs_embeds ������ ����)
        # PE ����: x * sqrt(dim) + pe
        pe_term = inputs_embeds * math.sqrt(self.pos_enc.dim)

        # ������ ������ ���� PE ��������
        seq_len = inputs_embeds.size(1)
        pe_term = pe_term + self.pos_enc.pe[:, :seq_len].to(device)

        # Dropout ����
        pos_emb = self.pos_enc.dropout(pe_term)

        # ���� ���� (���� ���� ���� ����: inputs + seg + pos_output)
        embeddings = inputs_embeds + seg_emb + pos_emb

        # -----------------------------------------------------------
        # 4. Padding Mask ����
        # -----------------------------------------------------------

        pad_vec = self.pre_weights(self.model.pad_id)

        # ���� ���� ���� (���� ����)
        is_pad = torch.all(inputs_embeds == pad_vec, dim=-1)

        # ������ ���� (1: Real, 0: Pad)
        mask_src = (~is_pad).float()

        # ���� ����
        mask = rearrange(mask_src, 'b s -> b 1 1 s')
        padding_mask = repeat(mask, 'b 1 1 s -> b 1 new s', new=mask_src.shape[-1])

        return embeddings, seg_emb, padding_mask

    def train(self, device, train_steps, valid_iter_fct=None, valid_steps=-1):
        training_loss_hist=[]
        training_acc_hist=[]

        logger.info("Start training...")        #
        print("============== TRAIN START ==============")
        for epoch in range(self.train_epoch):
            self.model.train()
            self.embedder.train()
            print('train model on')
            #
            Train_total_loss = 0
            Train_correct_predictions = 0
            start = time.time()

            some_step_correct_predictions = 0
            some_step_loss = 0
            for step, (videos, exercise_name) in enumerate(self.data_loader):
                # === DEBUG ===
                # if step == 0:
                #     original = {}
                #     for name, p in self.embedder.named_parameters():
                #         original[name] = p.detach().cpu().clone()
                # === DEBUG ===
                # videos = videos.to(device, non_blocking=True)
                tgt = exercise_name.to(device, non_blocking=True)
                output = self.embedder(videos)
                #
                input_embs, segs, pad_mask = self.emb_(output)
                # tgt = torch.tensor(exercise_name).to(device) - 22
                sent_scores = self.model(input_embs, segs, pad_mask)
                pred_exercise = torch.argmax(sent_scores, dim=1)

                # Calculate LOSS
                loss = self.loss(sent_scores, tgt)

                Train_total_loss += loss.item()
                some_step_loss += loss.item()

                # Calculate ACC
                some_step_correct_predictions += (pred_exercise == tgt).sum().item()
                Train_correct_predictions += (pred_exercise == tgt).sum().item()
                #
                correct = (pred_exercise == tgt).sum().item()
                # print("Correct predictions:", correct)
                # UPDATE
                self.optim.optimizer.zero_grad()
                #
                loss.backward()
                #
                self.optim.optimizer.step()

                # === DEBUG ===
                # updated = {}
                # for name, p in self.embedder.named_parameters():
                #     updated[name] = p.detach().cpu().clone()
                #
                # for name, p in self.embedder.named_parameters():
                #     diff = not torch.allclose(original[name], updated[name], atol=1e-6)
                #     if config.BASIS_FREEZE and config.RELATIVE_FREEZE and diff == True:
                #         raise ValueError("IN FREEZE, PARAMETERS ARE CHANGED")
                #     elif not config.BASIS_FREEZE and not config.RELATIVE_FREEZE and diff == False:
                #         raise ValueError( )
                # === DEBUG ===

                # if step % 20 == 0:
                #     accuracy = some_step_correct_predictions / (tgt.size(0) * 20) * 100
                #     print('step:{},  loss:{:.3f},  acc:{:.2f}%'.format(
                #         step, some_step_loss / 20, accuracy))
                    #
                    # some_step_correct_predictions = 0
                    # some_step_loss = 0
                # if step % 20 == 0:
                #     num_samples = tgt.size(0) * 20  # ���� 20 step ������ ���� ��
                #     accuracy = some_step_correct_predictions / num_samples * 100
                #     print('[TRAIN] step:{},  loss:{:.3f},  acc:{:.2f}%'.format(
                #         step, some_step_loss / 20, accuracy))
                #
                #     some_step_correct_predictions = 0
                #     some_step_loss = 0
                #     debug_cnt += 1
                # if debug_cnt == 3:
                #     break
            end = time.time()
            epoch_time = end - start
            throughput = len(self.data_loader) / epoch_time  # samples/sec
            print(f"Throughput: {throughput:.1f} samples/s")

            #
            hours, rem = divmod(epoch_time, 3600)
            minutes, seconds = divmod(rem, 60)
            print(f"Elapsed time per epoch: {int(hours)}h {int(minutes)}m {seconds:.1f}s")
            #
            avg_train_loss = Train_total_loss / len(self.data_loader)
            Train_accuracy = 100. * Train_correct_predictions / len(self.data_loader.dataset)
            print('[TRAIN] Epoch: {} Average loss: {:.6f}, Accuracy: {:.2f}%'
                  .format(1 + epoch, avg_train_loss, Train_accuracy))
            training_loss_hist.append(avg_train_loss)
            #
            # Scheduler
            if self.args.use_scheduler:
                self.optim.scheduler.step()
                # DEBUR - SCHEDULER
                lr_from_optimizer = self.optim.optimizer.param_groups[0]['lr']
                print("current_lr: {:.6f}".format(lr_from_optimizer))
            # training_acc_hist.append(Train_accuracy)
            #
            # DEBUG -> do not save ckpt
            # if min(training_loss_hist) == avg_train_loss:
            #     dir_name = self.config.DIR_PATH
            #     training_loss_hist = [avg_train_loss]
            #     self._save(dir_name)
            if self.args.mode == 'train-valid':
                print('******* Strat Validation *******')
                self.model.eval()
                self.embedder.eval()
                print('eval model on')
                stats = Statistics()

                with torch.no_grad():
                    Valid_acc_hist = []
                    Valid_correct_predictions = 0
                    for step, (videos, exercise_name) in enumerate(self.valid_data_loader):
                        output = self.embedder(videos)
                        #
                        input_embs, segs, pad_mask = self.emb_(output)
                        tgt = exercise_name.to(device, non_blocking=True)
                        sent_scores = self.model(input_embs, segs, pad_mask).to(self.device)
                        pred_exercise = torch.argmax(sent_scores, dim=1)

                        correct_predictions = (pred_exercise == tgt).sum().item()
                        Valid_correct_predictions += correct_predictions

                        accuracy = correct_predictions / tgt.size(0) * 100
                        # if step % 20 == 0:
                        #     print('[VALID] step:{},  acc:{:.2f}%'.format(step, accuracy))
                        Valid_acc_hist.append(accuracy)
                    # total_val_acc = sum(Valid_acc_hist) / len(Valid_acc_hist)
                    # print('[Final VALID] acc:{:.2f}%'.format(total_val_acc))
                    total_val_acc = 100. * Valid_correct_predictions / len(self.valid_data_loader.dataset)
                    print('[VALID] Epoch: {}, Accuracy: {:.2f}%'
                          .format(1 + epoch, total_val_acc))


    def validate(self, video_loader, video_dataset, step=0):
        """Validate model.
            valid_iter: validate data iterator
        Returns:
            :obj:`nmt.Statistics`: validation loss statistics
        """
        # Set model in validating mode.
        # valid_vocab = video_dataset.vocab
        self.model.eval()
        stats = Statistics()

        with torch.no_grad():
            Valid_acc_hist = []
            Valid_correct_predictions = 0
            for step, (videos, exercise_name) in enumerate(video_loader):
                output = self.embedder(videos)
                #
                input_embs, segs, pad_mask = self.emb_(output)
                tgt = torch.tensor(exercise_name).to(self.device) - 22
                sent_scores = self.model(input_embs, segs, pad_mask).to(self.device)
                pred_exercise = torch.argmax(sent_scores, dim=1)

                correct_predictions = (pred_exercise == tgt).sum().item()
                Valid_correct_predictions += correct_predictions

                accuracy = correct_predictions / tgt.size(0) * 100
                if step % 20 == 0:
                    print('[VALID] step:{},  acc:{:.2f}%'.format(step,accuracy))
                Valid_acc_hist.append(accuracy)
            total_val_acc = sum(Valid_acc_hist) / len(Valid_acc_hist)
            print('[Final VALID] acc:{:.2f}%'.format(total_val_acc))
            return stats

    def test(self, test_iter, step, cal_lead=False, cal_oracle=False):
        """Validate model.
            valid_iter: validate data iterator
        Returns:
            :obj:`nmt.Statistics`: validation loss statistics
        """

        # Set model in validating mode.
        def _get_ngrams(n, text):
            ngram_set = set()
            text_length = len(text)
            max_index_ngram_start = text_length - n
            for i in range(max_index_ngram_start + 1):
                ngram_set.add(tuple(text[i : i + n]))
            return ngram_set

        def _block_tri(c, p):
            tri_c = _get_ngrams(3, c.split())
            for s in p:
                tri_s = _get_ngrams(3, s.split())
                if len(tri_c.intersection(tri_s)) > 0:
                    return True
            return False

        if not cal_lead and not cal_oracle:
            self.model.eval()
        stats = Statistics()

        src_path = "%s_step%d.src" % (self.args.result_path, step)
        can_path = "%s_step%d.candidate" % (self.args.result_path, step)
        gold_path = "%s_step%d.gold" % (self.args.result_path, step)
        with open(can_path, "w") as save_pred:
            with open(gold_path, "w") as save_gold:
                with open(src_path, "w") as save_src:
                    with torch.no_grad():
                        for batch in test_iter:
                            src = batch.src
                            labels = batch.src_sent_labels
                            segs = batch.segs
                            clss = batch.clss
                            mask = batch.mask_src
                            mask_cls = batch.mask_cls

                            gold = []
                            pred = []

                            if cal_lead:
                                selected_ids = [list(range(batch.clss.size(1)))] * batch.batch_size
                            elif cal_oracle:
                                selected_ids = [
                                    [j for j in range(batch.clss.size(1)) if labels[i][j] == 1]
                                    for i in range(batch.batch_size)
                                ]
                            else:
                                sent_scores, mask = self.model(src, segs, clss, mask, mask_cls)

                                loss = self.loss(sent_scores, labels.float())
                                loss = (loss * mask.float()).sum()
                                batch_stats = Statistics(float(loss.cpu().data.numpy()), len(labels))
                                stats.update(batch_stats)

                                sent_scores = sent_scores + mask.float()
                                sent_scores = sent_scores.cpu().data.numpy()
                                selected_ids = np.argsort(-sent_scores, 1)
                            # selected_ids = np.sort(selected_ids,1)
                            for i, idx in enumerate(selected_ids):
                                _pred = []
                                if len(batch.src_str[i]) == 0:
                                    continue
                                for j in selected_ids[i][: len(batch.src_str[i])]:
                                    if j >= len(batch.src_str[i]):
                                        continue
                                    candidate = " ".join(batch.src_str[i][j]).strip()
                                    if self.args.block_trigram:
                                        if not _block_tri(candidate, _pred):
                                            _pred.append(candidate)
                                    else:
                                        _pred.append(candidate)

                                    if (
                                        (not cal_oracle)
                                        and (not self.args.recall_eval)
                                        and len(_pred) == 3
                                    ):
                                        break

                                _pred = " ".join(_pred).replace(" ", "")
                                if self.args.recall_eval:
                                    _pred = " ".join(_pred.split()[: len(batch.tgt_str[i].split())])

                                pred.append(_pred)
                                gold.append(batch.tgt_str[i])
                            for i in range(len(src)):
                                save_src.write("".join("".join(j) for j in batch.src_str[i]).strip().replace("▁", " ") + "\n")
                            for i in range(len(gold)):
                                save_gold.write("".join(gold[i]).strip().replace("▁", " ") + "\n")
                            for i in range(len(pred)):
                                save_pred.write("".join(pred[i]).strip().replace("▁", " ") + "\n")
        if step != -1 and self.args.report_rouge:
            rouges = test_rouge(self.args.temp_dir, can_path, gold_path)
            logger.info("Rouges at step %d \n%s" % (step, rouge_results_to_str(rouges)))
        self._report_step(0, step, valid_stats=stats)

        return stats

    def _gradient_accumulation(self, src_emb, segs, mask_src, tgt):

        loss_sum_weighted = 0.0
        correct_sum = 0
        sample_sum = 0

        for i, batch in enumerate(batch):
            if i == 0:
                self.model.zero_grad()


            # OUTPUT
            sent_scores = self.model(src_emb, segs, mask_src)

            loss = self.loss(sent_scores,tgt)
            # print(loss)

            # LOSS
            bs = batch.batch_size
            loss_sum_weighted += loss.item() / bs
            sample_sum += bs

            # ACC
            pred_idx = torch.argmax(sent_scores, dim=-1)
            correct_sum += (pred_idx == tgt_idx).sum().item()

            (loss / self.grad_accum_count).backward()

            # loss = (loss * mask.float()).sum()
            # (loss / loss.numel()).backward()
            # loss.div(float(normalization)).backward()
            # batch_stats = Statistics(
            #     float(loss.cpu().data.numpy()),
            #     normalization,
            #     n_correct=correct,
            #     n_total=total
            # )
            # batch_stats = Statistics(float(loss.cpu().data.numpy()), normalization)
            # total_stats.update(batch_stats)
            # epoch_stats.update(batch_stats)

            # 4. Update the parameters and statistics.
            if self.grad_accum_count == 1:
                # Multi GPU gradient gather
                if self.n_gpu > 1:
                    grads = [
                        p.grad.data
                        for p in self.model.parameters()
                        if p.requires_grad and p.grad is not None
                    ]
                    distributed.all_reduce_and_rescale_tensors(grads, float(1))
                self.optim.step()
        # in case of multi step gradient accumulation,
        # update only after accum batches
        if self.grad_accum_count > 1:
            if self.n_gpu > 1:
                grads = [
                    p.grad.data
                    for p in self.model.parameters()
                    if p.requires_grad and p.grad is not None
                ]
                distributed.all_reduce_and_rescale_tensors(grads, float(1))
            self.optim.step()

        return loss_sum_weighted, correct_sum, sample_sum

    def _save(self, dir_path):
        real_model = self.model
        # real_generator = (self.generator.module
        #                   if isinstance(self.generator, torch.nn.DataParallel)
        #                   else self.generator)

        model_state_dict = real_model.state_dict()
        # generator_state_dict = real_generator.state_dict()
        checkpoint = {
            "model": model_state_dict,
            # 'generator': generator_state_dict,
            "opt": self.args,
            "optim": self.optim,
        }
        dir_path = self.args.model_path + dir_path
        os.makedirs(dir_path, exist_ok=True)
        checkpoint_path = os.path.join(dir_path, "model_ENCLAYER_{}.pt".format(self.args.ext_layers))
        #
        logger.info("Saving checkpoint %s" % checkpoint_path)
        #
        if not os.path.exists(checkpoint_path):
            torch.save(checkpoint, checkpoint_path)
            return checkpoint, checkpoint_path

    def _start_report_manager(self, start_time=None):
        """
        Simple function to start report manager (if any)
        """
        if self.report_manager is not None:
            if start_time is None:
                self.report_manager.start()
            else:
                self.report_manager.start_time = start_time

    def _maybe_gather_stats(self, stat):
        """
        Gather statistics in multi-processes cases

        Args:
            stat(:obj:onmt.utils.Statistics): a Statistics object to gather
                or None (it returns None in this case)

        Returns:
            stat: the updated (or unchanged) stat object
        """
        if stat is not None and self.n_gpu > 1:
            return Statistics.all_gather_stats(stat)
        return stat

    def _maybe_report_training(self, step, num_steps, learning_rate, report_stats):
        """
        Simple function to report training stats (if report_manager is set)
        see `onmt.utils.ReportManagerBase.report_training` for doc
        """
        if self.report_manager is not None:
            return self.report_manager.report_training(
                step, num_steps, learning_rate, report_stats, multigpu=self.n_gpu > 1
            )

    def _report_step(self, learning_rate, step, train_stats=None, valid_stats=None):
        """
        Simple function to report stats (if report_manager is set)
        see `onmt.utils.ReportManagerBase.report_step` for doc
        """
        if self.report_manager is not None:
            return self.report_manager.report_step(
                learning_rate, step, train_stats=train_stats, valid_stats=valid_stats
            )

    def _maybe_save(self, step):
        """
        Save the model if a model saver is set
        """
        if self.model_saver is not None:
            self.model_saver.maybe_save(step)
