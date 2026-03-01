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
        self.condition_vocab, self.exercise_vocab = self.get_condition_vocab()
        self.cond_int2str, self.ex_int2str = self.reverse_vocab()
        #
        # 260226 single label classification
        # self.loss = torch.nn.CrossEntropyLoss(reduction="mean")
        #
        # 260226 multi label classification (not use weighted loss)
        # self.loss = torch.nn.MultiLabelSoftMarginLoss()
        #
        self.exercise_loss, self.conditions_loss = self.get_loss()
        self.threshold = self.args.threshold
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

    def get_loss(self):
        exercise_loss = torch.nn.CrossEntropyLoss()
        print("[Exercise Loss Setup] CrossEntropyLoss.")
        if self.args.weighted_loss:
            weight = self.args.weighted_loss_value
            pos_weight = torch.ones(self.config.NUM_CONDITIONS, device=self.device)
            pos_weight[:] = weight
            conditions_loss = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction='none')
            #
            print(f"[Conditions Loss Setup] Weighted BCEWithLogitsLoss activated (pos_weight={weight}).")

        else:
            conditions_loss = torch.nn.BCEWithLogitsLoss(reduction='none')
            print("[Conditions Loss Setup] Standard BCEWithLogitsLoss activated (no class weighting).")
            #
        return exercise_loss, conditions_loss
    #
    def get_condition_vocab(self):
        import pickle

        with open(self.config.CONDITION_VOCAB_PATH,'rb') as f:
            condition_vocab = pickle.load(f)

        with open(self.config.T_VOCAB_PATH, 'rb') as f:
            exercise_vocab = pickle.load(f)
        return condition_vocab, exercise_vocab

    def reverse_vocab(self):
        cond_int2str = {v: k for k, v in self.condition_vocab.items()}
        ex_int2str = {v: k for k, v in self.exercise_vocab.items()}

        return cond_int2str, ex_int2str

    def get_dataloader(self, dataset):
        video_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.config.BATCH_SIZE,
            shuffle=True,
            num_workers=self.config.WORKERS,
            pin_memory=True,
            collate_fn=self.video_dataset.collate_fn)

        return video_loader

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
        logger.info("Start training...")        #
        for epoch in range(self.train_epoch):
            print("\n============== [{}][Model: TRAIN_MODE] TRAIN START ==============".format(epoch))
            self.model.train()
            self.embedder.train()
            #
            Train_total_loss = 0

            exercise_classification_acc = 0
            #
            condition_tp = 0
            condition_fp = 0
            condition_fn = 0
            #
            total_samples_seen = 0
            start = time.time()

            for step, (videos, exercise_name, conditions) in enumerate(self.data_loader):
                step_start = time.time()
                #
                B = len(exercise_name)
                D = self.config.NUM_CONDITIONS

                # exercise target
                ex_idx = torch.tensor(exercise_name, dtype=torch.long, device=self.device) - 22  # [B]

                # condition label: [B, D]
                cond_label = torch.zeros((B, D), device=self.device, dtype=torch.float32)
                cond_mask = torch.zeros((B, D), device=self.device, dtype=torch.bool)
                flat = [(b, (int(c) - 22) - self.config.CLASS_NUM, 0.0 if f else 1.0)
                        for b, conds in enumerate(conditions)
                        for c, f in conds]
                if flat:
                    rr, cc, vv = zip(*flat)

                    if step == 0:
                        cc_cpu = torch.tensor(cc)  # ? ÀÌÁ¦ cc´Â python tuple
                        assert cc_cpu.min().item() >= 0 and cc_cpu.max().item() < D, \
                            f"Condition index out of range: [{cc_cpu.min().item()}, {cc_cpu.max().item()}], D={D}"

                    rr = torch.tensor(rr, device=self.device)
                    cc = torch.tensor(cc, device=self.device)
                    vv = torch.tensor(vv, device=self.device, dtype=torch.float32)
                    cond_label[rr, cc] = vv
                    cond_mask[rr,cc] = True

                # forward
                output = self.embedder(videos)
                input_embs, segs, pad_mask = self.emb_(output)
                ex_logits, cond_logits = self.model(input_embs, segs, pad_mask)

                # loss (CE + BCEWithLogits)
                exercise_loss = self.exercise_loss(ex_logits, ex_idx)
                condition_loss = self.args.cond_loss_weight * self.conditions_loss(cond_logits, cond_label)
                cond_mask_f = cond_mask.float()
                condition_loss = (condition_loss * cond_mask_f).sum() / (cond_mask_f.sum() + 1e-8)
                loss = exercise_loss + condition_loss
                Train_total_loss += loss.item()

                # exercise acc
                pred_ex = torch.argmax(ex_logits, dim=1)
                exercise_classification_acc += (pred_ex == ex_idx).sum().item()

                # condition metrics
                mask_bool = cond_mask.bool()
                pred_cond = (torch.sigmoid(cond_logits) > self.threshold) & mask_bool
                tgt_cond = (cond_label > self.threshold) & mask_bool

                # DEBUG
                # if step == 0:
                #     # ---- Exercise ----
                #     pred0 = int(pred_ex[0].item())  # 0~40
                #     tgt0 = int(ex_idx[0].item())  # 0~40
                #
                #     pred_raw = pred0 + 22
                #     tgt_raw = tgt0 + 22
                #
                #     print(f"Expected: {self.ex_int2str[tgt_raw]}\nPredicted: {self.ex_int2str[pred_raw]}")
                #
                #     # ---- Condition ----
                #     p_local = pred_cond[0].nonzero(as_tuple=True)[0]  # 0~NUM_COND-1
                #     t_local = tgt_cond[0].nonzero(as_tuple=True)[0]
                #
                #     # local -> raw (63~159)
                #     p_raw = p_local + (self.config.CLASS_NUM + 22)  # +63
                #     t_raw = t_local + (self.config.CLASS_NUM + 22)
                #
                #     p_condition_lst = [self.cond_int2str[int(idx.item())] for idx in p_raw]
                #     t_condition_lst = [self.cond_int2str[int(idx.item())] for idx in t_raw]
                #
                #     print("Expected: " + ", ".join(t_condition_lst))
                #     print("Predicted: " + ", ".join(p_condition_lst))

                tp = (pred_cond & tgt_cond).sum().item()
                fp = (pred_cond & ~tgt_cond).sum().item()
                fn = (~pred_cond & tgt_cond).sum().item()

                condition_tp += tp
                condition_fp += fp
                condition_fn += fn

                #
                batch_size = cond_label.size(0)
                total_samples_seen += batch_size
                if step % 1000 == 0 and step != 0:
                    ex_acc = 100.0 * exercise_classification_acc / total_samples_seen
                    precision = condition_tp / (condition_tp + condition_fp + 1e-8)
                    recall = condition_tp / (condition_tp + condition_fn + 1e-8)
                    f1 = 2 * precision * recall / (precision + recall + 1e-8)

                    print(f"[TRAIN][Step {step}] "
                          f"Loss: {Train_total_loss / (step + 1):.4f} | "
                          f"Exercise_ACC: {ex_acc:.2f}% | "
                          f"P: {precision:.4f} | "
                          f"R: {recall:.4f} | "
                          f"F1: {f1:.4f}")
                #
                # UPDATE
                self.optim.optimizer.zero_grad()
                loss.backward()
                self.optim.optimizer.step()

                step_end = time.time()
                step_time = step_end - step_start
                # print(f"[Step {step}/{len(self.data_loader)}] Step Time: {step_time:.4f} sec")
            #
            end = time.time()
            epoch_time = end - start
            #
            # Metric
            avg_train_loss = Train_total_loss / len(self.data_loader)
            Train_exercise_cls_accuracy = 100.0 * exercise_classification_acc / total_samples_seen
            precision = condition_tp / (condition_tp + condition_fp + 1e-8)
            recall = condition_tp / (condition_tp + condition_fn + 1e-8)
            f1 = 2 * precision * recall / (precision + recall + 1e-8)
            #
            print('[TRAIN] Epoch: {} Average loss: {:.6f}, Exercise_CLS_Accuracy: {:.2f}%, Condition_CLS_Precision: {:.4f}, Condition_CLS_Recall: {:.4f}, Condition_CLS_F1: {:.4f}'
                  .format(1 + epoch, avg_train_loss, Train_exercise_cls_accuracy, precision, recall, f1))
            # throughput
            throughput = len(self.data_loader.dataset) / epoch_time  # samples/sec
            print(f"Throughput: {throughput:.1f} samples/s")
            #
            # elapsed time
            hours, rem = divmod(epoch_time, 3600)
            minutes, seconds = divmod(rem, 60)
            print(f"Elapsed time per epoch: {int(hours)}h {int(minutes)}m {seconds:.1f}s")
            #
            # Scheduler
            if self.args.use_scheduler:
                self.optim.scheduler.step()
                # DEBUR - SCHEDULER
                lr_from_optimizer = self.optim.optimizer.param_groups[0]['lr']
                print("current_lr: {:.6f}".format(lr_from_optimizer))

            if self.args.mode == 'train-valid':
                print("\n============== [Model: EVAL_MODE] VALID START ==============")
                self.model.eval()
                self.embedder.eval()

                with torch.no_grad():
                    exercise_classification_acc = 0
                    #
                    condition_tp = 0
                    condition_fp = 0
                    condition_fn = 0
                    #
                    valid_samples_seen = 0
                    for step, (videos, exercise_name, conditions) in enumerate(self.valid_data_loader):

                        B = len(exercise_name)
                        D = self.config.NUM_CONDITIONS

                        # ---- exercise target ----
                        ex_idx = torch.tensor(exercise_name, dtype=torch.long, device=self.device) - 22

                        # ---- condition label ----
                        cond_label = torch.zeros((B, D), device=self.device, dtype=torch.float32)
                        cond_mask = torch.zeros((B, D), device=self.device, dtype=torch.bool)
                        flat = [(b, (int(c) - 22) - self.config.CLASS_NUM, 0.0 if f else 1.0)
                                for b, conds in enumerate(conditions)
                                for c, f in conds]

                        if flat:
                            rr, cc, vv = zip(*flat)

                            if step == 0:
                                cc_cpu = torch.tensor(cc)  # ? ÀÌÁ¦ cc´Â python tuple
                                assert cc_cpu.min().item() >= 0 and cc_cpu.max().item() < D, \
                                    f"Condition index out of range: [{cc_cpu.min().item()}, {cc_cpu.max().item()}], D={D}"

                            rr = torch.tensor(rr, device=self.device)
                            cc = torch.tensor(cc, device=self.device)
                            vv = torch.tensor(vv, device=self.device, dtype=torch.float32)
                            cond_label[rr, cc] = vv
                            cond_mask[rr,cc] = True
                        # ---- forward ----
                        output = self.embedder(videos)
                        input_embs, segs, pad_mask = self.emb_(output)
                        ex_logits, cond_logits = self.model(input_embs, segs, pad_mask)

                        # ---- Exercise ACC ----
                        pred_ex = torch.argmax(ex_logits, dim=1)
                        exercise_classification_acc += (pred_ex == ex_idx).sum().item()

                        # ---- Condition metrics ----
                        mask_bool = cond_mask.bool()
                        pred_cond = (torch.sigmoid(cond_logits) > self.threshold) & mask_bool
                        tgt_cond = (cond_label > self.threshold) & mask_bool

                        if step == 0:
                            # ---- Exercise debug ----
                            pred_raw = int(pred_ex[0].item()) + 22
                            tgt_raw = int(ex_idx[0].item()) + 22

                            print(f"Expected: {self.ex_int2str[tgt_raw]}\nPredicted: {self.ex_int2str[pred_raw]}")

                            # ---- Condition debug ----
                            mask0 = cond_mask[0].bool()  # [D]
                            p_local = (pred_cond[0] & mask0).nonzero(as_tuple=True)[0]
                            t_local = (tgt_cond[0] & mask0).nonzero(as_tuple=True)[0]

                            # local -> raw = local + CLASS_NUM + 22
                            p_raw = p_local + (self.config.CLASS_NUM + 22)
                            t_raw = t_local + (self.config.CLASS_NUM + 22)

                            p_condition_lst = [self.cond_int2str[int(idx.item())] for idx in p_raw]
                            t_condition_lst = [self.cond_int2str[int(idx.item())] for idx in t_raw]

                            print("Expected: " + ", ".join(t_condition_lst))
                            print("Predicted: " + ", ".join(p_condition_lst))

                        tp = (pred_cond & tgt_cond).sum().item()
                        fp = (pred_cond & ~tgt_cond).sum().item()
                        fn = (~pred_cond & tgt_cond).sum().item()

                        condition_tp += tp
                        condition_fp += fp
                        condition_fn += fn

                        batch_size = cond_label.size(0)
                        valid_samples_seen += batch_size
                    Valid_exercise_cls_accuracy = 100.0 * exercise_classification_acc / valid_samples_seen
                    precision = condition_tp / (condition_tp + condition_fp + 1e-8)
                    recall = condition_tp / (condition_tp + condition_fn + 1e-8)
                    f1 = 2 * precision * recall / (precision + recall + 1e-8)
                    #
                    print(
                        '[VALID] Exercise_CLS_Accuracy: {:.2f}%, Condition_CLS_Precision: {:.4f}, Condition_CLS_Recall: {:.4f}, Condition_CLS_F1: {:.4f}'
                        .format(Valid_exercise_cls_accuracy, precision, recall, f1))


    def validate(self, video_loader, video_dataset, step=0):
        """Validate model.
            valid_iter: validate data iterator
        Returns:
            :obj:`nmt.Statistics`: validation loss statistics
        """
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
