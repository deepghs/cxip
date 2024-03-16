import logging
import os
from tempfile import TemporaryDirectory
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from lighttuner.hpo import hpo, R, randint, uniform
from rainbowneko.infer import BasicAction, MemoryMixin, feedback_input
from sklearn.cluster import OPTICS
from sklearn.metrics import f1_score, precision_score, recall_score, adjusted_rand_score
from tqdm import tqdm


class ContrastiveAnalysisAction(BasicAction, MemoryMixin):
    def __init__(self, clses, max_samples: int = 500000):
        MemoryMixin.__init__(self)
        if isinstance(clses, np.ndarray):
            clses = clses.tolist()
        elif isinstance(clses, torch.Tensor):
            clses = clses.cpu().numpy().tolist()
        else:
            clses = clses

        id_map, max_id = {}, -1
        cls_ids = []
        for v in clses:
            if v not in id_map:
                max_id += 1
                id_map[v] = max_id
            cls_ids.append(id_map[v])
        self.cls_ids = np.array(cls_ids)

        self.max_samples = max_samples

    @classmethod
    def _plt_export(self):
        with TemporaryDirectory() as td:
            imgfile = os.path.join(td, 'image.png')
            plt.savefig(imgfile)

            image = Image.open(imgfile)
            image.load()
            image = image.convert('RGB')
            return image

    @feedback_input
    def forward(self, logits: np.ndarray, memory, **states):
        scale = logits[0, 0]
        logging.info(f'Model scale: {scale:.4f}')

        np.set_printoptions(precision=3)
        scores = np.clip((scale - logits) / (scale * 2), a_min=0.0, a_max=1.0)

        pn_matrix = self.cls_ids == self.cls_ids.reshape(-1, 1)
        pos_samples = scores[pn_matrix]
        neg_samples = scores[~pn_matrix]

        logging.info(f'Pos samples: {pos_samples.shape!r}, '
                     f'mean: {pos_samples.mean():.3f}, std: {pos_samples.std():.3f}.')
        logging.info(f'Neg samples: {neg_samples.shape!r}, '
                     f'mean: {neg_samples.mean():.3f}, std: {neg_samples.std():.3f}.')
        target_number = min(pos_samples.shape[0], neg_samples.shape[0], self.max_samples // 2)
        if pos_samples.shape[0] > target_number:
            logging.info(f'Sample positive samples from {pos_samples.shape[0]} to {target_number} ...')
            idx = np.random.choice(np.arange(pos_samples.shape[0]), target_number, replace=False)
            pos_samples = pos_samples[idx]
        if neg_samples.shape[0] > target_number:
            logging.info(f'Sample negative samples from {neg_samples.shape[0]} to {target_number} ...')
            idx = np.random.choice(np.arange(neg_samples.shape[0]), target_number, replace=False)
            neg_samples = neg_samples[idx]

        x = np.concatenate([pos_samples, neg_samples])
        y = np.concatenate([np.zeros_like(pos_samples), np.ones_like(neg_samples)])
        logging.info(f'X shape: {x.shape!r}, Y shape: {y.shape!r} ...')

        ths = np.linspace(0.0, 1.0, 1001)
        f1_scores, precision_scores, recall_scores = [], [], []
        for th in tqdm(ths, desc='Scan Thresholds'):
            y_pred = (x > th).astype(int)
            f1_scores.append(f1_score(y, y_pred, zero_division=1.0))
            precision_scores.append(precision_score(y, y_pred, zero_division=1.0))
            recall_scores.append(recall_score(y, y_pred, zero_division=1.0))

        iths = np.argmax(f1_scores)
        best_threshold = ths[iths]
        logging.info(f'Best threshold: {best_threshold:.4f}, F1 Score: {f1_scores[iths]:.4f}')

        logging.info('Plotting F1 Score ...')
        plt.cla()
        plt.plot(ths, f1_scores)
        plt.title(f'Best threshold: {best_threshold:.3f}, F1 score: {f1_scores[iths]:.3f}')
        plt.xlabel('Diff Value')
        plt.ylabel('F1 Score')
        f1_score_plot = self._plt_export()

        logging.info('Plotting Precision ...')
        plt.cla()
        plt.plot(ths, precision_scores)
        plt.title(f'Best threshold: {best_threshold:.3f}, F1 score: {f1_scores[iths]:.3f}')
        plt.xlabel('Diff Value')
        plt.ylabel('Precision')
        precision_plot = self._plt_export()

        logging.info('Plotting Recall ...')
        plt.cla()
        plt.plot(ths, recall_scores)
        plt.title(f'Best threshold: {best_threshold:.3f}, F1 score: {f1_scores[iths]:.3f}')
        plt.xlabel('Diff Value')
        plt.ylabel('Recall')
        recall_plot = self._plt_export()
        plt.cla()

        memory.post_info = {
            'scale': scale,
            'scores': scores,
            'threshold': best_threshold,
            'metrics': {
                'f1': f1_scores[iths],
                'precision': precision_scores[iths],
                'recall': recall_scores[iths],
            },
            'plots': {
                'f1_score': f1_score_plot,
                'precision': precision_plot,
                'recall': recall_plot,
            },
            'true': self.cls_ids,
        }


class ClusterTestAction(BasicAction, MemoryMixin):
    def __init__(self, init_steps: int = 50, max_steps: int = 350, min_samples_range: Tuple[int, int] = (3, 5)):
        MemoryMixin.__init__(self)
        self.init_steps = init_steps
        self.max_steps = max_steps
        self.min_samples_range = min_samples_range

    def cluster_it(self, memory, threshold: float, min_samples: int):
        logging.info(f'Clustering for threshold: {threshold}, min_samples: {min_samples} ...')
        post_info = memory.post_info
        batch_diff = post_info['scores']

        def _metric(x, y):
            return batch_diff[int(x), int(y)].item()

        samples = np.arange(post_info['true'].shape[0]).reshape(-1, 1)
        clustering = OPTICS(max_eps=threshold, min_samples=min_samples, metric=_metric).fit(samples)

        cluster_labels = clustering.labels_.tolist()
        processed_cluster_labels = cluster_labels.copy()
        max_clu_id = np.max(cluster_labels).item()
        for i in range(len(cluster_labels)):
            if processed_cluster_labels[i] < 0:
                max_clu_id += 1
                processed_cluster_labels[i] = max_clu_id
        cluster_score = adjusted_rand_score(post_info['true'], processed_cluster_labels)
        logging.info(f'Adjust rand score: {cluster_score:.4f}.')

        return cluster_score

    @feedback_input
    def forward(self, memory, **states):
        @hpo
        def opt_func(cfg):
            threshold: float = cfg['threshold']
            min_samples: int = cfg['min_samples']
            return self.cluster_it(memory, threshold, min_samples)

        logging.info('Waiting for HPO ...')
        params, score, _ = opt_func.bayes() \
            .init_steps(self.init_steps) \
            .max_steps(self.max_steps) \
            .maximize(R).max_workers(1).rank(10) \
            .spaces({
            'min_samples': randint(*self.min_samples_range),
            'threshold': uniform(0.0, 0.5),
        }).run()

        memory.hpo = {
            'min_min_samples': self.min_samples_range[0],
            'max_min_samples': self.min_samples_range[1],
            'params': params,
            'scores': score,
        }
