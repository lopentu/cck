# copied from lago/subliminal/src/subliminal/metrics.py

import numpy as np
from sklearn.neighbors import NearestNeighbors

def compute_nearestN_metrics_full(
      pred_mat, tgt_mat=None, nnInst=None,
      label_ids=None, 
      metric="cosine", 
      n_neighbors=20,
      verbose=False
    ):
  """
  compute_nn_metrics_full(pred_mat, tgt_mat, label_ids=None, metric="cosine")
  
  Compute nearest neighbor metrics for a set of predictions and targets.
  label_ids indicates the true index of each row in pred_mat in the tgt_mat, and
  it should be the same length as the number of rows in pred_mat.
  """

  assert tgt_mat is not None or nnInst is not None, \
    "tgt_mat and nnInst cannot be None at the same time"
  
  if nnInst is None:
    nnInst = NearestNeighbors(n_neighbors=n_neighbors, metric=metric)
    nnInst.fit(tgt_mat)
  _, idxs = nnInst.kneighbors(pred_mat)
  N = pred_mat.shape[0]
  acc_vec = []
  if label_ids is None:
    label_ids = np.arange(N)
  for k in range(n_neighbors):
    corrects = np.any(idxs[:,:k+1] == label_ids[:, np.newaxis], axis=1)
    n_correct = np.sum(corrects) / len(corrects)
    acc_vec.append(n_correct)
  return acc_vec, idxs