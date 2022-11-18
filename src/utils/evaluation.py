import math
import sys
import numpy as np
from utils.data_utils import *

# Input should be length of the sequence x 99
def evaluate(eulerchannels_pred,eulerchannels_gt):
	for j in np.arange( eulerchannels_pred.shape[0] ):
		for k in np.arange(3,97,3):
			eulerchannels_pred[j,k:k+3] = rotmat2euler(expmap2rotmat( eulerchannels_pred[j,k:k+3]))
	eulerchannels_pred[:,0:6] = 0

	# Pick only the dimensions with sufficient standard deviation. Others are ignored.
	idx_to_use = np.where( np.std( eulerchannels_pred, 0 ) > 1e-4 )[0]
	# Euclidean distance between Euler angles for sample i
	euc_error = np.power( eulerchannels_gt[:,idx_to_use] - eulerchannels_pred[:,idx_to_use], 2)
	euc_error = np.sum(euc_error, 1)
	euc_error = np.sqrt( euc_error )
	return euc_error

# Evaluate a whole batch (all errors at each timestep)
def evaluate_batch(euler_pred,eulerchannels_gt):
	nsamples    = len(euler_pred)
	mean_errors = np.zeros( (nsamples, euler_pred[0].shape[0]) )
	for i in np.arange(nsamples):
		mean_errors[i,:] = evaluate(euler_pred[i],eulerchannels_gt[i])
	return np.mean( mean_errors, 0 )
