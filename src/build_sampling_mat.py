import numpy as np
import librosa
from scipy.ndimage import median_filter, gaussian_filter
from tqdm import tqdm
from utils import *
import argparse
import logging
import multiprocessing as mp


def build_SSMS(track, args):

    # Loading beats
    file_struct = FileStruct(track)
    beat_frames, _ = read_beats(file_struct.beat_file)
    beat_frames = librosa.util.fix_frames(beat_frames)

    to_do = not os.path.isfile(file_struct.pos_matrix_file) or not os.path.isfile(file_struct.neg_matrix_file)
    
    print(track, to_do, os.path.isfile(file_struct.pos_matrix_file), os.path.isfile(file_struct.neg_matrix_file))
    
    if to_do:
    # Loading features
        mfcc = get_features(track, args.feature_types[0])
        chromas = get_features(track, args.feature_types[1])

        # Syncing features
        mfcc_synced = librosa.util.sync(mfcc, beat_frames, aggregate=np.median)[1:]
        mfcc_synced = np.minimum(mfcc_synced,
                           librosa.decompose.nn_filter(mfcc_synced,
                                                       aggregate=np.median,
                                                       metric='cosine'))

        chromas_synced = librosa.util.sync(chromas, beat_frames, aggregate=np.median)
        chromas_synced = np.minimum(chromas_synced,
                           librosa.decompose.nn_filter(chromas_synced,
                                                       aggregate=np.median,
                                                       metric='cosine'))

        if chromas_synced.shape[-1] == mfcc_synced.shape[-1]:
            SSM, SSM_neg = compute_ssms(mfcc_synced, chromas_synced, args)

            neg_mat_file = file_struct.get_matrix_filename('neg')
            pos_mat_file = file_struct.get_matrix_filename('pos')

            neg_mat_file.parent.mkdir(parents=True, exist_ok=True)
            pos_mat_file.parent.mkdir(parents=True, exist_ok=True)

            np.save(neg_mat_file, SSM_neg)
            np.save(pos_mat_file, SSM)
        else:
            print(track, chromas_synced.shape,mfcc_synced.shape, len(beat_frames))


def compute_ssm(X, delay, K):
    N = X.shape[1]
    X_stack = librosa.feature.stack_memory(X, n_steps=delay, delay=1)
    R_pos = librosa.segment.recurrence_matrix(X_stack, k=K*N, width=1, metric='euclidean', sym=True, sparse=False, mode='connectivity', self=True)
    return R_pos

    
def compute_ssms(mfcc, chroma, args):
    W = args.W 
    alpha_neg = args.alpha_neg 
    LAMBDA = args.gamma 
    DIAGONAL_FILTER = args.diag_filer 
    lag_mfcc = args.lag_mfcc 
    lag_chroma = args.lag_chroma 
    K = args.K 

    # convert features to time-larg and calculate SSM
    SSM_chromas = compute_ssm(chroma, lag_chroma, K=K)#
    SSM_mfcc = compute_ssm(mfcc, lag_mfcc, K=K)

    diagonal_median = librosa.segment.timelag_filter(median_filter)

    SSM = LAMBDA*SSM_chromas+(1-LAMBDA)*SSM_mfcc
    SSM = diagonal_median(SSM, size=(1, DIAGONAL_FILTER), mode='mirror')

    SSM_gaussian = gaussian_filter(SSM, sigma=1+W//2)
    for i in range(len(SSM_gaussian)):
        SSM_gaussian[i] = (SSM_gaussian[i]-np.min(SSM_gaussian[i]))/(max(np.max(SSM_gaussian[i])-np.min(SSM_gaussian[i]), 1e-6))

    SSM_gaussian = .5*SSM_gaussian + .5*SSM_gaussian.T
    SSM_gaussian_copy = np.copy(SSM_gaussian)
    negatives_gaussian = 1-SSM_gaussian_copy

    for i in range(len(negatives_gaussian)):
        for j in range(len(negatives_gaussian)):
            num = max(np.abs(i-j)/len(negatives_gaussian), SSM_gaussian[i,j])
            negatives_gaussian[i,j] *= np.exp(-alpha_neg*num) 

    return SSM_gaussian, negatives_gaussian


def main(args):
    tracklist = clean_tracklist_audio(args.data_path, annotations=False, n_samples=100000, sampling_mats=False)
    tracklist = tracklist[::-1]
    multi_process = True
    if multi_process:
        pool = mp.Pool(mp.cpu_count())
        funclist = []
        for file in tqdm(tracklist):
            f = pool.apply_async(build_SSMS, [file, args])
            funclist.append(f)
        pool.close()
        pool.join()
    else:
        for track in tqdm(tracklist):
            build_SSMS(track, args)

def parse_args():
    """
    Parses command-line arguments.

    Returns:
        Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Sampling args
    parser.add_argument('--feature_types', nargs="+", type=str, default=['mfcc', 'chroma'])
    parser.add_argument('--W', type=int, default=9)
    parser.add_argument('--alpha_neg', type=int, default=3)
    parser.add_argument('--gamma', type=float, default=.5)
    parser.add_argument('--diag_filer', type=int, default=31)
    parser.add_argument('--alpha', type=int, default=60)
    parser.add_argument('--beta_ssm', type=float, default=.5)
    parser.add_argument('--lag_mfcc', type=int, default=16)
    parser.add_argument('--lag_chroma', type=int, default=8)
    parser.add_argument('--K', type=float, default=.1)

    # Paths
    parser.add_argument('--data_path', type=str, default=None)

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    logging.info(f'Arguments: {args}')
    main(args)