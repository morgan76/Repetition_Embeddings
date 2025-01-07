import numpy as np
import torch
from torch.utils import data
import librosa
from utils import *
from tqdm import tqdm
import multiprocessing as mp
import joblib
import random

from scipy.special import softmax

class DatasetSimCLR(data.Dataset):
    """
    Dataset class for SimCLR-style contrastive learning.
    
    Attributes:
        split (str): Data split ('train', 'val', or 'test').
        tracklist (list): List of track identifiers.
        n_embedding (int): Number of embeddings for each frame.
        hop_length (int): Hop length for audio processing.
        sample_rate (int): Sampling rate of audio.
        n_conditions (int): Number of sampling conditions.
        n_anchors (int): Number of anchor samples.
        n_positives (int): Number of positive samples per anchor.
        n_negatives (int): Number of negative samples per anchor.
        n_samples (int): Number of samples to include.
        sampled_frames (list): List of sampled frames with their anchors, positives, and negatives.
    """
    def __init__(
        self, split, data_path, max_len, n_embedding, hop_length, sample_rate, 
        n_conditions, n_anchors, n_positives, n_negatives, temperature_positives, temperature_negatives, n_samples
    ):
        self.split = split
        self.data_path = data_path
        self.tracklist = clean_tracklist_audio(data_path=self.data_path, annotations=False, n_samples=int(n_samples), sampling_mats=True)
        self.n_embedding = n_embedding
        self.max_len = max_len
        self.hop_length = hop_length
        self.sample_rate = sample_rate
        self.n_conditions = n_conditions
        self.n_anchors = n_anchors
        self.n_positives = n_positives
        self.n_negatives = n_negatives
        self.temperature_positives = temperature_positives
        self.temperature_negatives = temperature_negatives 
        self.n_samples = n_samples
        #self.sampled_frames = self._sample_frames()
        self.sampled_frames = self.sample_frames_offline()
        print('Len sampled frams =', len(self.sampled_frames))

    def sample_frames_offline(self):
        #pool = mp.Pool(mp.cpu_count())
        #sampled_frames = []
        #for track in tqdm(self.tracklist):
        #    f = pool.apply_async(self._sample_frames, [track]).get()
        #    sampled_frames.append(f)
        #pool.close()
        #pool.join()
        jobs = [ joblib.delayed(self._sample_frames)(track=i) for i in tqdm(self.tracklist) ] #audio_file, embeddings, level
        sampled_frames = joblib.Parallel(n_jobs=32, verbose=1)(jobs)
        return [x for x in sampled_frames if x is not None]
    
    def _sample_frames(self, track):
        """
        Pre-samples anchors, positives, and negatives for each track.
        
        Returns:
            list: Precomputed frame triplets for each track.
        """

        #for track in tqdm(self.tracklist, desc="Sampling frames"):
        file_struct = FileStruct(track)
        beat_frames, _ = read_beats(file_struct.beat_file)
        pos_matrix = np.load(file_struct.pos_matrix_file)
        neg_matrix = np.load(file_struct.neg_matrix_file)

        points = np.arange(len(pos_matrix))
        nb_peaks = []
        for point in list(points):
            row = pos_matrix[point]
            peaks = librosa.util.peak_pick(row, pre_max=8, post_max=8, pre_avg=16, post_avg=16, delta=0.05, wait=16)
            nb_peaks.append(len(peaks))
        # ADD another condition, which samples n rows in pos_matrix, calculated number peaks, if mean of these peaks > threshold: ok (to avoid selecting single-diagonals sampling ssms)

        if len(beat_frames) > 200 and np.mean(pos_matrix) > .07 and np.mean(pos_matrix) < .3 and self.split == 'train' and np.mean(nb_peaks)>2.5:

            beat_frames = librosa.util.fix_frames(beat_frames)[:self.max_len]
            nb_embeddings = len(beat_frames) - 1
            pos_matrix = pos_matrix[:self.max_len, :self.max_len]
            neg_matrix = neg_matrix[:self.max_len, :self.max_len]
            # Sample anchor indexes
            anchor_indexes = np.random.choice(
                np.arange(0, nb_embeddings),
                size=min(self.n_anchors, nb_embeddings),
                replace=False
            )
            anchors, positives, negatives = [], [], []

            for anchor_index in anchor_indexes:
                pos_idx, neg_idx = self._sampler_matrices(anchor_index,
                            self.n_positives, self.n_negatives, 
                            pos_matrix, neg_matrix)
                anchors.append(anchor_index)
                positives.append(pos_idx)
                negatives.append(neg_idx)
            
            beat_frames = [i*256 for i in beat_frames]
            return (track, beat_frames, anchors, positives, negatives)

        elif self.split == 'valid':
            
            beat_frames = librosa.util.fix_frames(beat_frames)
            nb_embeddings = len(beat_frames) - 1

            anchor_indexes = np.random.choice(
                np.arange(0, nb_embeddings),
                size=min(self.n_anchors, nb_embeddings),
                replace=False
            )
            anchors, positives, negatives = [], [], []

            for anchor_index in anchor_indexes:
                pos_idx, neg_idx = self._sampler_matrices(anchor_index,
                            self.n_positives, self.n_negatives, 
                            pos_matrix, neg_matrix)
                anchors.append(anchor_index)
                positives.append(pos_idx)
                negatives.append(neg_idx)
                
            beat_frames = [i*256 for i in beat_frames]
            return (track, beat_frames, anchors, positives, negatives)


        #return sampled_frames

    def _sample_frames_(self):
        """
        Pre-samples anchors, positives, and negatives for each track.
        
        Returns:
            list: Precomputed frame triplets for each track.
        """
        sampled_frames = []

        for track in tqdm(self.tracklist, desc="Sampling frames"):
            file_struct = FileStruct(track)
            beat_frames, _ = read_beats(file_struct.beat_file)
            pos_matrix = np.load(file_struct.pos_matrix_file)
            neg_matrix = np.load(file_struct.neg_matrix_file)

            points = np.arange(len(pos_matrix))
            nb_peaks = []
            for point in list(points):
                row = pos_matrix[point]
                peaks = librosa.util.peak_pick(row, pre_max=8, post_max=8, pre_avg=16, post_avg=16, delta=0.05, wait=16)
                nb_peaks.append(len(peaks))
            # ADD another condition, which samples n rows in pos_matrix, calculated number peaks, if mean of these peaks > threshold: ok (to avoid selecting single-diagonals sampling ssms)

            if len(beat_frames) > 200 and np.mean(pos_matrix) > .07 and np.mean(pos_matrix) < .3 and self.split == 'train' and np.mean(nb_peaks)>2.5:

                beat_frames = librosa.util.fix_frames(beat_frames)[:self.max_len]
                nb_embeddings = len(beat_frames) - 1
                pos_matrix = pos_matrix[:self.max_len, :self.max_len]
                neg_matrix = neg_matrix[:self.max_len, :self.max_len]
                # Sample anchor indexes
                anchor_indexes = np.random.choice(
                    np.arange(0, nb_embeddings),
                    size=min(self.n_anchors, nb_embeddings),
                    replace=False
                )
                anchors, positives, negatives = [], [], []

                for anchor_index in anchor_indexes:
                    pos_idx, neg_idx = self._sampler_matrices(anchor_index,
                                self.n_positives, self.n_negatives, 
                                pos_matrix, neg_matrix)
                    anchors.append(anchor_index)
                    positives.append(pos_idx)
                    negatives.append(neg_idx)
                
                beat_frames = [i*256 for i in beat_frames]
                sampled_frames.append((track, beat_frames, anchors, positives, negatives))

            elif self.split == 'valid':
                
                beat_frames = librosa.util.fix_frames(beat_frames)
                nb_embeddings = len(beat_frames) - 1

                anchor_indexes = np.random.choice(
                    np.arange(0, nb_embeddings),
                    size=min(self.n_anchors, nb_embeddings),
                    replace=False
                )
                anchors, positives, negatives = [], [], []

                for anchor_index in anchor_indexes:
                    pos_idx, neg_idx = self._sampler_matrices(anchor_index,
                                self.n_positives, self.n_negatives, 
                                pos_matrix, neg_matrix)
                    anchors.append(anchor_index)
                    positives.append(pos_idx)
                    negatives.append(neg_idx)
                    
                beat_frames = [i*256 for i in beat_frames]
                sampled_frames.append((track, beat_frames, anchors, positives, negatives))


        return sampled_frames

    def _sampler_matrices(self, anchor_index, n_positives, n_negatives, pos_matrix, neg_matrix):
        """
        Samples positive and negative indexes for a given anchor.
        
        Args:
            anchor_index (int): Index of the anchor frame.
            nb_embeddings (int): Total number of embeddings.
        
        Returns:
            tuple: Positive and negative sample indexes.
        """
        random.seed(None)

        possible_positives = [i for i in range(len(pos_matrix))]
        possible_positives = [i for i in possible_positives if np.abs(i-anchor_index)>0]
        positive_prob = softmax( pos_matrix[anchor_index][possible_positives] / self.temperature_positives )
        try:
            positive_indexes = np.random.choice(possible_positives, size=n_positives, replace=False, p=positive_prob)
        except:
            positive_indexes = np.random.choice(possible_positives, size=n_positives, replace=True, p=positive_prob)

        possible_negatives = [i for i in range(len(neg_matrix))]
        possible_negatives = [i for i in possible_negatives if np.abs(i-anchor_index)>0]
        negative_prob = softmax( neg_matrix[anchor_index][possible_negatives] / self.temperature_negatives )
        try:
            negative_indexes = np.random.choice(possible_negatives, size=n_negatives, replace=False, p=negative_prob)
        except:
            negative_indexes = np.random.choice(possible_negatives, size=n_negatives, replace=True, p=negative_prob)

        return positive_indexes, negative_indexes

    def __getitem__(self, index):
        """
        Retrieves a data sample.
        
        Args:
            index (int): Index of the sample.
        
        Returns:
            tuple: Condition, features, anchors, positives, negatives.
        """
        #track, anchors, positives, negatives = self.sampled_frames[index]
        track, beat_frames, anchors, positives, negatives = self.sampled_frames[index]
        file_struct = FileStruct(track)

        # Read and pad waveform
        waveform = np.load(file_struct.audio_npy_file, mmap_mode='r', allow_pickle=True)
        pad_width = (self.hop_length * self.n_embedding - 2) // 2
        features_padded = np.pad(waveform, pad_width=(pad_width, pad_width), mode='edge')

        # Extract features at beat frames
        #beat_frames, _ = read_beats(file_struct.beat_file)
        #beat_frames = librosa.util.fix_frames(beat_frames)
        #beat_frames = [i*256 for i in beat_frames]
        
        features = np.stack([
            features_padded[i:i + pad_width * 2] for i in beat_frames
        ], axis=0)

        return (
            track,
            torch.tensor(np.array(features), dtype=torch.float32),  # Convert list of arrays to a single ndarray first
            torch.tensor(np.array(anchors), dtype=torch.long),
            torch.tensor(np.array(positives).flatten(), dtype=torch.long),  # Flatten before converting
            torch.tensor(np.array(negatives).flatten(), dtype=torch.long)   # Flatten before converting
        )

    def __len__(self):
        """
        Returns the total number of samples.
        """
        return len(self.sampled_frames)


def get_dataloader(
    split, data_path, max_len, n_embedding, hop_length, sample_rate,
    n_conditions, n_anchors, n_positives, n_negatives, 
    temperature_positives, temperature_negatives,
    n_samples, batch_size, num_workers
):
    """
    Creates a DataLoader for the SimCLR dataset.

    Args:
        split (str): Data split ('train', 'val', or 'test').
        tracklist (list): List of tracks to process.
        n_embedding (int): Number of embeddings.
        hop_length (int): Hop length for audio processing.
        sample_rate (int): Sampling rate of audio.
        n_conditions (int): Number of sampling conditions.
        n_anchors (int): Number of anchor samples.
        n_positives (int): Number of positive samples per anchor.
        n_negatives (int): Number of negative samples per anchor.
        n_samples (int): Number of samples to process.
        batch_size (int): Batch size for DataLoader.
        num_workers (int): Number of parallel workers.
    
    Returns:
        DataLoader: Configured DataLoader instance.
    """
    dataset = DatasetSimCLR(
        split, data_path, max_len, n_embedding, hop_length, sample_rate, 
        n_conditions, n_anchors, n_positives, n_negatives, temperature_positives, 
        temperature_negatives, n_samples
    )
    return data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=True
    )
