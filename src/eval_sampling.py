import numpy as np
from utils import *
from data_loader import get_dataloader
from tqdm import tqdm 
import logging
import argparse
import mir_eval


def main(args):


    TPs, TNs = [], []
    ds_an_pos, ds_an_neg = [], []
    results = {}

    dataset = get_dataloader(args.split, args.data_path, args.max_len, args.n_embedding, args.hop_length, args.sample_rate,
                            args.n_conditions, args.n_anchors, args.n_positives, args.n_negatives, 
                            args.temperature_positives, args.temperature_negatives,
                            args.n_samples, args.batch_size, args.num_workers)

    for triplet in tqdm(dataset.dataset.sampled_frames):
        
        ds_an_pos_track, ds_an_neg_track = [], []

        track, beat_frames, anchors_track, positives_track, negatives_track = triplet
        results[track] = {}
        file_struct = FileStruct(track)
        N = len(beat_frames)
        ref_labels, ref_times, duration_ = get_ref_labels(file_struct, 0)
        ref_inter = times_to_intervals(ref_times)
        (ref_inter, ref_labels) = mir_eval.util.adjust_intervals(ref_inter, list(ref_labels), t_min=0, t_max=ref_inter.max())
        ref_times = intervals_to_times(ref_inter)
        ref_labels_list = get_labels(beat_frames, ref_times, ref_labels)

        for i in range(len(anchors_track)):
            anchor_index = anchors_track[i]
            positive_indexes, negative_indexes = positives_track[i], negatives_track[i]
            positive_indexes = np.unique(positive_indexes)
            negative_indexes = np.unique(negative_indexes)
            pos_ok = 0
            neg_ok = 0
            label_anchor = ref_labels_list[anchor_index]
            if args.transform_labels:

                if 'refrain' in label_anchor:
                    label_anchor = 'refrain'
                if 'chorus' in label_anchor:
                    label_anchor = 'chorus'
                elif 'verse' in label_anchor:
                    label_anchor = 'verse'
                elif 'bridge' in label_anchor:
                    label_anchor = 'bridge'
                else:
                    label_anchor = label_anchor.lower()

            
                for j in positive_indexes:
                    if 'refrain' in ref_labels_list[j].lower():
                        label_pos = 'refrain'
                    if 'chorus' in ref_labels_list[j].lower():
                        label_pos = 'chorus'
                    elif 'verse' in ref_labels_list[j].lower():
                        label_pos = 'verse'
                    elif 'bridge' in ref_labels_list[j].lower():
                        label_pos = 'bridge'
                    else:
                        label_pos = ref_labels_list[j].lower()

                    if label_pos == label_anchor.lower():
                        pos_ok += 1
                    else:
                        if 'SALAMI' in args.data_path:
                            if label_pos[0] == label_anchor and np.abs(len(ref_labels_list[j])-len(label_anchor)) < 3:
                                pos_ok += 1
            else:
                for j in positive_indexes:
                    label_pos = ref_labels_list[j]
                    if label_pos == label_anchor:
                            pos_ok += 1
     
            ds_an_pos.append(np.abs(anchor_index-j)/N) #/N
            ds_an_pos_track.append(np.abs(anchor_index-j)/N) #/N
            
            if args.transform_labels:
                for j in negative_indexes:

                    negative_label = ref_labels_list[j].lower()
                    if 'refrain' in negative_label:
                        negative_label = 'refrain'
                    elif 'verse' in negative_label:
                        negative_label = 'verse'
                    elif 'bridge' in negative_label:
                        negative_label = 'bridge'
                    elif 'chorus' in negative_label:
                        negative_label = 'chorus'
                    else:
                        negative_label = negative_label.lower()
                    
                    if 'SALAMI' in args.data_path:
                        if negative_label[0] == label_anchor and np.abs(len(negative_label)-len(label_anchor)) < 3:
                            neg_ok += 1
                    else:
                        if negative_label != label_anchor:
                            neg_ok += 1

            else:
                for j in negative_indexes:
                    negative_label = ref_labels_list[j]
                    if negative_label != label_anchor:
                        neg_ok += 1
                

            ds_an_neg.append(np.abs(anchor_index-j)/N) #/N
            ds_an_neg_track.append(np.abs(anchor_index-j)/N) #
        
            TP = pos_ok/len(positive_indexes)
            TN = neg_ok/len(negative_indexes)

            TPs.append(TP)
            TNs.append(TN)

            results[track]['TP'] = TP
            results[track]['TN'] = TN
            results[track]['an_pos'] = np.mean(ds_an_pos_track)
            results[track]['an_neg'] = np.mean(ds_an_neg_track)
            

    dataset_name = args.data_path.split('/')[-2]

    
    print('---------------')
    print('Results on ', dataset_name, 'dataset :')
    print('TPs =', np.mean(TPs), '+/-', np.std(TPs))
    print('TNs =', np.mean(TNs), '+/-', np.std(TNs))
    print('Mean anchor pos', np.mean(ds_an_pos), '+/-', np.std(ds_an_pos))
    print('Mean anchor neg', np.mean(ds_an_neg), '+/-', np.std(ds_an_neg))

    
def parse_args():
    """
    Parses command-line arguments.

    Returns:
        Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Dataset args
    parser.add_argument('--split', type=str, default="train")
    parser.add_argument('--data_path', type=str, default="../../msaf_/datasets/RWC-Pop/")
    parser.add_argument('--max_len', type=int, default=2500)
    parser.add_argument('--n_embedding', type=int, default=64)
    parser.add_argument('--hop_length', type=int, default=256)
    parser.add_argument('--sample_rate', type=int, default=22050)
    parser.add_argument('--n_conditions', type=int, default=1)
    parser.add_argument('--n_anchors', type=int, default=32)
    parser.add_argument('--n_positives', type=int, default=32)
    parser.add_argument('--n_negatives', type=int, default=64)
    parser.add_argument('--temperature_positives', type=float, default=.1)
    parser.add_argument('--temperature_negatives', type=float, default=.1)
    parser.add_argument('--n_samples', type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--transform_labels', type=int, default=1)

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    logging.info(f'Arguments: {args}')
    main(args)
