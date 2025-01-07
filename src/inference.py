import os
import torch
import argparse
import numpy as np
import pytorch_lightning as pl
from models import *
from lightning_model import PLModel
from data_loader import get_dataloader as ssm_dataloader
from predict_async import apply_async_with_callback
from sklearn.manifold import TSNE
import pickle
from tqdm import tqdm
import joblib
from losses import *
from collections import OrderedDict
from pathlib import Path



def tsne_projection(track, embeddings):
    tsnes = []
    embeddings_tensor = torch.tensor(embeddings)
    embeddings_tensor = F.normalize(embeddings_tensor, p=2, dim=-1)
    embeddings_ = embeddings_tensor.numpy()
    X_embedded = TSNE(n_components=2).fit_transform(embeddings_)
    tsnes.append(X_embedded)
    
    return [track, embeddings, tsnes]


def inference(args):
    
    test_dataloader = ssm_dataloader(split='train', data_path=args.data_path, max_len=args.max_len, n_embedding=args.n_embedding, hop_length=args.hop_length, sample_rate=args.sample_rate, 
        n_conditions=args.n_conditions, n_anchors=args.n_anchors, n_positives=args.n_positives, n_negatives=args.n_negatives, temperature_positives=args.temperature_positives, 
        temperature_negatives=args.temperature_negatives, n_samples=args.n_samples, batch_size=1, num_workers=args.num_workers)


    num_gpus = torch.cuda.device_count()

    trainer = pl.Trainer(devices=num_gpus, accelerator="auto")

    if 'epoch' not in args.checkpoint_path and '.ckpt' not in args.checkpoint_path:
        check_point_name = os.listdir()[-1]
        check_point_name = max(Path(args.checkpoint_path).glob('*.ckpt'), key=os.path.getmtime)
        checkpoint_path = os.path.join(args.checkpoint_path, check_point_name)
    else:
        checkpoint_path = args.checkpoint_path

    print(checkpoint_path)
    
    model = PLModel.load_from_checkpoint(checkpoint_path)

    torch.save(model.state_dict(), '/tsi/data_doctorants/mbuisson/LinkSeg/data/backbone_repetition_2.pt')

    predictions = trainer.validate(model, dataloaders=test_dataloader)

    
    model.verbose = True


    embeddings_list = model.embeddings_list
    tracklist = [i[0] for i in embeddings_list]

    
    if args.split_number > -1 : 
        dataset_name = args.data_path.split('/')[-1]
        splits_dict_path = '/tsi/data_doctorants/mbuisson/GNN/SPLITS/splits_dict_'+dataset_name+'.pkl'
        with open(splits_dict_path, 'rb') as fp:
            split_dict = pickle.load(fp)
        overall_dict = split_dict[args.split_number]
        tracklist_split = overall_dict['test']
    
        new_embeddings_list = []
        for i in range(len(embeddings_list)):
            if embeddings_list[i][0] in tracklist_split:
                new_embeddings_list.append(embeddings_list[i])
        
        embeddings_list = new_embeddings_list
        tracklist = tracklist_split


    level = int(args.annot_level)
    print('Annot level =', level)
    

    out = apply_async_with_callback(embeddings_list, tracklist, True, False, 0, None, embedding_levels=[], annot=0)
    

    dataset_name = embeddings_list[0][0].split('/')[-3]


    if args.tsne == 1:
        print('Calculating T-SNE projections ...')
        jobs = [ joblib.delayed(tsne_projection)(track=i[0], embeddings=i[1]) for i in tqdm(embeddings_list) ]
        new_embeddings_list = joblib.Parallel(n_jobs=32, verbose=1)(jobs)
        
        print('Length embeddings list =', len(new_embeddings_list))
        #pickle.dump(new_embeddings_list, open( "embeddings_REPETITION_"+dataset_name+".pkl", "wb" ) )
        #pickle.dump(new_embeddings_list, open( "embeddings_MULTI_LEVEL_"+dataset_name+".pkl", "wb" ) )
        #pickle.dump(new_embeddings_list, open( "embeddings_MULTI_LEVEL_"+dataset_name+".pkl", "wb" ) )
    
    #else:
        #pickle.dump(out, open( "results_defense_multi_level"+dataset_name+"_"+str(embedding_levels[0])+"_"+str(level)+"_res_dict_.pkl", "wb" ) )


def parse_args():
    """
    Parses command-line arguments.

    Returns:
        Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Trainer args
    parser.add_argument('--num_nodes', type=int, default=1)
    parser.add_argument('--max_epochs', type=int, default=200)
    parser.add_argument('--gradient_clip_val', type=float, default=1)
    parser.add_argument('--accumulate_grad_batches', type=int, default=1)
    parser.add_argument('--enable_progress_bar', type=int, default=1)

    # Model args
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--temperature_loss', type=float, default=.1)
    parser.add_argument('--symmetrical', type=bool, default=False)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--check_val_every_n_epoch', type=int, default=1)
    parser.add_argument('--random_seed', type=int, default=42)

    # STFT args
    parser.add_argument('--n_mels', type=int, default=64)
    parser.add_argument('--n_fft', type=int, default=1024)
    parser.add_argument('--hop_length', type=int, default=256)
    parser.add_argument('--f_min', type=int, default=0)
    parser.add_argument('--f_max', type=int, default=11025)
    parser.add_argument('--sample_rate', type=int, default=22050)

    # Frame encoder args
    parser.add_argument('--n_embedding', type=int, default=64)
    parser.add_argument('--embedding_dim', type=int, default=32)
    parser.add_argument('--conv_ndim', type=int, default=32)
    parser.add_argument('--attention_ndim', type=int, default=32)
    parser.add_argument('--attention_nheads', type=int, default=8)
    parser.add_argument('--attention_nlayers', type=int, default=2)
    parser.add_argument('--hidden_dim', type=int, default=32)
    parser.add_argument('--dropout', type=float, default=0.1)

    # Sampling args
    parser.add_argument('--n_anchors', type=int, default=16)
    parser.add_argument('--n_positives', type=int, default=32)
    parser.add_argument('--n_negatives', type=int, default=64)
    parser.add_argument('--temperature_positives', type=float, default=.1)
    parser.add_argument('--temperature_negatives', type=float, default=.1)
    parser.add_argument('--n_conditions', type=int, default=1)
    parser.add_argument('--max_len', type=int, default=2500)
    parser.add_argument('--n_samples', type=int, default=1e6)
    parser.add_argument('--n_val_samples', type=int, default=1e6)
    parser.add_argument('--double_beats', type=int, default=2)

    # Paths
    parser.add_argument('--data_path', type=str, default=None)
    parser.add_argument('--checkpoint_path', type=str, default=None)
    parser.add_argument('--val_data_path', type=str, default=None)
    parser.add_argument('--output_dir', type=str, default='./results/exp')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    inference(args)