# coding:utf-8
import torch
import numpy as np
import json
import opennre
from opennre import encoder, model, framework
import sys
import os
import argparse
import logging
import random

from sklearn.model_selection import train_test_split

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def reset_weights(m):
  '''
    Try resetting model weights to avoid
    weight leakage.
  '''
  for layer in m.children():
   if hasattr(layer, 'reset_parameters'):
    print(f'Reset trainable parameters of layer = {layer}')
    layer.reset_parameters()

parser = argparse.ArgumentParser()
parser.add_argument('--pretrain_path', default='bert-base-uncased',
        help='Pre-trained ckpt path / model name (hugginface)')
parser.add_argument('--ckpt', default='',
        help='Checkpoint name')
parser.add_argument('--pooler', default='entity', choices=['cls', 'entity'],
        help='Sentence representation pooler')
parser.add_argument('--only_test', action='store_true',
        help='Only run test')
parser.add_argument('--mask_entity', action='store_true',
        help='Mask entity mentions')

# Data
parser.add_argument('--metric', default='micro_f1', choices=['micro_f1', 'acc'],
        help='Metric for picking up best checkpoint')
parser.add_argument('--input_file', default='', type=str,
        help='Input data file')
parser.add_argument('--rel2id_file', default='', type=str,
        help='Relation to ID file')

# Cross Validation
parser.add_argument('--folds', default='5', type=int,
        help='Number of folds for cross validation')

parser.add_argument('--train-split', default='0.8', type=float,
        help='Size of training partition')
parser.add_argument('--val-split', default='0.1', type=float,
        help='Size of validation partition')
parser.add_argument('--test-split', default='0.1', type=float,
        help='Size of test partition')

# Hyper-parameters
parser.add_argument('--batch_size', default=128, type=int,
        help='Batch size')
parser.add_argument('--lr', default=2e-5, type=float,
        help='Learning rate')
parser.add_argument('--max_length', default=128, type=int,
        help='Maximum sentence length')
parser.add_argument('--max_epoch', default=1, type=int,
        help='Max number of training epochs')

# Seed
parser.add_argument('--seed', default=42, type=int,
        help='Seed')

args = parser.parse_args()

# Set random seed
set_seed(args.seed)

# Some basic settings
root_path = '.'
sys.path.append(root_path)


logging.info('Arguments:')
for arg in vars(args):
    logging.info('    {}: {}'.format(arg, getattr(args, arg)))

rel2id = json.load(open(args.rel2id_file))



#iterate through folds for k-fold cross validation
for iteration in range(1,args.folds):
    print("Starting {}. fold.".format(iteration))
    # Define the sentence encoder
    if args.pooler == 'entity':
        sentence_encoder = opennre.encoder.BERTEntityEncoder(
            max_length=args.max_length,
            pretrain_path=args.pretrain_path,
            mask_entity=args.mask_entity
        )
    elif args.pooler == 'cls':
        sentence_encoder = opennre.encoder.BERTEncoder(
            max_length=args.max_length,
            pretrain_path=args.pretrain_path,
            mask_entity=args.mask_entity
        )
    else:
        raise NotImplementedError

    # Define the model
    model = opennre.model.SoftmaxNN(sentence_encoder, len(rel2id), rel2id)

    # create cross validation splits
    with open(args.input_file,'r') as f:
        lines = f.readlines()
        #splitting input file
        train, test = train_test_split(lines, test_size=args.val_split+args.test_split)

        val_test_proportion = args.val_split / (args.test_split+args.val_split)
        test, val = train_test_split(test, test_size=val_test_proportion)

    #Writing train, val, test file to root folder
    train_path = "{}_train.txt".format(root_path)
    val_path = "{}_val.txt".format(root_path)
    test_path = "{}_test.txt".format(root_path)
    with open(train_path, 'w') as f:
        for line in train:
            f.write(line)
    with open(val_path, 'w') as f:
        for line in val:
            f.write(line)
    with open(test_path, 'w') as f:
        for line in test:
            f.write(line)

    if not os.path.exists('ckpt'):
        os.mkdir('ckpt')

    ckpt = 'ckpt/{}_fold_{}.pth.tar'.format(args.ckpt,iteration)

    # Define the whole training framework
    framework = opennre.framework.SentenceRE(
        train_path=train_path,
        val_path=val_path,
        test_path=test_path,
        model=model,
        ckpt=ckpt,
        batch_size=args.batch_size,
        max_epoch=args.max_epoch,
        lr=args.lr,
        opt='adamw'
    )

    # Train the model
    framework.train_model('micro_f1')

    # Test
    framework.load_state_dict(torch.load(ckpt)['state_dict'])
    result = framework.eval_model(framework.test_loader)

    # Print the result
    logging.info('Test set results:')
    logging.info('Accuracy: {}'.format(result['acc']))
    logging.info('Micro precision: {}'.format(result['micro_p']))
    logging.info('Micro recall: {}'.format(result['micro_r']))
    logging.info('Micro F1: {}'.format(result['micro_f1']))


    #delete splits
    os.remove(train_path)
    os.remove(val_path)
    os.remove(test_path)
    # delete temporary files
    #reset_weights(model)
   # torch.cuda.empty_cache()