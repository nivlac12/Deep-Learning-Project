import sys
import argparse
import os
import json
from time import time
from datetime import timedelta
from os.path import join, exists
import pdb
import tensorflow as tf
import keras_nlp

from utils import read_jsonl, get_data_path, get_result_path

from dataloader import MatchSumPipe
from model import MatchSum
from metrics import MarginRankingLoss, ValidMetric, MatchRougeMetric
from callback import MyLRSchedule


def configure_training(args):
    devices = [int(gpu) for gpu in args.gpus.split(',')]
    params = {}
    params['encoder']       = args.encoder
    params['candidate_num'] = args.candidate_num
    params['batch_size']    = args.batch_size
    params['accum_count']   = args.accum_count
    params['max_lr']        = args.max_lr
    params['margin']        = args.margin
    params['warmup_steps']  = args.warmup_steps
    params['n_epochs']      = args.n_epochs
    params['valid_steps']   = args.valid_steps
    return devices, params

def train_model(args):
    num_samples=2
    # check if the data_path and save_path exists
    data_paths = get_data_path(args.mode, 'bert')
    print(data_paths)
    for name in data_paths:
        assert exists(data_paths[name])
    if not exists(args.save_path):
        os.makedirs(args.save_path)

    # data_paths = get_data_path(args.mode, 'bert')
    # for name in data_paths:
    #     assert exists(data_paths[name])
    # if not exists(args.save_path):
    #     os.makedirs(args.save_path)
    
    # load summarization datasets
    #imported from dataloader.py file
    train_cand_dataset, train_text_dataset, train_summ_dataset = MatchSumPipe(num_samples=num_samples).process_from_file(data_paths['train'])
    valid_cand_dataset, valid_text_dataset, valid_summ_dataset = MatchSumPipe(num_samples=num_samples).process_from_file(data_paths['val'])
    
    # configure training
    # devices, train_params = configure_training(args)
    _, train_params = configure_training(args)
    with open(join(args.save_path, 'params.json'), 'w') as f:
        json.dump(train_params, f, indent=4)
    # print('Devices is:')
    # print(devices)
    
    x = read_jsonl(data_paths['val'], num_samples=num_samples)
    summaries = []
    for i in x:
        summaries.append(' '.join(i['summary']))
    val_summaries = tf.convert_to_tensor(summaries)

    candidate_summaries = []
    for i in x:
        # i = a text
        lst = []
        indices = i['indices']
        text = i['text'] 
        #for ind in indices:
        cand_sum = []
        for j in indices[0]:
            cand_sum.append(text[j])
        lst.append(' '.join(cand_sum))
        candidate_summaries.append(' '.join(cand_sum))
    val_candidate_summaries = tf.convert_to_tensor(candidate_summaries)

    x = read_jsonl(data_paths['train'], num_samples=num_samples)

    summaries = []
    for i in x:
        summaries.append(' '.join(i['summary']))
    train_summaries = tf.convert_to_tensor(summaries)

    candidate_summaries = []
    for i in x:
        lst = []
        indices = i['indices']
        text = i['text']
        for ind in indices:
            cand_sum = []
            for j in ind:
                cand_sum.append(text[j])
            lst.append(' '.join(cand_sum))
        candidate_summaries.append(lst)
    train_candidate_summaries = tf.convert_to_tensor(candidate_summaries)

    # configure model
    model = MatchSum(args.candidate_num, args.encoder)
    optimizer = tf.keras.optimizers.Adam(learning_rate=MyLRSchedule(0.0, max_lr=args.max_lr, warmup_steps = args.warmup_steps, update_every = args.accum_count)) # Learning rate may need to be changed depending on issues

    criterion = MarginRankingLoss(args.margin)
    # val_metric = ValidMetric(save_path=save_path, data=read_jsonl(data_paths['val']), batch_size=params['batch_size'], max_idx=valid_cand_dataset.shape[0])

    # assert args.batch_size % len(devices) == 0
    
    model.compile(
        optimizer = optimizer,
        loss = criterion,
        metrics = [keras_nlp.metrics.RougeL(), keras_nlp.metrics.RougeN(order=2)]
    )

    print('Start training with the following hyper-parameters:')
    print(train_params)

    pdb.set_trace()

    # pdb.set_trace()
    model.fit(
        x=[train_text_dataset, train_cand_dataset, train_summ_dataset, train_candidate_summaries, train_summaries], # not sure what data structure this is
        y=None,
        batch_size=args.batch_size,
        epochs=args.n_epochs,
        verbose=2, # Can be changed
        # callbacks=[MyCallback()],
        # validation_split=0.0,
        validation_data=[[valid_text_dataset, valid_cand_dataset, valid_summ_dataset], [val_candidate_summaries, val_summaries]],
        shuffle=False, # This could be something that is important
        # class_weight=None,
        # sample_weight=None,
        # initial_epoch=0,
        # steps_per_epoch=None,
        # validation_steps=1,
        # validation_batch_size=valid_cand_dataset.shape[0],
        # validation_freq=1,
        # max_queue_size=10,
        # workers=1,
        # use_multiprocessing=False
    )
    # trainer = Trainer(
    #   train_data=train_set,
    #   model=model, optimizer=optimizer,
    #   loss=criterion,
    #   batch_size=args.batch_size,
    #   update_every=args.accum_count,
    #   n_epochs=args.n_epochs, 
    #   print_every=10,
    #   dev_data=valid_set,
    #   metrics=val_metric, 
    #   metric_key='ROUGE',
    #   validate_every=args.valid_steps, 
    #   save_path=args.save_path, device=devices,
    #   callbacks=callbacks)

    # model.evaluate(
    # #x_test,
    # #y_test,
    # #batch_size=128,
    # #verbose=0, 
    # #callbacks=tf.keras.callbacks.CallbackList(
    # #callbacks=MyCallback(args), add_progbar=True,
    # )



# def test_model(args):

#     models = os.listdir(args.save_path)
    
#     # load dataset
#     data_paths = get_data_path(args.mode, args.encoder)
#     #from dataloader file
#     datasets = MatchSumPipe(args.candidate_num, args.encoder).process_from_file(data_paths)
#     print('Information of dataset is:')
#     print(datasets)
#     test_set = datasets.datasets['test']
    
#     # need 1 gpu for testing
#     device = int(args.gpus)
    
#     args.batch_size = 1

#     for cur_model in models:
        
#         print('Current model is {}'.format(cur_model))

#         # load model
#         model = torch.load(join(args.save_path, cur_model))
    
#         # configure testing
#         dec_path, ref_path = get_result_path(args.save_path, cur_model)
#         test_metric = MatchRougeMetric(data=read_jsonl(data_paths['test']), dec_path=dec_path, 
#                                   ref_path=ref_path, n_total = len(test_set))
#         tester = Tester(data=test_set, model=model, metrics=[test_metric], 
#                         batch_size=args.batch_size, device=device, use_tqdm=False)
#         tester.test()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='training/testing of MatchSum'
    )
    parser.add_argument('--mode', required=True,
                        help='training or testing of MatchSum', type=str)

    parser.add_argument('--save_path', required=True,
                        help='root of the model', type=str)
    # example for gpus input: '0,1,2,3'
    parser.add_argument('--gpus', required=True,
                        help='available gpus for training(separated by commas)', type=str)
    parser.add_argument('--encoder', required=True,
                        help='the encoder for matchsum (bert/roberta)', type=str)

    parser.add_argument('--batch_size', default=2,
                        help='the training batch size', type=int)
    parser.add_argument('--accum_count', default=2,
                        help='number of updates steps to accumulate before performing a backward/update pass', type=int)
    parser.add_argument('--candidate_num', default=2,
                        help='number of candidates summaries', type=int)
    parser.add_argument('--max_lr', default=2e-5,
                        help='max learning rate for warm up', type=float)
    parser.add_argument('--margin', default=0.01,
                        help='parameter for MarginRankingLoss', type=float)
    parser.add_argument('--warmup_steps', default=100,
                        help='warm up steps for training', type=int)
    parser.add_argument('--n_epochs', default=5,
                        help='total number of training epochs', type=int)
    parser.add_argument('--valid_steps', default=100,
                        help='number of update steps for validation and saving checkpoint', type=int)

    args = parser.parse_known_args()[0]
    
    if args.mode == 'train':
        print('Training process of MatchSum !!!')
        train_model(args)
    # else:
    #     print('Testing process of MatchSum !!!')
    #     test_model(args)

