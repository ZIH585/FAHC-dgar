import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import random
from tqdm import tqdm
import torch
import numpy as np
import dgl
import copy
import fitlog
import pickle
import time
import itertools
import argparse
from pathlib import Path
from src.cli import build_arg_parser, validate_args
from src.runtime import (PROTOCOL, prepare_run, restore_evaluation_rng,
                         save_training_state, restore_training_state, validate_history_cache)
from rgcn import utils
from rgcn.utils import *
from src.rrgcn import RecurrentRGCN
from src.hyperparameter_range import hp_range
import torch.nn.modules.rnn
from collections import defaultdict
from rgcn.knowledge_graph import _read_triplets_as_list
from diffusion.model_21 import create_model_diffu, Att_Diffuse_model
from diffusion.dcr import Recorder
from torch.optim import *

import copy
import math
    
    
def getReverseSample(test_triplets, num_rels):
    inverse_test_triplets = test_triplets[:, [2, 1, 0]]  # inverse triplets
    inverse_test_triplets[:, 1] = inverse_test_triplets[:, 1] + num_rels
    a_triples = torch.concat(
        (test_triplets, inverse_test_triplets))  # [2,n_triplets,3]
    return a_triples


@torch.no_grad()
def test(args, model, history_list, test_list, num_rels, num_nodes,
         use_cuda, all_ans_list, all_ans_r_list, model_name,
         static_graph, mode, last_history, last_output,
         static_graph_diffu, model_diffpre):
    """
    :param model: model used to test
    :param history_list:    all input history snap shot list, not include output label train list or valid list
    :param test_list:   test triple snap shot list
    :param num_rels:    number of relations
    :param num_nodes:   number of nodes
    :param use_cuda:
    :param all_ans_list:     dict used to calculate filter mrr (key and value are all int variable not tensor)
    :param all_ans_r_list:     dict used to calculate filter mrr (key and value are all int variable not tensor)
    :param model_name:
    :param static_graph
    :param mode
    :return mrr_raw, mrr_filter, mrr_raw_r, mrr_filter_r
    """
    ranks_raw, ranks_filter, mrr_raw_list, mrr_filter_list = [], [], [], []
    ranks_raw_r, ranks_filter_r, mrr_raw_list_r, mrr_filter_list_r = [], [], [], []
    MRRs=[]
    H1=[]
    H3=[]
    H10=[]
    
    
    idx = 0
    if mode == "test":
        # test mode: load parameter form file
        if use_cuda:
            checkpoint = torch.load(
                model_name, map_location=torch.device("cuda:{}".format(args.gpu)))
        else:
            checkpoint = torch.load(
                model_name, map_location=torch.device('cpu'))
        # print("Load Model name: {}. Using best epoch : {}".format(
        #     model_name, checkpoint['epoch']))  # use best stat checkpoint
        print("\n"+"-"*10+"start testing"+"-"*10+"\n")
        model.load_state_dict(checkpoint)

    model.eval()
    device = next(model.parameters()).device
    # do not have inverse relation in test input
    his_list = [snap for snap in history_list]

    last_history = last_history
    last_output = last_output
    
    mrr_1, hits1_1, hits3_1, hits10_1=0.0,0.0,0.0,0.0
    mrr_sum_avg,hit1_sum_avg,hit3_sum_avg,hit10_sum_avg=[],[],[],[]

    for time_idx, test_snap in enumerate(tqdm(test_list)):
        if time_idx==0:
            continue
        # Match the author's history-window condition, including nondefault lengths.
        if time_idx - args.train_history_len < 0:
            input_list = his_list[:time_idx]
        else:
            input_list = his_list[time_idx - args.test_history_len:time_idx]
        history_glist = [build_sub_graph(
            num_nodes, num_rels, g, use_cuda, args.gpu) for g in input_list]
        
        # query_er_history[str(element[0])][str(element[1])]=list(set(query_er_history[str(element[0])][str(element[1])]))
        current_query_history_triples = set()
        current_query_history_graph = None
        
        if time_idx-1<args.reply_batch:
            reply_time_index=list(range(time_idx))
        else:
            reply_time_index=random.sample(list(range(time_idx)), args.reply_batch)
                    
        for i in reply_time_index:
            with open(args.data_root / args.dataset / "history_snap_v2" / f"snap_{i}.pkl", "rb") as stream:
                related_history = pickle.load(stream)
            for element in test_snap:
                if str(element[0]) in related_history.keys():
                    current_query_history_triples.update(related_history[str(element[0])])
        
        test_triples_input = torch.LongTensor(
            test_snap).cuda() if use_cuda else torch.LongTensor(test_snap)
        test_triples_input = test_triples_input.to(device)

        sequence = None
        output_reply_triple = torch.as_tensor(list(current_query_history_triples),
            dtype=torch.long, device=device).reshape(-1, 3)
        if args.diffuc and len(output_reply_triple)>0 and args.delete_feature_reply is False:
            scores, diffu_rep, weights, t, _, ent_emb, noise, _, targets = model_diffpre(
                sequence, output_reply_triple, args, False, use_cuda, static_graph=static_graph_diffu, ct=time_idx, model=model, history_glist=last_history, triples=last_output)

            model_output_o=diffu_rep[:,-1,:]
        else:
            diffu_rep=None
            model_output_o=None
        test_triples, final_score, final_r_score = model.predict(
            history_glist, num_rels, static_graph, test_triples_input, diffu_rep,model_output_o, output_reply_triple, use_cuda, current_query_history_graph)

        mrr_filter_snap_r, mrr_snap_r, rank_raw_r, rank_filter_r = utils.get_total_rank(
            test_triples, final_r_score, all_ans_r_list[time_idx], eval_bz=1000, rel_predict=1)
        mrr_filter_snap, mrr_snap, rank_raw, rank_filter = utils.get_total_rank(
            test_triples, final_score, all_ans_list[time_idx], eval_bz=1000, rel_predict=0)
        
        mrr,hits=utils.stat_ranks_record(rank_filter,"predict_filter_ent")
        # One score per time slice, with inverse queries included by predict().
        mrr_value = mrr.item()
        hit_values = [hit.item() for hit in hits]
        MRRs.append(mrr_value)
        H1.append(hit_values[0])
        H3.append(hit_values[1])
        H10.append(hit_values[2])
        if time_idx == len(test_list) - 1:
            print("Current MRR (filter_ent): {:.6f}".format(mrr_value))
            print("Current Hits@1 (filter_ent): {:.6f}".format(hit_values[0]))
            print("Current Hits@10 (filter_ent): {:.6f}".format(hit_values[2]))
        else:
            mrr_sum_avg.append(mrr_value)
            hit1_sum_avg.append(hit_values[0])
            hit3_sum_avg.append(hit_values[1])
            hit10_sum_avg.append(hit_values[2])
        
        
        # used to global statistic
        ranks_raw.append(rank_raw)
        ranks_filter.append(rank_filter)
        # used to show slide results
        mrr_raw_list.append(mrr_snap)
        mrr_filter_list.append(mrr_filter_snap)

        # relation rank
        ranks_raw_r.append(rank_raw_r)
        ranks_filter_r.append(rank_filter_r)
        mrr_raw_list_r.append(mrr_snap_r)
        mrr_filter_list_r.append(mrr_filter_snap_r)

        idx += 1
        last_output = [snap for snap in test_triples_input]
        last_history = [snap for snap in history_glist]
    # The original loop skips time_idx=0; Average covers 1..T-2.
    if mrr_sum_avg:
        print("Average MRR (macro, time_idx=1..T-2): {:.6f}".format(np.mean(mrr_sum_avg)))
        print("Average Hits@1 (macro, time_idx=1..T-2): {:.6f}".format(np.mean(hit1_sum_avg)))
        print("Average Hits@10 (macro, time_idx=1..T-2): {:.6f}".format(np.mean(hit10_sum_avg)))
    else:
        print("Average metrics unavailable: no previous evaluated slice")
    mrr_raw = utils.stat_ranks(ranks_raw, "raw_ent")
    mrr_filter = utils.stat_ranks(ranks_filter, "filter_ent")
    mrr_raw_r = utils.stat_ranks(ranks_raw_r, "raw_rel")
    mrr_filter_r = utils.stat_ranks(ranks_filter_r, "filter_rel")
    
    if args.dcr_mode != 'original':
        args._dcr_recorder.flush()
    return mrr_raw, mrr_filter, mrr_raw_r, mrr_filter_r, last_history, last_output,MRRs,H1,H3,H10,mrr_sum_avg,hit1_sum_avg,hit3_sum_avg,hit10_sum_avg

@torch.no_grad()
def test_single(args, model, history_glist, test_triples_input,
                time_index, num_rels, num_nodes, use_cuda,
                all_ans_list, static_graph,
                diffu_rep=None, model_output_o=None,
                output_reply_triple=None):

    model.eval()

    if diffu_rep is not None:
        model_output_o = diffu_rep[:, -1, :]
    else:
        model_output_o = None

    test_triples, final_score, final_r_score = model.predict(
        history_glist,
        num_rels,
        static_graph,
        test_triples_input,
        diffu_rep,
        model_output_o,
        output_reply_triple,
        use_cuda
    )

    mrr_filter_snap, mrr_snap, rank_raw, rank_filter = \
        utils.get_total_rank(
            test_triples,
            final_score,
            all_ans_list[time_index],
            eval_bz=1000,
            rel_predict=0
        )

    mrr, hits = utils.stat_ranks_record(
        rank_filter,
        "predict_filter_ent"
    )

    return mrr, hits, rank_filter


def run_experiment(args, n_hidden=None, n_layers=None, dropout=None, n_bases=None):
    # load configuration for grid search the best configuration
    if n_hidden is not None:
        args.n_hidden = n_hidden
    if n_layers is not None:
        args.n_layers = n_layers
    if dropout is not None:
        args.dropout = dropout
    if n_bases is not None:
        args.n_bases = n_bases
    validate_args(args)
    print(f"Baseline protocol: {getattr(args, '_protocol', PROTOCOL)}")
    # Upstream declares --seed but never applies it. No flag preserves that
    # initialization policy; an explicit seed is a documented reproducibility opt-in.
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed % (2**32))
        torch.default_generator.manual_seed(args.seed)
    print(f"Seed policy: {'upstream (no explicit reseeding)' if args.seed is None else args.seed}")
    use_cuda = args.gpu >= 0 and torch.cuda.is_available()
    run_device = torch.device(f"cuda:{args.gpu}" if use_cuda else "cpu")
    if use_cuda:
        torch.cuda.set_device(args.gpu)
        if args.seed is not None:
            torch.cuda.manual_seed(args.seed)
    # load graph data
    print("loading graph data")
    data_list = []
    data = utils.load_data(args.dataset, data_root=args.data_root)
    train_list = utils.split_by_time(data.train)
    valid_list = utils.split_by_time(data.valid)
    test_list = utils.split_by_time(data.test)

    data_list = []
    data_list.extend(train_list)
    data_list.extend(valid_list)
    data_list.extend(test_list)
    train_list,valid_list,test_list = utils.spilt_data(data_list)
    data_history = args.data_root / args.dataset / "history_snap_v2"
    validate_history_cache(train_list, data.num_rels, data_history)
    prepare_run(args)
    if args.dcr_mode != 'original':
        args._dcr_recorder = Recorder(args.run_dir / 'dcr.jsonl')
    fitlog.set_log_dir(str(args.run_dir / "fitlog" if args.run_dir else Path(__file__).resolve().parents[1] / "logs"))
    fitlog.add_hyper({key: str(value) if isinstance(value, Path) else value
                      for key, value in vars(args).items() if not key.startswith('_')})
    
    train_times,val_times,test_times=[],[],[]
    for i in range(len(train_list)):
        train_times.append(i)
        val_times.append(i)
        test_times.append(i)
 
    
    history_times=train_times
    
    test_multi_time=math.ceil(len(train_times)*0.9)
    test_multi_list=[]
    # test_multi_times=[]
    for i in range(test_multi_time,len(train_times)):
        test_step=[]
        test_step.extend(train_list[i])
        test_step.extend(valid_list[i])
        test_step.extend(test_list[i])
        test_multi_list.append(test_step)

    num_nodes = data.num_nodes
    num_rels = data.num_rels
    args.num_rels = num_rels
    args.num_nodes = num_nodes

    all_ans_list_test = utils.load_all_answers_for_time_filter(
        test_list, num_rels, num_nodes, False)
    all_ans_list_r_test = utils.load_all_answers_for_time_filter(
        test_list, num_rels, num_nodes, True)
    all_ans_list_valid = utils.load_all_answers_for_time_filter(
        valid_list, num_rels, num_nodes, False)
    all_ans_list_r_valid = utils.load_all_answers_for_time_filter(
        valid_list, num_rels, num_nodes, True)
    
    all_ans_list = utils.load_all_answers_for_time_filter(data_list, num_rels, num_nodes, False)
    all_ans_list_r = utils.load_all_answers_for_time_filter(data_list, num_rels, num_nodes, True)

    model_name = "{}-{}-{}-ly{}-dilate{}-his{}-weight:{}-discount:{}-angle:{}-dp{}|{}|{}|{}-gpu{}"\
        .format(args.dataset, args.encoder, args.decoder, args.n_layers, args.dilate_len, args.train_history_len, args.weight, args.discount, args.angle,
                args.dropout, args.input_dropout, args.hidden_dropout, args.feat_dropout, args.gpu)
    root = Path(__file__).resolve().parents[1]
    model_state_file = str(args.run_dir / "model.pt" if args.run_dir else root / "models" / model_name)
    print("Sanity Check: stat name : {}".format(model_state_file))
    print("Sanity Check: Is cuda available ? {}".format(torch.cuda.is_available()))
    
    diffu_model_state_file = str(args.run_dir / "diffusion.pt" if args.run_dir else root / "diffu_models" / model_name)
    if not args.test:
        Path(model_state_file).parent.mkdir(parents=True, exist_ok=True)
        Path(diffu_model_state_file).parent.mkdir(parents=True, exist_ok=True)
    print("Sanity Check: diffu stat name : {}".format(diffu_model_state_file))

    use_cuda = args.gpu >= 0 and torch.cuda.is_available()

    if args.add_static_graph:
        static_triples = np.array(_read_triplets_as_list(
            str(args.data_root / args.dataset / "e-w-graph.txt"), {}, {}, load_time=False))
        num_static_rels = len(np.unique(static_triples[:, 1]))
        num_words = len(np.unique(static_triples[:, 2]))
        static_triples[:, 2] = static_triples[:, 2] + num_nodes
        static_node_id = torch.from_numpy(np.arange(num_words + data.num_nodes)).view(-1, 1).long().cuda(args.gpu) \
            if use_cuda else torch.from_numpy(np.arange(num_words + data.num_nodes)).view(-1, 1).long()
    else:
        num_static_rels, num_words, static_triples, static_graph,static_graph_diffu = 0, 0, [], None,None
        

    # diffusion
    diffu_rec = create_model_diffu(args)
    diffu_rec_re = create_model_diffu(args)

    # initial_h = Initial_h(args, num_words, num_nodes, num_static_rels, args.layer_norm)
    # initial_h_re = Initial_h(args, num_words, num_nodes, num_static_rels, args.layer_norm)

    diffu = Att_Diffuse_model(diffu_rec, args, args.encoder,
                              num_nodes,
                              num_rels,
                              #   initial_h,
                              dropout=args.dropout,
                              max_time=len(history_times),
                              num_words=num_words,
                              num_static_rels=num_static_rels,
                              num_bases=args.n_bases)

    model_diffpre = Att_Diffuse_model(diffu_rec_re, args, args.encoder,
                                      num_nodes,
                                      num_rels,
                                      #   initial_h_re,
                                      dropout=args.dropout,
                                      max_time=len(history_times),
                                      num_words=num_words,
                                      num_static_rels=num_static_rels,
                                      num_bases=args.n_bases)

    # create stat
    model = RecurrentRGCN(args.decoder,
                          args.encoder,
                          num_nodes,
                          num_rels,
                          num_static_rels,
                          num_words,
                          args.n_hidden,
                          args.attn_heads,
                          args.opn,
                          #   initial_h,
                          sequence_len=args.train_history_len,
                          num_bases=args.n_bases,
                          num_basis=args.n_basis,
                          num_hidden_layers=args.n_layers,
                          dropout=args.dropout,
                          self_loop=args.self_loop,
                          skip_connect=args.skip_connect,
                          layer_norm=args.layer_norm,
                          input_dropout=args.input_dropout,
                          hidden_dropout=args.hidden_dropout,
                          feat_dropout=args.feat_dropout,
                          aggregation=args.aggregation,
                          weight=args.weight,
                          discount=args.discount,
                          angle=args.angle,
                          use_static=args.add_static_graph,
                          entity_prediction=args.entity_prediction,
                          relation_prediction=args.relation_prediction,
                          use_cuda=use_cuda,
                          gpu=args.gpu,
                          analysis=args.run_analysis,
                          args=args)

    model_repre = RecurrentRGCN(args.decoder,
                                args.encoder,
                                num_nodes,
                                num_rels,
                                num_static_rels,
                                num_words,
                                args.n_hidden,
                                args.attn_heads,
                                args.opn,
                                # initial_h_re,
                                sequence_len=args.train_history_len,
                                num_bases=args.n_bases,
                                num_basis=args.n_basis,
                                num_hidden_layers=args.n_layers,
                                dropout=args.dropout,
                                self_loop=args.self_loop,
                                skip_connect=args.skip_connect,
                                layer_norm=args.layer_norm,
                                input_dropout=args.input_dropout,
                                hidden_dropout=args.hidden_dropout,
                                feat_dropout=args.feat_dropout,
                                aggregation=args.aggregation,
                                weight=args.weight,
                                discount=args.discount,
                                angle=args.angle,
                                use_static=args.add_static_graph,
                                entity_prediction=args.entity_prediction,
                                relation_prediction=args.relation_prediction,
                                use_cuda=use_cuda,
                                gpu=args.gpu,
                                analysis=args.run_analysis,
                                args=args)


    fitlog.add_hyper_in_file(__file__)
    if use_cuda:
        torch.cuda.set_device(args.gpu)
        model.cuda()
        diffu.cuda()
        model_diffpre.cuda()
        model_repre.cuda()
    # -------------------------------------------------------
    # Freeze feature extraction models
    # -------------------------------------------------------
    for p in model_repre.parameters():
        p.requires_grad_(False)

    for p in model_diffpre.parameters():
        p.requires_grad_(False)

    # Upstream keeps both feature generators in TRAIN mode, even at final TEST.
    # Freezing parameters/no_grad saves memory; eval() would also change dropout
    # and BatchNorm and is therefore a different baseline.
    model_repre.train()
    model_diffpre.train()

    fitlog.add_other(model, "model")
    fitlog.add_other(diffu, "diffu")
    if args.add_static_graph:
        static_graph = build_sub_graph(
            len(static_node_id), num_static_rels, static_triples, use_cuda, args.gpu)
        static_graph_diffu = build_sub_graph(
            len(static_node_id), num_static_rels, static_triples, use_cuda, args.gpu)

    # optimizer
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=1e-5)
    optimizer_diffu = torch.optim.Adam(
        diffu.parameters(), lr=args.lr, weight_decay=1e-5)
    per_epochs = (len(train_list)//args.accumulation_steps)
    accumulation_steps = args.accumulation_steps
    
    

    # scheduler = lr_scheduler.CosineAnnealingWarmRestarts(
    #     optimizer_diffu, T_0=50 * per_epochs, T_mult=2, eta_min=args.lr/100)

    first_stage = 1
    resumed_auxiliary = None
    if args.resume:
        first_stage, resumed_auxiliary = restore_training_state(
            args, model, diffu, model_repre, model_diffpre, optimizer, optimizer_diffu, run_device)

    if args.test:
        if not os.path.exists(model_state_file):
            raise FileNotFoundError(model_state_file)
        if args.diffuc and not os.path.exists(diffu_model_state_file):
            raise FileNotFoundError(diffu_model_state_file)

        # The training branch normally transfers this final diffusion state
        # into the feature generator before calling test().
        if args.diffuc:
            model_diffpre.load_state_dict(torch.load(diffu_model_state_file, map_location="cpu", weights_only=True))
        if args.run_dir:
            restore_evaluation_rng(args, run_device, len(train_list) - 1)

        final_idx = len(train_list) - 1
        input_list = train_list[max(0, final_idx - args.train_history_len):final_idx]
        last_history = [build_sub_graph(num_nodes, num_rels, snap, use_cuda, args.gpu)
                        for snap in input_list]
        device = torch.device("cuda:{}".format(args.gpu)) if use_cuda else torch.device("cpu")
        final_train = torch.from_numpy(train_list[final_idx]).long().to(device)
        last_output = [snap for snap in final_train]

        print("Final full-pipeline TEST")
        result = test(args, model, train_list, test_list,
                      num_rels, num_nodes, use_cuda, all_ans_list_test,
                      all_ans_list_r_test, model_state_file, static_graph,
                      "test", last_history, last_output, static_graph_diffu,
                      model_diffpre)
        mrr_raw, mrr_filter, mrr_raw_r, mrr_filter_r = result[:4]
        if args.final_validation:
            print("Final full-pipeline VALIDATION (supplementary; after TEST)")
            test(args, model, train_list, valid_list, num_rels, num_nodes, use_cuda,
                 all_ans_list_valid, all_ans_list_r_valid, model_state_file, static_graph,
                 "test", result[4], result[5], static_graph_diffu, model_diffpre)
        fitlog.finish()
    else:
        print("----------------------------------------start training----------------------------------------\n")
        best_mrr = 0
        # model_repre=None
        # model_diffpre=None

        # all_sequence = np.array(np.load(
        #     '/data/ChenWei/ZhangZhiyu/REGCN/RE-GCN-0829/data/{}/history_seq/h_r_seen_triple_all.npy'.format(args.dataset), allow_pickle=True)).tolist()

        for outer_epoch in range(args.n_epochs):
            model.train()
            losses = []
            losses_diff = []
            losses_e = []
            losses_r = []
            losses_static = []
            history_triple = []
            
            ranks_filter=[]
            ranks_filter_predict=[]
            
            MRRs=[]
            H1=[]
            H3=[]
            H10=[]

            last_history = None
            last_output = None

            if args.resume:
                previous = first_stage - 1
                last_history = [build_sub_graph(num_nodes, num_rels, snap, use_cuda, args.gpu)
                    for snap in train_list[max(0, previous - args.train_history_len):previous]]
                last_output = list(torch.as_tensor(train_list[previous], dtype=torch.long, device=run_device))
            idx = list(range(first_stage, len(train_list)))
            
            for batch_idx, train_sample_num in enumerate(tqdm(idx)):
                if train_sample_num == 0:
                    continue
                output = train_list[train_sample_num:train_sample_num+1]
                if train_sample_num - args.train_history_len < 0:
                    input_list = train_list[0: train_sample_num]
                else:
                    input_list = train_list[train_sample_num - args.train_history_len:
                                            train_sample_num]
                # history storage

                output_re = train_list[train_sample_num]

                # generate history graph
                history_glist = [build_sub_graph(
                    num_nodes, num_rels, snap, use_cuda, args.gpu) for snap in input_list]
                current_query_history_graph = None

                output = [torch.from_numpy(_).long().cuda() for _ in output] if use_cuda else [torch.from_numpy(_).long() for _ in output]
                output=output[0]
                
                with torch.no_grad():
                    output, diffu_rep, output_reply_triple = \
                        model_diffpre.gereration_feature(
                            args,
                            train_sample_num,
                            model_diffpre,
                            model_repre,
                            last_history,
                            last_output,
                            static_graph_diffu,
                            output_re,
                            output,
                            use_cuda,
                            train_list
                        )
                
                # best_mrr=0
                early_stopping = EarlyStopping(patience=5, verbose=False, path=model_state_file)
                early_stopping_dif = EarlyStopping(patience=5, verbose=False, path=diffu_model_state_file)
                if train_sample_num > 1:
                    # Author protocol: roll back weights only. Adam retains all
                    # updates, including those after the best validation epoch.
                    model.load_state_dict(torch.load(model_state_file, map_location="cpu", weights_only=True))
                if diffu_rep is not None:
                    model_output_o=diffu_rep[:,-1,:]
                else:
                    model_output_o=None    
                    
                for epoch in range(args.regcn_epochs):
                    model.train()
                    loss_e, loss_r, loss_static, emebding_entity,loss_reply = model.get_loss(
                        history_glist, output, static_graph, use_cuda, model_output=diffu_rep,model_output_x_t=model_output_o, tag=output_reply_triple, diffuc=args.diffuc, output_reply_triple=output_reply_triple, current_query_history_graph=None)
                    
                    loss = args.task_weight*loss_e+ \
                        (1-args.task_weight)*loss_r + loss_static

                    losses.append(loss.item())
                    losses_e.append(loss_e.item())
                    losses_r.append(loss_r.item())
                    losses_static.append(loss_static.item())

                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), args.grad_norm)  # clip gradients
                    optimizer.step()
                    optimizer.zero_grad()
                    
                    test_triples_input = torch.LongTensor(
                        valid_list[train_sample_num]).cuda() if use_cuda else torch.LongTensor(valid_list[train_sample_num])
                    test_triples_input = test_triples_input.to(run_device)
                    diffu_rep_test=None
                    output_reply_triple_test=None
                    if args.delete_feature_reply is False:
                        with torch.no_grad():
                            _, diffu_rep_test, output_reply_triple_test = \
                                model_diffpre.gereration_feature(
                                    args,
                                    train_sample_num,
                                    model_diffpre,
                                    model_repre,
                                    last_history,
                                    last_output,
                                    static_graph_diffu,
                                    valid_list[train_sample_num],
                                    test_triples_input,
                                    use_cuda,
                                    train_list
                            )
                    mrr, hits, _ = test_single(
                        args, model, history_glist, test_triples_input,
                        train_sample_num, num_rels, num_nodes, use_cuda,
                        all_ans_list_valid, static_graph,
                        diffu_rep_test, model_output_o,
                        output_reply_triple_test
                    )
                    
                    early_stopping(mrr, model)
                    if early_stopping.early_stop:
                        break
                    
                          
                
                # Upstream executes this diagnostic before final diffusion training.
                # It consumes RNG / generator buffers; removing it changes training.
                # Its TEST metric is printed only, never used for early stopping.
                if train_sample_num == idx[-1]:
                    diffu_rep_test = output_reply_triple_test = None
                    test_triples_input = torch.as_tensor(test_list[train_sample_num], dtype=torch.long, device=run_device)
                    if args.delete_feature_reply is False:
                        with torch.no_grad():
                            _, diffu_rep_test, output_reply_triple_test = model_diffpre.gereration_feature(
                                args, train_sample_num, model_diffpre, model_repre,
                                last_history, last_output, static_graph_diffu,
                                test_list[train_sample_num], test_triples_input, use_cuda, train_list)
                    mrr, hits, _ = test_single(
                        args, model, history_glist, test_triples_input, train_sample_num,
                        num_rels, num_nodes, use_cuda, all_ans_list_test, static_graph,
                        diffu_rep_test, model_output_o, output_reply_triple_test)
                    print("Upstream last-stage TEST diagnostic (before diffusion):",
                          mrr.item(), *(hit.item() for hit in hits))

                # diffu
                if args.diffuc:

                    sequence = None
                    true_triples = getReverseSample(output, num_rels)

                    emebding_entity = emebding_entity.detach()
                    # best_loss=10000
                    mrr_diff=0
                    if train_sample_num > 1:
                        diffu.load_state_dict(torch.load(diffu_model_state_file, map_location="cpu", weights_only=True))
                    for epoch in range(args.diffu_epochs):
                        diffu.train()
                        scores, diffu_rep_1, weights, t, mask_seq,emb_ent,noise,query_object3,_ = diffu(sequence,true_triples,args, True, use_cuda,static_graph_diffu,train_sample_num,model_output=diffu_rep, targets=output_reply_triple)
                        loss = diffu.loss_diffu_ce(diffu_rep_1, true_triples,query_object3,true_triples,mask_seq,emb_ent)
                        # loss = loss_diffu_value / accumulation_steps  # 将损失除以累积次数，这样使得每次累积的梯度相当于一个较大批次的梯度
                        losses_diff.append(loss.item())

                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(
                            diffu.parameters(), args.grad_norm)  # clip gradients
                        optimizer_diffu.step()
                        optimizer_diffu.zero_grad()  # 重置梯度
                        
                        #val
                        diffu.eval()
                        args.vl = True

                        valid_triples_input = (
                            torch.LongTensor(valid_list[train_sample_num]).cuda()
                            if use_cuda
                            else torch.LongTensor(valid_list[train_sample_num])
                        )

                        valid_triples_input = valid_triples_input.to(run_device)

                        true_triples_valid = getReverseSample(
                            valid_triples_input,
                            num_rels
                        )

                        with torch.no_grad():

                            diffu_reps1 = []

                            scores, diffu_rep1, weights, t, _, ent_emb, noise, _, _ = \
                                diffu(
                                    sequence,
                                    true_triples_valid,
                                    args,
                                    False,
                                    use_cuda,
                                    static_graph_diffu
                                )

                            diffu_reps1.append(diffu_rep1)

                            scores_rec_diffu = diffu.diffu_rep_pre(
                                diffu_reps1,
                                ent_emb
                            )

                            mrr_filter_snap, mrr_snap, rank_raw, rank_filter = \
                                utils.get_total_rank(
                                    true_triples_valid,
                                    scores_rec_diffu,
                                    all_ans_list_valid[train_sample_num],
                                    eval_bz=1000,
                                    rel_predict=0
                                )

                            mrr, hits = utils.stat_ranks_record(
                                rank_filter,
                                "predict_filter_ent"
                            )

                        args.vl = False

                        early_stopping_dif(mrr, diffu)

                        if early_stopping_dif.early_stop:
                            break
                            
                
                last_history = [snap for snap in history_glist]
                last_output = [snap for snap in output]
                
                checkpoint = torch.load(model_state_file, map_location="cpu", weights_only=True)
                model_repre.load_state_dict(checkpoint)
                if args.diffuc:
                    checkpoint_diffu = torch.load(diffu_model_state_file, map_location="cpu", weights_only=True)
                    model_diffpre.load_state_dict(checkpoint_diffu)
                # The next stage starts from these BEST feature weights and the
                # LAST optimizer states, exactly as in the uninterrupted loop.
                if args.dcr_mode != 'original':
                    args._dcr_recorder.flush()
                save_training_state(args, train_sample_num, model_repre, model_diffpre,
                                    optimizer, optimizer_diffu, run_device, auxiliary=None)

        print("Final full-pipeline TEST")
        mrr_raw, mrr_filter, mrr_raw_r, mrr_filter_r, last_history, last_output,mrrs,h1s,h3s,h10s,m_,h1,h3,h10 = test(args,
                                                                                       model,
                                                                                       train_list,
                                                                                       test_list,
                                                                                       num_rels,
                                                                                       num_nodes,
                                                                                       use_cuda,
                                                                                       all_ans_list_test,
                                                                                       all_ans_list_r_test,
                                                                                       model_state_file,
                                                                                       static_graph,
                                                                                       mode="test",
                                                                                       last_history=last_history,
                                                                                       last_output=last_output,
                                                                                       static_graph_diffu=static_graph_diffu,
                                                                                       model_diffpre=model_diffpre)
        if args.final_validation:
            print("Final full-pipeline VALIDATION (supplementary; after TEST)")
            test(args, model, train_list, valid_list, num_rels, num_nodes, use_cuda,
                 all_ans_list_valid, all_ans_list_r_valid, model_state_file, static_graph,
                 "test", last_history, last_output, static_graph_diffu, model_diffpre)
        fitlog.finish()
        # pca(entity_old,entity_new,args.dataset,mrr_old,mrr_new)
    return mrr_raw, mrr_filter, mrr_raw_r, mrr_filter_r


if __name__ == '__main__':
    args = validate_args(build_arg_parser().parse_args())
    print(args)
    if args.grid_search:
        out_log = '{}.{}.gs'.format(
            args.dataset, args.encoder+"-"+args.decoder)
        o_f = open(out_log, 'w')
        print("** Grid Search **")
        o_f.write("** Grid Search **\n")
        hyperparameters = args.tune.split(',')

        if args.tune == '' or len(hyperparameters) < 1:
            print("No hyperparameter specified.")
            sys.exit(0)
        grid = hp_range[hyperparameters[0]]
        for hp in hyperparameters[1:]:
            grid = itertools.product(grid, hp_range[hp])
        hits_at_1s = {}
        hits_at_10s = {}
        mrrs = {}
        grid = list(grid)
        print('* {} hyperparameter combinations to try'.format(len(grid)))
        o_f.write('* {} hyperparameter combinations to try\n'.format(len(grid)))
        o_f.close()

        for i, grid_entry in enumerate(list(grid)):

            o_f = open(out_log, 'a')

            if not (type(grid_entry) is list or type(grid_entry) is list):
                grid_entry = [grid_entry]
            grid_entry = utils.flatten(grid_entry)
            print('* Hyperparameter Set {}:'.format(i))
            o_f.write('* Hyperparameter Set {}:\n'.format(i))
            signature = ''
            print(grid_entry)
            o_f.write("\t".join([str(_) for _ in grid_entry]) + "\n")
            # def run_experiment(args, n_hidden=None, n_layers=None, dropout=None, n_bases=None):
            mrr_raw, mrr_filtered, _, _ = run_experiment(
                copy.deepcopy(args), grid_entry[0], grid_entry[1], grid_entry[2], grid_entry[3])
            o_f.write(f"MRR (raw): {mrr_raw:.6f}; MRR (filtered): {mrr_filtered:.6f}\n")
            o_f.close()
    # single run
    else:
        run_experiment(args)
    sys.exit()
