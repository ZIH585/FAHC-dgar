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
import json
from functools import wraps
from datetime import datetime
from pathlib import Path
from rgcn import utils
from rgcn.utils import *
from src.rrgcn import RecurrentRGCN
from src.hyperparameter_range import hp_range
import torch.nn.modules.rnn
from collections import defaultdict
from rgcn.knowledge_graph import _read_triplets_as_list
from diffusion.model_21 import create_model_diffu, Att_Diffuse_model
from diffusion.fahc_selector import (load_hcp_candidates, materialize_replay,
                                     recent_candidate_indices,
                                     sample_candidate_indices,
                                     select_hcp)
from src.fahc import (probe_hcp_risks, replay_kd_loss,
                      teacher_replay_logits)
from src.fahc_config import (ALGORITHM_VERSION, experiment_signature, parse_bool, validate_args)
from src.fahc_runtime import cpu_optimizer_state, isolated_rng
from torch.optim import *

import copy
import math
    
    
def getReverseSample(test_triplets, num_rels):
    inverse_test_triplets = test_triplets[:, [2, 1, 0]]  # inverse triplets
    inverse_test_triplets[:, 1] = inverse_test_triplets[:, 1] + num_rels
    a_triples = torch.concat(
        (test_triplets, inverse_test_triplets))  # [2,n_triplets,3]
    return a_triples


def deterministic_evaluation(function):
    """Use a fixed evaluation RNG without consuming training randomness."""
    @wraps(function)
    def wrapped(args, model, *positional, **keywords):
        with isolated_rng(args.seed + 7000003, next(model.parameters()).device):
            return function(args, model, *positional, **keywords)
    return wrapped


@deterministic_evaluation
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
        input_list = his_list[max(0, time_idx - args.test_history_len):time_idx]
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
            with open("../data/{}/history_snap_v2/snap_{}.pkl".format(args.dataset, i), "rb") as history_file:
                related_history = pickle.load(history_file)
            for element in test_snap:
                if str(element[0]) in related_history.keys():
                    current_query_history_triples.update(related_history[str(element[0])])
        
        test_triples_input = torch.LongTensor(
            test_snap).cuda() if use_cuda else torch.LongTensor(test_snap)
        test_triples_input = test_triples_input.to(device)

        sequence = None
        output_reply_triple = torch.as_tensor(
            sorted(current_query_history_triples), dtype=torch.long,
            device=device).reshape(-1, 3)
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


@torch.no_grad()
def evaluate_history_only_snapshots(args, model, train_list, valid_list,
                                    all_ans_list_valid, num_nodes, num_rels,
                                    static_graph, use_cuda, current_time,
                                    past_samples):
    """Measure current and old validation MRR with the same history-only path.

    This diagnostic is separate from DGAR's diffusion-conditioned validation.
    A positive ``past_samples`` selects evenly spaced old snapshots; zero
    evaluates every old snapshot; a negative value evaluates only current.
    """
    past = [] if past_samples < 0 else list(range(1, current_time))
    if past_samples > 0 and len(past) > past_samples:
        positions = np.linspace(0, len(past) - 1, past_samples, dtype=int)
        past = [past[int(position)] for position in positions]
    indices = past + [current_time]
    device = next(model.parameters()).device
    results = {}
    was_training = model.training
    model.eval()
    try:
        for time_index in indices:
            if len(valid_list[time_index]) == 0:
                continue
            history = train_list[max(0, time_index - args.train_history_len):time_index]
            if not history:
                continue
            graphs = [build_sub_graph(num_nodes, num_rels, snap, use_cuda, args.gpu)
                      for snap in history]
            triples = torch.as_tensor(valid_list[time_index], dtype=torch.long,
                                      device=device)
            all_triples, scores, _ = model.predict(
                graphs, num_rels, static_graph, triples, None, None, None,
                use_cuda)
            filtered_mrr, _, _, _ = utils.get_total_rank(
                all_triples, scores, all_ans_list_valid[time_index],
                eval_bz=1000, rel_predict=0)
            results[time_index] = float(filtered_mrr)
    finally:
        model.train(was_training)
    return results


def run_experiment(args, n_hidden=None, n_layers=None, dropout=None, n_bases=None):
    # Apply grid overrides before validation and checkpoint naming.
    for name, value in (("n_hidden", n_hidden), ("n_layers", n_layers),
                        ("dropout", dropout), ("n_bases", n_bases)):
        if value is not None:
            setattr(args, name, value)
    validate_args(args)
    settings_hash = experiment_signature(args)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f") + "_" + settings_hash
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.random.default_generator.manual_seed(args.seed)
    use_cuda = args.gpu >= 0 and torch.cuda.is_available()
    run_device = torch.device("cuda:{}".format(args.gpu)) if use_cuda else torch.device("cpu")
    if use_cuda:
        torch.cuda.set_device(args.gpu)
        torch.cuda.manual_seed(args.seed)

    fahc_enabled = args.hcp_selection != "original" or args.fahc_kd_weight > 0
    diagnostics_enabled = (fahc_enabled or args.fahc_diagnostics or
                           getattr(args, "fahc_final_validation", False))
    if diagnostics_enabled and not args.test:
        log_path = (Path(args.fahc_log) if args.fahc_log else
                    Path(__file__).resolve().parents[1] / "logs" /
                    "fahc_{}_{}_{}_{}.jsonl".format(
                        args.dataset, args.hcp_selection, args.seed,
                        run_id))
        log_path.parent.mkdir(parents=True, exist_ok=True)
        print("FAHC log: {}".format(log_path))
        best_history_mrr = {}
        with log_path.open("a", encoding="utf-8") as log_file:
            log_file.write(json.dumps({
                "event": "run_config", "run_id": run_id,
                "algorithm_version": ALGORITHM_VERSION, "settings_hash": settings_hash,
                "settings": vars(args).copy(),
                "risk_context": "history_only_current_stage",
                "coverage_context": "full_candidate_hcp",
                "split_protocol": "chronological_merge_then_per_snapshot_80_10_10",
                "filter_protocol": "same_snapshot_evaluation_split_only",
                "history_cache": "existing_history_snap_v2",
            }) + "\n")
    # set_seed(args.seed)
    # load graph data
    print("loading graph data")
    data_list = []
    data = utils.load_data(args.dataset)
    train_list = utils.split_by_time(data.train)
    valid_list = utils.split_by_time(data.valid)
    test_list = utils.split_by_time(data.test)

    data_list = []
    data_list.extend(train_list)
    data_list.extend(valid_list)
    data_list.extend(test_list)
    train_list,valid_list,test_list = utils.spilt_data(data_list)
    data_history="../data/{}/history_snap_v2/".format(args.dataset)
    if not os.path.exists(data_history):
        get_history(train_list,data.num_rels,args.dataset)
    
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
    if fahc_enabled:
        model_name += "-fahc-{}-{}".format(args.hcp_selection, settings_hash)
    model_state_file = '../models/' + model_name
    print("Sanity Check: stat name : {}".format(model_state_file))
    print("Sanity Check: Is cuda available ? {}".format(torch.cuda.is_available()))
    
    diffu_model_state_file = '../diffu_models/' + model_name
    print("Sanity Check: diffu stat name : {}".format(diffu_model_state_file))

    if not args.test:
        Path(model_state_file).parent.mkdir(parents=True, exist_ok=True)
        Path(diffu_model_state_file).parent.mkdir(parents=True, exist_ok=True)

    if args.add_static_graph:
        static_triples = np.array(_read_triplets_as_list(
            "../data/" + args.dataset + "/e-w-graph.txt", {}, {}, load_time=False))
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

    model_repre.eval()
    model_diffpre.eval()

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

    if args.test:
        if not os.path.exists(model_state_file):
            raise FileNotFoundError(model_state_file)
        if args.diffuc and not os.path.exists(diffu_model_state_file):
            raise FileNotFoundError(diffu_model_state_file)

        # The training branch normally transfers this final diffusion state
        # into the feature generator before calling test().
        if args.diffuc:
            model_diffpre.load_state_dict(torch.load(diffu_model_state_file, map_location="cpu"))
        model_diffpre.eval()

        final_idx = len(train_list) - 1
        input_list = train_list[max(0, final_idx - args.train_history_len):final_idx]
        last_history = [build_sub_graph(num_nodes, num_rels, snap, use_cuda, args.gpu)
                        for snap in input_list]
        device = torch.device("cuda:{}".format(args.gpu)) if use_cuda else torch.device("cpu")
        final_train = torch.from_numpy(train_list[final_idx]).long().to(device)
        last_output = [snap for snap in final_train]

        result = test(args, model, train_list + valid_list, test_list,
                      num_rels, num_nodes, use_cuda, all_ans_list_test,
                      all_ans_list_r_test, model_state_file, static_graph,
                      "test", last_history, last_output, static_graph_diffu,
                      model_diffpre)
        mrr_raw, mrr_filter, mrr_raw_r, mrr_filter_r = result[:4]
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

            idx = [_ for _ in range(len(train_list))]
            
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

                # Begin each stage from the previous stage's best checkpoint.
                # model_repre is the frozen teacher used by diffusion and KD.
                if train_sample_num > 1:
                    checkpoint = torch.load(model_state_file, map_location="cpu")
                    model.load_state_dict(checkpoint)
                    model_repre.load_state_dict(checkpoint)
                    del checkpoint

                selected_hcp = None
                selection_start = time.perf_counter()
                selection_record = {
                    "event": "stage", "run_id": run_id,
                    "outer_epoch": outer_epoch,
                    "snapshot": train_sample_num,
                    "mode": args.hcp_selection,
                    "k": args.reply_batch,
                    "fact_budget": args.fahc_fact_budget,
                    "fact_budget_active": args.hcp_selection != "original",
                    "selected_indices": [],
                    "selected_nonempty_count": 0,
                    "selected_full_fact_count": 0,
                    "selected_fact_count": 0,
                    "replay_retained_fraction": None,
                    "probe_steps": 0,
                }
                if args.hcp_selection != "original" and train_sample_num > 1:
                    selector_rng = random.Random(
                        args.seed + 1000003 * train_sample_num +
                        9176 * outer_epoch)
                    if args.hcp_selection == "recent":
                        candidate_indices = recent_candidate_indices(
                            train_sample_num, args.reply_batch)
                    else:
                        candidate_indices = sample_candidate_indices(
                            train_sample_num, args.reply_batch,
                            args.fahc_candidate_multiplier, selector_rng)
                    full_candidates = load_hcp_candidates(
                        args.dataset, candidate_indices, output_re[:, 0])
                    score_seed = (args.seed + 1000003 * train_sample_num +
                                  9176 * outer_epoch)
                    # Risk and coverage describe the complete HCP. The replay
                    # budget is applied uniformly only after choosing slices.
                    candidates = full_candidates
                    risks = None
                    if args.hcp_selection in ("risk", "risk+coverage"):
                        probe_output = output
                        if len(output) > args.fahc_probe_facts:
                            positions = sorted(selector_rng.sample(
                                range(len(output)), args.fahc_probe_facts))
                            probe_output = output[positions]
                        selection_record["probe_fact_count"] = len(probe_output)
                        risks = probe_hcp_risks(
                            model, model_repre, candidates, probe_output,
                            history_glist, static_graph, use_cuda,
                            args.task_weight,
                            (args.fahc_probe_lr if args.fahc_probe_lr is not None
                             else args.lr),
                            args.grad_norm, args.fahc_probe_steps,
                            args.fahc_score_batch_size,
                            training_optimizer=optimizer)
                        selection_record["probe_steps"] = (
                            args.fahc_probe_steps if any(
                                candidate.triples for candidate in candidates) else 0)
                    selected_hcp = select_hcp(
                        candidates, args.reply_batch, risks=risks,
                        mode=args.hcp_selection, eta=args.fahc_eta,
                        time_bin_size=args.fahc_time_bin_size,
                        rng=selector_rng,
                        entity_weight=args.fahc_entity_weight,
                        relation_modulus=num_rels)
                    full_fact_count = selected_hcp.fact_count
                    greedy_gain = sum(selected_hcp.score_by_index.values())
                    selected_risk = (sum(risks[candidate.time_index]
                                         for candidate in selected_hcp.selected)
                                     if risks is not None else None)
                    selection_record.update({
                        "selected_full_fact_count": full_fact_count,
                        "selected_risk_sum": selected_risk,
                        "selected_weighted_coverage_gain_sum": (
                            greedy_gain - (selected_risk or 0.0)
                            if args.hcp_selection in ("coverage", "risk+coverage")
                            else None),
                        "risk_share_of_greedy_gain": (
                            selected_risk / greedy_gain
                            if args.hcp_selection == "risk+coverage" and greedy_gain > 0
                            else None),
                    })
                    selected_hcp = materialize_replay(
                        selected_hcp, full_candidates,
                        args.fahc_fact_budget, score_seed)
                    selection_record.update({
                        "replay_retained_fraction": (
                            selected_hcp.fact_count / full_fact_count
                            if full_fact_count else None),
                        "candidate_indices": candidate_indices,
                        "candidate_fact_counts": {
                            str(candidate.time_index): len(candidate.triples)
                            for candidate in full_candidates},
                        "scored_fact_counts": {
                            str(candidate.time_index): (
                                len(candidate.triples) if risks is not None else 0)
                            for candidate in candidates},
                        "candidate_inverse_fact_counts": {
                            str(candidate.time_index): sum(
                                triple[1] >= num_rels for triple in candidate.triples)
                            for candidate in candidates},
                        "risks": risks or {},
                        "selected_indices": selected_hcp.indices,
                        "selected_nonempty_count": selected_hcp.nonempty_count,
                        "selected_fact_count": selected_hcp.fact_count,
                        "greedy_marginal_scores": selected_hcp.score_by_index,
                        "selected_full_relation_count": len({
                            triple[1] % num_rels for candidate in selected_hcp.selected
                            for triple in candidate.triples}),
                        "selected_full_time_bin_count": len({
                            candidate.time_index // args.fahc_time_bin_size
                            for candidate in selected_hcp.selected if candidate.triples}),
                        "replay_relation_count": len({
                            triple[1] % num_rels for triple in selected_hcp.triples}),
                        "replay_entity_count": len({
                            entity for s, _, o in selected_hcp.triples for entity in (s, o)}),
                    })
                selection_record["selection_seconds"] = time.perf_counter() - selection_start
                
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
                            train_list,
                            hcp_triples=(selected_hcp.triples
                                         if selected_hcp is not None else None)
                        )
                
                # best_mrr=0
                early_stopping = EarlyStopping(patience=5, verbose=False, path=model_state_file)
                early_stopping_dif = EarlyStopping(patience=5, verbose=False, path=diffu_model_state_file)
                best_optimizer_state = None
                best_diffu_optimizer_state = None
                if diffu_rep is not None:
                    model_output_o=diffu_rep[:,-1,:]
                else:
                    model_output_o=None    

                teacher_chunks = None
                if (args.fahc_kd_weight > 0 and train_sample_num > 1
                        and output_reply_triple is not None
                        and len(output_reply_triple) > 0):
                    teacher_chunks = teacher_replay_logits(
                        model_repre, output_reply_triple, history_glist,
                        static_graph, use_cuda, args.fahc_score_batch_size)
                    offset = active_count = 0
                    for teacher in teacher_chunks:
                        confidence, predictions = teacher.float().softmax(dim=-1).max(dim=-1)
                        labels = output_reply_triple[offset:offset + len(teacher), 2].cpu()
                        active_count += int(((predictions == labels) &
                                             (confidence >= args.fahc_kd_confidence)).sum())
                        offset += len(teacher)
                    selection_record["kd_teacher_active_fact_count"] = active_count
                kd_losses = []

                # The feature teachers do not change within the RGCN loop.
                # Reuse one seeded validation prompt/noise realization at every
                # epoch and restore RNG so validation cannot alter training.
                test_triples_input = torch.as_tensor(
                    valid_list[train_sample_num], dtype=torch.long, device=run_device)
                diffu_rep_test = output_reply_triple_test = None
                if not args.delete_feature_reply:
                    with isolated_rng(args.seed + 3000017 * train_sample_num, run_device), torch.no_grad():
                        _, diffu_rep_test, output_reply_triple_test = model_diffpre.gereration_feature(
                            args, train_sample_num, model_diffpre, model_repre,
                            last_history, last_output, static_graph_diffu,
                            valid_list[train_sample_num], test_triples_input,
                            use_cuda, train_list)
                    
                for epoch in range(args.regcn_epochs):
                    model.train()
                    loss_e, loss_r, loss_static, emebding_entity,loss_reply = model.get_loss(
                        history_glist, output, static_graph, use_cuda, model_output=diffu_rep,model_output_x_t=model_output_o, tag=output_reply_triple, diffuc=args.diffuc, output_reply_triple=output_reply_triple, current_query_history_graph=None)
                    
                    loss = args.task_weight*loss_e+ \
                        (1-args.task_weight)*loss_r + loss_static

                    if teacher_chunks is not None:
                        kd_loss = replay_kd_loss(
                            model, teacher_chunks, output_reply_triple,
                            history_glist, static_graph, use_cuda,
                            args.fahc_kd_temperature,
                            args.fahc_kd_confidence)
                        loss = loss + args.fahc_kd_weight * kd_loss
                        kd_losses.append(float(kd_loss.detach()))

                    losses.append(loss.item())
                    losses_e.append(loss_e.item())
                    losses_r.append(loss_r.item())
                    losses_static.append(loss_static.item())

                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), args.grad_norm)  # clip gradients
                    optimizer.step()
                    optimizer.zero_grad()
                    
                    mrr, hits, _ = test_single(
                        args, model, history_glist, test_triples_input,
                        train_sample_num, num_rels, num_nodes, use_cuda,
                        all_ans_list_valid, static_graph,
                        diffu_rep_test, model_output_o,
                        output_reply_triple_test
                    )
                    
                    previous_best = early_stopping.best_score
                    if not torch.isfinite(mrr):
                        raise ValueError("non-finite RGCN validation MRR")
                    early_stopping(mrr,model)
                    if previous_best is None or mrr >= previous_best + early_stopping.delta:
                        best_optimizer_state = cpu_optimizer_state(optimizer)
                    if early_stopping.early_stop:
                        break
                    
                          
                
                if train_sample_num==idx[-1]:
                    diffu_rep_test=None
                    output_reply_triple_test=None
                    test_triples_input = torch.LongTensor(
                        test_list[train_sample_num]).cuda() if use_cuda else torch.LongTensor(test_list[train_sample_num])
                    test_triples_input = test_triples_input.to(run_device)
                    if args.delete_feature_reply is False:
                        with isolated_rng(args.seed + 5000011 * train_sample_num, run_device), torch.no_grad():
                            _, diffu_rep_test, output_reply_triple_test = \
                                model_diffpre.gereration_feature(
                                    args,
                                    train_sample_num,
                                    model_diffpre,
                                    model_repre,
                                    last_history,
                                    last_output,
                                    static_graph_diffu,
                                    test_list[train_sample_num],
                                    test_triples_input,
                                    use_cuda,
                                    train_list
                                )
                    mrr, hits, rank_filter = test_single(args,model, history_glist, test_triples_input, train_sample_num, num_rels, num_nodes, use_cuda, all_ans_list_test, static_graph, diffu_rep_test,model_output_o,output_reply_triple_test)
                    print("the lastet time metric",mrr.item(),hits[0].item(),hits[1].item(),hits[2].item())     
                    
                # diffu
                # Diffusion training belongs to each snapshot. start_reply is
                # checked against configured RGCN epochs before starting a run;
                # an early stop must not leave a missing/stale diffusion file.
                if args.diffuc:

                    sequence = None
                    true_triples = getReverseSample(output, num_rels)

                    emebding_entity = emebding_entity.detach()
                    # best_loss=10000
                    mrr_diff=0
                    if train_sample_num>1:
                        checkpoint = torch.load(
                            diffu_model_state_file,
                            map_location="cpu"
                        )
                        diffu.load_state_dict(checkpoint)
                        del checkpoint
                    
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

                        with isolated_rng(args.seed + 4000037 * train_sample_num, run_device), torch.no_grad():

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

                        previous_best = early_stopping_dif.best_score
                        if not torch.isfinite(mrr):
                            raise ValueError("non-finite diffusion validation MRR")
                        early_stopping_dif(mrr, diffu)
                        if previous_best is None or mrr >= previous_best + early_stopping_dif.delta:
                            best_diffu_optimizer_state = cpu_optimizer_state(optimizer_diffu)

                        if early_stopping_dif.early_stop:
                            break
                            
                
                last_history = [snap for snap in history_glist]
                last_output = [snap for snap in output]
                
                checkpoint = torch.load(
                    model_state_file,
                    map_location="cpu"
                )
                model.load_state_dict(checkpoint)
                model_repre.load_state_dict(checkpoint)
                optimizer.load_state_dict(best_optimizer_state)
                if args.diffuc:
                    checkpoint_diffu = torch.load(diffu_model_state_file, map_location="cpu")
                    diffu.load_state_dict(checkpoint_diffu)
                    model_diffpre.load_state_dict(checkpoint_diffu)
                    optimizer_diffu.load_state_dict(best_diffu_optimizer_state)
                    del checkpoint_diffu

                del checkpoint

                if diagnostics_enabled:
                    if args.hcp_selection == "original" and output_reply_triple is not None:
                        selection_record["selected_fact_count"] = len(output_reply_triple)
                        selection_record["selected_full_fact_count"] = len(output_reply_triple)
                        selection_record["replay_retained_fraction"] = (
                            1.0 if len(output_reply_triple) else None)
                    selection_record["kd_weight"] = args.fahc_kd_weight
                    selection_record["kd_mean_loss"] = (
                        float(np.mean(kd_losses)) if kd_losses else None)
                    if args.fahc_eval_every > 0:
                        evaluate_past = (
                            train_sample_num % args.fahc_eval_every == 0 or
                            train_sample_num == idx[-1])
                        history_mrr = evaluate_history_only_snapshots(
                            args, model, train_list, valid_list,
                            all_ans_list_valid, num_nodes, num_rels,
                            static_graph, use_cuda, train_sample_num,
                            (args.fahc_past_eval_samples if evaluate_past
                             else -1))
                        past_mrr = {i: value for i, value in history_mrr.items()
                                    if i < train_sample_num}
                        forgetting = [max(best_history_mrr.get(i, value) - value, 0)
                                      for i, value in past_mrr.items()]
                        for i, value in history_mrr.items():
                            best_history_mrr[i] = max(
                                best_history_mrr.get(i, value), value)
                        selection_record["history_only_current_filtered_mrr"] = (
                            history_mrr.get(train_sample_num))
                        selection_record["history_only_past_filtered_mrr"] = past_mrr
                        selection_record["history_only_past_macro_mrr"] = (
                            float(np.mean(list(past_mrr.values())))
                            if past_mrr else None)
                        selection_record["history_only_sampled_forgetting"] = (
                            float(np.mean(forgetting)) if forgetting else None)
                    with log_path.open("a", encoding="utf-8") as log_file:
                        log_file.write(json.dumps(selection_record) + "\n")

        final_training_history, final_training_output = last_history, last_output
        print("\nFinal full-pipeline TEST\n")
        mrr_raw, mrr_filter, mrr_raw_r, mrr_filter_r, last_history, last_output,mrrs,h1s,h3s,h10s,m_,h1,h3,h10 = test(args,
                                                                                       model,
                                                                                       train_list+valid_list,
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
        if getattr(args, "fahc_final_validation", False):
            with log_path.open("a", encoding="utf-8") as log_file:
                log_file.write(json.dumps({
                    "event": "final_test", "run_id": run_id,
                    "filtered_entity_mrr": float(mrr_filter),
                    "past_macro_filtered_entity_mrr": (
                        float(np.mean(mrrs[:-1])) if len(mrrs) > 1 else None),
                    "current_filtered_entity_mrr": (
                        float(mrrs[-1]) if mrrs else None),
                    "per_snapshot_filtered_entity_mrr": {
                        index: float(value) for index, value in
                        enumerate(mrrs, start=1)},
                }) + "\n")
            print("\nFinal full-pipeline VALIDATION (configuration selection)\n")
            validation = test(
                args, model, train_list + valid_list, valid_list,
                num_rels, num_nodes, use_cuda, all_ans_list_valid,
                all_ans_list_r_valid, model_state_file, static_graph,
                mode="valid", last_history=final_training_history,
                last_output=final_training_output,
                static_graph_diffu=static_graph_diffu,
                model_diffpre=model_diffpre)
            per_snapshot = validation[6]
            with log_path.open("a", encoding="utf-8") as log_file:
                log_file.write(json.dumps({
                    "event": "final_validation", "run_id": run_id,
                    "filtered_entity_mrr": float(validation[1]),
                    "past_macro_filtered_entity_mrr": (
                        float(np.mean(per_snapshot[:-1]))
                        if len(per_snapshot) > 1 else None),
                    "current_filtered_entity_mrr": (
                        float(per_snapshot[-1]) if per_snapshot else None),
                    "per_snapshot_filtered_entity_mrr": {
                        index: float(value) for index, value in
                        enumerate(per_snapshot, start=1)},
                }) + "\n")
        fitlog.finish()
        # pca(entity_old,entity_new,args.dataset,mrr_old,mrr_new)
    return mrr_raw, mrr_filter, mrr_raw_r, mrr_filter_r


def build_arg_parser():
    parser = argparse.ArgumentParser(description='REGCN')

    parser.add_argument("--gpu", type=int, default=2,
                        help="gpu")
    parser.add_argument("--batch-size", type=int, default=1,
                        help="batch-size")
    parser.add_argument("--seed", type=int, default=1,
                        help="random seed")
    parser.add_argument("-d", "--dataset", type=str,  default="ICEWS14s",
                        help="dataset to use")
    parser.add_argument("--test", action='store_true', default=False,
                        help="load stat from dir and directly test")
    parser.add_argument("--run-analysis", action='store_true', default=False,
                        help="print log info")
    parser.add_argument("--run-statistic", action='store_true', default=False,
                        help="statistic the result")
    parser.add_argument("--multi-step", action='store_true', default=False,
                        help="do multi-steps inference without ground truth")
    parser.add_argument("--topk", type=int, default=10,
                        help="choose top k entities as results when do multi-steps without ground truth")
    # parser.add_argument("--add-static-graph",  action='store_true',
    #                  help="use the info of static graph")
    parser.add_argument("--add-static-graph", type=parse_bool, nargs="?", const=True, default=True,
                       help="use the info of static graph")
    parser.add_argument("--add-rel-word", action='store_true', default=False,
                        help="use words in relaitons")
    parser.add_argument("--relation-evaluation", action='store_true', default=False,
                        help="save model accordding to the relation evalution")

    # configuration for encoder RGCN stat
    parser.add_argument("--weight", type=float, default=0.5,
                        help="weight of static constraint")
    parser.add_argument("--task-weight", type=float, default=0.7,
                        help="weight of entity prediction task")
    parser.add_argument("--discount", type=float, default=1,
                        help="discount of weight of static constraint")
    parser.add_argument("--angle", type=int, default=10,
                        help="evolution speed")

    parser.add_argument("--encoder", type=str, default="uvrgcn",
                        help="method of encoder")
    parser.add_argument("--aggregation", type=str, default="none",
                        help="method of aggregation")
    parser.add_argument("--dropout", type=float, default=0.2,
                        help="dropout probability")
    parser.add_argument("--skip-connect", action='store_true', default=False,
                        help="whether to use skip connect in a RGCN Unit")
    parser.add_argument("--n-hidden", type=int, default=200,
                        help="number of hidden units")
    parser.add_argument("--opn", type=str, default="sub",
                        help="opn of compgcn")

    parser.add_argument("--n-bases", type=int, default=100,
                        help="number of weight blocks for each relation")
    parser.add_argument("--n-basis", type=int, default=100,
                        help="number of basis vector for compgcn")
    parser.add_argument("--n-layers", type=int, default=2,
                        help="number of propagation rounds")
    parser.add_argument("--self-loop", type=parse_bool, nargs="?", const=True, default=True,
                        help="perform layer normalization in every layer of gcn ")
    parser.add_argument("--layer-norm", type=parse_bool, nargs="?", const=True, default=True,
                        help="perform layer normalization in every layer of gcn ")
    parser.add_argument("--relation-prediction", type=parse_bool, nargs="?", const=True, default=True,
                        help="add relation prediction loss")
    parser.add_argument("--entity-prediction", type=parse_bool, nargs="?", const=True, default=True,
                        help="add entity prediction loss")
    parser.add_argument("--split_by_relation", action='store_true', default=False,
                        help="do relation prediction")

    # configuration for stat training
    parser.add_argument("--n-epochs", type=int, default=1,
                        help="number of minimum training epochs on each time step")
    parser.add_argument("--regcn_epochs", type=int, default=10,
                    help="number of minimum training epochs on each time step")#5
    parser.add_argument("--diffu_epochs", type=int, default=10,
                    help="number of minimum training epochs on each time step")
    parser.add_argument("--lr", type=float, default=0.001,
                        help="learning rate")
    parser.add_argument("--grad-norm", type=float, default=1.0,
                        help="norm to clip gradient to")
    parser.add_argument("--reply_batch_num", type=int, default=128)

    # configuration for evaluating
    parser.add_argument("--evaluate-every", type=int, default=1,
                        help="perform evaluation every n epochs")

    # configuration for decoder
    parser.add_argument("--decoder", type=str, default="convtranse",
                        help="method of decoder")
    parser.add_argument("--input-dropout", type=float, default=0.2,
                        help="input dropout for decoder ")
    parser.add_argument("--hidden-dropout", type=float, default=0.2,
                        help="hidden dropout for decoder")
    parser.add_argument("--feat-dropout", type=float, default=0.2,
                        help="feat dropout for decoder")

    # configuration for sequences stat
    parser.add_argument("--train-history-len", type=int, default=3,
                        help="history length")
    parser.add_argument("--test-history-len", type=int, default=3,
                        help="history length for test")
    parser.add_argument("--dilate-len", type=int, default=1,
                        help="dilate history graph")
    parser.add_argument("--reply_batch", type=int, default=35,
                help="number of reply_batch")
    parser.add_argument("--hcp-selection", default="original",
                        choices=("original", "random", "recent", "risk",
                                 "coverage", "risk+coverage"),
                        help="HCP selection mode; original keeps DGAR sampling")
    parser.add_argument("--fahc-candidate-multiplier", type=float, default=2.0,
                        help="historical candidate pool size divided by k")
    parser.add_argument("--fahc-fact-budget", type=int, default=128,
                        help="maximum unique HCP facts for controlled modes")
    parser.add_argument("--fahc-eta", type=float, default=1.0,
                        help="weight of diminishing-return coverage")
    parser.add_argument("--fahc-entity-weight", type=float, default=1.0,
                        help="entity coverage weight; set 0 for relation/time only")
    parser.add_argument("--fahc-time-bin-size", type=int, default=10,
                        help="historical snapshots per time coverage bin")
    parser.add_argument("--fahc-probe-steps", type=int, default=1,
                        help="current-only probe updates before HCP scoring")
    parser.add_argument("--fahc-probe-facts", type=int, default=128,
                        help="maximum current facts in each probe update")
    parser.add_argument("--fahc-probe-lr", type=float, default=None,
                        help="probe learning rate; defaults to --lr")
    parser.add_argument("--fahc-score-batch-size", type=int, default=256,
                        help="HCP scoring and KD decoder chunk size")
    parser.add_argument("--fahc-kd-weight", type=float, default=0.0,
                        help="lambda for gated old-model entity KL")
    parser.add_argument("--fahc-kd-temperature", type=float, default=2.0,
                        help="KD softmax temperature")
    parser.add_argument("--fahc-kd-confidence", type=float, default=0.5,
                        help="minimum correct teacher confidence at T=1")
    parser.add_argument("--fahc-eval-every", type=int, default=1,
                        help="stage interval for history-only current/past validation; 0 disables")
    parser.add_argument("--fahc-past-eval-samples", type=int, default=4,
                        help="old snapshots per evaluation; 0 evaluates all")
    parser.add_argument("--fahc-log", type=str, default=None,
                        help="JSONL path for HCP choices, risks, KD, and MRR")
    parser.add_argument("--fahc-diagnostics", action="store_true",
                        help="also record the same stage diagnostics for original DGAR")
    parser.add_argument("--fahc-final-validation", action="store_true",
                        help="evaluate all validation snapshots with final DGAR inference for configuration selection")

    # configuration for optimal parameters
    parser.add_argument("--grid-search", action='store_true', default=False,
                        help="perform grid search for best configuration")
    parser.add_argument("-tune", "--tune", type=str, default="n_hidden,n_layers,dropout,n_bases",
                        help="stat to use")
    parser.add_argument("--num-k", type=int, default=500,
                        help="number of triples generated")

    
    # diffusion
    parser.add_argument("--diffu", default=True, type=parse_bool,
                        help="do or don't reply")
    parser.add_argument("--hidden_size", default=200,
                        type=int, help="hidden size of model")
    parser.add_argument('--emb_dropout', type=float,
                        default=0.2, help='Dropout of embedding')
    parser.add_argument("--hidden_act", default="gelu", type=str)  # gelu relu
    parser.add_argument('--num_blocks', type=int, default=2,
                        help='Number of Transformer blocks')
    parser.add_argument('--num_blocks_cross', type=int,
                        default=0, help='Number of Transformer blocks')
    parser.add_argument('--attn_heads', type=int, default=2,
                        help='Number of Transformer blocks')

    parser.add_argument('--schedule_sampler_name', type=str,
                        default='lossaware', help='Diffusion for t generation')
    parser.add_argument('--diffusion_steps', type=int,
                        default=35, help='Diffusion step')
    parser.add_argument("--k_step", type=int, default=0,
                        help="number of propagation rounds")
    parser.add_argument('--lambda_uncertainty', type=float,
                        default=0.01, help='uncertainty weight')
    # cosine, linear, trunc_cos, trunc_lin, pw_lin, sqrt
    parser.add_argument('--noise_schedule',
                        default='linear', help='Beta generation')
    parser.add_argument('--rescale_timesteps',
                        type=parse_bool, default=False, help='rescal timesteps')
    parser.add_argument("--pattern_noise_radio", type=float, default=1)
    parser.add_argument("--max_len", type=int, default=64)
    parser.add_argument("--temperature_object", type=float, default=0.5)
    parser.add_argument("--his_max_len", type=int, default=128)
    parser.add_argument('--vl',type=parse_bool, default=False)

    parser.add_argument('--history_sample', default=64,
                        type=int, help='rescal timesteps')
    
    parser.add_argument('--delete_score', type=parse_bool, default=False)
    parser.add_argument('--delete_his_prompt', type=parse_bool, default=False)
    parser.add_argument('--delete_temporal_attention', type=parse_bool, default=False)
    parser.add_argument('--delete_feature_reply', type=parse_bool, default=False)
    parser.add_argument('--mu', type=float, default=1.0)
    parser.add_argument('--mu_r', type=float, default=1.0)
    
    # parser.add_argument("--reply_batch_num", type=int, default=128)
    parser.add_argument("--heads", type=int, default=4,
                        help="perform layer normalization")
    parser.add_argument("--diffuc",  default="add",
                        type=lambda value: value if value in ("add", "Linear") else parse_bool(value),
                        help="perform layer normalization")
    parser.add_argument('--classifier_scale', type=float,
                        default=1, help='梯度权重')
    parser.add_argument('--grad_epoch', type=int,
                    default=3, help='梯度循环')
    parser.add_argument("--start_reply", type=int,
                        default=0, help="从第几个epoch开始回放")
    parser.add_argument('--bordline', type=int, default=0, help='大于多少开始回放')

    parser.add_argument('--scale', type=float, default=50, help='scale weight')
    parser.add_argument('--beta_start', type=float,
                        default=0.0001, help='beta_start weight')
    parser.add_argument('--beta_end', type=float,
                        default=0.02, help='beta_end weight')

    parser.add_argument("--grad_norm", type=float,
                        default=1.0, help="norm to clip gradient to")
    parser.add_argument("--accumulation_steps", type=int,
                        default=2, help="norm to clip gradient to")

    return parser


if __name__ == '__main__':
    args = build_arg_parser().parse_args()
    print(args)
    fitlog.set_log_dir("logs/")         # 设定日志存储的目录
    fitlog.add_hyper(args)
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
            print("MRR (raw): {:.6f}; MRR (filtered): {:.6f}".format(mrr_raw, mrr_filtered))
            o_f.write("MRR (raw): {:.6f}; MRR (filtered): {:.6f}\n".format(mrr_raw, mrr_filtered))
            o_f.close()
    # single run
    else:
        run_experiment(args)
    sys.exit()
