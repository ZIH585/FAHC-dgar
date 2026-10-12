"""Original DGAR arguments and shared run configuration."""
import argparse
import math
from pathlib import Path
from src.qcdr import effective_candidate_alpha


def parse_bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in ("true", "1", "yes"):
        return True
    if value.lower() in ("false", "0", "no"):
        return False
    raise argparse.ArgumentTypeError("expected true or false")


def parse_fusion(value):
    return value if value in ("add", "Linear") else parse_bool(value)


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
    # parser.add_argument("--add-static-graph", type=parse_bool,  action='store_true',
    #                  help="use the info of static graph")
    parser.add_argument("--add-static-graph", type=parse_bool,   default=True,
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
    parser.add_argument("--self-loop", action='store_true', default=True,
                        help="perform layer normalization in every layer of gcn ")
    parser.add_argument("--layer-norm", action='store_true', default=True,
                        help="perform layer normalization in every layer of gcn ")
    parser.add_argument("--relation-prediction", action='store_true', default=True,
                        help="add relation prediction loss")
    parser.add_argument("--entity-prediction", action='store_true', default=True,
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
    parser.add_argument('--rescale_timesteps', type=parse_bool,
                        default=False, help='rescal timesteps')
    parser.add_argument("--pattern_noise_radio", type=float, default=1)
    parser.add_argument("--max_len", type=int, default=64)
    parser.add_argument("--temperature_object", type=float, default=0.5)
    parser.add_argument("--his_max_len", type=int, default=128)
    parser.add_argument('--vl',type=parse_bool, default=False)

    parser.add_argument('--history_sample', type=int, default=64,
                        help='rescal timesteps')
    
    parser.add_argument('--delete_score', type=parse_bool, default=False)
    parser.add_argument('--delete_his_prompt', type=parse_bool, default=False)
    parser.add_argument('--delete_temporal_attention', type=parse_bool, default=False)
    parser.add_argument('--delete_feature_reply', type=parse_bool, default=False)
    parser.add_argument('--mu', type=float, default=1.0)
    parser.add_argument('--mu_r', type=float, default=1.0)
    
    # parser.add_argument("--reply_batch_num", type=int, default=128)
    parser.add_argument("--heads", type=int, default=4,
                        help="perform layer normalization")
    parser.add_argument("--diffuc", type=parse_fusion, default="add",
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

    parser.add_argument("--data-root", type=Path, default=Path(__file__).resolve().parents[1] / "data")
    parser.add_argument("--run-dir", type=Path, default=None, help="Separate config, checkpoints and outputs for this run")
    parser.add_argument("--resume", type=Path, default=None, help="Resume a completed chronological stage from latest.pt")
    parser.add_argument("--final-validation", action="store_true", help="Also report final full-pipeline validation")
    parser.add_argument("--qcdr-mode", default="original",
                        choices=("original", "candidate-stopgrad", "candidate-scaled"),
                        help="Route only historical entity replay's candidate gradient")
    parser.add_argument("--qcdr-candidate-alpha", type=float, default=1.0,
                        help="Candidate gradient multiplier in candidate-scaled mode, in [0,1]; stopgrad uses 0")
    parser.add_argument("--qcdr-target", choices=("entity",), default="entity",
                        help="Only entity candidates are supported; relation prediction stays Original")
    return parser


def validate_args(args):
    effective_candidate_alpha(args.qcdr_mode, args.qcdr_candidate_alpha, args.qcdr_target)
    for name in ("n_epochs", "regcn_epochs", "diffu_epochs", "train_history_len", "test_history_len", "n_hidden", "hidden_size", "attn_heads", "diffusion_steps"):
        if getattr(args, name) < 1:
            raise ValueError(f"{name} must be positive")
    if args.diffusion_steps < 2:
        raise ValueError("diffusion_steps must be at least 2")
    if args.diffuc is True:
        args.diffuc = "add"
    if args.hidden_size != args.n_hidden or args.n_hidden % args.attn_heads:
        raise ValueError("hidden sizes must match and divide evenly into attention heads")
    if args.gpu < -1 or args.seed < 0 or args.reply_batch < 0 or args.reply_batch_num < 1:
        raise ValueError("invalid device, seed or replay size")
    if not 0 <= args.task_weight <= 1 or not math.isfinite(args.lr) or args.lr <= 0:
        raise ValueError("invalid task weight or learning rate")
    if args.diffuc and args.start_reply != 0:
        raise ValueError("start_reply must be 0: every chronological stage needs a diffusion checkpoint")
    if args.resume and (args.test or args.n_epochs != 1 or args.run_dir is None):
        raise ValueError("resume requires --run-dir, one chronological pass, and no --test")
    if args.run_dir is not None:
        args.run_dir = args.run_dir.resolve()
    args.data_root = args.data_root.resolve()
    return args
