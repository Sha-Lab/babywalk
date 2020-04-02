RESULT_DIR = 'follower/results/'
SNAPSHOT_DIR = 'follower/snapshots/'
PLOT_DIR = 'follower/plots/'
SUMMARY_DIR = 'follower/summary/'
FOLLOWER_PATH = None
SPEAKER_PATH = None
ACTION_EMBEDDING_SIZE = 2048 + 128
HIDDEN_SIZE = 512
WEIGHT_DECAY = 0.00005
FEATURE_SIZE = 2048 + 128
LOG_EVERY = 500
SAVE_EVERY = 5000

def add_general_args(parser):

    # data
    parser.add_argument("--split_postfix", type=str, default='')
    parser.add_argument("--add_augment", action='store_true')
    parser.add_argument("--augment_data", type=str, default='literal_speaker_data_augmentation')
    parser.add_argument("--task_name", type=str, default='R2R')

    # learning algorithm
    parser.add_argument("--reward", action='store_true')
    parser.add_argument("--curriculum_rl", action='store_true')
    parser.add_argument("--count_curriculum", type=int, default=0)
    parser.add_argument("--max_curriculum", type=int, default=4)
    parser.add_argument("--curriculum_iters", type=int, default=10000)
    parser.add_argument("--learning_method", type=str, default="adam")
    parser.add_argument("--feedback_method", type=str, default="sample")
    parser.add_argument("--il_mode", type=str, default=None)

    # learning params
    parser.add_argument("--n_iters", type=int, default=20000)
    parser.add_argument("--batch_size", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--max_ins_len", type=int, default=100)
    parser.add_argument("--max_steps", type=int, default=10)
    parser.add_argument("--beam_size", type=int, default=8)
    parser.add_argument("--action_embed_size", type=int, default=ACTION_EMBEDDING_SIZE)
    parser.add_argument("--feature_size", type=int, default=FEATURE_SIZE)
    parser.add_argument("--weight_decay", type=float, default=WEIGHT_DECAY)
    parser.add_argument("--hidden_size", type=int, default=HIDDEN_SIZE)

    # network
    parser.add_argument("--coground", action='store_true')
    parser.add_argument("--history", action='store_true')
    parser.add_argument("--wemb", type=int, default=300)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--reward_type", type=str, default='dis')
    parser.add_argument("--exp_forget", type=float, default=0.5)

    # load model
    parser.add_argument("--no_speaker", action='store_true')
    parser.add_argument("--load_opt", action='store_true')
    parser.add_argument("--speaker_prefix", type=str, default=SPEAKER_PATH)
    parser.add_argument("--follower_prefix", type=str, default=FOLLOWER_PATH)

    # save and log in training
    parser.add_argument("--no_save", action='store_true')
    parser.add_argument("--model_name", type=str, default="follower")
    parser.add_argument("--result_dir", default=RESULT_DIR)
    parser.add_argument("--snapshot_dir", default=SNAPSHOT_DIR)
    parser.add_argument("--plot_dir", default=PLOT_DIR)
    parser.add_argument("--summary_dir", default=SUMMARY_DIR)
    parser.add_argument("--log_every", type=int, default=LOG_EVERY)
    parser.add_argument("--save_every", type=int, default=SAVE_EVERY)
    
    # evaluation
    parser.add_argument("--use_test", action='store_true')
    parser.add_argument("--one_by_one", action='store_true')
    parser.add_argument("--one_by_one_mode", type=str, default=None)
