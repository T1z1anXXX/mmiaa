from sacred import Experiment

ex = Experiment("ViLT")


def _loss_names(d):
    ret = {
        "itm": 0,
        "mlm": 0,
        "mpp": 0,
        "ava": 0,
    }
    ret.update(d)
    return ret


@ex.config
def config():
    exp_name = "finetune_ava"
    seed = 0
    datasets = ["ava"]
    loss_names = _loss_names({"ava": 1})
    batch_size = 256

    # eval config (for bash execution)
    test_ratio = None
    test_type = None
    test_exp_name = None

    # fix backbone model (ViLT) weights
    fix_model = True

    # missing modality config
    missing_ratio = {'train': 0.7, 'val': 0.7, 'test': 0.7}
    missing_type = {'train': 'both', 'val': 'both', 'test': 'both'}  # ['text', 'image', 'both']
    both_ratio = 0.5  # missing both ratio
    missing_table_root = '/root/autodl-tmp/project/datasets/missing_tables'
    simulate_missing = False

    # missing_modal_prompts config
    prompt_length = 16
    learnt_p = True
    prompt_layers = [0, 1, 2, 3, 4, 5]

    # missing_modal_digests config
    digest_length = 1
    digest_layers = [0, 1, 2, 3, 4, 5]
    multi_layer_digest = True

    multi_layer_pandd = True

    # visual config
    visual = True

    # Image setting
    train_transform_keys = ["pixelbert"]
    val_transform_keys = ["pixelbert"]
    image_size = 384
    max_image_len = -1
    patch_size = 32
    draw_false_image = 0
    image_only = False

    # Text Setting
    vqav2_label_size = 3129
    max_text_len = 512
    tokenizer = "bert-base-uncased"
    vocab_size = 30522
    whole_word_masking = False
    mlm_prob = 0.15
    draw_false_text = 0

    # Transformer Setting
    vit = "vit_base_patch32_384"
    hidden_size = 768
    num_heads = 12
    num_layers = 12
    mlp_ratio = 4
    drop_rate = 0.1

    # Optimizer Setting
    optim_type = "adamw"
    learning_rate = 1e-2
    weight_decay = 2e-2
    decay_power = 1
    max_epoch = 15
    max_steps = None
    warmup_steps = 0.1
    end_lr = 0
    lr_mult = 1  # multiply lr for downstream heads

    # Downstream Setting
    get_recall_metric = False
    mmimdb_class_num = 23
    hatememes_class_num = 2
    food101_class_num = 101
    ava_class_num = 10
    pccd_class_num = 10

    # PL Trainer Setting
    resume_from = None
    fast_dev_run = False
    val_check_interval = 0.2
    test_only = True
    finetune_first = False

    # below params varies with the environment
    data_root = "/root/autodl-tmp/project/datasets/ava"
    log_dir = "result"
    per_gpu_batchsize = 16  # you should define this manually with per_gpu_batch_size=#
    num_gpus = 1
    num_nodes = 1

    # load_path = "/root/autodl-tmp/pretrained/vilt_200k_mlm_itm.ckpt"
    load_path = "/root/autodl-tmp/project/result/finetune_ava_seed0_from_vilt_200k_mlm_itm/version_43/checkpoints/last.ckpt"
    num_workers = 8
    precision = 16

@ex.named_config
def task_finetune_ava():
    exp_name = "finetune_ava"
    datasets = ["ava"]
    loss_names = _loss_names({"ava": 1})
    batch_size = 32
    max_epoch = 15
    max_steps = None
    warmup_steps = 0.1
    draw_false_image = 0
    learning_rate = 1e-2
    val_check_interval = 0.2
    weight_decay = 2e-2
    optim_type = "adam"
    max_text_len = 512
