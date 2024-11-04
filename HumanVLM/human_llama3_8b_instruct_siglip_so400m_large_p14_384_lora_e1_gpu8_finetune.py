# Copyright (c) OpenMMLab. All rights reserved.
from mmengine.hooks import (CheckpointHook, DistSamplerSeedHook, IterTimerHook,
                            LoggerHook, ParamSchedulerHook)
from mmengine.optim import AmpOptimWrapper, CosineAnnealingLR, LinearLR
from peft import LoraConfig
from torch.optim import AdamW
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          SiglipImageProcessor, SiglipVisionModel)
from mmengine.visualization import Visualizer, TensorboardVisBackend

from xtuner.dataset import ConcatDataset, LLaVADataset
from xtuner.dataset.collate_fns import default_collate_fn
from xtuner.dataset.map_fns import llava_map_fn, template_map_fn_factory
from mmengine.dataset import DefaultSampler
from xtuner.dataset.samplers import LengthGroupedSampler
from xtuner.engine.hooks import DatasetInfoHook, EvaluateChatHook
from xtuner.engine.runner import TrainLoop
from xtuner.model import LLaVAModel
from xtuner.utils import PROMPT_TEMPLATE

#######################################################################
#                          PART 1  Settings                           #
#######################################################################
# Model
llm_name_or_path = 'meta-llama/Meta-Llama-3-8B-Instruct'
visual_encoder_name_or_path = 'google/siglip-so400m-patch14-384'
# Specify the pretrained pth
pretrained_pth = './work_dirs/human_llama3_8b_instruct_siglip_so400m_large_p14_384_e1_gpu8_pretrain/iter_54000.pth'  # noqa: E501

# Data
data_root = '/home/ubuntu/public-Datasets/HumanSFT/'
data_path = data_root + 'ft_hfformat_base_attr_keypoint_0616_clean'
# data_path = data_root + 'ft_json_base_attr_keypoint_0616'
image_folder = data_root + 'data'
prompt_template = PROMPT_TEMPLATE.llama3_chat
max_length = int(4096 - 728)

# Scheduler & Optimizer
batch_size = 2  # per_device
accumulative_counts = 8
dataloader_num_workers = 16
max_epochs = 1
optim_type = AdamW
lr = 5e-5
betas = (0.9, 0.999)
weight_decay = 0
max_norm = 1  # grad clip
warmup_ratio = 0.03

# Save
save_steps = 1000
save_total_limit = 20  # Maximum checkpoints to keep (-1 means unlimited)

# Evaluate the generation performance during the training
evaluation_freq = 1000
SYSTEM = ''
evaluation_images = '/home/ubuntu/san/LYT/UniDetRet-exp/HumanLlama3/test1.jpg'
evaluation_inputs = ['Please describe this picture', 'List the facial atrributes of the person in the photograph with a markdown table.', 'how many people are in the picture?', 'Locate a person on a motorcycle wearing a helmet in image and provide its coordinates, please.',
                     'The head is located at the apex of the human body, serving as the control center and housing crucial sensory receptors.\nWhere is the exact location of the head of this person in the image? Provide the coordinates, please.', 
                     'The left shoulder marks the region where the shoulder blade (scapula) interfaces with the arm, usually located near the upper lateral part of the rib cage.\nCan you locate the left shoulder of this person in the photo and provide its coordinates?']

#######################################################################
#            PART 2  Model & Tokenizer & Image Processor              #
#######################################################################
tokenizer = dict(
    type=AutoTokenizer.from_pretrained,
    pretrained_model_name_or_path=llm_name_or_path,
    trust_remote_code=True,
    padding_side='right')

image_processor = dict(
    type=SiglipImageProcessor.from_pretrained,
    pretrained_model_name_or_path=visual_encoder_name_or_path,
    trust_remote_code=True)

model = dict(
    type=LLaVAModel,
    freeze_llm=True,
    freeze_visual_encoder=True,
    pretrained_pth=pretrained_pth,
    llm=dict(
        type=AutoModelForCausalLM.from_pretrained,
        pretrained_model_name_or_path=llm_name_or_path,
        trust_remote_code=True),
    llm_lora=dict(
        type=LoraConfig, r=128, lora_alpha=256, lora_dropout=0.05, bias='none', task_type='CAUSAL_LM'),
    visual_encoder=dict(
        type=SiglipVisionModel.from_pretrained,
        pretrained_model_name_or_path=visual_encoder_name_or_path),
    visual_encoder_lora=dict(
        type=LoraConfig, r=128, lora_alpha=256, lora_dropout=0.05, bias='none'))

#######################################################################
#                      PART 3  Dataset & Dataloader                   #
#######################################################################
# data_path=data_path,
llava_dataset = dict(
    type=LLaVADataset,
    offline_processed_text_folder=data_path,
    image_folder=image_folder,
    tokenizer=tokenizer,
    image_processor=image_processor,
    dataset_map_fn=llava_map_fn,
    template_map_fn=dict(
        type=template_map_fn_factory, template=prompt_template),
    max_length=max_length,
    pad_image_to_square=False)

# train_dataloader = dict(
#     batch_size=batch_size,
#     num_workers=dataloader_num_workers,
#     dataset=llava_dataset,
#     sampler=dict(type=DefaultSampler, shuffle=True),
#     collate_fn=dict(type=default_collate_fn))

train_dataloader = dict(
    batch_size=batch_size,
    num_workers=dataloader_num_workers,
    pin_memory=True,
    dataset=llava_dataset,
    sampler=dict(
        type=LengthGroupedSampler,
        length_property='modality_length',
        per_device_batch_size=batch_size * accumulative_counts),
    collate_fn=dict(type=default_collate_fn))

#######################################################################
#                    PART 4  Scheduler & Optimizer                    #
#######################################################################
# optimizer
optim_wrapper = dict(
    type=AmpOptimWrapper,
    optimizer=dict(
        type=optim_type, lr=lr, betas=betas, weight_decay=weight_decay),
    clip_grad=dict(max_norm=max_norm, error_if_nonfinite=False),
    accumulative_counts=accumulative_counts,
    loss_scale='dynamic',
    dtype='float16')

# learning policy
# More information: https://github.com/open-mmlab/mmengine/blob/main/docs/en/tutorials/param_scheduler.md  # noqa: E501
param_scheduler = [
    dict(
        type=LinearLR,
        start_factor=1e-5,
        by_epoch=True,
        begin=0,
        end=warmup_ratio * max_epochs,
        convert_to_iter_based=True),
    dict(
        type=CosineAnnealingLR,
        eta_min=0.0,
        by_epoch=True,
        begin=warmup_ratio * max_epochs,
        end=max_epochs,
        convert_to_iter_based=True)
]

# train, val, test setting
train_cfg = dict(type=TrainLoop, max_epochs=max_epochs)

#######################################################################
#                           PART 5  Runtime                           #
#######################################################################
# Log the dialogue periodically during the training process, optional
custom_hooks = [
    dict(type=DatasetInfoHook, tokenizer=tokenizer),
    dict(
        type=EvaluateChatHook,
        tokenizer=tokenizer,
        image_processor=image_processor,
        every_n_iters=evaluation_freq,
        evaluation_inputs=evaluation_inputs,
        evaluation_images=evaluation_images,
        system=SYSTEM,
        prompt_template=prompt_template)
]

# configure default hooks
default_hooks = dict(
    # record the time of every iteration.
    timer=dict(type=IterTimerHook),
    # print log every 10 iterations.
    logger=dict(type=LoggerHook, log_metric_by_epoch=False, interval=10),
    # enable the parameter scheduler.
    param_scheduler=dict(type=ParamSchedulerHook),
    # save checkpoint per `save_steps`.
    checkpoint=dict(
        type=CheckpointHook,
        by_epoch=False,
        interval=save_steps,
        max_keep_ckpts=save_total_limit),
    # set sampler seed in distributed evrionment.
    sampler_seed=dict(type=DistSamplerSeedHook),
)

# configure environment
env_cfg = dict(
    # whether to enable cudnn benchmark
    cudnn_benchmark=False,
    # set multi process parameters
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    # set distributed parameters
    dist_cfg=dict(backend='nccl'),
)

# set visualizer
visualizer = dict(type=Visualizer, vis_backends=[dict(type=TensorboardVisBackend)])

# set log level
log_level = 'INFO'

# # load from which checkpoint
# load_from = './work_dirs/human_llama3_8b_instruct_siglip_so400m_large_p14_384_lora_e1_gpu8_finetune/iter_32000.pth'

# # whether to resume training from the loaded checkpoint
# resume = True

# load from which checkpoint
load_from = None

# whether to resume training from the loaded checkpoint
resume = False

# Defaults to use random seed and disable `deterministic`
randomness = dict(seed=None, deterministic=False)

# set log processor
log_processor = dict(by_epoch=False)
