# -*- coding: utf-8 -*-
# @Time : 2024/10/23 9:51 PM
# @Author : Caster Lee
# @Email : li949777411@gmail.com
# @File : load_demo.py
# @Project : graphconvnet_pretrained

import torch
from xtuner.humanllava.vig.load_vigbackbone_pretrained import load_pretrained
if __name__ == '__main__':

    model_name = 'graphconvnet_s'  # graphconvnet_b
    ckpt_path = f'/home/ubuntu/san/LYT/UniDetRet-exp/HumanLlama3/ckpt/graphconvnet_s/model_best.pth.tar'

    model = load_pretrained(model_name, ckpt_path)

    params = sum([p.numel() for p in model.parameters() if p.requires_grad])
    print(f"Trainable params number: {params:,}")
    x = torch.randn((4, 3, 384, 384))
    print(x.shape)
    output = model(x)
    print(output.shape)
