import math
import torch
import torch.nn as nn

import numpy as np

import functools

uniform_ = None

def next_power_of_2(x):
    return 1 if x == 0 else 2**math.ceil(math.log2(x))

def build_init(args):
    global uniform_
    args.init_method = getattr(args, 'init_method', 'xavier')
    if args.init_method == 'kaiming':
        uniform_ = kaiming_uniform_
    elif args.init_method == 'kaiming_fanout':
        uniform_ = functools.partial(kaiming_uniform_, mode='fan_out')
    elif args.init_method == 'xavier1_2':
        uniform_ = xavier_uniform1_2_
    elif 'xavier_origin_ratio' in args.init_method:
        origin_ffn_ratio = float(args.init_method.split(':')[1])
        uniform_ = functools.partial(xavier_uniform_origin_ratio_, ratio=origin_ffn_ratio)
    elif args.init_method == 'xavier2exp':
        uniform_ = xavier_uniform_2exp_
    elif args.init_method == 'xavier2exp_ratio':
        uniform_ = xavier_uniform_2exp_same_ratio_
    elif 'gain' in args.init_method:
        gain = float(args.init_method.split(':')[1])
        print("initialization gain:", gain)
        uniform_ = functools.partial(xavier_uniform_gain_, gain=gain)
    elif 'xavier_non_linear' in args.init_method:
        gain_ratio = float(args.init_method.split(':')[1])
        uniform_ = functools.partial(xavier_uniform_non_linear_, gain_ratio=gain_ratio)
    else:
        print("[WARNING] Fallback to xavier initializer")
        uniform_ = None

def xavier_uniform_non_linear_(tensor, gain_ratio=1., non_linear='linear'):
    return nn.init.xavier_uniform_(tensor, gain=gain_ratio * nn.init.calculate_gain(non_linear))

def xavier_uniform_origin_ratio_(tensor, gain=1., ratio=2, **kwargs):
    fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(tensor)
    fan_out = ratio * fan_in
    std = gain * math.sqrt(2.0 / float(fan_in + fan_out))
    a = math.sqrt(3.0) * std  # Calculate uniform bounds from standard deviation

    return nn.init._no_grad_uniform_(tensor, -a, a)

def kaiming_uniform_(tensor, non_linear, mode='fan_in'):
    return nn.init.kaiming_uniform_(tensor, mode=mode, nonlinearity=non_linear)

def xavier_uniform_gain_(tensor, gain=1., **kwargs):
    return nn.init.xavier_uniform_(tensor, gain)

def xavier_uniform1_2_(tensor, gain=1., **kwargs):
    fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(tensor)
    if fan_out < 2 * fan_in:
        fan_out = 2*fan_in
    std = gain * math.sqrt(2.0 / float(fan_in + fan_out))
    a = math.sqrt(3.0) * std  # Calculate uniform bounds from standard deviation

    return nn.init._no_grad_uniform_(tensor, -a, a)

def xavier_uniform_2exp_(tensor, gain=1., **kwargs):
    fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(tensor)
    fan_in = next_power_of_2(fan_in)
    fan_out = 2 * fan_in
    std = gain * math.sqrt(2.0 / float(fan_in + fan_out))
    a = math.sqrt(3.0) * std  # Calculate uniform bounds from standard deviation

    return nn.init._no_grad_uniform_(tensor, -a, a)

def xavier_uniform_2exp_same_ratio_(tensor, gain=1., **kwargs):
    fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(tensor)
    ratio = fan_out / fan_in
    fan_in = next_power_of_2(fan_in)
    fan_out = fan_in * ratio    
    std = gain * math.sqrt(2.0 / float(fan_in + fan_out))
    a = math.sqrt(3.0) * std  # Calculate uniform bounds from standard deviation

    return nn.init._no_grad_uniform_(tensor, -a, a)

