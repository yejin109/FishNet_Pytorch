import torch


device = 'cuda' if torch.cuda.is_available() else 'cpu'


net_config = dict()
net_config['num_cls'] = 10

net_config['tail_ch_in'] = [64, 128]
net_config['tail_ch_out'] = [128, 256]
net_config['tail_res_blks'] = [2, 2]

net_config['body_ch_in'] = [256, 256+128]
net_config['body_ch_out'] = [256, 192]
net_config['body_res_blks'] = [1, 1]
net_config['body_trans_blks'] = [1, 1]

net_config['head_ch_in'] = [192+64, 256+256]
net_config['head_ch_out'] = [256, 512]
net_config['head_res_blks'] = [2, 2]
net_config['head_trans_blks'] = [1, 2]
# net_config['channel_sizes'] = [128, 256, 256, 256, 384, 440]
# net_config['num_residual_blocks'] = [1] * 16
# net_config['num_transition_blocks'] = [2, 2,  2, 4]
# net_config['num_down_sample'] = 2
# net_config['num_up_sample'] = 2

envs = dict()
envs['lr'] = 0.1
envs['momentum'] = 0.9 # for SGD
envs['weight_decay'] = 1e-4
envs['batch_size'] = 32
envs['epochs'] = 80
envs['Implement_ID'] = "Imp_5"
