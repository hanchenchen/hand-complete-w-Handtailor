from .core import config
from .model.factory import create_model
import os
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from pathlib import Path
from torchvision import transforms


def get_hrnet():

	args_config = '.cache/w32_224x224_adam_lr1e-3.yaml'
	args_resume = '.cache/best_7.542.pth'
	args_mGPUs = False

	args_config = Path(args_config)
	cfg = config.read_config(args_config)

	# if hasattr(cfg.TRAIN, 'CUDNN'):
	#     cudnn.benchmark = cfg.TRAIN.CUDNN.BENCHMARK
	#     cudnn.deterministic = cfg.TRAIN.CUDNN.DETERMINISTIC
	#     cudnn.enabled = cfg.TRAIN.CUDNN.ENABLED
	if hasattr(cfg.TRAIN, 'SEED'):
	    np.random.seed(cfg.TRAIN.SEED)
	    torch.manual_seed(cfg.TRAIN.SEED)


	# create model, optimizer and scheduler
	hrnet_model = create_model(cfg, None).cuda()
	# TODO: model summary

	# multi GPUs
	if args_mGPUs:
	    hrnet_model = torch.nn.DataParallel(hrnet_model)
	    cfg.TRAIN.BATCH_SIZE *= torch.cuda.device_count()

	img_normalize = transforms.Normalize(
	    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
	)
	img_transform = transforms.Compose([
	    transforms.ToTensor(),
	    img_normalize
	])

	state = torch.load(args_resume)
	if isinstance(hrnet_model, torch.nn.DataParallel):
	    hrnet_model.module.load_state_dict(state['model_state_dict'])
	else:
	    hrnet_model.load_state_dict(state['model_state_dict'])
	print('loading ckpt from ', args_resume)
	hrnet_model.eval()
	return hrnet_model

