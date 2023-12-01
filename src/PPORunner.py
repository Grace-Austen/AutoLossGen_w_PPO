import numpy as np
import torch.backends.cudnn as cudnn
import torch

import argparse
import logging
import time
import os
import sys

from utils.global_p import *
from models.BiasedMF import BiasedMF
from models.DeepModel import DeepModel
from models.controller import Controller
from models.LossFormula import LossFormula

from data_loader.DataLoader import DataLoader
from data_processor.DataProcessor import DataProcessor
from PPO import PPO

def build_optimizer(model, op_name, learning_rate, l2_weight):
		optimizer_name = op_name.lower()
		if optimizer_name == 'gd':
			logging.info("Optimizer: GD")
			optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=l2_weight)
			# optimizer = torch.optim.SGD(model.parameters(), lr=self.learning_rate)
		elif optimizer_name == 'adagrad':
			logging.info("Optimizer: Adagrad")
			optimizer = torch.optim.Adagrad(model.parameters(), lr=learning_rate, weight_decay=l2_weight)
			# optimizer = torch.optim.Adagrad(model.parameters(), lr=self.learning_rate)
		elif optimizer_name == 'adam':
			logging.info("Optimizer: Adam")
			optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=l2_weight)
			# optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
		else:
			logging.error("Unknown Optimizer: " + optimizer_name)
			assert op_name in ['GD', 'Adagrad', 'Adam']
			optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=l2_weight)
			# optimizer = torch.optim.SGD(model.parameters(), lr=self.learning_rate)
		return optimizer

def main():
	parser = argparse.ArgumentParser(description='Model')
	# Running
	parser.add_argument('--verbose', type=int, default=logging.INFO, help='Logging Level, 0, 10, ..., 50')
	parser.add_argument('--log_file', type=str, default='../log/log_0.txt', help='Logging file path')
	parser.add_argument('--model_path', type=str, help='Model save path.',
						default=os.path.join(MODEL_DIR, 'biasedMF.pt'))  # '%s/%s.pt' % (model_name, model_name)))
	parser.add_argument('--controller_model_path', type=str, help='Controller Model save path.',
						default=os.path.join(MODEL_DIR, 'controller.pt'))
	parser.add_argument('--shared_cnn_model_path', type=str, help='Shared CNN Model save path.',
						default=os.path.join(MODEL_DIR, 'loss_formula.pt'))
	parser.add_argument('--formula_path', type=str, help='Loss Formula save path.',
						default=os.path.join(MODEL_DIR, 'Formula.txt'))
	parser.add_argument('--search_loss', action='store_true', help="To search a loss or verify a loss")
	parser.add_argument('--train_with_optim', action='store_true')
	
	#data
	# parser.add_argument('--data', type=str, default='./mnist') # dealth with by dataloader
	# parser.add_argument('--train_portion', type=float, default=0.9) # not needed, I'm assuming something else deals with it
	# parser.add_argument('--batch_size', type=int, default=256) # PPO.py

	#model
	parser.add_argument('--model_name', type=str, default='BiasedMF', help='Choose model to run.')
	# parser.add_argument('--model_epochs', type=int, default=5) # PPO.py search_train_epoch
	parser.add_argument('--model_lr', type=float, default=0.001)
	# parser.add_argument('--model_weight_decay', type=float, default=3e-4) # PPO.py l2_weight
	# parser.add_argument('--model_momentum', type=float, default=0.9) # not sure if I need to use this...
	
	parser.add_argument('--u_vector_size', type=int, default=64, help='Size of user vectors.')
	parser.add_argument('--i_vector_size', type=int, default=64, help='Size of item vectors.')
	parser.add_argument('--smooth_coef', type=float, default=1e-6)
	parser.add_argument('--layers', type=str, default='[64, 16]',
						help="Size of each layer. (For Deep RS Model.)")
	parser.add_argument('--loss_func', type=str, default='BCE',
						help='Loss Function. Choose from ["BCE", "MSE", "Hinge", "Focal", "MaxR", "SumR", "LogMin"]')
	parser.add_argument('--child_num_layers', type=int, default=12)
	parser.add_argument('--child_num_branches', type=int, default=8)  # different layers
	parser.add_argument('--child_out_filters', type=int, default=36)
				

	#architecture
	parser.add_argument('--arch_epochs', type=int, default=100) # epochs
	parser.add_argument('--arch_lr', type=float, default=3.5e-4) 
	parser.add_argument('--episodes', type=int, default=20)
	parser.add_argument('--entropy_weight', type=float, default=1e-5)
	parser.add_argument('--baseline_weight', type=float, default=0.95)
	# parser.add_argument('--embedding_size', type=int, default=32)
	# parser.add_argument('--algorithm', type=str, choices=['PPO', 'PG', 'RS'], default='PPO') # only using PPO
	parser.add_argument('--sample_branch_id', action='store_true')
	parser.add_argument('--sample_skip_id', action='store_true')
	
	#PPO
	parser.add_argument('--ppo_epochs', type=int, default=10)
	parser.add_argument('--clip_epsilon', type=float, default=0.2)

	parser.add_argument('--gpu', type=int, default=0)
	parser.add_argument('--seed', type=int, default=2, help='random seed')

	parser = DataLoader.parse_data_args(parser)
	parser = DataProcessor.parse_dp_args(parser)
	parser = PPO.parse_runner_args(parser)
	parser = Controller.parse_Ctrl_args(parser)
	parser = LossFormula.parse_Formula_args(parser)
	args, extras = parser.parse_known_args()

	# logging
	for handler in logging.root.handlers[:]:
		logging.root.removeHandler(handler)
	logging.basicConfig(filename=args.log_file, level=args.verbose)
	logging.info(vars(args))

	# set seed, deal with cuda
	torch.manual_seed(args.seed)
	torch.cuda.manual_seed(args.seed)
	np.random.seed(args.seed)
	if torch.cuda.is_available():
		device = torch.device('cuda:{}'.format(str(args.gpu)))
		cudnn.benchmark = True
		cudnn.enable = True
		logging.info('using gpu : {}'.format(args.gpu))
		torch.cuda.manual_seed(args.seed)
	else:
		device = torch.device('cpu')
		logging.info('using cpu')

	controller = Controller(search_for=args.search_for,
							search_whole_channels=True,
							num_layers=args.child_num_layers + 3,
							num_branches=args.child_num_branches,
							out_filters=args.child_out_filters,
							lstm_size=args.controller_lstm_size,
							lstm_num_layers=args.controller_lstm_num_layers,
							tanh_constant=args.controller_tanh_constant,
							temperature=None,
							skip_target=args.controller_skip_target,
							skip_weight=args.controller_skip_weight,
							# entropy_weight=args.controller_entropy_weight,
							# bl_dec=args.controller_bl_dec,
							# num_aggregate=args.controller_num_aggregate,
							model_path=args.controller_model_path,
							sample_branch_id=args.sample_branch_id,
							sample_skip_id=args.sample_skip_id)
	controller = controller.cuda()

	loss_formula = LossFormula(num_layers=args.child_num_layers + 3,
							num_branches=args.child_num_branches,
							out_filters=args.child_out_filters,
							keep_prob=args.child_keep_prob,
							model_path=args.shared_cnn_model_path,
							epsilon=args.epsilon)
	loss_formula = loss_formula.cuda()

	controller_optimizer = torch.optim.Adam(params=controller.parameters(),
												lr=args.controller_lr,
												betas=(0.0, 0.999),
												eps=1e-3)

	model_name = eval(args.model_name)
	data_loader = DataLoader(path=args.path, dataset=args.dataset, label=args.label, sep=args.sep)


	model = model_name(user_num=data_loader.user_num, item_num=data_loader.item_num,
					   u_vector_size=args.u_vector_size, i_vector_size=args.i_vector_size, model_path=args.model_path,
					   smooth_coef=args.smooth_coef, layers=args.layers, loss_func=args.loss_func
					   )
	model.optimizer = build_optimizer(model, args.optimizer, args.model_lr, args.l2)

	# model_optimizer

	if torch.cuda.device_count() > 0:
		model = model.cuda()

	data_processor = DataProcessor(data_loader, model, rank=False, test_neg_n=args.test_neg_n)
	ppo = PPO(args=args, device=device, controller=controller, controller_optimizer=controller_optimizer,
				 loss_formula=loss_formula, batch_size=args.batch_size, eval_batch_size=args.eval_batch_size,
				 dropout=args.dropout, l2=1e-5, metrics=args.metric)
	ppo.multi_solve_environment(model, data_processor)

if __name__ == '__main__':
	main()
