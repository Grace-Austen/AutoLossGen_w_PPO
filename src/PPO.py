# from models.controller import Controller
from models.Worker import Worker, get_acc, train_model, evaluate_model
import numpy as np
import torch
import torch.optim as optim
import logging
from multiprocessing import Process, Queue
import multiprocessing
import copy

multiprocessing.set_start_method('spawn', force=True)

def consume(worker, results_queue, model, data, data_processor, epoch=-1, loss_fun=None):
	get_acc(worker, model, data, data_processor, epoch, loss_fun)
	results_queue.put(worker)

class PPO(object):
	@staticmethod
	def parse_runner_args(parser):
		parser.add_argument('--load', type=int, default=0,
							help='Whether load model and continue to train')
		parser.add_argument('--epoch', type=int, default=100,
							help='Number of epochs.')
		parser.add_argument('--check_epoch', type=int, default=1,
							help='Check every epochs.')
		parser.add_argument('--early_stop', type=int, default=1,
							help='whether to early-stop.')
		parser.add_argument('--lr', type=float, default=0.01,
							help='Learning rate.')
		parser.add_argument('--batch_size', type=int, default=128,
							help='Batch size during training.')
		parser.add_argument('--eval_batch_size', type=int, default=128 * 128,
							help='Batch size during testing.')
		parser.add_argument('--dropout', type=float, default=0.2,
							help='Dropout probability for each deep layer')
		parser.add_argument('--l2', type=float, default=1e-5,
							help='Weight of l2_regularize in loss.')
		parser.add_argument('--optimizer', type=str, default='GD',
							help='optimizer: GD, Adam, Adagrad')
		parser.add_argument('--metric', type=str, default="AUC",
							help='metrics: RMSE, MAE, AUC, F1, Accuracy, Precision, Recall')
		parser.add_argument('--skip_eval', type=int, default=0,
							help='number of epochs without evaluation')
		parser.add_argument('--skip_rate', type=float, default=1.005, help='bad loss skip rate')
		parser.add_argument('--rej_rate', type=float, default=1.005, help='bad training reject rate')
		parser.add_argument('--skip_lim', type=float, default=1e-5, help='bad loss skip limit')
		parser.add_argument('--rej_lim', type=float, default=1e-5, help='bad training reject limit')
		parser.add_argument('--lower_bound_zero_gradient', type=float, default=1e-4, help='bound to check zero gradient')
		parser.add_argument('--search_train_epoch', type=int, default=1, help='epoch num for training when searching loss')
		parser.add_argument('--step_train_epoch', type=int, default=1, help='epoch num for training each step')
		
		return parser
	

	def __init__(self, args, device, controller, controller_optimizer,
			  loss_formula=None, batch_size=128, eval_batch_size=128 * 128,
			  dropout=0.2, l2=1e-5, metrics='AUC,RMSE'):
		
		self.batch_size = batch_size
		self.eval_batch_size = eval_batch_size
		self.dropout = dropout
		self.no_dropout = 0.0
		self.l2_weight = l2

		self.metrics = metrics.lower().split(',')
		self.loss_formula = loss_formula

		self.args = args
		self.device = device

		self.arch_epochs = args.arch_epochs
		self.episodes = args.episodes
		self.entropy_weight = args.entropy_weight

		self.ppo_epochs = args.ppo_epochs

		self.controller = controller
		self.controller_optimizer = controller_optimizer

		self.baseline = None
		self.baseline_weight = self.args.baseline_weight

		self.clip_epsilon = 0.2

	def multi_solve_environment(self, model, data_processor):
		workers_top20 = []

		train_data = data_processor.get_train_data(epoch=-1)
		validation_data = data_processor.get_validation_data()
		test_data = data_processor.get_test_data()

		min_reward = torch.tensor(-1.0).cuda()

		data = {
		'min_reward': min_reward,
		'search_loss': self.args.search_loss,
		'train': False,
		'search_train_epoch': self.args.search_train_epoch,
		'metrics': self.metrics,
		
		'lower_bound_zero_gradient': self.args.lower_bound_zero_gradient,
		'regularizer': False,
		'l2_weight': self.l2_weight,
		'dropout': self.dropout,
		'no_dropout': self.no_dropout,

		'batch_size': self.batch_size,
		'eval_batch_size': self.eval_batch_size,
		'train_data': None, #should be epoch_train_data
		'validation_data': validation_data,
		}

		for arch_epoch in range(self.arch_epochs):
			cur_model = copy.deepcopy(model)
			results_queue = Queue()
			processes = []

			epoch_train_data = data_processor.get_train_data(epoch=arch_epoch)

			for episode in range(self.episodes):
				actions_p, actions_log_p, actions_index = self.controller.sample()
				actions_p = actions_p.cpu().detach().numpy().tolist()
				actions_log_p = actions_log_p.cpu().detach().numpy().tolist()
				# actions_index = actions_index.cpu().detach().numpy().tolist()

				if episode < self.episodes // 3:
					worker = Worker(actions_p, actions_log_p, actions_index, self.args, 'cuda:0')
				elif self.episodes // 3 <= episode < 2 * self.episodes // 3:
					worker = Worker(actions_p, actions_log_p, actions_index, self.args, 'cuda:1')
				else:
					worker = Worker(actions_p, actions_log_p, actions_index, self.args, 'cuda:3')

				worker_data = copy.deepcopy(data)
				worker_data['train_data'] = epoch_train_data

				worker_model = copy.deepcopy(cur_model)
				process = Process(target=consume, args=(worker, results_queue, worker_model, worker_data, data_processor, arch_epoch, self.loss_formula))
				process.start()
				processes.append(process)

			for process in processes:
				process.join()

			workers = []
			for episode in range(self.episodes):
				worker = results_queue.get()
				worker.actions_p = torch.Tensor(worker.actions_p).to(self.device)
				worker.actions_index = torch.LongTensor(worker.actions_index).to(self.device)
				workers.append(worker)

			for episode, worker in enumerate(workers):
				if self.baseline == None:
					self.baseline = worker.acc
				else:
					self.baseline = self.baseline * self.baseline_weight + worker.acc * (1 - self.baseline_weight)

			# sort worker retain top20
			workers_total = workers_top20 + workers
			workers_total.sort(key=lambda worker: worker.acc, reverse=True)
			workers_top20 = workers_total[:20]
			top1_acc = workers_top20[0].acc
			top5_avg_acc = np.mean([worker.acc for worker in workers_top20[:5]])
			top20_avg_acc = np.mean([worker.acc for worker in workers_top20])
			logging.info('arch_epoch {:0>3d} top1_acc {:.4f} top5_avg_acc {:.4f} top20_avg_acc {:.4f} baseline {:.4f} '.format(
				arch_epoch, top1_acc, top5_avg_acc, top20_avg_acc, self.baseline))
			for i in range(5):
				print(workers_top20[i].loss_string)

			worker_data = copy.deepcopy(data)
			worker_data['train_data'] = epoch_train_data
			train_model(model, worker_data, data_processor, arch_epoch, workers_top20[0].actions_index, self.loss_formula)

			for ppo_epoch in range(self.ppo_epochs):
				loss = 0

				for worker in workers:
					actions_p, actions_log_p = self.controller.get_p(worker.actions_index)

					loss += self.cal_loss(actions_p, actions_log_p, worker, self.baseline)

				loss /= len(workers)
				logging.info('ppo_epoch {:0>3d} loss {:.4f} '.format(ppo_epoch, loss))

				self.controller_optimizer.zero_grad()
				loss.backward()
				self.controller_optimizer.step()

	def clip(self, actions_importance):
		lower = torch.ones_like(actions_importance).to(self.device) * (1 - self.clip_epsilon)
		upper = torch.ones_like(actions_importance).to(self.device) * (1 + self.clip_epsilon)

		actions_importance, _ = torch.min(torch.cat([actions_importance.unsqueeze(0), upper.unsqueeze(0)], dim=0), dim=0)
		actions_importance, _ = torch.max(torch.cat([actions_importance.unsqueeze(0), lower.unsqueeze(0)], dim=0), dim=0)

		return actions_importance

	def cal_loss(self, actions_p, actions_log_p, worker, baseline):
		actions_importance = actions_p / worker.actions_p
		clipped_actions_importance = self.clip(actions_importance)

		if worker.no_grad:
			reward = -1 - baseline

		else:
			reward = worker.acc - baseline
		
		actions_reward = actions_importance * reward
		clipped_actions_reward = clipped_actions_importance * reward
		
		actions_reward, _ = torch.min(torch.cat([actions_reward.unsqueeze(0), clipped_actions_reward.unsqueeze(0)], dim=0), dim=0)
		policy_loss = -1 * torch.sum(actions_reward)
		entropy = -1 * torch.sum(actions_p * actions_log_p)
		entropy_bonus = -1 * entropy * self.entropy_weight

		return policy_loss + entropy_bonus