import torch
import torch.nn as nn
from tqdm import tqdm
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import numpy as np
from utils import utils, global_p

class Worker(object):
	def __init__(self, actions_p, actions_log_p, actions_index, args, device):
		self.actions_p = actions_p
		self.actions_log_p = actions_log_p
		self.actions_index = actions_index

		self.args = args
		self.device = device

		self.params_size = None
		self.acc = 0
		self.grad = None
		self.no_grad = False

		self.loss_string = None

def get_acc(worker, model, data, data_processor, epoch=-1, loss_fun=None):
	sample_arc = worker.actions_index
	worker.loss_string = loss_fun.log_formula(sample_arc=sample_arc, id=loss_fun.num_layers-1)
	'''Zero grad detection'''
	test_pred = torch.rand(20).cuda() * 0.8 + 0.1 # change range here
	test_label = torch.rand(20).cuda()
	test_pred.requires_grad = True    
	test_loss = loss_fun(test_pred, test_label, sample_arc, small_epsilon=True)
	try:
		test_loss.backward()
	except RuntimeError:
		pass
	worker.grad = test_pred.grad
	
	if test_pred.grad is None or torch.norm(test_pred.grad, float('inf')) < data['lower_bound_zero_gradient']:
		worker.no_grad = True
		return

	'''Valid gradient, Train'''
	train_model(model, data, data_processor, epoch, sample_arc, loss_fun)

	'''Get reward'''
	worker.acc = evaluate_model(model, data, data_processor)


def train_model(model, data, data_processor, epoch, sample_arc, loss_fun):
	'''Get data'''
	train_data = data['train_data']

	train_epoch = data['search_train_epoch']
	for i in range(train_epoch):
		'''Transform Train Data'''
		batch_size = data['batch_size']
		batches = data_processor.prepare_batches(train_data, batch_size, train=True)
		for batch in batches: # Add control
			batch['train'] = True
			batch['dropout'] = data['dropout']

		batch_size = batch_size if data_processor.rank == 0 else batch_size * 2
		accumulate_size = 0
		to_show = batches if data['search_loss'] else tqdm(batches, leave=False, desc='Epoch %5d' % (epoch + 1), ncols=100, mininterval=1)
		
		'''Train model'''
		model.train()
		for batch in to_show:
				accumulate_size += len(batch['Y'])
				model.optimizer.zero_grad()
				output_dict = model(batch)
				loss = output_dict['loss'] + model.l2() * data['l2_weight']
				if loss_fun is not None and sample_arc is not None:
					loss = loss_fun(output_dict['prediction'], batch['Y'], sample_arc)
					if data['regularizer']:
						loss += model.l2() * data['l2_weight']
				loss.backward()
				torch.nn.utils.clip_grad_value_(model.parameters(), 50)
				if accumulate_size >= batch_size or batch is batches[-1]:
					model.optimizer.step()
					accumulate_size = 0
		model.eval()

	
def evaluate_model(model, data, data_processor):
	'''Transform Predict Validation Data'''
	eval_batch_size = data['eval_batch_size']
	validation_data = data['validation_data']
	train = data['train']
	eval_batch_size = data['eval_batch_size']
	validate_batches = data_processor.prepare_batches(validation_data, eval_batch_size, train=train)
	for batch in validate_batches: # Add control
		batch['train'] = train
		batch['dropout'] = data['dropout'] if train else data['no_dropout']

	'''Predict with validation data'''
	model.eval()
	predictions = []
	for batch in tqdm(validate_batches, leave=False, ncols=100, mininterval=1, desc='Predict'):
		prediction = model.predict(batch)['prediction']
		predictions.append(prediction.detach().cpu())

	predictions = np.concatenate(predictions)
	sample_ids = np.concatenate([b[global_p.K_SAMPLE_ID] for b in validate_batches])

	reorder_dict = dict(zip(sample_ids, predictions))
	predictions = np.array([reorder_dict[i] for i in validation_data[global_p.K_SAMPLE_ID]]).cuda()

	'''Calculate Reward'''
	return model.evaluate_method(predictions, validation_data, metrics=data['metrics'])[0]
