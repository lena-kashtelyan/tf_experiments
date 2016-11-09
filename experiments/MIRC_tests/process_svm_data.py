import numpy as np
from sklearn.svm import SVC

def attention_SVM(fc8,gt_ids):

	#Leave one out crossvalidation
	y_hat = np.zeros((gt_ids.shape[0]))
	y_gt = np.zeros((gt_ids.shape[0]))
	cv_range = np.arange(gt_ids.shape[0])
	for lo in cv_range:
		train_ind = np.where(cv_range != lo)[0]
		train_X = fc8[train_ind,:]
		train_y = gt_ids[train_ind]
		test_X = fc8[lo,:]
		y_gt[lo] = gt_ids[lo]

		#Preprocess data
		mu = np.mean(train_X,axis=0)
		sd = np.std(train_X,axis=0)
		train_X = (train_X - mu) / sd
		test_X = (test_X - mu) / sd

		#Train an svm
		clf = SVC(kernel='linear')
		clf.fit(train_X, train_y) 
		y_hat[lo] = clf.predict(test_X.reshape(1,-1))
	return np.mean(y_hat == y_gt)

def prepare_data(data):
	fc8 = data['fc_batches']
	test_names = data['test_names']
	gt = data['gt']
	gt_ids = data['gt_ids']
	full_syn = data['full_syn']

	#Remove MIRCS
	fc8 = fc8[gt_ids != -1,:]
	test_names = test_names[gt_ids != -1]
	gt = gt[gt_ids != -1]
	gt_ids = gt_ids[gt_ids != -1]

	return fc8, test_names, gt, gt_ids, full_syn

attention_data_pointer = 'svm_data/attention_svm_data_vgg16.npz'
baseline_data_pointer = 'svm_data/baseline_svm_data_vgg16.npz'
#attention_data_pointer = 'svm_data/attention_svm_data_vgg19.npz'
#baseline_data_pointer = 'svm_data/baseline_svm_data_vgg19.npz'

#Prepare data and run svms
att_fc8, att_test_names, att_gt, att_gt_ids, att_full_syn = \
	prepare_data(np.load(attention_data_pointer))
attention_accuracy = attention_SVM(att_fc8,att_gt_ids)

base_fc8, base_test_names, base_gt, base_gt_ids, base_full_syn = \
	prepare_data(np.load(baseline_data_pointer))
baseline_accuracy = attention_SVM(base_fc8,base_gt_ids)

print(attention_accuracy,baseline_accuracy)
