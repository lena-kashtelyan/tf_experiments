import sys
import numpy as np
from sklearn.svm import SVC

def attention_SVM(fc,gt_ids):
	#Leave one out crossvalidation
	y_hat = np.zeros((gt_ids.shape[0]))
	y_gt = np.zeros((gt_ids.shape[0]))
	cv_range = np.arange(gt_ids.shape[0])
	for lo in cv_range:
		train_ind = np.where(cv_range != lo)[0]
		train_X = fc[train_ind,:]
		train_y = gt_ids[train_ind]
		test_X = fc[lo,:]
		y_gt[lo] = gt_ids[lo]

		#Preprocess data
		mu = np.mean(train_X,axis=0)
		sd = np.std(train_X,axis=0)
		train_X = (train_X - mu) / (sd + 1e-1)
		test_X = (test_X - mu) / (sd + 1e-1)

		#Train an svm
		clf = SVC(kernel='linear')
		clf.fit(train_X, train_y) 
		y_hat[lo] = clf.predict(test_X.reshape(1,-1))
	return np.mean(y_hat == y_gt)

def prepare_data(data):
	fc = data['fc_batches']
	test_names = data['test_names']
	gt = data['gt']
	gt_ids = data['gt_ids']
	full_syn = data['full_syn']

	#Remove MIRCS
	fc = fc[gt_ids != -1,:]
	test_names = test_names[gt_ids != -1]
	gt = gt[gt_ids != -1]
	gt_ids = gt_ids[gt_ids != -1]

	return fc, test_names, gt, gt_ids, full_syn

def run_svm(svm_type='vgg16'):
	if svm_type == 'vgg16':
		attention_data_pointer = 'svm_data/attention_svm_data_vgg16.npz'
		baseline_data_pointer = 'svm_data/baseline_svm_data_vgg16.npz'
	elif svm_type == 'vgg19':
		attention_data_pointer = 'svm_data/attention_svm_data_vgg19.npz'
		baseline_data_pointer = 'svm_data/baseline_svm_data_vgg19.npz'

	#Prepare data and run svms
	att_fc7, att_test_names, att_gt, att_gt_ids, att_full_syn = \
		prepare_data(np.load(attention_data_pointer))
	attention_accuracy = attention_SVM(att_fc7,att_gt_ids)

	base_fc7, base_test_names, base_gt, base_gt_ids, base_full_syn = \
		prepare_data(np.load(baseline_data_pointer))
	baseline_accuracy = attention_SVM(base_fc7,base_gt_ids)

	print(attention_accuracy,baseline_accuracy)

if __name__ == '__main__':
    if len(sys.argv) > 1:
        run_svm(svm_type=sys.argv[1])
    else:
        run_svm()
