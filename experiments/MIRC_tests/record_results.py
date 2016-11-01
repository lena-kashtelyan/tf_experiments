import os, sys
sys.path.append('/home/drew/Documents/tensorflow-vgg/')
sys.path.append('alt_resnet')
import tensorflow as tf
from exp_ops.helper_functions import *
from experiments.config import * # Path configurations


#from pg import DB
#db = DB()
#db = DB(dbname='mircs', host='pgserver', port=5432, user='drew', passwd='serrelab')
#db.query("""CREATE TABLE model_performance (
#...     model_name varchar(),
#...     accuracy float32, 
#...     pvalue float32,
#...     date date)""")
#db.query("""INSERT INTO weather
#...     VALUES ('San Francisco', 46, 50, 0.25, '11/27/1994')""")


net_config = {
	'models': ['vgg16','resnet50','resnet101','resnet152'],
	'attention':[['none'], 
		['/home/drew/Documents/MIRC_behavior/heat_map_output/pooled_p2p_alt/uniform_weight_overlap_human/heatmaps.npz'], #clicks
		['/home/drew/Documents/MIRC_behavior/click_comparisons/output/labelme.npz'], #labelme
		['/home/drew/Documents/MIRC_behavior/heat_map_output/pooled_p2p_alt/uniform_weight_overlap_human/heatmaps.npz',
        '/home/drew/Documents/MIRC_behavior/click_comparisons/output/labelme.npz']], #Both labelme and clicks
	'pvalues': True
}

class_accuracy, t1_true_acc, t5_true_acc, t1_preds, t5_preds, t1_pval, t5_pval =\
	run_analyses(net_config)

