import re
import numpy as np
from glob import glob
from scipy import misc
from sklearn import svm
import skimage
from matplotlib import pyplot as plt
from tqdm import tqdm

def import_model(mtype,attention=False):
    if attention:
        mtype = 'att_' + mtype
    from model_depo import mtype as nn_model

def preproc_im(im):
    im[:,:,0] -= 103.939
    im[:,:,1] -= 116.779
    im[:,:,2] -= 123.68
    im = im[...,[2,1,0]] #RGB -> BGR
    im = im.transpose((2,0,1)) # H/W/C -> C/H/W
    return im

def find_syn(syn,cat,skeys):
    out_sys = ''
    for s in skeys:
        if s == cat:
            out_sys = syn[s]
    if out_sys == '':
        out_sys = str(-1)
    return out_sys

def find_sidx(skey,syn):
    sidx = -1
    for s in range(len(skey)):
        if skey[s] == syn:
            sidx = s
    return sidx

def get_argmax(data):
    out = np.zeros((data.shape[0],1))
    for idx in range(data.shape[0]):
        out[idx] = np.argmax(data[idx,:])
    return out

def read_color(im):
    return load_image(im).astype(np.float32)[None,:,:,:]

def read_gray(im):
    return np.repeat(skimage.color.rgb2gray(load_image(im))[:,:,None],3,axis=2)[None,:,:,:]

def process_image(im,im_size,togray=True): #TODO, remove im_size from this. 224x224 is hardcoded
    if togray:
        proc_im = read_gray(im)
    else:
        try:
            proc_im = read_color(im)
        except:
            print('Image read as grayscale:',im)
            proc_im = read_gray(im)
    return proc_im

def get_synkeys():
    syn = {};
    syn['border_collie'] = 'n02106166'
    syn['panther'] = 'n02128925'
    syn['bald_eagle'] = 'n01614925'
    syn['sorrel'] = 'n02389026'
    syn['great_white_shark'] = 'n01484850'#n02106166
    syn['airliner'] = 'n02690373'#n02106166
    syn['school_bus'] ='n04146614'# n02106166
    syn['sportscar'] = 'n04285008'#n02106166
    syn['trailer_truck'] = 'n04467665'#n02106166
    syn['speedboat'] = 'n04273569'#n02106166
    #syn['bike'] = 'n03792782'#n02106166
    #syn['warship'] = 'n02687172'#n02106166
    #syn['bug'] = 'n02190166'#n02106166
    #syn['glasses'] = 'n04356056'#n02106166
    #syn['suit'] = 'n04350905'#n02106166
    #syn['eye'] = '-1'
    skeys = syn.keys()
    return syn, skeys

def zscore_attention(maps):
    mus = np.mean(np.mean(maps,axis=0),axis=0)
    sds = np.std(np.std(maps,axis=0),axis=0)
    return 3 + ((maps - mus) / sds) #rectify/ortranslate to 0.. doesn't work quite yet. probably not a good idea anyway.

def scale_attention(maps):
    return maps/np.max(np.max(maps,axis=0),axis=0)[None,None,:]

def get_attention_maps(attention_path,im_size):
    att_dict = np.load(attention_path)
    att_maps = att_dict['image_maps']
    att_labels = att_dict['im_files']
    #Normalize each map to [0,1]
    att_maps = np.nanmean(att_maps,axis=3)
    for a in range(att_maps.shape[2]):
        res_map = misc.imresize(att_maps[:,:,a],im_size)[None,:,:]
        if a == 0:
            out_a = res_map
        else:
            out_a = np.concatenate((out_a,res_map),axis=0)
    #out_a = scale_attention(out_a.astype(np.float32))
    #out_a = scale_attention(out_a.astype(np.float32)) + 0.5
    out_a = scale_attention(out_a.astype(np.float32)) + 1
    #out_a = scale_attention(out_a.astype(np.float32)) + 1e50
    #out_a = zscore_attention(out_a.astype(np.float32))
    #out_a = np.ones((out_a.shape)) + 1 #+100 ##+ 1e10
    return out_a[:,:,:,None]# > 0).astype(np.float32)

def prepare_training_images(train_im_dir, im_size, im_ext, grayscale=False, keep_number = 500):
    #train_im_dir is a directory of directories
    cat_folders = glob(train_im_dir + '/*')
    X = []
    y = []
    y_names = []
    for count,idx in enumerate(cat_folders):
        img_pointers = np.asarray(glob(idx + '/*' + im_ext))
        np.random.shuffle(img_pointers)
        img_pointers = img_pointers[:keep_number] #keep some proportion of the training images
        for il, im in enumerate(img_pointers):
            it_im = process_image(im,im_size,grayscale)
            if count == 0 and il == 0:
                im_array = it_im
            else:
                im_array = np.concatenate((im_array,it_im),axis=0)
        y = np.append(y,np.repeat(count,img_pointers.shape[0]))
        y_names = np.append(y_names,np.repeat(idx,img_pointers.shape[0]))
    return im_array,y,y_names

def prepare_testing_images(test_im_dir, im_size, im_ext, grayscale=False, keep_number = 500):
    #train_im_dir is a directory of directories
    X = []
    y = []
    y_names = []
    test_ims = sorted(glob(test_im_dir + '/*' + im_ext))

    for il, im in enumerate(test_ims):
        it_im = process_image(im,im_size,grayscale)
        if il == 0:
            im_array = it_im
        else:
            im_array = np.concatenate((im_array,it_im),axis=0)
        y = np.append(y,il)
        y_names = np.append(y_names,im)
    return im_array,y,y_names

def train_svm(X,y):
    return svm.LinearSVC(C=1).fit(X, y)

def get_labels(all_ims,syn,skeys,syn_file):

    syn_key = [line.rstrip('\n') for line in open(syn_file)]

    gt = []
    gt_ids = []
    for it, imn in enumerate(all_ims):
        im_cat = re.split('\d',re.split('\.',re.split('/',imn)[-1])[0])[0]
        out_syn = find_syn(syn,im_cat,skeys)
        out_syn_ids = (find_sidx(syn_key,out_syn))
        gt.append(out_syn)
        gt_ids.append(out_syn_ids)
    return gt,gt_ids

def factors(n):    
    return list(reduce(list.__add__, 
                ([i, n//i] for i in range(1, int(n**0.5) + 1) if n % i == 0)))

def im_mosaic(ims,save=False):
    cm = 'jet'
    sp_factors = factors(ims.shape[-1])
    s1 = sp_factors[np.argmin(np.abs(map(lambda x: x - np.sqrt(ims.shape[-1]),sp_factors)))]
    s2 = ims.shape[-1] // s1
    f = plt.figure()
    for p in tqdm(range(ims.shape[-1])):
        a = plt.subplot(s1,s2,p+1)
        plt.imshow(np.squeeze(ims[:,:,p]),cmap=cm)
        a.axes.get_xaxis().set_visible(False)
        a.axes.get_yaxis().set_visible(False)
    plt.subplots_adjust(wspace=0.01,hspace=0.01,right=0.8)    
    cbar_ax = f.add_axes([0.85, 0.15, 0.05, 0.7])
    plt.colorbar(cax=cbar_ax)
    #if save != False:
    #    plt.savefig(save + '.png')
    plt.show()

def get_class_accuracy(cat_guesses,gt_ids,file_path):
    synset = [l.strip() for l in open(file_path).readlines()]
    cat_guesses = cat_guesses.ravel()
    uni_classes = np.unique(gt_ids)
    return [[synset[it_gt],np.mean(cat_guesses[gt_ids==it_gt] == it_gt)] for it_gt in uni_classes]

def get_t5_accuracy(t5_preds,gt_syn):
    eval_pred = np.zeros((len(t5_preds)))
    for idx, preds in enumerate(t5_preds):
        t5_labs = [re.split(' ', x[0])[0] for x in preds]
        eval_pred[idx] = gt_syn[idx] in t5_labs
    return eval_pred

def evaluate_model(gt_syn,gt_ids,prob,train_names,im_ext,file_path,print_results=True):
    gt_ids = np.asarray(gt_ids)
    gt_syn = np.asarray(gt_syn)
    prob = prob[gt_ids!=-1]
    gt_syn = gt_syn[gt_ids!=-1]
    gt_ids = gt_ids[gt_ids!=-1]

    proc_names = [re.split(im_ext,re.split('/',x)[-1])[0] for x in train_names]
    t1_preds = []
    t5_preds = []
    for idx in prob:
        t1,t5 = print_prob(idx, file_path)
        t1_preds.append(t1)
        t5_preds.append(t5)

    cat_guesses = get_argmax(prob).transpose()
    class_accuracy = get_class_accuracy(cat_guesses,gt_ids,file_path)
    #print(zip(t1_preds,proc_names))
    t1_acc = np.mean((cat_guesses == gt_ids).astype(np.float32))
    t5_acc = np.mean(get_t5_accuracy(t5_preds,gt_syn).astype(np.float32))
    if print_results:
        print('top-1 accuracy is', t1_acc)
        print('top-5 accuracy is', t5_acc)
        print('class-by-class accuracy is',class_accuracy)
    return class_accuracy, t1_preds, t5_preds, t1_acc, t5_acc

# returns the top1 string
def print_prob(prob, file_path):
    synset = [l.strip() for l in open(file_path).readlines()]

    # print prob
    pred = np.argsort(prob)[::-1]

    # Get top1 label
    top1 = synset[pred[0]]
    # Get top5 label
    top5 = [(synset[pred[i]], prob[pred[i]]) for i in range(5)]
    return top1, top5

def load_image(path):
    # load image
    img = skimage.io.imread(path)
    img = img / 255.0 #Scale images to [0,1] for Vvgg16.py
    assert (0 <= img).all() and (img <= 1.0).all()
    # print "Original Image Shape: ", img.shape
    # we crop image from center
    short_edge = min(img.shape[:2])
    yy = int((img.shape[0] - short_edge) / 2)
    xx = int((img.shape[1] - short_edge) / 2)
    crop_img = img[yy: yy + short_edge, xx: xx + short_edge]
    # resize to 224, 224
    resized_img = skimage.transform.resize(crop_img, (224, 224))
    return resized_img

def shuffle_attention(attention_batch,shuffle_or_warp):
    for idx in range(attention_batch.shape[0]):
        if shuffle_or_warp == 'scramble':
            it_map = np.squeeze(attention_batch[idx,:,:,:])[np.random.permutation(attention_batch.shape[1]),:]
            attention_batch[idx,:,:,:] = it_map[:,np.random.permutation(attention_batch.shape[1])][:,:,None] 
        if shuffle_or_warp == 'shuffle':
            attention_batch = attention_batch[np.random.permutation(attention_batch.shape[0])]
        elif shuffle_or_warp == 'warp':
            #tform = skimage.transform.AffineTransform(scale=np.random.uniform(.9,1.1), \
            #    rotation=np.random.uniform(-np.pi/8,np.pi/8),\
            #    translation=(np.random.randint(np.round(-attention_batch.shape[1]*.05),np.round(attention_batch.shape[1]*.5))))
            tform = skimage.transform.AffineTransform(scale=np.repeat(np.random.uniform(.75,1.25),2),\
                translation=(np.random.randint(-np.round(attention_batch.shape[1]*.25),np.round(attention_batch.shape[1]*.25)),\
                np.random.randint(-np.round(attention_batch.shape[1]*.25),np.round(attention_batch.shape[1]*.25))))
            attention_batch[idx,:,:,:] = (skimage.transform.warp(np.squeeze(attention_batch[idx,:,:,:]-1),tform)+1)[:,:,None] #HARDCODED FOR THE [1,2] ATTENTION CASE
    return attention_batch