# import _init_paths
import numpy as np
import scipy.io as sio
import sys,os
import caffe
import argparse
import time
import skimage.io;skimage.io.use_plugin('matplotlib') # cannot convert object to float64
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

dataset_dict = {'fish':'fish','dog':'stanford_dog','bird':'CUB_200_2011'}
def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Find bounding-box related to dog')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [0]',
                        default=0, type=int)
    parser.add_argument('--cpu', dest='cpu_mode',
                        help='Use CPU mode (overrides --gpu)',
                        action='store_true')
    parser.add_argument('--dataset', dest='dataset', help='which dataset to test',
                        default='dog',type=str)
    parser.add_argument('--model_type', dest='model_type', help='model type alx or vgg16',
                        default='alex', type=str)
    parser.add_argument('--test_model', dest='test_model', help='choose caffemodel for test',
                        default='alex_dog_or_65000_iter_65000',type=str)
    parser.add_argument('--test_prototxt', dest='test_prototxt', help='test prototxt file',
                        default='test_alex_or', type=str)
    parser.add_argument('--mean_file', dest='mean_file', help='mean_file to be loaded',
                        default='ilsvrc_2012_mean.npy', type=str)
    args = parser.parse_args()
    return args


if __name__ == '__main__':

    # parameter preparation

    args = parse_args()
    work_dir = '/home/minshang/fine_grained'
    prototxt_file = os.path.join(work_dir,'prototxt',dataset_dict[args.dataset],args.model_type,args.test_prototxt+'.prototxt')
    model_file = os.path.join(work_dir,'output',dataset_dict[args.dataset],'snapshot',args.test_model+'.caffemodel')
    image_dir = os.path.join(work_dir,'dataset',dataset_dict[args.dataset],'images')
    image_list = os.path.join(work_dir,'dataset',dataset_dict[args.dataset],'test_'+args.dataset+'.txt')
    mean_file = os.path.join(work_dir,'model',args.mean_file)
    # save the prediction results format:image_name\ttrue_label 1\tclass1 score1
    save_file1 = os.path.join(work_dir,'output',dataset_dict[args.dataset],'test',args.test_model+'_top10.txt')
    save_file2 = os.path.join(work_dir, 'output', dataset_dict[args.dataset], 'test', args.test_model + '_score.txt')

    caffe.set_mode_gpu()
    caffe.set_device(args.gpu_id)
    mean_value=np.load(mean_file).mean(1).mean(1)

    net = caffe.Classifier(prototxt_file, model_file,
                mean=mean_value,
                channel_swap=(2,1,0),
                raw_scale=255,
                image_dims=(256, 256))
    sum = 0
    with open(save_file1,'w') as fout1:
        with open(save_file2,'w') as fout2:
            with open(image_list) as fid:
                for idx,line in enumerate (fid):
                    str = line.strip().split(' ')
                    image_related_name = str[0].strip()
                    true_label = int(str[1].strip())
                    image_file=os.path.join(image_dir,image_related_name)
                    start= time.time()
                    img = caffe.io.load_image(image_file)
                    prediction = net.predict([img])
          #          print type(prediction)
          #          print prediction.shape
                    top_k = prediction[0].flatten().argsort()[-1:-11:-1]
                    if((prediction[0].argmax())==true_label):
                        sum =sum+1
                    print("image %d: %0.2f s, predicted class: %d %d" %(idx,(time.time()-start),prediction[0].argmax(),true_label))
                    print >>fout1, '%s %d %d %d %d %d %d %d %d %d %d\n'%(image_related_name,top_k[0],top_k[1],top_k[2],
                                                                         top_k[3],top_k[4],top_k[5],top_k[6],top_k[7],
                                                                         top_k[8],top_k[9]),
                    #print >>fout2, '%s'%(image_related_name),
                    #print >>fout2, prediction
                   # fout2.write(prediction)
                    #print >>fout2, ' %0.4f'%(prediction[:,i] for i in range(120))
                    for i in range(prediction.shape[1]):
                       print >>fout2, '%0.4f '%(prediction[:,i]),
                    print >>fout2, '\n',

    print sum


