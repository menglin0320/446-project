import json
import numpy as np
import h5py
import gc
import matplotlib.pyplot as plt
import random
class dataset(object):
    def __init__(self, json_path='/home/menglin/446_project/neuraltalk2/coco/cocotalk.json',
                h5_path = '/home/menglin/446_project/neuraltalk2/coco/cocotalk.h5'):
        self.open_json(json_path)
        self.ix_to_word = self.info['ix_to_word']
        self.vocab_size = len(self.ix_to_word.keys())
        print('vocab size is ' + str(self.vocab_size))
        self.VGG_MEAN = [103.939, 116.779, 123.68]
        self.VGG_MEAN_image = np.asarray(self.VGG_MEAN).reshape(1,3).repeat(224*224,axis=0).reshape(224,224,3)
        self.open_hdf5(h5_path)
        image_array_shape = self.hf.get('images').shape
        self.num_images = image_array_shape[0]
        self.num_channels = image_array_shape[3]
        self.max_image_size = image_array_shape[1]
        print ('get %(num_images)d images of size  %(max_image_size)d, %(max_image_size)d, %(num_channels)d'\
         %{'num_images':self.num_images,'num_channels':self.num_channels, 'max_image_size':self.max_image_size})

        seq_size = self.hf.get('labels').shape
        self.seq_length = seq_size[1]
        print('max sequence length in data is '+ str(self.seq_length))
        self.label_start_ix = np.array(self.hf.get('label_start_ix')) 
        self.label_end_ix = np.array(self.hf.get('label_end_ix'))

        self.split_ix = {}
        self.iterators = {}
        for i,img in enumerate(self.info['images']):
            split = img['split']
            if(not split in self.split_ix):
                self.split_ix[split] = []
                self.iterators[split] = 0
            self.split_ix[split].append(i)     
        for k,v in enumerate(self.split_ix):
            print('assigned %(n_img)d images to split %(k)s' %{'n_img':len(self.split_ix[v]),'k':v}) 

    def open_hdf5(self,h5_path):
        print('loading h5 file: ' + h5_path)
        self.hf = h5py.File(h5_path,'r')


    def open_json(self,json_path): 
        print('loading json file: ' + json_path)
        with open(json_path) as json_file:   
            self.info = json.load(json_file) 

    def resetIterator(self,split):
        self.iterators[split] = 1

    def getVocabSize(self):
        return self.vocab_size

    def getVocab(self):
        return self.ix_to_word

    def getSeqLength(self):
        return self.seq_length
    
    def n_images(self):
        return self.n_images

    def get_batch(self,split,batch_size,seq_per_img = 5):
        split_ix = self.split_ix[split]
        max_index = len(split_ix)
        img_batch_raw = np.zeros((batch_size,self.max_image_size,self.max_image_size,self.num_channels),dtype=np.float32)
        label_batch = np.zeros((batch_size*seq_per_img,self.seq_length))
        length_batch = np.zeros((batch_size*seq_per_img))
        wrapped = False
        infos = []
        for i in range(0,batch_size):
            ri = self.iterators[split]
            ri_next = ri + 1
            if(ri_next >= max_index):
                ri_next = 0
                wrapped = True
            self.iterators[split] = ri_next
            ix = split_ix[ri]
            img = np.array(self.hf.get('images')[ix])[:,:,[2,1,0]].astype(np.float32) - self.VGG_MEAN_image
            img_batch_raw[i] = img/256
            ix1 = self.label_start_ix[ix]-1
            ix2 = self.label_end_ix[ix]
            ncap = ix2-ix1
            if(ncap < seq_per_img):
                seq = np.array((seq_per_img,self.seq_length))
                for q in range(0,seq_per_img):
                    ixl = random.choice(range(ix1,ix2))
                    seq[q,:] =  np.array(self.hf.get('labels')[ixl,:])
                    seq_lengths = np.array(self.hf.get('label_length')[ixl])
            else:
                ixl =  random.choice(range(ix1,ix2-seq_per_img+1))
                seq = np.array(self.hf['labels'][ixl:ixl+seq_per_img,0:self.seq_length])
                seq_lengths = np.array(self.hf.get('label_length')[ixl:ixl+seq_per_img])
            il = seq_per_img*i
            label_batch[il:il + seq_per_img,:]= seq
            length_batch[il:il + seq_per_img] = seq_lengths
            info_struct = {}
            info_struct['id'] = self.info['images'][ix]['id']
            info_struct['file_path'] = self.info['images'][ix]['file_path']
            infos.append(info_struct)
        data = {}
        data['images'] = img_batch_raw
        data['labels'] = label_batch
        data['length'] = length_batch
        data['bounds'] = {'it_pos_now':self.iterators[split],'it_max' : len(split_ix),'wrapped':wrapped}
        data['infos'] = infos
        return data

    def close_hdf5(self):
        for obj in gc.get_objects():   # Browse through ALL objects
            if isinstance(obj, h5py.File):   # Just HDF5 files
                try:
                    obj.close()
                except:
                    pass # Was algety closed

    def deprocess_image(self,image):
        return np.uint8((image*256 + self.VGG_MEAN_image)[:,:,[2,1,0]])
                       
    def show_image_in_batch(self,data,idx,seq_per_img):
        image = self.deprocess_image(data['images'][idx])
        plt.imshow(image)

        for j in range(0,seq_per_img):
            setence = ''
            for i in range(0,self.seq_length):
                if(i<data['length'][idx*seq_per_img+j]):
                    setence = setence + self.ix_to_word[str(int(data['labels'][idx*seq_per_img+j,i]))] + ' '
            print setence
            