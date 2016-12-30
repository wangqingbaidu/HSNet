# -*- coding: UTF-8 -*- 
'''
Authorized  by Vlon Jang
Created on Oct 9, 2016
Blog: www.wangqingbaidu.cn
Email: wangqingbaidu@gmail.com
From Institute of Computing Technology
©2015-2016 All Rights Reserved.
'''
import os, random, sys
from sys import maxint
class PrepareData:
    def __init__(self,
                 data = None,
                 unique_data = set(),
                 negative_label = 'nfuck',
                 train_list_save_name = 'pnet_train.list',
                 valid_list_save_name = 'pnet_valid.list',
                 lables_list_save_name = 'pnet_labels.list',
                 valid_ratio = .1,
                 rename = False,
                 load_balance = False,
                 debug = False,
                 pic_format = ['bmp', 'jpg', 'jpeg', 'png']):
        
        if not data:
            print 'Train data can not be empty!'
            assert data
        elif type(data) != dict:            
            print 'Data input must be label-directory pair in type of dict!'
            assert type(data) == dict
            
        self.load_balance = load_balance
        self.debug = debug
        self.rename = rename
        self.data = data
        self.unique_data = unique_data
        self.negative_label = negative_label
        self.pic_format = pic_format
        self.train_list_save_name = train_list_save_name
        self.valid_list_save_name = valid_list_save_name
        self.lables_list_save_name = lables_list_save_name
        self.valid_ratio = valid_ratio
        
        self.train_list = {}
        self.label_prog = {}
        self.train_labels = set()
        self.dir2expand = []
        
        for k in data:
            self.dir2expand.append((data[k], k))
            if self.rename:
                self.train_labels.add('/' + k + '_')
            else:
                self.train_labels.add(k)
                
            self.train_list[k] = []
            self.label_prog[k] = set()
            
    def shuffle_train_data(self):
        for l in self.train_list:
            random.shuffle(self.train_list[l])
        
    def balance_samples(self):
        min_samples = maxint
        for l in self.train_list:
            if len(self.train_list[l]) < min_samples:
                min_samples = len(self.train_list[l])
            if self.debug:
                print 'label:', l, 'contains', len(self.train_list[l]), 'files' 
        for l in self.train_list:
            self.train_list[l] = self.train_list[l][:min_samples]
            
    def balance_negative(self, label = 'nfuck'):
        min_samples = 0
        for l in self.train_list:
            if l == label:
                continue
            min_samples += len(self.train_list[l])
        
        assert label in self.train_labels
        self.train_list[label] = self.train_list[label][:min_samples]
        
    def genList(self):
        while len(self.dir2expand):
            td_dir, label = self.dir2expand.pop(0)
            files = os.listdir(td_dir)
            for f in files:
                original_file_name = td_dir + '/' + f
                if os.path.isfile(original_file_name):
                    sufix = os.path.splitext(f)[1][1:]
                    if sufix in self.pic_format:
                        new_file_name = original_file_name
                        if not f.startswith(label + '_'):
                            new_file_name = td_dir + '/' + label + '_' + f
                            
                        if self.rename:
                            os.rename(original_file_name, new_file_name)
                        else:
                            new_file_name = original_file_name
                        
                        if label in self.unique_data: 
                            progID = new_file_name.split('-')[-3]
                            if not progID in self.label_prog[label]:
                                self.train_list[label].append(new_file_name)
                                self.label_prog[label].add(progID)
                            else:
                                continue
                        else:
                            self.train_list[label].append(new_file_name)
                            
                        if self.debug:
                            print 'file:', new_file_name, 'label:', label
                            
                elif os.path.isdir(original_file_name):
                    self.dir2expand.append((original_file_name, label))
                    if self.debug:
                        print 'dir:', original_file_name, 'label', label
    
    def saveList(self):
        if len(self.train_labels) != len(self.train_labels):
            print 'Number of train data and labels does not match!'
            assert len(self.train_labels) == len(self.train_labels)
        
        self.shuffle_train_data()
        
        if self.load_balance:
            self.balance_samples()
            
        self.balance_negative(self.negative_label)
        
        train_list_combine = []
        for l in self.train_list:
            train_list_combine += self.train_list[l]
            print 'label:', l, '; number:', len(self.train_list[l])
        random.shuffle(train_list_combine)
        
        train_index = int(len(train_list_combine) * (1 - self.valid_ratio))
              
        train_list_file = open(self.train_list_save_name, 'w')
        valid_list_file = open(self.valid_list_save_name, 'w')
        labels_list_file = open(self.lables_list_save_name, 'w')
        
        for i in train_list_combine[:train_index]:
            train_list_file.write(i + '\n')
        
        for i in train_list_combine[train_index:]:
            valid_list_file.write(i + '\n')
        
        for l in self.train_labels:
            labels_list_file.write(l + '\n')
        
        train_list_file.close()
        valid_list_file.close()
        labels_list_file.close()
        
        
        if self.valid_ratio == 1:
            os.remove(self.train_list_save_name)
        elif self.valid_ratio == 0:            
            os.remove(self.valid_list_save_name)
                        
    def go(self):
        self.genList()
        self.saveList()

if __name__ == '__main__':
#     data = {'fuck': './pimg', 'nfuck': './npimg'}
    
    data = {'koujiao': './new_porn/koujiao', 'xingjiao': './new_porn/xingjiao',
            'jj': './new_porn/jj', 'yd': './new_porn/yd',
            'qunp': './new_porn/qunp', 'nfuck': './npimg'}
    unique_data = set({'xingjiao', 'koujiao', 'yd'})
    #data = {'fuck': './pimg'}
    #data = {'fuck': './porn_train', 'nfuck': './npimg'}
    debug = False    
    if '-d' in sys.argv:
        debug = True
        sys.argv.remove('-d')
        
    if len(sys.argv) > 1:
        if (len(sys.argv) / 2 == 0):
            print 'Input parameters must be label path pair!'
            exit()
        else:
            data = {}
            label = None
            x = sys.argv[1:]
            lp = [(x[2 * i], x[2 * i + 1]) for i in range(len(x) / 2)]
            for label, path in lp:
                if (os.path.exists(path)):
                    data[label] = path
                else:
                    print 'Label %s, Path %s not exist!' %(label, path)
                    exit()

    print data
    p = PrepareData(data, unique_data= unique_data, debug=debug, valid_ratio = 0.1)
    p.go()
        
