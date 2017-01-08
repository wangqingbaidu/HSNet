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
import argparse

def color_print(text, color=None):
    try:
        from termcolor import cprint
        cprint(text, color=color)
    except:
        print text
        
class PrepareData:
    def __init__(self,args = None, pic_format = ['bmp', 'jpg', 'jpeg', 'png']):     
        self.args = args
        self.load_balance = args.balance
        self.debug = args.debug
        self.unique_data = set(args.ulabel)
        self.valid_ratio = args.ratio
        
        self.pic_format = pic_format
        self.train_list_save_name = args.train_list_save_name
        self.valid_list_save_name = args.valid_list_save_name
        self.labels_list_save_name = args.labels_list_save_name
        self.dataset_name = args.name + '.dataset'
        
        self.__valid_settings()
        
        self.train_list = {}
        self.label_prog = {}
        self.dir2expand = []
        
        for k in self.data:
            self.dir2expand += self.data[k]
            self.train_list[k] = []
            self.label_prog[k] = set()
    
    def go(self):
        self.__find_file()
        self.__saveList()
        
    def __shuffle_train_data(self):
        for l in self.train_list:
            random.shuffle(self.train_list[l])
        
    def __balance_samples(self):
        min_samples = maxint
        for l in self.train_list:
            if len(self.train_list[l]) < min_samples:
                min_samples = len(self.train_list[l])
            if self.debug:
                print 'label:', l, 'contains', len(self.train_list[l]), 'files' 
        for l in self.train_list:
            self.train_list[l] = self.train_list[l][:min_samples]
            
    def __balance_negative(self, label = 'nfuck'):
        min_samples = 0
        for l in self.train_list:
            if l == label:
                continue
            min_samples += len(self.train_list[l])
        
        assert label in self.train_labels
        self.train_list[label] = self.train_list[label][:min_samples]
    
    def __get_file_label(self, fname = None):
        rct = None
        for l in self.label:
            if l in fname:
                if rct == None:
                    rct = l
                else:
                    color_print("Warning: %s ignored. Contains more than one label." %fname, 'yellow')
                    return None
        if rct == None:
            color_print("Warning: %s ignored. Does't contain any label." %fname, 'yellow')
        
        return rct
        
    
    def __find_file(self):
        cwd = os.getcwd()
        while len(self.dir2expand):
            td_dir = self.dir2expand.pop(0)
            files = os.listdir(td_dir)
            for f in files:
                fname = os.path.join(td_dir, f)
                if os.path.isfile(fname):
                    sufix = os.path.splitext(f)[1][1:]
                    if sufix in self.pic_format:
                        label = self.__get_file_label(fname)
                        if label == None:
                            continue
                        
                        if label in self.unique_data: 
                            progID = f.split('-')[-3]
                            if not progID in self.label_prog[label]:
                                self.train_list[label].append(fname)
                                self.label_prog[label].add(progID)
                            else:
                                continue
                        else:
                            self.train_list[label].append(os.path.join(cwd, fname))
                            
                        if self.debug:
                            print 'file:', fname, 'label:', label
                            
                elif os.path.isdir(fname):
                    self.dir2expand.append(fname)
                    if self.debug:
                        print 'dir:', fname
    
    def __saveList(self):        
        self.__shuffle_train_data()
        if self.load_balance:
            self.__balance_samples()
            
#         self.balance_negative(self.negative_label)
        
        train_list_combine = []
        for l in self.train_list:
            train_list_combine += self.train_list[l]
            print 'label:%s\tnumber:%d' %(l, len(self.train_list[l]))
        random.shuffle(train_list_combine)
        
        train_index = int(len(train_list_combine) * (1 - self.valid_ratio))
              
        train_list_file = open(self.train_list_save_name, 'w')
        valid_list_file = open(self.valid_list_save_name, 'w')
        labels_list_file = open(self.labels_list_save_name, 'w')
        dataset_file = open(self.dataset_name, 'w')
        
        for i in train_list_combine[:train_index]:
            train_list_file.write(i + '\n')
        
        for i in train_list_combine[train_index:]:
            valid_list_file.write(i + '\n')
        
        for l in self.label:
            labels_list_file.write(l + '\n')
        
        dataset = ("labels={0}\n"+\
        "train={1}\n"+\
        "valid={2}\n"+\
        "backup=/home/mcg/darknet/train_data/backup_{3}\n"+\
        "classes=2\n" +\
        "upperbound=0.9\n"+\
        "early_stop=0\n"+\
        "console=1\n").format(self.labels_list_save_name, 
                             self.train_list_save_name, 
                             self.valid_list_save_name, 
                             self.args.name)
        dataset_file.write(dataset.replace('\t', ''))
        os.system("mkdir backup_%s" %self.args.name)
        
        train_list_file.close()
        valid_list_file.close()
        labels_list_file.close()
        dataset_file.close()        
        
        if self.valid_ratio == 1:
            os.remove(self.train_list_save_name)
        elif self.valid_ratio == 0:            
            os.remove(self.valid_list_save_name)

    def __valid_settings(self):
        if not os.path.exists(self.args.dir):
            color_print("Path %s not exists!" %self.args.dir, 'red')
            exit(0)
                    
        self.data = {}
        self.dir_map_label = {}
        self.label = self.args.label if self.args.label else os.listdir(args.dir)
        for l in self.label:
            self.data[l] = []
        
#         find dir label
        for d in os.listdir(self.args.dir):
            if os.path.isdir(os.path.join(self.args.dir, d)):
                label_count = []
                for l in self.label:
                    if l in d:
                        self.data[l].append(os.path.join(self.args.dir, d))
                        self.dir_map_label[os.path.join(self.args.dir, d)] = l
                        label_count.append(l)
                if len(label_count) > 1:
                    color_print("Dir %s can't contains more than one label. %s" %(d, label_count), 'red')
                    exit(0)
                elif len(label_count) == 0:
                    color_print("Warning: Subdir %s in %s contains no label" %(d, self.args.dir), 'yellow')
        
        if not self.data:
            color_print("No data in %s on given labels %s." %(self.args.dir, ' '.join(self.label)), 'red')
            
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate training list.')
    parser.add_argument("dir", default=None, 
                        help = "Directory of all training data.")
    parser.add_argument("-n", "--name", default='pnet', 
                        help = "Name of this training data.")
    parser.add_argument("--labels_list_save_name", default = 'pnet_labels.list',
                        help = "File name of label list.")
    parser.add_argument("--train_list_save_name", default = 'pnet_train.list',
                        help = "File name of train list.")
    parser.add_argument("--valid_list_save_name", default = 'pnet_valid.list',
                        help = "File name of valid list.")
    
    parser.add_argument("-label", nargs='*', default = [], 
                        help = "Labels of training data, if None, use all sub-directories as labels.")
    parser.add_argument("-ulabel", nargs='*', default = [],
                        help = "Unique data which remove duplicate data in the given label.")
    parser.add_argument('-debug', action = 'store_true',
                        help = "Show data info.")
    parser.add_argument("-b", "--balance", action = 'store_true',
                        help = "Balance of the samples of each label.")
    parser.add_argument('-r', '--ratio', default = 0.1, type=float,
                        help = "How many samples are to be used as validation data.")
    args = parser.parse_args()
    
    p = PrepareData(args=args)
    
    p.go()
        
