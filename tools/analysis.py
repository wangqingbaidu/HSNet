# -*- coding: UTF-8 -*- 
'''
Authorized  by vlon Jang
Created on Nov 22, 2016
Email:zhangzhiwei@ict.ac.cn
From Institute of Computing Technology
All Rights Reserved.
'''
import os, sys

if __name__ == '__main__':
    filename = 'valid.log'
    if len(sys.argv) > 1:
        filename = sys.argv[-1]
    
    if not os.path.exists(filename):
        print 'file %s not exist!' %filename
        exit()
    
    error_num = 0
    for i in open(filename).readlines():
        items = i.split() 
        try:
            true_label = int(items[3][:-1])
            pred_label = int(items[7])
            if (true_label == 5 and pred_label < 5) or (true_label < 5 and pred_label == 5):
                error_num += 1
                print i[:-1]
        except:
            pass
    
    valid_num = len(open('pnet_valid.list').readlines())
    
    print 'acc = ',1 - error_num / float(valid_num)