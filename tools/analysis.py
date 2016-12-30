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
    
    final_valid = open('final_valid.log', 'w')
    error_num = 0
    error_num_per_class = {}
    for i in open(filename).readlines():
        items = i.split() 
        try:
            true_label = int(items[3][:-1])
            pred_label = int(items[7])
            if (true_label == 5 and pred_label < 5) or (true_label < 5 and pred_label == 5):
                error_num += 1
                if error_num_per_class.has_key(true_label):
                    error_num_per_class[true_label] += 1
                else:
                    error_num_per_class[true_label] = 1
                
                final_valid.write(i)
                print i[:-1]
        except:
            pass
        
    final_valid.close()
    valid_num = len(open('pnet_valid.list').readlines())
    
    error_num_per_class = sorted(error_num_per_class.iteritems(), key = lambda k:k[1], reverse = True)
    for c, num in error_num_per_class:
        print 'Class', c, "Rate", num / float(error_num)
    print 'acc = ',1 - error_num / float(valid_num)