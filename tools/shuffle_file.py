# -*- coding: UTF-8 -*- 
'''
Authorized  by Vlon Jang
Created on Nov 15, 2016
Blog: www.wangqingbaidu.cn
Email: wangqingbaidu@gmail.com
From Institute of Computing Technology
©2015-2016 All Rights Reserved.
'''
import sys, os
if __name__ == '__main__':
    assert len(sys.argv) == 3
    num = int(sys.argv[1])
    path = sys.argv[2]
    f_list = os.listdir(path)
    
    num_per_person = len(f_list) / num
    for i in range(num):
        os.system('mkdir porn_%d' %i)
        
    count = 0
    for f in f_list:
        os.system('cp %s/%s porn_%d' %(path, f, count/num_per_person))
        count += 1
    
    print 'CP %d done!' %count