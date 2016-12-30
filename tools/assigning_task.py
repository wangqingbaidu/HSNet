# -*- coding: UTF-8 -*- 
'''
Authorized  by Vlon Jang
Created on Nov 15, 2016
Blog: www.wangqingbaidu.cn
Email: wangqingbaidu@gmail.com
From Institute of Computing Technology
©2015-2016 All Rights Reserved.
'''
import sys, os, argparse
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='assign task')
    parser.add_argument('path')
    parser.add_argument('-n', type = int, default = 9)
    args = parser.parse_args()
    
    if not os.path.exists(args.path):
        print 'path %s not exist!' %args.path
        
    num = args.n
    path = args.path
    f_list = os.listdir(path)
    
    num_per_person = len(f_list) / num
    for i in range(num):
        os.system('mkdir porn_%d' %i)
        
    count = 0
    for f in f_list:
        os.system('cp %s/%s porn_%d' %(path, f, count/num_per_person))
        count += 1
    
    print 'CP %d done!' %count