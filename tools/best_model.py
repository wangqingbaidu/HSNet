# -*- coding: UTF-8 -*- 
'''
Authorized  by Vlon Jang
Created on Nov 14, 2016
Blog: www.wangqingbaidu.cn
Email: wangqingbaidu@gmail.com
From Institute of Computing Technology
©2015-2016 All Rights Reserved.
'''
import os, sys
import argparse
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Get best model")
    parser.add_argument("dir", help='Directory of weights file.', default='.')
    parser.add_argument('-cfg', help='Configuration of network.', default='./cfg/darknet_ori.cfg')
    parser.add_argument('-dataset', help='Dataset of network.', default='pnet.dataset')
    parser.add_argument('-output', help='Output file of validation.', default='validation.log')
    
    args = parser.parse_args()
    
    weights_list = []
    cfgfile = args.cfg
    if os.path.exists(args.dir):
        weights_list = os.system('ls ' + os.path.join(args.dir, '*.weights'))
    
    if len(weights_list) == 0:
        print 'No weights file!'
        exit()
    else:
        print weights_list
        result = []
        for w in weights_list:
            print './darknet -i 1 classifier valid %s %s %s' %(args.dataset, cfgfile, w)
            os.system('./darknet -i 1 classifier valid %s %s %s' %(args.dataset, cfgfile, w))
            f = open('valid.log').readlines()
            acc = float(f[-2].split()[-1])
            eff = float(f[-1].split()[-2])
            result.append((w, acc, eff))
                          
        f = open(args.output, 'w')
        result = sorted(result, key=lambda x:x[1], reverse=1)
        for item in result:
            f.write('weights: %s acc:%f eff:%f\n' %(item[0], item[1], item[2]))
        
        f.close()
