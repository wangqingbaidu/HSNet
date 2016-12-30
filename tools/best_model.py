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
if __name__ == '__main__':
    weights_list = []
    cfgfile = 'cfg/darknet_es_shortcut_22.cfg'
    cfgfile = './cfg/darknet_ori_shortcut3.cfg'
    cfgfile = './cfg/darknet_ori_shortcut_mul.cfg'
    cfgfile = './cfg/darknet_l_shortcut_mul.cfg'
    #cfgfile = './cfg/darknet_ori_shortcut_sig.cfg'
    if len(sys.argv) == 1:
        flist = os.listdir('.')
        for f in flist:
            if '.weights' in f:
                weights_list.append(f)
    else:
        if '-d' in sys.argv:
            sys.argv = sys.argv[1:]
            dirpath = sys.argv[sys.argv.index('-d') + 1]
            sys.argv.remove('-d')
            sys.argv.remove(dirpath)
            for i in sys.argv:
                weights_list.append(i)
            
            for i in os.listdir(dirpath):
                if '.weights' in i:
                    weights_list.append(dirpath + '/' + i)
    
    if len(weights_list) == 0:
        print 'No weights file!'
        exit()
    else:
        print weights_list
        result = []
        for w in weights_list:
            print './darknet -i 1 classifier valid cfg/pnet.dataset %s %s' %(cfgfile, w)
            os.system('./darknet -i 1 classifier valid cfg/pnet_mul.dataset %s %s' %(cfgfile, w))
            f = open('valid.log').readlines()
            acc = float(f[-2].split()[-1])
            eff = float(f[-1].split()[-2])
            result.append((w, acc, eff))
                          
        f = open('validation_sig.log', 'w')
        result = sorted(result, key=lambda x:x[1], reverse=1)
        for item in result:
            f.write('weights: %s acc:%f eff:%f\n' %(item[0], item[1], item[2]))
        
        f.close()
