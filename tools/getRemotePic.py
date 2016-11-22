# -*- coding: UTF-8 -*- 
'''
Authorized  by vlon Jang
Created on Nov 7, 2016
Email:zhangzhiwei@ict.ac.cn
From Institute of Computing Technology
All Rights Reserved.
'''
import sys, os
data_dir = '/home/mcg/darknet/train_data/'
def getCfgFile(model = None):
    cfgfile = './pnet_valid.list'
    if not os.path.exists(cfgfile):
        if model != 'l':
            os.system('scp mcg@10.25.0.118:' + data_dir + cfgfile + ' ./')
        else:
            os.system('cp ' + data_dir + cfgfile + ' ./')
            
    
def paraseDataList(filterList=None, model = None):
    picList = []
    count = 0
    getCfgFile(model)
    cfgfile = './pnet_valid.list'
    f = open(cfgfile).readlines()
    for item in f:
        if str(count) in filterList:
            picList.append(data_dir + item.replace('\n', ''))
        count += 1
#     os.remove(cfgfile)
    return picList

def getPics(c = 'cp ', picList = None):
    assert picList
    for pic in picList:
        cmd = c + pic
        cmd += ' ./'
        print cmd 
        os.system(cmd)
    
if __name__ == '__main__':
    remote = True
    if len(sys.argv) > 1 and sys.argv[1] == 'l':
        getPics('cp ', paraseDataList(sys.argv[2:], 'l'))
    else:
        getPics('scp mcg@10.25.0.118:', paraseDataList(sys.argv[1:]))
        
