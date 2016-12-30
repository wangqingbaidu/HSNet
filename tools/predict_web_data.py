# -*- coding: UTF-8 -*- 
'''
Authorized  by Vlon Jang
Created on Dec 16, 2016
Blog: www.wangqingbaidu.cn
Email: wangqingbaidu@gmail.com
From Institute of Computing Technology
©2015-2016 All Rights Reserved.
'''
import urllib, argparse, os

def get_domain(url = None):
    assert type(url) == str
    
    _, rest = urllib.splittype(url)
    domain, rest = urllib.splithost(rest)
    if domain:
        return domain
    else:
        print 'Unkonw %s' %url
        return None
    
def get_url_path_list(ip):
    if not os.path.exists(ip):
        print 'File %s not exists!' %ip
        exit()
    domain_list = []
    path_list = []
    pic_format = ['bmp', 'jpg', 'jpeg', 'png']
    ifile = open(ip).readlines()[1:]
    for item in ifile:
        try:
            url = item.split(',')[1]
            path = item.split(',')[-5][1:]
            sufix = os.path.splitext(path)[1][1:]
            if not sufix in pic_format:
                continue
            domain = get_domain(url)
            if domain and os.path.exists(path) and path:
                domain_list.append(domain)
                path_list.append(path)
        except Exception, e:
            print "Can't parser item %s! due to %s" %(item.replace('\n', ''), e)
    
    return (domain_list, path_list)
    
    
if __name__ == '__main__':    
    parser = argparse.ArgumentParser(description='Predict web data.')
    parser.add_argument('-i', default = 'db.csv')
    parser.add_argument('-o', default = 'domain_path.csv')
    parser.add_argument('-olist', default = 'pnet_test.list')
    parser.add_argument('--dataset', default='cfg/pnet_mul.dataset')
    parser.add_argument('--netcfg', default='cfg/darknet_l_shortcut_mul.cfg')
    parser.add_argument('--weights', default='weights/backup_1205_l_shortcut_mul/darknet_l_shortcut_mul_762.weights')
    
    args = parser.parse_args()
    
    if os.path.exists('predictions.log'):
        os.remove('predictions.log')
    
    print '-------------------------PHASE ONE parser file----------------------------'
    domain_list, path_list = get_url_path_list(args.i)
    assert len(domain_list) == len(path_list)
    
    domain_path_file = open(args.o, 'w')
    web_valid_file = open(args.olist, 'w')
    
    for domain, path in zip(domain_list, path_list):
        domain_path_file.write('%s,%s\n' %(domain, path))
        web_valid_file.write('%s\n' %path)
        
    domain_path_file.close()
    web_valid_file.close()
    
    print 'Parser file %s done...' %args.i

    print '-------------------------PHASE TWO predict data----------------------------'
    cmd = './darknet -i 1 classifier label %s %s %s' %(args.dataset, args.netcfg, args.weights)
    print cmd
    os.system(cmd)
    
    if not os.path.exists('predictions.log'):
        print "Didn't get predictions from %" %args.olist
        exit()
    
    predictions_list = [int(x) for x in open('predictions.log').readlines()]
    max_result = max(predictions_list)
    assert len(predictions_list) == len(domain_list)
    
    domain_hits = {}
    results_file = open('results.csv', 'w')
    for domain, path, result in zip(domain_list, path_list, predictions_list):
        results_file.write('%s,%s,%d\n'%(domain, path, result))
        if domain_hits.has_key(domain):
            domain_hits[domain]['total'] += 1
        else:
            domain_hits[domain] = {}
            domain_hits[domain]['hit'] = 0
            domain_hits[domain]['uhit'] = 0
            domain_hits[domain]['total'] = 1
    
    results_file.close()
    print 'Predict %s data done...' %args.olist
    
    print '-------------------------PHASE THREE combine results----------------------------'
    total_samples = len(predictions_list)
    
    for domain, result in zip(domain_list, predictions_list):
        if result < max_result:
            domain_hits[domain]['hit'] += 1
        else:
            domain_hits[domain]['uhit'] += 1
        
    report_file = open('report.csv', 'w')
    for domain in domain_hits.keys():
        print domain, domain_hits[domain]['hit'], domain_hits[domain]['uhit'], domain_hits[domain]['total']
        report_file.write('%s,%d,%d,%d\n' \
            %(domain, domain_hits[domain]['hit'], domain_hits[domain]['uhit'], domain_hits[domain]['total']))
        
    report_file.close()
    print 'Combine results done...'
    
    print 'All run successfully.'
    
    
    
    