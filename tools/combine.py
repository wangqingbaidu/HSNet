import os
valid_data = open("pnet_test.list").readlines()
predictions = open("predictions.log").readlines()

res = open("ce.csv", 'w')
count = 0
for v, p in zip(valid_data, predictions):
	os.system("cp %s ./p/%s.jpg" %(v[:-1], p[:-1] + '_' + str(count)))
	res.write("%s,%s" %(p[:-1], v))
	count += 1
res.close()

