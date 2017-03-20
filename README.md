# RedNet
基于[darknet](http://pjreddie.com/darknet/)源码修改的网络，干什么的，不告诉你~~


目录结构:
"""python
RedNet
|___forward_src 			前馈网络模型代码
	|___pnet.c				动态库入口文件，其中封装了模型加载、预测等相关接口
	
|___src						RedNet网络核心代码，包括模型训练以及预测
	
|___tools					相关脚本
	|___analysis.py			用于多分类然后融合判断正确率的脚本。
	|___assigning_task.py	分配任务脚本。
	|___best_model.py		RedNet本身不支持训练时候保存最好的模型，所以这个脚本用来从模型中选择最好的模型。
	|___get_list.py			生成训练数据以及测试数据集脚本。
	|___predict_web_data.py	去年郭老师的预测网络数据代码，其中包括预测已经生成相应的报告。
	
	
|___cfg			 			存储网络参数相关配置以及对应的dataset
	|___pnet_list			最近一批数据生成的训练以及测试集。
	|___darknet_S_SC_C.cfg 	其中S对应的是模型的大小，es非常小、sm小、ori原始、l大，SC对应有无shortcut，C代表分类个数
	|___*.dataset			RedNet网络参数配置，包括训练集、测试集list，分类中心，半停阈值等。
	
|___Makefile				生成可以训练以及预测的RedNet可执行程序

|___makelib					生成动态链接库

|___trainNet.sh				训练网络的启动脚本
	
"""
tips:
1. 所有的脚本通过-h可以输出参数使用说明。

2. Make文件中EXEC=../train_data/darknet用于执行生成的文件目录。

3. 生成的可执行程序分类任务使用classifier，训练使用train，验证使用valid，预测使用label。

	例如训练网络使用./darknet -i 0 classifier train pnet.dataset ./cfg/darknet_es_shortcut_22.cfg
