#include "cuda_runtime.h"
#include "curand.h"
#include "cublas_v2.h"

extern "C" {
#include <stdio.h>
#include <time.h>
#include <assert.h>

#include "network.h"
#include "image.h"
#include "data.h"
#include "utils.h"
#include "parser.h"

#include "crop_layer.h"
#include "connected_layer.h"
#include "rnn_layer.h"
#include "gru_layer.h"
#include "crnn_layer.h"
#include "detection_layer.h"
#include "region_layer.h"
#include "convolutional_layer.h"
#include "activation_layer.h"
#include "maxpool_layer.h"
#include "reorg_layer.h"
#include "avgpool_layer.h"
#include "normalization_layer.h"
#include "batchnorm_layer.h"
#include "cost_layer.h"
#include "local_layer.h"
#include "softmax_layer.h"
#include "dropout_layer.h"
#include "route_layer.h"
#include "shortcut_layer.h"
#include "blas.h"
}

float * get_network_output_gpu_layer(network net, int i);
float * get_network_delta_gpu_layer(network net, int i);
float * get_network_output_gpu(network net);
float * get_network_output_layer_gpu(network net, int i);

void forward_network_gpu(network net, network_state state)
{
    state.workspace = net.workspace;
    int i;
    for(i = 0; i < net.n; ++i){
        state.index = i;
        layer l = net.layers[i];
        if(l.delta_gpu){
            fill_ongpu(l.outputs * l.batch, 0, l.delta_gpu, 1);
        }
        l.forward_gpu(l, state);
        state.input = l.output_gpu;
    }
}

void backward_network_gpu(network net, network_state state)
{
    state.workspace = net.workspace;
    int i;
    float * original_input = state.input;
    float * original_delta = state.delta;
    for(i = net.n-1; i >= 0; --i){
        state.index = i;
        layer l = net.layers[i];
        if(i == 0){
            state.input = original_input;
            state.delta = original_delta;
        }else{
            layer prev = net.layers[i-1];
            state.input = prev.output_gpu;
            state.delta = prev.delta_gpu;
        }
        l.backward_gpu(l, state);
    }
}

void update_network_gpu(network net)
{
    int i;
    int update_batch = net.batch*net.subdivisions;
    float rate = get_current_rate(net);
    for(i = 0; i < net.n; ++i){
        layer l = net.layers[i];
        l.t = get_current_batch(net);
        if(l.update_gpu){
            l.update_gpu(l, update_batch, rate, net.momentum, net.decay);
        }
    }
}


void forward_network_gpu_use_flag(network net, network_state state, int* flag, int isTrain)
{
    state.workspace = net.workspace;
    int i;
    for(i = 0; i < net.n; ++i){
        state.index = i;
        layer l = net.layers[i];
        if(l.delta_gpu){
            fill_ongpu(l.outputs * l.batch, 0, l.delta_gpu, 1);
        }
        l.forward_gpu(l, state);
        state.input = l.output_gpu;
        
        flag[i] = 1;
        if (l.type == COST)
        {
//        	float* out = (float*)calloc(net.layers[i - 1].outputs*net.layers[i - 1].batch, sizeof(float));
//          cuda_copy_array(net.layers[i - 1].output_gpu, out, net.layers[i - 1].outputs*net.layers[i - 1].batch);
        	if (net.early_stop)
        	{
        		//get upper threashold
				float upper = net.upperbound;
				if(isTrain)
				{
					float precentage = (float)(*net.seen) / net.N / 50;
					float prob_rand = 1.0 / net.nclasses;
		//        	printf("%d,%d,%f", *net.seen, net.N, precentage);				
					upper = (net.upperbound - prob_rand) * precentage + prob_rand;
					upper = upper > net.upperbound ? net.upperbound : upper;
				}
				else
				
				//if train use voting to deciside whether to stop
				//else use one sample.
				float* out = get_network_output_layer_gpu(net, i - 1);
				int outputs = net.layers[i - 1].outputs;
				int batch_size = net.batch;
				int indexes;
				
				int b;
				int early_stop_number = 0;
				float mean_prob = 0;
				for (b = 0; b < batch_size; b++)
				{
					top_k(out + outputs * b, outputs, 1, &indexes);
					if(out[indexes + outputs * b, outputs] >= upper)
					{
						early_stop_number++;
						mean_prob += out[indexes + outputs * b, outputs];
					}
				}
				
				if (net.print2console)
					if (batch_size == 1)
						printf("Cost layer AT %d with probability %.6f of type %d and threshold: %.6f", i, out[indexes], indexes, upper);
					else
						printf("Cost layer AT %d higher than threshold: %.6f with mean probability %.6f of %d samples", 
								i, upper, mean_prob / batch_size, early_stop_number);
						
				if(early_stop_number >= batch_size / 2)
				{
					if (net.print2console)
						printf("----------------------------STOP!\n");
					break;
				}
        	}
			if (i != net.n - 1)
			{
				if (net.early_stop && net.print2console)
					printf("----------------------------DOESN'T STOP!\n");
				int i_forward = i;
				//Cost layer set to be false
				flag[i_forward--] = 0;
				while(net.layers[i_forward].type != CONVOLUTIONAL)
					flag[i_forward--] = 0;
				//last fully convolutional layer set to be false
				flag[i_forward--] = 0;
				state.input = net.layers[i_forward].output_gpu;
			}
			else
			{
				if (net.early_stop && net.print2console)
					printf("----------------------------STOP!\n");
			}
        }
    }    

	if (net.early_stop && net.print2console)
	{
		printf("layer");
		int total_ignored = 0;
		for (i = 0; i < net.n; i++)
			if (!flag[i])
			{
				printf(" %d", i);
				total_ignored++;
			}
		if (total_ignored)
			printf(" is ignored!\n");
		else
			printf("None is ignored!\n");
	}
}

void backward_network_gpu_use_flag(network net, network_state state, int* flag)
{
    state.workspace = net.workspace;
    int i;
    float * original_input = state.input;
    float * original_delta = state.delta;
    int last_layer, first_layer;
    for (i = net.n - 1; i >= 0; i--)
    	if (flag[i]) break;
    last_layer = i;
    
    for(i = last_layer; i > 0; i--)
    	if (!flag[i - 1]) break;
    first_layer = i;
    
    printf("Backward and Update layer:");
    for(i = last_layer; i >= first_layer; --i){    
    	printf(" %d", i);
    	
        state.index = i;
        layer l = net.layers[i];
        if(i == 0){
            state.input = original_input;
            state.delta = original_delta;
        }else{
            layer prev = net.layers[i-1];
            state.input = prev.output_gpu;
            state.delta = prev.delta_gpu;
        }
        l.backward_gpu(l, state);
    }
    printf("\n");
}

void update_network_gpu_use_flag(network net, int* flag)
{
    int i;
    int update_batch = net.batch*net.subdivisions;
    float rate = get_current_rate(net);
    
    int last_layer, first_layer;
    for (i = net.n - 1; i >= 0; i--)
    	if (flag[i]) break;
    last_layer = i;
    
    for(i = last_layer; i > 0; i--)
    	if (!flag[i - 1]) break;
    first_layer = i;
    
    for(i = first_layer; i < last_layer; ++i){    		
        layer l = net.layers[i];
        l.t = get_current_batch(net);
        if(l.update_gpu){
            l.update_gpu(l, update_batch, rate, net.momentum, net.decay);
        }
    }
}

void forward_backward_network_gpu(network net, float *x, float *y)
{
    network_state state;
    state.index = 0;
    state.net = net;
    int x_size = get_network_input_size(net)*net.batch;
    int y_size = get_network_output_size(net)*net.batch;
    if(net.layers[net.n-1].truths) y_size = net.layers[net.n-1].truths*net.batch;
    if(!*net.input_gpu){
        *net.input_gpu = cuda_make_array(x, x_size);
        *net.truth_gpu = cuda_make_array(y, y_size);
    }else{
        cuda_push_array(*net.input_gpu, x, x_size);
        cuda_push_array(*net.truth_gpu, y, y_size);
    }
    state.input = *net.input_gpu;
    state.delta = 0;
    state.truth = *net.truth_gpu;
    state.train = 1;
    forward_network_gpu(net, state);
    backward_network_gpu(net, state);
}

void forward_backward_network_gpu_use_flag(network net, float *x, float *y, int* flag)
{
    network_state state;
    state.index = 0;
    state.net = net;
    int x_size = get_network_input_size(net)*net.batch;
    int y_size = get_network_output_size(net)*net.batch;
    if(net.layers[net.n-1].truths) y_size = net.layers[net.n-1].truths*net.batch;
    if(!*net.input_gpu){
        *net.input_gpu = cuda_make_array(x, x_size);
        *net.truth_gpu = cuda_make_array(y, y_size);
    }else{
        cuda_push_array(*net.input_gpu, x, x_size);
        cuda_push_array(*net.truth_gpu, y, y_size);
    }
    state.input = *net.input_gpu;
    state.delta = 0;
    state.truth = *net.truth_gpu;
    state.train = 1;
    forward_network_gpu_use_flag(net, state, flag, 1);
    backward_network_gpu_use_flag(net, state, flag);
}


float train_network_datum_gpu(network net, float *x, float *y)
{
	int* flag = (int*)calloc(net.n, sizeof(int));
    *net.seen += net.batch;
    forward_backward_network_gpu_use_flag(net, x, y, flag);
    float error = get_network_cost(net);
    if (((*net.seen) / net.batch) % net.subdivisions == 0) update_network_gpu_use_flag(net, flag);
	
	free(flag);
    return error;
}

typedef struct {
    network net;
    data d;
    float *err;
} train_args;

void *train_thread(void *ptr)
{
    train_args args = *(train_args*)ptr;
    free(ptr);
    cuda_set_device(args.net.gpu_index);
    *args.err = train_network(args.net, args.d);
    return 0;
}

pthread_t train_network_in_thread(network net, data d, float *err)
{
    pthread_t thread;
    train_args *ptr = (train_args *)calloc(1, sizeof(train_args));
    ptr->net = net;
    ptr->d = d;
    ptr->err = err;
    if(pthread_create(&thread, 0, train_thread, ptr)) error("Thread creation failed");
    return thread;
}

void pull_updates(layer l)
{
    if(l.type == CONVOLUTIONAL){
        cuda_pull_array(l.bias_updates_gpu, l.bias_updates, l.n);
        cuda_pull_array(l.weight_updates_gpu, l.weight_updates, l.n*l.size*l.size*l.c);
        if(l.scale_updates) cuda_pull_array(l.scale_updates_gpu, l.scale_updates, l.n);
    } else if(l.type == CONNECTED){
        cuda_pull_array(l.bias_updates_gpu, l.bias_updates, l.outputs);
        cuda_pull_array(l.weight_updates_gpu, l.weight_updates, l.outputs*l.inputs);
    }
}

void push_updates(layer l)
{
    if(l.type == CONVOLUTIONAL){
        cuda_push_array(l.bias_updates_gpu, l.bias_updates, l.n);
        cuda_push_array(l.weight_updates_gpu, l.weight_updates, l.n*l.size*l.size*l.c);
        if(l.scale_updates) cuda_push_array(l.scale_updates_gpu, l.scale_updates, l.n);
    } else if(l.type == CONNECTED){
        cuda_push_array(l.bias_updates_gpu, l.bias_updates, l.outputs);
        cuda_push_array(l.weight_updates_gpu, l.weight_updates, l.outputs*l.inputs);
    }
}

void update_layer(layer l, network net)
{
    int update_batch = net.batch*net.subdivisions;
    float rate = get_current_rate(net);
    l.t = get_current_batch(net);
    if(l.update_gpu){
        l.update_gpu(l, update_batch, rate, net.momentum, net.decay);
    }
}

void merge_weights(layer l, layer base)
{
    if (l.type == CONVOLUTIONAL) {
        axpy_cpu(l.n, 1, l.biases, 1, base.biases, 1);
        axpy_cpu(l.n*l.size*l.size*l.c, 1, l.weights, 1, base.weights, 1);
        if (l.scales) {
            axpy_cpu(l.n, 1, l.scales, 1, base.scales, 1);
        }
    } else if(l.type == CONNECTED) {
        axpy_cpu(l.outputs, 1, l.biases, 1, base.biases, 1);
        axpy_cpu(l.outputs*l.inputs, 1, l.weights, 1, base.weights, 1);
    }
}

void scale_weights(layer l, float s)
{
    if (l.type == CONVOLUTIONAL) {
        scal_cpu(l.n, s, l.biases, 1);
        scal_cpu(l.n*l.size*l.size*l.c, s, l.weights, 1);
        if (l.scales) {
            scal_cpu(l.n, s, l.scales, 1);
        }
    } else if(l.type == CONNECTED) {
        scal_cpu(l.outputs, s, l.biases, 1);
        scal_cpu(l.outputs*l.inputs, s, l.weights, 1);
    }
}


void pull_weights(layer l)
{
    if(l.type == CONVOLUTIONAL){
        cuda_pull_array(l.biases_gpu, l.biases, l.n);
        cuda_pull_array(l.weights_gpu, l.weights, l.n*l.size*l.size*l.c);
        if(l.scales) cuda_pull_array(l.scales_gpu, l.scales, l.n);
    } else if(l.type == CONNECTED){
        cuda_pull_array(l.biases_gpu, l.biases, l.outputs);
        cuda_pull_array(l.weights_gpu, l.weights, l.outputs*l.inputs);
    }
}

void push_weights(layer l)
{
    if(l.type == CONVOLUTIONAL){
        cuda_push_array(l.biases_gpu, l.biases, l.n);
        cuda_push_array(l.weights_gpu, l.weights, l.n*l.size*l.size*l.c);
        if(l.scales) cuda_push_array(l.scales_gpu, l.scales, l.n);
    } else if(l.type == CONNECTED){
        cuda_push_array(l.biases_gpu, l.biases, l.outputs);
        cuda_push_array(l.weights_gpu, l.weights, l.outputs*l.inputs);
    }
}

void distribute_weights(layer l, layer base)
{
    if(l.type == CONVOLUTIONAL){
        cuda_push_array(l.biases_gpu, base.biases, l.n);
        cuda_push_array(l.weights_gpu, base.weights, l.n*l.size*l.size*l.c);
        if(base.scales) cuda_push_array(l.scales_gpu, base.scales, l.n);
    } else if(l.type == CONNECTED){
        cuda_push_array(l.biases_gpu, base.biases, l.outputs);
        cuda_push_array(l.weights_gpu, base.weights, l.outputs*l.inputs);
    }
}


void merge_updates(layer l, layer base)
{
    if (l.type == CONVOLUTIONAL) {
        axpy_cpu(l.n, 1, l.bias_updates, 1, base.bias_updates, 1);
        axpy_cpu(l.n*l.size*l.size*l.c, 1, l.weight_updates, 1, base.weight_updates, 1);
        if (l.scale_updates) {
            axpy_cpu(l.n, 1, l.scale_updates, 1, base.scale_updates, 1);
        }
    } else if(l.type == CONNECTED) {
        axpy_cpu(l.outputs, 1, l.bias_updates, 1, base.bias_updates, 1);
        axpy_cpu(l.outputs*l.inputs, 1, l.weight_updates, 1, base.weight_updates, 1);
    }
}

void distribute_updates(layer l, layer base)
{
    if(l.type == CONVOLUTIONAL){
        cuda_push_array(l.bias_updates_gpu, base.bias_updates, l.n);
        cuda_push_array(l.weight_updates_gpu, base.weight_updates, l.n*l.size*l.size*l.c);
        if(base.scale_updates) cuda_push_array(l.scale_updates_gpu, base.scale_updates, l.n);
    } else if(l.type == CONNECTED){
        cuda_push_array(l.bias_updates_gpu, base.bias_updates, l.outputs);
        cuda_push_array(l.weight_updates_gpu, base.weight_updates, l.outputs*l.inputs);
    }
}

void sync_layer(network *nets, int n, int j)
{
    //printf("Syncing layer %d\n", j);
    int i;
    network net = nets[0];
    layer base = net.layers[j];
    cuda_set_device(net.gpu_index);
    pull_weights(base);
    for (i = 1; i < n; ++i) {
        cuda_set_device(nets[i].gpu_index);
        layer l = nets[i].layers[j];
        pull_weights(l);
        merge_weights(l, base);
    }
    scale_weights(base, 1./n);
    for (i = 0; i < n; ++i) {
        cuda_set_device(nets[i].gpu_index);
        layer l = nets[i].layers[j];
        distribute_weights(l, base);
    }
    //printf("Done syncing layer %d\n", j);
}

typedef struct{
    network *nets;
    int n;
    int j;
} sync_args;

void *sync_layer_thread(void *ptr)
{
    sync_args args = *(sync_args*)ptr;
    sync_layer(args.nets, args.n, args.j);
    free(ptr);
    return 0;
}

pthread_t sync_layer_in_thread(network *nets, int n, int j)
{
    pthread_t thread;
    sync_args *ptr = (sync_args *)calloc(1, sizeof(sync_args));
    ptr->nets = nets;
    ptr->n = n;
    ptr->j = j;
    if(pthread_create(&thread, 0, sync_layer_thread, ptr)) error("Thread creation failed");
    return thread;
}

void sync_nets(network *nets, int n, int interval)
{
    int j;
    int layers = nets[0].n;
    pthread_t *threads = (pthread_t *) calloc(layers, sizeof(pthread_t));

    *nets[0].seen += interval * (n-1) * nets[0].batch * nets[0].subdivisions;
    for (j = 0; j < n; ++j){
        *nets[j].seen = *nets[0].seen;
    }
    for (j = 0; j < layers; ++j) {
        threads[j] = sync_layer_in_thread(nets, n, j);
    }
    for (j = 0; j < layers; ++j) {
        pthread_join(threads[j], 0);
    }
    free(threads);
}

float train_networks(network *nets, int n, data d, int interval)
{
    int i;
    int batch = nets[0].batch;
    int subdivisions = nets[0].subdivisions;
    assert(batch * subdivisions * n == d.X.rows);
    pthread_t *threads = (pthread_t *) calloc(n, sizeof(pthread_t));
    float *errors = (float *) calloc(n, sizeof(float));

    float sum = 0;
    for(i = 0; i < n; ++i){
        data p = get_data_part(d, i, n);
        threads[i] = train_network_in_thread(nets[i], p, errors + i);
    }
    for(i = 0; i < n; ++i){
        pthread_join(threads[i], 0);
        //printf("%f\n", errors[i]);
        sum += errors[i];
    }
    //cudaDeviceSynchronize();
    if (get_current_batch(nets[0]) % interval == 0) {
        printf("Syncing... ");
        fflush(stdout);
        sync_nets(nets, n, interval);
        printf("Done!\n");
    }
    //cudaDeviceSynchronize();
    free(threads);
    free(errors);
    return (float)sum/(n);
}

float *get_network_output_layer_gpu(network net, int i)
{
    layer l = net.layers[i];
    cuda_pull_array(l.output_gpu, l.output, l.outputs*l.batch);
    return l.output;
}

float *get_network_output_gpu(network net)
{
    int i;
    for(i = net.n-1; i > 0; --i) if(net.layers[i].type != COST) break;
    return get_network_output_layer_gpu(net, i);
}

float *get_network_output_gpu_from_index(network net, int index)
{
    int i;
    for(i = index; i > 0; --i) if(net.layers[i].type != COST) break;
    return get_network_output_layer_gpu(net, i);
}

float *network_predict_gpu(network net, float *input)
{
    cuda_set_device(net.gpu_index);
    int size = get_network_input_size(net) * net.batch;
    int i, lindex = 0;
    network_state state;
    state.index = 0;
    state.net = net;
    state.input = cuda_make_array(input, size);
    state.truth = 0;
    state.train = 0;
    state.delta = 0;
    int* flag = (int*)calloc(net.n, sizeof(int));
    forward_network_gpu_use_flag(net, state, flag, 0);
    for (i = 0; i < net.n; i++)
    	if (flag[i]) lindex = i;
    float *out = get_network_output_gpu_from_index(net, lindex - 1);
    cuda_free(state.input);
    free(flag);
    return out;
}

