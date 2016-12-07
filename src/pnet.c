#include "network.h"
#include "utils.h"
#include "parser.h"
#include "option_list.h"
#include "blas.h"
#include "assert.h"
#include "classifier.h"
#include "cuda.h"
#include <sys/time.h>
#define max(x,y)  ( x>y?x:y )
#define min(x,y)  ( x<y?x:y )
#ifdef OPENCV
#include "opencv2/highgui/highgui_c.h"
#endif

list *read_data_cfg(char *filename)
{
    FILE *file = fopen(filename, "r");
    if(file == 0) file_error(filename);
    char *line;
    int nu = 0;
    list *options = make_list();
    while((line=fgetl(file)) != 0){
        ++ nu;
        strip(line);
        switch(line[0]){
            case '\0':
            case '#':
            case ';':
                free(line);
                break;
            default:
                if(!read_option(line, options)){
                    fprintf(stderr, "Config file error line %d, could parse: %s\n", nu, line);
                    free(line);
                }
                break;
        }
    }
    fclose(file);
    return options;
}

float *get_regression_values(char **labels, int n)
{
    float *v = calloc(n, sizeof(float));
    int i;
    for(i = 0; i < n; ++i){
        char *p = strchr(labels[i], ' ');
        *p = 0;
        v[i] = atof(p+1);
    }
    return v;
}

void train_classifier_multi(char *datacfg, char *cfgfile, char *weightfile, int *gpus, int ngpus, int clear)
{

}


void train_classifier(char *datacfg, char *cfgfile, char *weightfile, int clear)
{
    int nthreads = 8;
    int i;

//    data_seed = time(0);
    srand(time(0));
    float avg_loss = -1;
    char *base = basecfg(cfgfile);
    printf("%s\n", base);
    network net = parse_network_cfg(cfgfile);
    if(weightfile){
        load_weights(&net, weightfile);
    }
    if(clear) *net.seen = 0;
    printf("Learning Rate: %g, Momentum: %g, Decay: %g\n", net.learning_rate, net.momentum, net.decay);
    int imgs = net.batch*net.subdivisions/nthreads;
    assert(net.batch*net.subdivisions % nthreads == 0);

    list *options = read_data_cfg(datacfg);

    char *backup_directory = option_find_str(options, "backup", "/backup/");
    char *label_list = option_find_str(options, "labels", "data/labels.list");
    char *train_list = option_find_str(options, "train", "data/train.list");
    int classes = option_find_int(options, "classes", 2);

    char **labels = get_labels(label_list);
    list *plist = get_paths(train_list);
    char **paths = (char **)list_to_array(plist);
    printf("%d\n", plist->size);
    int N = plist->size;
    clock_t time;

    pthread_t *load_threads = calloc(nthreads, sizeof(pthread_t));
    data *trains  = calloc(nthreads, sizeof(data));
    data *buffers = calloc(nthreads, sizeof(data));

    load_args args = {0};
    args.w = net.w;
    args.h = net.h;

    args.min = net.min_crop;
    args.max = net.max_crop;
    args.angle = net.angle;
    args.aspect = net.aspect;
    args.exposure = net.exposure;
    args.saturation = net.saturation;
    args.hue = net.hue;
    args.size = net.w;

    args.paths = paths;
    args.classes = classes;
    args.n = imgs;
    args.m = N;
    args.labels = labels;
    args.type = CLASSIFICATION_DATA;

    for(i = 0; i < nthreads; ++i){
        args.d = buffers + i;
        load_threads[i] = load_data_in_thread(args);
    }

    int epoch = (*net.seen)/N;
    while(get_current_batch(net) < net.max_batches || net.max_batches == 0){
        time=clock();
        for(i = 0; i < nthreads; ++i){
            pthread_join(load_threads[i], 0);
            trains[i] = buffers[i];
        }
        data train = concat_datas(trains, nthreads);

        for(i = 0; i < nthreads; ++i){
            args.d = buffers + i;
            load_threads[i] = load_data_in_thread(args);
        }

        printf("Loaded: %lf seconds\n", sec(clock()-time));
        time=clock();

#ifdef OPENCV
        if(0){
            int u;
            for(u = 0; u < imgs; ++u){
                image im = float_to_image(net.w, net.h, 3, train.X.vals[u]);
                show_image(im, "loaded");
                cvWaitKey(0);
            }
        }
#endif

        float loss = train_network(net, train);
        if(avg_loss == -1) avg_loss = loss;
        avg_loss = avg_loss*.9 + loss*.1;
        printf("%d, %.3f: %f, %f avg, %f rate, %lf seconds, %d images\n", get_current_batch(net), (float)(*net.seen)/N, loss, avg_loss, get_current_rate(net), sec(clock()-time), *net.seen);
        free_data(train);
        for(i = 0; i < nthreads; ++i){
            free_data(trains[i]);
        }
        if(*net.seen/N > epoch){
            epoch = *net.seen/N;
            char buff[256];
            sprintf(buff, "%s/%s_%d.weights",backup_directory,base, epoch);
            save_weights(net, buff);
        }
        if(get_current_batch(net)%100 == 0){
            char buff[256];
            sprintf(buff, "%s/%s.backup",backup_directory,base);
            save_weights(net, buff);
        }
    }
    char buff[256];
    sprintf(buff, "%s/%s.weights", backup_directory, base);
    save_weights(net, buff);

    for(i = 0; i < nthreads; ++i){
        pthread_join(load_threads[i], 0);
        free_data(buffers[i]);
    }
    free(buffers);
    free(trains);
    free(load_threads);

    free_network(net);
    free_ptrs((void**)labels, classes);
    free_ptrs((void**)paths, plist->size);
    free_list(plist);
    free(base);
}

void validate_classifier_crop(char *datacfg, char *filename, char *weightfile)
{
    int i = 0;
    network net = parse_network_cfg(filename);
    if(weightfile){
        load_weights(&net, weightfile);
    }
    srand(time(0));

    list *options = read_data_cfg(datacfg);

    char *label_list = option_find_str(options, "labels", "data/labels.list");
    char *valid_list = option_find_str(options, "valid", "data/train.list");
    int classes = option_find_int(options, "classes", 2);
    int topk = option_find_int(options, "top", 1);

    char **labels = get_labels(label_list);
    list *plist = get_paths(valid_list);

    char **paths = (char **)list_to_array(plist);
    int m = plist->size;
    free_list(plist);

    clock_t time;
    float avg_acc = 0;
    float avg_topk = 0;
    int splits = m/1000;
    int num = (i+1)*m/splits - i*m/splits;

    data val, buffer;

    load_args args = {0};
    args.w = net.w;
    args.h = net.h;

    args.paths = paths;
    args.classes = classes;
    args.n = num;
    args.m = 0;
    args.labels = labels;
    args.d = &buffer;
    args.type = OLD_CLASSIFICATION_DATA;

    pthread_t load_thread = load_data_in_thread(args);
    for(i = 1; i <= splits; ++i){
        time=clock();

        pthread_join(load_thread, 0);
        val = buffer;

        num = (i+1)*m/splits - i*m/splits;
        char **part = paths+(i*m/splits);
        if(i != splits){
            args.paths = part;
            load_thread = load_data_in_thread(args);
        }
        printf("Loaded: %d images in %lf seconds\n", val.X.rows, sec(clock()-time));

        time=clock();
        float *acc = network_accuracies(net, val, topk);
        avg_acc += acc[0];
        avg_topk += acc[1];
        printf("%d: top 1: %f, top %d: %f, %lf seconds, %d images\n", i, avg_acc/i, topk, avg_topk/i, sec(clock()-time), val.X.rows);
        free_data(val);
    }
}

void validate_classifier_10(char *datacfg, char *filename, char *weightfile)
{
    int i, j;
    network net = parse_network_cfg(filename);
    set_batch_network(&net, 1);
    if(weightfile){
        load_weights(&net, weightfile);
    }
    srand(time(0));

    list *options = read_data_cfg(datacfg);

    char *label_list = option_find_str(options, "labels", "data/labels.list");
    char *valid_list = option_find_str(options, "valid", "data/train.list");
    int classes = option_find_int(options, "classes", 2);
    int topk = option_find_int(options, "top", 1);

    char **labels = get_labels(label_list);
    list *plist = get_paths(valid_list);

    char **paths = (char **)list_to_array(plist);
    int m = plist->size;
    free_list(plist);

    float avg_acc = 0;
    float avg_topk = 0;
    int *indexes = calloc(topk, sizeof(int));

    for(i = 0; i < m; ++i){
        int class = -1;
        char *path = paths[i];
        for(j = 0; j < classes; ++j){
            if(strstr(path, labels[j])){
                class = j;
                break;
            }
        }
        int w = net.w;
        int h = net.h;
        int shift = 32;
        image im = load_image_color(paths[i], w+shift, h+shift);
        image images[10];
        images[0] = crop_image(im, -shift, -shift, w, h);
        images[1] = crop_image(im, shift, -shift, w, h);
        images[2] = crop_image(im, 0, 0, w, h);
        images[3] = crop_image(im, -shift, shift, w, h);
        images[4] = crop_image(im, shift, shift, w, h);
        flip_image(im);
        images[5] = crop_image(im, -shift, -shift, w, h);
        images[6] = crop_image(im, shift, -shift, w, h);
        images[7] = crop_image(im, 0, 0, w, h);
        images[8] = crop_image(im, -shift, shift, w, h);
        images[9] = crop_image(im, shift, shift, w, h);
        float *pred = calloc(classes, sizeof(float));
        for(j = 0; j < 10; ++j){
            float *p = network_predict(net, images[j].data);
            axpy_cpu(classes, 1, p, 1, pred, 1);
            free_image(images[j]);
        }
        free_image(im);
        top_k(pred, classes, topk, indexes);
        free(pred);
        if(indexes[0] == class) avg_acc += 1;
        for(j = 0; j < topk; ++j){
            if(indexes[j] == class) avg_topk += 1;
        }

        printf("%d: top 1: %f, top %d: %f\n", i, avg_acc/(i+1), topk, avg_topk/(i+1));
    }
}

void validate_classifier_full(char *datacfg, char *filename, char *weightfile)
{
    int i, j;
    network net = parse_network_cfg(filename);
    set_batch_network(&net, 1);
    if(weightfile){
        load_weights(&net, weightfile);
    }
    srand(time(0));

    list *options = read_data_cfg(datacfg);

    char *label_list = option_find_str(options, "labels", "data/labels.list");
    char *valid_list = option_find_str(options, "valid", "data/train.list");
    int classes = option_find_int(options, "classes", 2);
    int topk = option_find_int(options, "top", 1);

    char **labels = get_labels(label_list);
    list *plist = get_paths(valid_list);

    char **paths = (char **)list_to_array(plist);
    int m = plist->size;
    free_list(plist);

    float avg_acc = 0;
    float avg_topk = 0;
    int *indexes = calloc(topk, sizeof(int));

    int size = net.w;
    for(i = 0; i < m; ++i){
        int class = -1;
        char *path = paths[i];
        for(j = 0; j < classes; ++j){
            if(strstr(path, labels[j])){
                class = j;
                break;
            }
        }
        image im = load_image_color(paths[i], 0, 0);
        image resized = resize_min(im, size);
        resize_network(&net, resized.w, resized.h);
        //show_image(im, "orig");
        //show_image(crop, "cropped");
        //cvWaitKey(0);
        float *pred = network_predict(net, resized.data);

        free_image(im);
        free_image(resized);
        top_k(pred, classes, topk, indexes);

        if(indexes[0] == class) avg_acc += 1;
        for(j = 0; j < topk; ++j){
            if(indexes[j] == class) avg_topk += 1;
        }

        printf("%d: top 1: %f, top %d: %f\n", i, avg_acc/(i+1), topk, avg_topk/(i+1));
    }
}


void validate_classifier_single(char *datacfg, char *filename, char *weightfile)
{
    int i, j;
    network net = parse_network_cfg(filename);
    if(weightfile){
        load_weights(&net, weightfile);
    }
    set_batch_network(&net, 1);
    srand(time(0));

    list *options = read_data_cfg(datacfg);

    char *label_list = option_find_str(options, "labels", "data/labels.list");
    char *valid_list = option_find_str(options, "valid", "data/train.list");
    int classes = option_find_int(options, "classes", 2);
    int topk = option_find_int(options, "top", 1);

    char **labels = get_labels(label_list);
    list *plist = get_paths(valid_list);

    char **paths = (char **)list_to_array(plist);
    int m = plist->size;
    free_list(plist);

    float avg_acc = 0;
    float avg_topk = 0;
    int *indexes = calloc(topk, sizeof(int));

    for(i = 0; i < m; ++i){
        int class = -1;
        char *path = paths[i];
        for(j = 0; j < classes; ++j){
            if(strstr(path, labels[j])){
                class = j;
                break;
            }
        }
        image im = load_image_color(paths[i], 0, 0);
        image resized = resize_min(im, net.w);
        image crop = crop_image(resized, (resized.w - net.w)/2, (resized.h - net.h)/2, net.w, net.h);
        //show_image(im, "orig");
        //show_image(crop, "cropped");
        //cvWaitKey(0);
        float *pred = network_predict(net, crop.data);

        if(resized.data != im.data) free_image(resized);
        free_image(im);
        free_image(crop);
        top_k(pred, classes, topk, indexes);

        if(indexes[0] == class) avg_acc += 1;
        for(j = 0; j < topk; ++j){
            if(indexes[j] == class) avg_topk += 1;
        }

        printf("%d: top 1: %f, top %d: %f\n", i, avg_acc/(i+1), topk, avg_topk/(i+1));
    }
}

void validate_classifier_multi(char *datacfg, char *filename, char *weightfile)
{
    int i, j;
    network net = parse_network_cfg(filename);
    set_batch_network(&net, 1);
    if(weightfile){
        load_weights(&net, weightfile);
    }
    srand(time(0));

    list *options = read_data_cfg(datacfg);

    char *label_list = option_find_str(options, "labels", "data/labels.list");
    char *valid_list = option_find_str(options, "valid", "data/train.list");
    int classes = option_find_int(options, "classes", 2);
    int topk = option_find_int(options, "top", 1);

    char **labels = get_labels(label_list);
    list *plist = get_paths(valid_list);
    int scales[] = {224, 288, 320, 352, 384};
    int nscales = sizeof(scales)/sizeof(scales[0]);

    char **paths = (char **)list_to_array(plist);
    int m = plist->size;
    free_list(plist);

    float avg_acc = 0;
    float avg_topk = 0;
    int *indexes = calloc(topk, sizeof(int));

    for(i = 0; i < m; ++i){
        int class = -1;
        char *path = paths[i];
        for(j = 0; j < classes; ++j){
            if(strstr(path, labels[j])){
                class = j;
                break;
            }
        }
        float *pred = calloc(classes, sizeof(float));
        image im = load_image_color(paths[i], 0, 0);
        for(j = 0; j < nscales; ++j){
            image r = resize_min(im, scales[j]);
            resize_network(&net, r.w, r.h);
            float *p = network_predict(net, r.data);
            axpy_cpu(classes, 1, p, 1, pred, 1);
            flip_image(r);
            p = network_predict(net, r.data);
            axpy_cpu(classes, 1, p, 1, pred, 1);
            if(r.data != im.data) free_image(r);
        }
        free_image(im);
        top_k(pred, classes, topk, indexes);
        free(pred);
        if(indexes[0] == class) avg_acc += 1;
        for(j = 0; j < topk; ++j){
            if(indexes[j] == class) avg_topk += 1;
        }

        printf("%d: top 1: %f, top %d: %f\n", i, avg_acc/(i+1), topk, avg_topk/(i+1));
    }
}

void try_classifier(char *datacfg, char *cfgfile, char *weightfile, char *filename, int layer_num)
{
    network net = parse_network_cfg(cfgfile);
    if(weightfile){
        load_weights(&net, weightfile);
    }
    set_batch_network(&net, 1);
    srand(2222222);

    list *options = read_data_cfg(datacfg);

    char *name_list = option_find_str(options, "names", 0);
    if(!name_list) name_list = option_find_str(options, "labels", "data/labels.list");
    int top = option_find_int(options, "top", 1);

    int i = 0;
    char **names = get_labels(name_list);
    clock_t time;
    int *indexes = calloc(top, sizeof(int));
    char buff[256];
    char *input = buff;
    while(1){
        if(filename){
            strncpy(input, filename, 256);
        }else{
            printf("Enter Image Path: ");
            fflush(stdout);
            input = fgets(input, 256, stdin);
            if(!input) return;
            strtok(input, "\n");
        }
        image orig = load_image_color(input, 0, 0);
        image r = resize_min(orig, 256);
        image im = crop_image(r, (r.w - 224 - 1)/2 + 1, (r.h - 224 - 1)/2 + 1, 224, 224);
        float mean[] = {0.48263312050943, 0.45230225481413, 0.40099074308742};
        float std[] = {0.22590347483426, 0.22120921437787, 0.22103996251583};
        float var[3];
        var[0] = std[0]*std[0];
        var[1] = std[1]*std[1];
        var[2] = std[2]*std[2];

        normalize_cpu(im.data, mean, var, 1, 3, im.w*im.h);

        float *X = im.data;
        time=clock();
        float *predictions = network_predict(net, X);

        layer l = net.layers[layer_num];
        for(i = 0; i < l.c; ++i){
            if(l.rolling_mean) printf("%f %f %f\n", l.rolling_mean[i], l.rolling_variance[i], l.scales[i]);
        }
#ifdef GPU
        cuda_pull_array(l.output_gpu, l.output, l.outputs);
#endif
        for(i = 0; i < l.outputs; ++i){
            printf("%f\n", l.output[i]);
        }
        /*

           printf("\n\nWeights\n");
           for(i = 0; i < l.n*l.size*l.size*l.c; ++i){
           printf("%f\n", l.filters[i]);
           }

           printf("\n\nBiases\n");
           for(i = 0; i < l.n; ++i){
           printf("%f\n", l.biases[i]);
           }
         */

        top_predictions(net, top, indexes);
        printf("%s: Predicted in %f seconds.\n", input, sec(clock()-time));
        for(i = 0; i < top; ++i){
            int index = indexes[i];
            printf("%s: %f\n", names[index], predictions[index]);
        }
        free_image(im);
        if (filename) break;
    }
}


void predict_classifier(char *datacfg, char *cfgfile, char *weightfile, char *filename)
{
    network net = parse_network_cfg(cfgfile);
    if(weightfile){
        load_weights(&net, weightfile);
    }
    set_batch_network(&net, 1);
    srand(2222222);

    list *options = read_data_cfg(datacfg);

    char *name_list = option_find_str(options, "names", 0);
    if(!name_list) name_list = option_find_str(options, "labels", "data/labels.list");
    int top = option_find_int(options, "top", 1);

    int i = 0;
    char **names = get_labels(name_list);
    clock_t time;
    int *indexes = calloc(top, sizeof(int));
    char buff[256];
    char *input = buff;
    int size = net.w;
    while(1){
        if(filename){
            strncpy(input, filename, 256);
        }else{
            printf("Enter Image Path: ");
            fflush(stdout);
            input = fgets(input, 256, stdin);
            if(!input) return;
            strtok(input, "\n");
        }
        image im = load_image_color(input, 0, 0);
        image r = resize_min(im, size);
        resize_network(&net, r.w, r.h);
        printf("%d %d\n", r.w, r.h);

        float *X = r.data;
        time=clock();
        float *predictions = network_predict(net, X);
        top_predictions(net, top, indexes);
        printf("%s: Predicted in %f seconds.\n", input, sec(clock()-time));
        for(i = 0; i < top; ++i){
            int index = indexes[i];
            printf("%s: %f\n", names[index], predictions[index]);
        }
        if(r.data != im.data) free_image(r);
        free_image(im);
        if (filename) break;
    }
}


void label_classifier(char *datacfg, char *filename, char *weightfile)
{
    int i;
    network net = parse_network_cfg(filename);
    set_batch_network(&net, 1);
    if(weightfile){
        load_weights(&net, weightfile);
    }
    srand(time(0));

    list *options = read_data_cfg(datacfg);

    char *label_list = option_find_str(options, "names", "data/labels.list");
    char *test_list = option_find_str(options, "test", "data/train.list");
    int classes = option_find_int(options, "classes", 2);

    char **labels = get_labels(label_list);
    list *plist = get_paths(test_list);

    char **paths = (char **)list_to_array(plist);
    int m = plist->size;
    free_list(plist);

    for(i = 0; i < m; ++i){
        image im = load_image_color(paths[i], 0, 0);
        image resized = resize_min(im, net.w);
        image crop = crop_image(resized, (resized.w - net.w)/2, (resized.h - net.h)/2, net.w, net.h);
        float *pred = network_predict(net, crop.data);

        if(resized.data != im.data) free_image(resized);
        free_image(im);
        free_image(crop);
        int ind = max_index(pred, classes);

        printf("%s\n", labels[ind]);
    }
}


void test_classifier(char *datacfg, char *cfgfile, char *weightfile, int target_layer)
{
    int curr = 0;
    network net = parse_network_cfg(cfgfile);
    if(weightfile){
        load_weights(&net, weightfile);
    }
    srand(time(0));

    list *options = read_data_cfg(datacfg);

    char *test_list = option_find_str(options, "test", "data/test.list");
    int classes = option_find_int(options, "classes", 2);

    list *plist = get_paths(test_list);

    char **paths = (char **)list_to_array(plist);
    int m = plist->size;
    free_list(plist);

    clock_t time;

    data val, buffer;

    load_args args = {0};
    args.w = net.w;
    args.h = net.h;
    args.paths = paths;
    args.classes = classes;
    args.n = net.batch;
    args.m = 0;
    args.labels = 0;
    args.d = &buffer;
    args.type = OLD_CLASSIFICATION_DATA;

    pthread_t load_thread = load_data_in_thread(args);
    for(curr = net.batch; curr < m; curr += net.batch){
        time=clock();

        pthread_join(load_thread, 0);
        val = buffer;

        if(curr < m){
            args.paths = paths + curr;
            if (curr + net.batch > m) args.n = m - curr;
            load_thread = load_data_in_thread(args);
        }
        fprintf(stderr, "Loaded: %d images in %lf seconds\n", val.X.rows, sec(clock()-time));

        time=clock();
        matrix pred = network_predict_data(net, val);

        int i, j;
        if (target_layer >= 0){
            //layer l = net.layers[target_layer];
        }

        for(i = 0; i < pred.rows; ++i){
            printf("%s", paths[curr-net.batch+i]);
            for(j = 0; j < pred.cols; ++j){
                printf("\t%g", pred.vals[i][j]);
            }
            printf("\n");
        }

        free_matrix(pred);

        fprintf(stderr, "%lf seconds, %d images, %d total\n", sec(clock()-time), val.X.rows, curr);
        free_data(val);
    }
}


void threat_classifier(char *datacfg, char *cfgfile, char *weightfile, int cam_index, const char *filename)
{
#ifdef OPENCV
    float threat = 0;
    float roll = .2;

    printf("Classifier Demo\n");
    network net = parse_network_cfg(cfgfile);
    if(weightfile){
        load_weights(&net, weightfile);
    }
    set_batch_network(&net, 1);
    list *options = read_data_cfg(datacfg);

    srand(2222222);
    CvCapture * cap;

    if(filename){
        cap = cvCaptureFromFile(filename);
    }else{
        cap = cvCaptureFromCAM(cam_index);
    }

    int top = option_find_int(options, "top", 1);

    char *name_list = option_find_str(options, "names", 0);
    char **names = get_labels(name_list);

    int *indexes = calloc(top, sizeof(int));

    if(!cap) error("Couldn't connect to webcam.\n");
    //cvNamedWindow("Threat", CV_WINDOW_NORMAL); 
    //cvResizeWindow("Threat", 512, 512);
    float fps = 0;
    int i;

    int count = 0;

    while(1){
        ++count;
        struct timeval tval_before, tval_after, tval_result;
        gettimeofday(&tval_before, NULL);

        image in = get_image_from_stream(cap);
        if(!in.data) break;
        image in_s = resize_image(in, net.w, net.h);

        image out = in;
        int x1 = out.w / 20;
        int y1 = out.h / 20;
        int x2 = 2*x1;
        int y2 = out.h - out.h/20;

        int border = .01*out.h;
        int h = y2 - y1 - 2*border;
        int w = x2 - x1 - 2*border;

        float *predictions = network_predict(net, in_s.data);
        float curr_threat = predictions[0] * 0 + predictions[1] * .6 + predictions[2];
        threat = roll * curr_threat + (1-roll) * threat;

        draw_box_width(out, x2 + border, y1 + .02*h, x2 + .5 * w, y1 + .02*h + border, border, 0,0,0);
        if(threat > .97) {
            draw_box_width(out,  x2 + .5 * w + border,
                    y1 + .02*h - 2*border, 
                    x2 + .5 * w + 6*border, 
                    y1 + .02*h + 3*border, 3*border, 1,0,0);
        }
        draw_box_width(out,  x2 + .5 * w + border,
                y1 + .02*h - 2*border, 
                x2 + .5 * w + 6*border, 
                y1 + .02*h + 3*border, .5*border, 0,0,0);
        draw_box_width(out, x2 + border, y1 + .42*h, x2 + .5 * w, y1 + .42*h + border, border, 0,0,0);
        if(threat > .57) {
            draw_box_width(out,  x2 + .5 * w + border,
                    y1 + .42*h - 2*border, 
                    x2 + .5 * w + 6*border, 
                    y1 + .42*h + 3*border, 3*border, 1,1,0);
        }
        draw_box_width(out,  x2 + .5 * w + border,
                y1 + .42*h - 2*border, 
                x2 + .5 * w + 6*border, 
                y1 + .42*h + 3*border, .5*border, 0,0,0);

        draw_box_width(out, x1, y1, x2, y2, border, 0,0,0);
        for(i = 0; i < threat * h ; ++i){
            float ratio = (float) i / h;
            float r = (ratio < .5) ? (2*(ratio)) : 1;
            float g = (ratio < .5) ? 1 : 1 - 2*(ratio - .5);
            draw_box_width(out, x1 + border, y2 - border - i, x2 - border, y2 - border - i, 1, r, g, 0);
        }
        top_predictions(net, top, indexes);
        char buff[256];
        sprintf(buff, "/home/pjreddie/tmp/threat_%06d", count);
        save_image(out, buff);

        printf("\033[2J");
        printf("\033[1;1H");
        printf("\nFPS:%.0f\n",fps);

        for(i = 0; i < top; ++i){
            int index = indexes[i];
            printf("%.1f%%: %s\n", predictions[index]*100, names[index]);
        }

        if(0){
            show_image(out, "Threat");
            cvWaitKey(10);
        }
        free_image(in_s);
        free_image(in);

        gettimeofday(&tval_after, NULL);
        timersub(&tval_after, &tval_before, &tval_result);
        float curr = 1000000.f/((long int)tval_result.tv_usec);
        fps = .9*fps + .1*curr;
    }
#endif
}


void demo_classifier(char *datacfg, char *cfgfile, char *weightfile, int cam_index, const char *filename)
{
#ifdef OPENCV
    printf("Classifier Demo\n");
    network net = parse_network_cfg(cfgfile);
    if(weightfile){
        load_weights(&net, weightfile);
    }
    set_batch_network(&net, 1);
    list *options = read_data_cfg(datacfg);

    srand(2222222);
    CvCapture * cap;

    if(filename){
        cap = cvCaptureFromFile(filename);
    }else{
        cap = cvCaptureFromCAM(cam_index);
    }

    int top = option_find_int(options, "top", 1);

    char *name_list = option_find_str(options, "names", 0);
    char **names = get_labels(name_list);

    int *indexes = calloc(top, sizeof(int));

    if(!cap) error("Couldn't connect to webcam.\n");
    cvNamedWindow("Classifier", CV_WINDOW_NORMAL); 
    cvResizeWindow("Classifier", 512, 512);
    float fps = 0;
    int i;

    while(1){
        struct timeval tval_before, tval_after, tval_result;
        gettimeofday(&tval_before, NULL);

        image in = get_image_from_stream(cap);
        image in_s = resize_image(in, net.w, net.h);
        show_image(in, "Classifier");

        float *predictions = network_predict(net, in_s.data);
        top_predictions(net, top, indexes);

        printf("\033[2J");
        printf("\033[1;1H");
        printf("\nFPS:%.0f\n",fps);

        for(i = 0; i < top; ++i){
            int index = indexes[i];
            printf("%.1f%%: %s\n", predictions[index]*100, names[index]);
        }

        free_image(in_s);
        free_image(in);

        cvWaitKey(10);

        gettimeofday(&tval_after, NULL);
        timersub(&tval_after, &tval_before, &tval_result);
        float curr = 1000000.f/((long int)tval_result.tv_usec);
        fps = .9*fps + .1*curr;
    }
#endif
}


void run_classifier(int argc, char **argv)
{
    if(argc < 4){
        fprintf(stderr, "usage: %s %s [train/test/valid] [cfg] [weights (optional)]\n", argv[0], argv[1]);
        return;
    }

    char *gpu_list = find_char_arg(argc, argv, "-gpus", 0);
    int *gpus = 0;
    int ngpus = 0;
    if(gpu_list){
        printf("%s\n", gpu_list);
        int len = strlen(gpu_list);
        ngpus = 1;
        int i;
        for(i = 0; i < len; ++i){
            if (gpu_list[i] == ',') ++ngpus;
        }
        gpus = calloc(ngpus, sizeof(int));
        for(i = 0; i < ngpus; ++i){
            gpus[i] = atoi(gpu_list);
            gpu_list = strchr(gpu_list, ',')+1;
        }
    }

    int cam_index = find_int_arg(argc, argv, "-c", 0);
    int clear = find_arg(argc, argv, "-clear");
    char *data = argv[3];
    char *cfg = argv[4];
    char *weights = (argc > 5) ? argv[5] : 0;
    char *filename = (argc > 6) ? argv[6]: 0;
    char *layer_s = (argc > 7) ? argv[7]: 0;
    int layer = layer_s ? atoi(layer_s) : -1;
    if(0==strcmp(argv[2], "predict")) predict_classifier(data, cfg, weights, filename);
    else if(0==strcmp(argv[2], "try")) try_classifier(data, cfg, weights, filename, atoi(layer_s));
    else if(0==strcmp(argv[2], "train")) train_classifier(data, cfg, weights, clear);
    else if(0==strcmp(argv[2], "trainm")) train_classifier_multi(data, cfg, weights, gpus, ngpus, clear);
    else if(0==strcmp(argv[2], "demo")) demo_classifier(data, cfg, weights, cam_index, filename);
    else if(0==strcmp(argv[2], "threat")) threat_classifier(data, cfg, weights, cam_index, filename);
    else if(0==strcmp(argv[2], "test")) test_classifier(data, cfg, weights, layer);
    else if(0==strcmp(argv[2], "label")) label_classifier(data, cfg, weights);
    else if(0==strcmp(argv[2], "valid")) validate_classifier_single(data, cfg, weights);
    else if(0==strcmp(argv[2], "validmulti")) validate_classifier_multi(data, cfg, weights);
    else if(0==strcmp(argv[2], "valid10")) validate_classifier_10(data, cfg, weights);
    else if(0==strcmp(argv[2], "validcrop")) validate_classifier_crop(data, cfg, weights);
    else if(0==strcmp(argv[2], "validfull")) validate_classifier_full(data, cfg, weights);
}

//std::vector<mx_float> image_data = std::vector<mx_float>(image_size);
network* net = 0;  // alias for void *
//pthread_mutex_t out_mutex = PTHREAD_MUTEX_INITIALIZER;

int YUV420_To_BGR24(unsigned char *puc_y, unsigned char *puc_u, unsigned char *puc_v, unsigned char *puc_rgb, int width_y, int height_y)
{

    int R,G,B,Y,U,V;
	int nWidth = width_y >> 1;
	int widthStep = width_y * 3;
	int y,x;
    for(y=0; y < height_y; y++)
    {
        for(x=0; x < width_y; x++)
        {

			Y = *(puc_y + y*width_y + x);
			U = *(puc_u + (y >> 1)*nWidth + (x>>1));
	        V = *(puc_v + (y >> 1)*nWidth + (x>>1));

			R = Y + 1.402*(V-128);
		    G = Y - 0.34414*(U-128) - 0.71414*(V-128);
			B = Y + 1.772*(U-128);

			if(R > 255) R = 255;
			if(R < 0) R = 0;
			if(G > 255) G = 255;
			if(G < 0) G = 0;
			if(B > 255) B = 255;
			if(B < 0) B = 0;

			//cout << int(rData) << " " << int(gData) << " " << int(bData) << endl;
            puc_rgb[(height_y - y - 1) * widthStep + x * 3 + 2] = R;   //R
            puc_rgb[(height_y - y - 1) * widthStep + x * 3 + 1] = G;   //G
            puc_rgb[(height_y - y - 1) * widthStep + x * 3 + 0] = B;   //B
        }
    }

    return 1;
}

//int init_mxnet()
//{
//	char cfgfile[] = "darknet.cfg";
//	char weightfile[] = "darknet.weights";
//	net = parse_network_cfg2pointer(cfgfile);
//	load_weights(net, weightfile);
//	set_batch_network(net, 1);
//    return 1;
//}

int init_net(const char* cfg)
{
	char cfgfile[100] = "darknet.cfg";
	char weightfile[100] = "darknet.weights";
	float upper = 0.9;
	int early_stop = 1, print2console = 0;
	FILE* f = NULL;
	if (f)
	{
		f = fopen(cfg, "r");
		fscanf(f, "%s", cfgfile);
		fscanf(f, "%s", weightfile);
		fscanf(f, "%f", &upper);
		fscanf(f, "%d", &early_stop);
		fscanf(f, "%d", &print2console);
		fclose(f);
	}

	puts(cfgfile);
	puts(weightfile);
	net = parse_network_cfg2pointer(cfgfile);
	net->upperbound = upper;
	net->early_stop =  early_stop;
	net->print2console = print2console;
    printf("threshold of network is upper %.4f\n", net->upperbound);

	load_weights(net, weightfile);
	set_batch_network(net, 1);
    return 1;
}

short mySaturateCast(float x){
	    if (x <= 1e-15){
	        return 0;
	    }
	    else if(x >= 32767){
	        return  32767;
	    }
	    return (short)x;
}

#define uchar unsigned char
//resize to 224 * 224
void myResize(uchar* dataDst ,unsigned char *pdata,int width,int height,int new_width,int new_height)
{
    int channel = 3;
	int iWidthSrc = width;
	int iHiehgtSrc = height;

	double scale_x = (double)width / new_width;
	double scale_y = (double)height / new_height;
	int stepDst = new_width * channel;

	uchar* dataSrc = pdata;
	int stepSrc = width * channel;
	int i,j;
	for (j = 0; j < new_height; ++j)
	{
			float fy = (float)((j + 0.5) * scale_y - 0.5);
			int sy = (int)fy;
			fy -= sy;
			sy = min(sy, iHiehgtSrc - 2);
			sy = max(0, sy);
			short cbufy[2];
			cbufy[0] = mySaturateCast((1.f - fy) * 2048);  //
			cbufy[1] = 2048 - cbufy[0];
			for (i = 0; i < new_width; ++i)
			{
				float fx = (float)((i + 0.5) * scale_x - 0.5);
				int sx = (int)fx;
				fx -= sx;
				if (sx < 0) {
					fx = 0, sx = 0;
				}
				if (sx >= iWidthSrc - 1) {
					fx = 0, sx = iWidthSrc - 2;
				}
				short cbufx[2];
				cbufx[0] = mySaturateCast((1.f - fx) * 2048);  //
				cbufx[1] = 2048 - cbufx[0];
				int k;
				for (k = 0; k < channel; ++k)
				{
					*(dataDst+ j*stepDst + 3*i + k) = (*(dataSrc + sy*stepSrc + 3*sx + k) * cbufx[0] * cbufy[0] +
					*(dataSrc + (sy+1)*stepSrc + 3*sx + k) * cbufx[0] * cbufy[1] +
					*(dataSrc + sy*stepSrc + 3*(sx+1) + k) * cbufx[1] * cbufy[0] +
					*(dataSrc + (sy+1)*stepSrc + 3*(sx+1) + k) * cbufx[1] * cbufy[1]) >> 22;
				}
			}
	}
}

void yuv_resize(unsigned char* pData1, int w1,int h1, unsigned char* pData2, int w2, int h2)
{
    int x1,y1,x2,y2;
    float xr,yr; //反向缩放比例
    xr=(float)(w1) / (float)(w2);
    yr=(float)(h1) / (float)(h2);
    for(x2=0;x2<w2;x2++)
        for(y2=0;y2<h2;y2++)
        {
            x1=(int)((float)(x2)*xr+0.5);
            y1=(int)((float)(y2)*yr+0.5);
            if(x1>=w1)
                x1=w1-1;
            if(y1>=h1)
                y1=h1-1;
            pData2[y2*w2+x2]=pData1[y1*w1+x1];
        }
}

void im_resize(unsigned char* pdata,int width,int height, unsigned char* rgb_data, int new_width, int new_height)
{
    unsigned char* rgbOdata = (unsigned char *)malloc(width * height * 3);
	//for(int i  = 0; i < 10; i++)
	//	cout << int(pdata[i]) << endl;
	//exit(1);

	/*
    unsigned char *puc_y = yuv_rdata;
    unsigned char *puc_u = puc_y + new_width * new_height;
    unsigned char *puc_v = puc_u + new_width * new_height/4;
	*/
	/*
	for (int i = 0;i < 5;i++){
	      for(int j = 0;j < 100;j++){
	          cout << (int)(*(pdata + i * 3 * width + j)) << " ";
	       }
	      cout << endl;
    }
	*/

	unsigned char *puc_y = pdata;
	unsigned char *puc_u = puc_y + width * height;
	unsigned char *puc_v = puc_u + width * height / 4;

	//cout << new_width << " " << new_height <<endl;

	YUV420_To_BGR24(puc_y, puc_u, puc_v, rgbOdata, width, height);

	/*
	for(int i = 0;i < 2;i++){
		for(int j = 0;j < 20;j++){
			cout << (int)(*(rgbOdata + i * 3 * width + j)) << " ";
		}
		cout << endl;
	}
	*/

    myResize(rgb_data,rgbOdata,width,height,new_width,new_height);


	//for(int i  = 0; i < 224*224*3; i++)
	//	cout << int(rgb_data[i]) << endl;

	//exit(1);
    free(rgbOdata);
    //return rgb_data;
}

void GetMeanFile(unsigned char* data, float* image_data,
                const int channels) {
	int rows = 224;
	int cols = 224;
    //float mean = 117.0;
    float mean = 0;
	//cout << channels <<endl;
    int size = rows * cols * 3;

    float* ptr_image_r = image_data;
    float* ptr_image_g = image_data + size / 3;
    float* ptr_image_b = image_data + size / 3 * 2 ;

    int i,j;
    for (i = 0; i < rows; i++) {
        for (j = 0; j < cols; j++) {
            if (channels > 1)
            {
                float b = (float)(*data++) - mean;
                float g = (float)(*data++) - mean;
		
		b /= 255;
		g /= 255;
                *ptr_image_g++ = g;
                *ptr_image_b++ = b;
            }

            float r = (float)(*data++) - mean;
	    r /= 255;
            *ptr_image_r++ = r;
        }
    }
}

extern int open(const char* cfgfile)
{
    int res = init_net(cfgfile);
    return res;
}

extern int close()
{
    return 1;
}

extern long long detect(unsigned char* pdata,int width,int height)
{

    int indexes[1];
    indexes[0] = 0;
    unsigned char* im_data = (unsigned char*)malloc(224 * 224 * 3);
    float*X  = (float*)calloc(224 * 224 * 3, sizeof(float));
    clock_t time;
	time = clock();
    im_resize(pdata, width, height, im_data, 224, 224);
    GetMeanFile(im_data, X, 3);
	printf("Get data in %f seconds.\n",sec(clock()-time));
/*
        int size = net->w;
        image im = load_image_color("00014152.jpg", 0, 0);
	printf("--------%d------%d\n", im.w,net->batch);
        image r = resize_min(im, size);
        //resize_network(net, r.w, r.h);
	float *X = r.data;
*/
    //resize_network(net, 224, 224);
	time = clock();
    float *predictions = network_predict(*net, X);
    top_predictions(*net, 1, indexes);
	printf("Predicted in %f seconds.\n",sec(clock()-time));
    printf("%f\n", predictions[indexes[0]]);
    free(X);
    free(im_data);
    if (indexes[0] == 0)
      return (1 << 8) | 37;
    else
      return 0;
//    return indexes[0];
}

