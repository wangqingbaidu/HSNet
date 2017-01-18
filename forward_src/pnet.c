#include "network.h"
#include "utils.h"
#include "parser.h"
#include <sys/time.h>
#define max(x,y)  ( x>y?x:y )
#define min(x,y)  ( x<y?x:y )

//std::vector<mx_float> image_data = std::vector<mx_float>(image_size);
network* net = 0;  // alias for void *
int rednet_classes = 2;
int print2console = 0;
pthread_mutex_t rednet_out = PTHREAD_MUTEX_INITIALIZER;
int rednet_use_flag = 0;
int rednet_type = 0;
float rednet_threshold = 0.9;

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
	int early_stop = 1;
	print2console = 0;
	FILE* f = NULL;
	f = fopen(cfg, "r");
	if (f)
	{
		fscanf(f, "%s", cfgfile);
		fscanf(f, "%s", weightfile);
		fscanf(f, "%f", &upper);
		fscanf(f, "%d", &early_stop);
		fscanf(f, "%d", &print2console);
		fscanf(f, "%d", &rednet_classes);
		fscanf(f, "%d", &rednet_type);
		fscanf(f, "%f", &rednet_threshold);
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
    float xr,yr;
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

extern long long detect_old(unsigned char* pdata,int width,int height)
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
    top_k(predictions, rednet_classes, 1, indexes);
	printf("Predicted in %f seconds.\n",sec(clock()-time));
    //printf("%f\n", predictions[indexes[0]]);
    printf("%d\t%f\n", indexes[0], predictions[indexes[0]]);
    free(X);
    free(im_data);
    if (indexes[0] == rednet_classes - 1 || predictions[indexes[0]] < rednet_threshold)
    	return 0;
    else
      return (1 << 9) | (0xa5 <<1) | 1;
//    return indexes[0];
}

extern long long detect(unsigned long long pid,unsigned char* pdata,int width,int height)
{
	pthread_mutex_lock(&rednet_out);
	if (rednet_use_flag == 0)
	{
		rednet_use_flag = 1;
		pthread_mutex_unlock(&rednet_out);
	}
	else
	{
		pthread_mutex_unlock(&rednet_out);
		return 0;
	}
    int indexes[1];
    indexes[0] = 0;
    unsigned char* im_data = (unsigned char*)malloc(224 * 224 * 3);
    float*X  = (float*)calloc(224 * 224 * 3, sizeof(float));
    clock_t time;
	time = clock();
    im_resize(pdata, width, height, im_data, 224, 224);
    GetMeanFile(im_data, X, 3);
    if (print2console)
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
    top_k(predictions, rednet_classes, 1, indexes);
	if (print2console)
		printf("Predicted in %f seconds.\n",sec(clock()-time));
	if(print2console)
		printf("%lld\t%d\t%f\n", pid, indexes[0], predictions[indexes[0]]);

	pthread_mutex_lock(&rednet_out);
	rednet_use_flag = 0;
	pthread_mutex_unlock(&rednet_out);

    free(X);
    free(im_data);
    if (indexes[0] == rednet_classes - 1 || predictions[indexes[0]] < rednet_threshold )
    	return 0;
    else
      return (1 << 9) | (rednet_type <<1) | 1;
    	//return 0xa5;
}
