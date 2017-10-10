#ifndef __UTILIS_H__
#define __UTILIS_H__

#include <iostream>
#include <deque>
#include <opencv2/core/core.hpp>

namespace shw{

    typedef struct _split_node{
        int ids;
        float threshold;
    }split_node;

    void load_samples(cv::Mat_<float> &features, cv::Mat_<float> &test_features, const char* file_name, int cnt, int test_cnt);
    void report(std::vector<float> &origin_samples, std::vector<float> &cur_samples);
    void quick_sort(float *dist, int start, int end);
    void quick_sort(float *dist, int *indexes, int start, int end);
    void quick_sort(cv::Mat_<float> &feature, cv::Mat_<int> &features_sorted_ids, int &index, int start, int end);
}

#endif
