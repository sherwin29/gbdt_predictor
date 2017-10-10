#include <utilis.h>
#include <stdlib.h>
#include <unistd.h>
#include <dirent.h>
#include <fstream>
#include <map>
#include <limits>

namespace shw
{
    #define feature_size 19
    #define thres 50000

    std::string features_name[feature_size] = //data_type;
    #define compound_feature_size 8
    std::string compound_features_name[compound_feature_size] = //data_type;

    void load_samples(cv::Mat_<float> &features, cv::Mat_<float> &test_features, const char* file_name, int cnt, int test_cnt)
    {
        std::map<std::string, int> lookup_table;
        // specifiy based on input data.
    }

    void report(std::vector<float> &origin_samples, std::vector<float> &cur_samples)
    {
        float score  = 0;
        float aver = 0;
        for (int i = 0; i < origin_samples.size(); i++)
        {
            score += std::pow((origin_samples[i] - cur_samples[i]), 2);
            aver += std::abs(origin_samples[i] - cur_samples[i]);
        }
        score = score / origin_samples.size();
        score = std::sqrt(score);
        std::cout << "std: " << score << " aver: " << aver / origin_samples.size() << std::endl;
    }

    void quick_sort(float *dist, int *indexes, int start, int end)
    {
        int i = start;
        int j = end;
        float mid = dist[(start + end) / 2];
        while(i <= j)
        {
            while(dist[i] < mid && i <= j)
                i++;
            while(dist[j] > mid && i <= j)
                j--;
            if(i <= j)
            {
                float tmp = dist[i];
                dist[i] = dist[j];
                dist[j] = tmp;
                int tmp_index = indexes[i];
                indexes[i] = indexes[j];
                indexes[j] = tmp_index;
                i++;
                j--;
            }
        }
        if (i < end)
            quick_sort(dist, indexes, i, end);
        if (j > start)
            quick_sort(dist, indexes, start, j);
    }

    void quick_sort(float *dist, int start, int end)
    {
        int i = start;
        int j = end;
        float mid = dist[(start + end) / 2];
        while(i <= j)
        {
            while(dist[i] < mid && i <= j)
                i++;
            while(dist[j] > mid && i <= j)
                j--;
            if(i <= j)
            {
                float tmp = dist[i];
                dist[i] = dist[j];
                dist[j] = tmp;
                i++;
                j--;
            }
        }
        if (i < end)
            quick_sort(dist, i, end);
        if (j > start)
            quick_sort(dist, start, j);
    }

    void quick_sort(cv::Mat_<float> &feature, cv::Mat_<int> &features_sorted_ids, int &index, int start, int end)
    {
        int i = start;
        int j = end;
        float mid = feature(0, (start + end) / 2);
        while (i <= j)
        {
            while (feature(0, i) < mid && i <= j)
                i++;
            while (feature(0, j) > mid && i <= j)
                j--;
            if (i <= j)
            {
                float tmp = feature(0, i);
                feature(0, i) = feature(0, j);
                feature(0, j) = tmp;
                int ids = features_sorted_ids(index, i);
                features_sorted_ids(index, i) = features_sorted_ids(index, j);
                features_sorted_ids(index, j) = ids;
                i++;
                j--;
            }
        }
        if (i < end)
            quick_sort(feature, features_sorted_ids, index, i, end);
        if (j > start)
            quick_sort(feature, features_sorted_ids, index, start, j);
    }

}
