#include <omp.h>
#include <tree.h>
#include <stdio.h>

namespace shw
{
    tree::tree(int tree_depth, int split_feature_num, float learning_rate)
    {
        this->tree_depth = tree_depth;
        this->split_feature_num = split_feature_num;
        this->learning_rate = learning_rate;
    }

    tree::tree() {}

    int tree::left_child(int index)
    {
        return index * 2 + 1;
    }

    int tree::right_child(int index)
    {
        return index * 2 + 2;
    }

    split_node tree::split_tree(const std::vector<float> &origin_samples, const std::vector<float> &cur_samples, const cv::Mat_<float> &features, int start, int end, const float &parent_sum, float &left_child_sum, float &right_child_sum)
    {
        std::vector<split_node> generate_split_feature;

        float base = 0.5;
        float bias = 0.2;

        float *tmp_feature = new float[end - start];

        for(int i = 0; i < split_feature_num; i++)
        {
            split_node _sp;
            for (int j = 0; j < end - start; j++)
                tmp_feature[j] = features(start + j, i);
            quick_sort(tmp_feature, 0, end - start - 1);
            int ids = (base + std::rand() / (RAND_MAX + 1.0) * bias) * (end - start);
            _sp.threshold = tmp_feature[ids];
            _sp.ids = i;
            generate_split_feature.push_back(_sp);
            //ids = (base + std::rand() / (RAND_MAX + 1.0) * bias) * (end - start);
            //_sp.threshold = tmp_feature[ids];
            //_sp.ids = i;
            //generate_split_feature.push_back(_sp);
        }
        std::vector<float> sum_left(split_feature_num);
        std::vector<int> left_cnt(split_feature_num);

        for(int m = 0; m < split_feature_num; m++)
        {

            //#pragma omp parallel for
            for(int i = start; i < end; i++)
            {
                if(features(i, m) < generate_split_feature[m].threshold)
                {
                    left_cnt[m]++;
                    sum_left[m] += origin_samples[i] - cur_samples[i];
                }
            }
        }

        float best_score = -1;
        int best_feat = 0;
        for(int m = 0; m < split_feature_num; m++)
        {
            float score = -1;
            unsigned long right_cnt = end - start - left_cnt[m];
            if (left_cnt[m] != 0 && right_cnt != 0)
            {
                float temp = parent_sum - sum_left[m];
                score = sum_left[m] * (sum_left[m]) / left_cnt[m] + temp * (temp) / right_cnt;

                if (score > best_score)
                {
                    best_score = score;
                    best_feat = m;
                }
            }
        }
        left_child_sum = sum_left[best_feat];

        if(left_cnt[best_feat] != 0)
        {
            right_child_sum = parent_sum - left_child_sum;
        }
        else
        {
            left_child_sum = 0;
            right_child_sum = parent_sum;
        }

        delete [] tmp_feature;
        return generate_split_feature[best_feat];
    }

    void tree::build(std::vector<float> &origin_samples, std::vector<float> &cur_samples, cv::Mat_<float> &features, std::vector<int> &samples_ids)
    {
        std::deque<std::pair<int, int> > piecewire_constant;
        piecewire_constant.push_back(std::pair<int, int>(0, origin_samples.size()));
        int split_node_num = std::pow(2, tree_depth) - 1;
        std::vector<float> sum(split_node_num * 2 + 1, 0.0);
        //#pragma omp parallel for
        for(size_t i = 0; i < origin_samples.size(); i++)
        {
            //sum[0] += (2 * origin_samples[i] / (1.0 + std::exp(cur_samples[i] * 2 * origin_samples[i])));
            sum[0] += origin_samples[i] - cur_samples[i];
        }
        tree_node_split_feature.resize(split_node_num);
        //std::cout << "***************************" << std::endl;
        for(int i = 0; i < split_node_num; i++)
        {
            std::pair<int, int> parts = piecewire_constant.front();
            piecewire_constant.pop_front();
            int m = parts.first;
            //std::cout << "node index: " << i << ": " << std::endl;
            split_node node = split_tree(origin_samples, cur_samples, features, parts.first, parts.second, sum[i], sum[left_child(i)], sum[right_child(i)]);

            tree_node_split_feature[i] = node;
            for(int j = parts.first; j < parts.second; j++)
            {
                if(features(j, node.ids) < node.threshold)
                {
                    cv::Mat_<float> tmp_fea = cv::Mat_<float>(1, features.cols);
                    memcpy(tmp_fea.data, features.data + j * sizeof(float) * features.cols, sizeof(float) * features.cols);
                    memcpy(features.data + j * sizeof(float) * features.cols, features.data + m * sizeof(float) * features.cols, sizeof(float) * features.cols);
                    memcpy(features.data + m * sizeof(float) * features.cols, tmp_fea.data, sizeof(float) * features.cols);
                    //std::cout << "switch " << j << " with " << m << " (" << origin_samples[j] << ", " << origin_samples[m] << ")" << std::endl;
                    float tmp = origin_samples[j];
                    origin_samples[j] = origin_samples[m];
                    origin_samples[m] = tmp;
                    tmp = cur_samples[j];
                    cur_samples[j] = cur_samples[m];
                    cur_samples[m] = tmp;
                    //std::cout << "after switch: (" << origin_samples[j] << ", " << origin_samples[m] << ")" << std::endl;
                    //std::cout << m << " : " << j << " " << samples_ids[j] << " (" << origin_samples[j] << ")" << std::endl;
                    //for (int i = 0; i < samples_ids.size(); i++)
                    //    std::cout << samples_ids[i] << ", ";
                    //std::cout << std::endl;
                    int tmp_ids = samples_ids[j];
                    samples_ids[j] = samples_ids[m];
                    samples_ids[m] = tmp_ids;
                    //std::cout << "switch " << j << " with " << m << std::endl;
                    //for (int i = 0; i < samples_ids.size(); i++)
                    //    std::cout << samples_ids[i] << ", ";
                    //std::cout << std::endl;
                    m++;
                }
            }
            piecewire_constant.push_back(std::pair<int, int>(parts.first, m));
            piecewire_constant.push_back(std::pair<int, int>(m, parts.second));
        }
        tree_leaf_regression_value.resize(piecewire_constant.size());

        for(size_t i = 0; i < tree_leaf_regression_value.size(); i++)
        {
            tree_leaf_regression_value[i] = 0;
            if(piecewire_constant[i].second != piecewire_constant[i].first)
            {
                //float nominator = 0;
                //for (int iter = piecewire_constant[i].first; iter < piecewire_constant[i].second; iter++){
                //    float residual_ = (2 * origin_samples[iter] / (1.0 + std::exp(cur_samples[iter] * 2 * origin_samples[iter])));
                //    nominator += std::abs(residual_) * (2 - std::abs(residual_));
                //}
                tree_leaf_regression_value[i] += (sum[split_node_num + i] * learning_rate) / (piecewire_constant[i].second - piecewire_constant[i].first);
                //#pragma omp parallel for
                for(int m = piecewire_constant[i].first; m < piecewire_constant[i].second; m++)
                {
                    cur_samples[m] += tree_leaf_regression_value[i];
                }
            }
        }
        //for (int i = 0; i < samples_ids.size(); i++)
        //    std::cout << samples_ids[i] << "-> " << origin_samples[i] << ": " << cur_samples[i] << std::endl;
    }

    int tree::predict(float &score, cv::Mat_<float> &feature)
    {
        int index = 0;
        int split_node_num = std::pow(2, tree_depth) - 1;
        //int split_node_num = 32;
        //std::cout << "tree_path: ";
        //static int i = 0;
        while(index < split_node_num)
        {
            //if (i < 18){
            //    std::cout << "split feature: [" << tree_node_split_feature[index].ids << ", " << tree_node_split_feature[index].threshold << "], ";
            //    i++;
            //}
            if(feature(0, tree_node_split_feature[index].ids) < tree_node_split_feature[index].threshold)
                index = index * 2 + 1;
            else
                index = index * 2 + 2;
        }
        //if (i < 18)
        //    std::cout << std::endl;
        //std::cout << " final index: " << index - split_node_num << std::endl;
        //std::cout << index << ", " << split_node_num << std::endl;
        //std::cout << index - split_node_num << std::endl;
        //cv::Mat_<float> tmp = tree_leaf_regression_value[index - split_node_num].rowRange(0, 5).t();
        //std::cout << "index: " << index << ", " << tmp.rowRange(0, 1) << std::endl;
        //score += tree_leaf_regression_value[index - split_node_num];
        //std::cout << score << ", " << split_node_num << std::endl;
        return index - split_node_num;
    }

    void tree::save_tree_model(FILE *file)
    {
        int len = tree_node_split_feature.size();

        fwrite(&len, sizeof(int), 1, file);
        for (int i = 0; i < tree_node_split_feature.size(); i++)
        {
            fwrite(&tree_node_split_feature[i].ids, sizeof(int), 1, file);
            fwrite(&tree_node_split_feature[i].threshold, sizeof(float), 1, file);
        }

        len = tree_leaf_regression_value.size();
        fwrite(&len, sizeof(int), 1, file);
        for (int i = 0; i < tree_leaf_regression_value.size(); i++)
            fwrite(&tree_leaf_regression_value[i], sizeof(float), 1, file);
    }

    void tree::save_tree(FILE *file, int *stat)
    {
        char tmp[32];
        for (int i = 0; i < tree_node_split_feature.size(); i++)
        {
            memset(tmp, 0, sizeof(tmp));
            char end;
            stat[tree_node_split_feature[i].ids]++;
            sprintf(tmp, "%d,%f ", tree_node_split_feature[i].ids, tree_node_split_feature[i].threshold);
            //printf(tmp);
            std::string str_tmp(tmp);
            if (i == tree_node_split_feature.size() - 1)
                str_tmp = str_tmp.substr(0, str_tmp.size() - 1) + std::string("\n");
            fwrite(str_tmp.c_str(), str_tmp.size(), 1, file);
        }
    }
    void tree::load_tree_model(FILE *file)
    {
        int len = 0;
        fread(&len, sizeof(int), 1, file);
        //std::cout << len << std::endl;
        tree_node_split_feature.resize(len);
        for (int i = 0; i < len; i++)
        {
            fread(&tree_node_split_feature[i].ids, sizeof(int), 1, file);
            fread(&tree_node_split_feature[i].threshold, sizeof(float), 1, file);
        }

        fread(&len, sizeof(int), 1, file);
        //std::cout << len << std::endl;
        tree_leaf_regression_value.resize(len);
        for (int i = 0; i < len; i++)
            fread(&tree_leaf_regression_value[i], sizeof(float), 1, file);
    }
}
