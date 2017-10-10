#include <gbdt.h>
#include <omp.h>
#include <stdio.h>

namespace shw
{

    gbdt::gbdt(int tree_size, float learning_rate, int tree_depth)
        : tree_size(tree_size), learning_rate(learning_rate), tree_depth(tree_depth)
    {
    }

    void gbdt::attributes(int &_tree_size, int &_leaf_node_cnt)
    {
        _tree_size = tree_size;
        _leaf_node_cnt = std::pow(2, tree_depth);
    }

    void gbdt::train(std::vector<float> &origin_samples, std::vector<float> &cur_samples, cv::Mat_<float> &features, std::vector<int> &samples_ids)
    {
        int tree_cnt = 0;
        split_feature_num = features.cols;
        //float sum = 0;
        //for (int i = 0; i < origin_samples.size(); i++)
        //    sum += origin_samples[i] - cur_samples[i];
        //sum = sum / origin_samples.size();
        //sum = std::log((1 + sum) / (1 - sum)) / 2;
        //for (int i = 0; i < origin_samples.size(); i++)
        //    cur_samples[i] = sum;
        //std::vector<int> loc(origin_samples.size());
        //for (int i = 0; i < origin_samples.size(); i++)
        //    loc[i] = i;
        //cv::Mat_<int> features_sorted_ids = cv::Mat_<int>(features.rows, features.cols);
        //for (int i = 0; i < features_sorted_ids.rows; i++)
        //    for (int j = 0; j < features_sorted_ids.cols; j++)
        //        features_sorted_ids(i, j) = j;
        //cv::Mat_<float> particular_features = cv::Mat_<float>(1, features.cols);
        //clock_t time_begin = clock();
        //for (int i = 0; i < features_sorted_ids.rows; i++)
        //{
        //    memcpy(particular_features.data, features.data + sizeof(float) * i * features.cols, sizeof(float) * features.cols);
        //    quick_sort(particular_features, features_sorted_ids, i, 0, features.cols - 1);
        //}
        //clock_t time_end = clock();
        //std::cout << "pre-sorting the training set: " << (time_end - time_begin) / (double)CLOCKS_PER_SEC << std::endl;
        while (true)
        {
            tree tr(tree_depth, features.cols, learning_rate);
            tr.build(origin_samples, cur_samples, features, samples_ids);

            forest.push_back(tr);
            tree_cnt++;
            if (tree_cnt % 10 == 0)
            {
                std::cout << "iter " << tree_cnt << ": ";
                report(origin_samples, cur_samples);
            }
            if (tree_cnt >= tree_size)
                break;
        }
    }

    void gbdt::predict(cv::Mat_<float> &features, cv::Mat_<float> &out_features, float &score)
    {

        int tree_cnt = 0;
        score = 0;
        //int tree_leaf_cnt = std::pow(2, tree_depth);
        //std::cout << tree_leaf_cnt << std::endl;
        //int sparse_feature_len = tree_leaf_cnt * tree_size;
        //out_features = cv::Mat_<float>(1, tree_size).zeros(1, tree_size);
        //std::cout << out_features.colRange(0, 10) << std::endl;
        //while (true){

        for (int tree_cnt = 0; tree_cnt < tree_size; tree_cnt++)
        {
            float _score = 0;
            //int index = 0;
            int index = forest[tree_cnt].predict(_score, features);
            //std::cout << index << std::endl;
            out_features(0, tree_cnt) = index;
            //tree_cnt++;
            //if (tree_cnt >= tree_size)
            //    break;
            //std::cout << tree_cnt << ", ";
        }
        //std::cout << out_features << std::endl;
        //std::cout << std::endl;
    }

    void gbdt::load_gbdt_model(const char *file_name)
    {
        FILE *file = fopen(file_name, "r");
        fread(&tree_size, sizeof(int), 1, file);
        fread(&learning_rate, sizeof(float), 1, file);
        fread(&tree_depth, sizeof(int), 1, file);
        fread(&split_feature_num, sizeof(int), 1, file);
        //forest.resize(tree_size);
        for (int i = 0; i < tree_size; i++)
        {
            tree tr(tree_depth, split_feature_num, learning_rate);
            tr.load_tree_model(file);
            forest.push_back(tr);
        }
        fclose(file);
    }

    void gbdt::save_gbdt_model(const char *file_name)
    {
        FILE *file = fopen(file_name, "wb");
        fwrite(&tree_size, sizeof(int), 1, file);
        fwrite(&learning_rate, sizeof(float), 1, file);
        fwrite(&tree_depth, sizeof(int), 1, file);
        fwrite(&split_feature_num, sizeof(int), 1, file);
        for (int i = 0; i < tree_size; i++)
            forest[i].save_tree_model(file);
        fclose(file);
    }

    void gbdt::save_tree(const char *file_name, int *stat)
    {
        std::cout << "save model to " << file_name << " tree size: " << tree_size << std::endl;
        FILE *file = fopen(file_name, "w");
        for (int i = 0; i < tree_size; i++)
            forest[i].save_tree(file, stat);
        fclose(file);
    }
}
