#include <stdio.h>
#include <ctime>
#include <gbdt.h>
#include <lr.h>
#include <pthread.h>

//#define len 10

//typedef struct{
//    shw::gbdt *gbdt_processor;
//    cv::Mat_<float> *features;
//    cv::Mat_<float> *gbdt_features;
//    int index;
//    int tree_size;
//}processor_info;


//void *thread_processor(void *data)
//{
//    processor_info *info = (processor_info*)data;
//    //std::cout << info->index << std::endl;
//    cv::Mat_<float> gbdt_features = *(info->gbdt_features);
//    cv::Mat_<float> out_features = cv::Mat_<float>(1, info->tree_size);
//    float score = 0;
//    //std::cout << "tree_size: " << info->tree_size << " index: " << info->index << std::endl;
//    cv::Mat_<float> tmp_feature = (info->features)->rowRange(info->index, info->index + 1);
//    info->gbdt_processor->predict(tmp_feature, out_features, score);
//    //std::cout << out_features << std::endl;
//    for (int j = 0; j < out_features.cols; j++)
//        gbdt_features(info->index, j) = out_features(0, j);
//    pthread_exit(NULL);
//}

int main(int argv, char **argc)
{
    std::srand(std::time(0));

    // path
    const char* pos_dataset_path = "data/_negative_bk.txt";
    const char* neg_dataset_path = "data/_positive_bk.txt";
    const char* lr_model_path = "data/model_save/light_lr";
    const char* gbdt_model_path = "data/model_save/light_gbdt";
    const char* train_result_path = "./test_result.txt";
    const char* test_result_path = "./test_result.txt";
    

    // learning para
    int tree_size = 100;
    int tree_depth = 5;
    float learning_ratio = 0.2;

    // data num
    int negative_cnt = 800000, positive_cnt = 500000;
    float portion = 0.3;
    

    std::vector<float> samples, cur_samples;
    cv::Mat_<float> negative_features, positive_features, negative_test_features, positive_test_features;
    int negative_test_cnt = negative_cnt * portion * 2;
    int positive_test_cnt = positive_cnt * portion;

    shw::load_samples(negative_features, negative_test_features, pos_dataset_path, negative_cnt, negative_test_cnt);
    shw::load_samples(positive_features, positive_test_features, neg_dataset_path, positive_cnt, positive_test_cnt);
    std::cout << "finish loading training set" << std::endl;


    samples.resize(negative_cnt + positive_cnt);
    cur_samples.resize(negative_cnt + positive_cnt);

    memset(samples.data(), 0, sizeof(int) * samples.size());
    for (int i = negative_cnt; i < samples.size(); i++)
    {
        samples[i] = 1;
    }
    std::vector<int> samples_ids(negative_cnt + positive_cnt);
    for (int i = 0; i < samples_ids.size(); i++)
    {
        samples_ids[i] = i;
    }
    memset(cur_samples.data(), 0, sizeof(int) * cur_samples.size());
    cv::Mat_<float> features = cv::Mat_<float>(positive_cnt + negative_cnt, negative_features.cols);
    for (int i = 0; i < negative_cnt + positive_cnt; i++)
    {
        for (int j = 0; j < negative_features.cols; j++)
        {
            if (i < negative_cnt)
            {
                features(i, j) = negative_features(i, j);
            }
            else
            {
                features(i, j) = positive_features(i - negative_cnt, j);
            }
        }
    }
    
    shw::gbdt forest(tree_size, learning_ratio, tree_depth);
    shw::report(samples, cur_samples);
    clock_t time_begin = clock();
    forest.train(samples, cur_samples, features, samples_ids);
    clock_t time_end = clock();
    forest.save_gbdt_model(gbdt_model_path);

    cv::Mat_<float> out_features = cv::Mat_<float>(1, tree_size);
    cv::Mat_<float> gbdt_features = cv::Mat_<float>(negative_features.rows + positive_features.rows, tree_size);
 
    for (int i = 0; i < negative_cnt; i++)
    {
        float score = 0;
        cv::Mat_<float> tmp_feature = negative_features.rowRange(i, i + 1);
        forest.predict(tmp_feature, out_features, score);
        for (int j = 0; j < out_features.cols; j++)
        {
            gbdt_features(i, j) = out_features(0, j);
        }
    }

    for (int i = 0; i < positive_cnt; i++)
    {
        float score = 0;
        cv::Mat_<float> tmp_feature = positive_features.rowRange(i, i + 1);
        forest.predict(tmp_feature, out_features, score);
        for (int j = 0; j < out_features.cols; j++)
        {
            gbdt_features(i + negative_features.rows, j) = out_features(0, j);
        }
    }

    cv::Mat_<float> test_features = cv::Mat_<float>(negative_test_cnt + positive_test_cnt, positive_features.cols);
    for (int i = 0; i < test_features.rows; i++)
    {
        for (int j = 0; j < test_features.cols; j++)
        {
            if (i < negative_test_cnt)
            {
                test_features(i, j) = negative_test_features(i, j);
            }
            else
            {
                test_features(i, j) = positive_test_features(i - negative_test_cnt, j);
            }
        }
    }

    std::vector<int> test_labels(negative_test_cnt + positive_test_cnt, 0);
    for (int i = negative_test_cnt; i < test_labels.size(); i++)
    {
        test_labels[i] = 1;
    }
    
    cv::Mat_<float> test_gbdt_features = cv::Mat_<float>(test_labels.size(), gbdt_features.cols);
    for (int i = 0; i < test_labels.size(); i++)
    {
        float score = 0;
        cv::Mat_<float> tmp_feature = test_features.rowRange(i, i + 1);
        forest.predict(tmp_feature, out_features, score);
        for (int j = 0; j < out_features.cols; j++)
        {
            test_gbdt_features(i, j) = out_features(0, j);
        }
    }
    std::cout << "gbdt_features: " << gbdt_features.rows << ", " << gbdt_features.cols << std::endl;
    
    memset(samples_ids.data(), 0, sizeof(int) * samples_ids.size());
    for (int i = negative_cnt; i < samples_ids.size(); i++)
    {
        samples_ids[i] = 1;
    }    
    int interval = std::pow(2, tree_depth);
    shw::lr lr_train;
    lr_train.lr_train(gbdt_features, samples_ids, interval);
    lr_train.save_lr_model(lr_model_path);

    std::vector<float> result;
    lr_train.lr_predict(gbdt_features, result, interval);
    FILE *result_file = fopen(train_result_path, "w");
    char header[32];
    memset(header, 0, sizeof(header));
    sprintf(header, "%d,%d\n", negative_cnt, positive_cnt);
    fwrite(header, strlen(header), 1, result_file);
    for (int i = 0; i < result.size(); i++)
    {
        char tmp[32];
        memset(tmp, 0, sizeof(tmp));
        sprintf(tmp, "%f\n", result[i]);
        fwrite(tmp, strlen(tmp), 1, result_file);
    }
    fclose(result_file);
    result.clear();
    lr_train.lr_predict(test_gbdt_features, result, interval);
    result_file = fopen(test_result_path, "w");
    memset(header, 0, sizeof(header));
    sprintf(header, "%d,%d\n", negative_test_cnt, positive_test_cnt);
    fwrite(header, strlen(header), 1, result_file);
    for (int i = 0; i < result.size(); i++)
    {
        char tmp[32];
        memset(tmp, 0, sizeof(tmp));
        sprintf(tmp, "%f\n", result[i]);
        fwrite(tmp, strlen(tmp), 1, result_file);
    }
    fclose(result_file);

    return 0;
}
