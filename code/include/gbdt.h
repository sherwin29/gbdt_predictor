#ifndef __GBDT__H__
#define __GBDT__H__

#include <tree.h>

namespace shw{

    class gbdt{
    
        public:
            gbdt(int tree_size = 0, float learning_rate = 0, int tree_depth = 0);
        public:
            void train(std::vector<float> &origin_samples, std::vector<float> &cur_samples, cv::Mat_<float> &features, std::vector<int> &samples_ids);
            void predict(cv::Mat_<float> &features, cv::Mat_<float> &out_features, float &score);
            void attributes(int &_tree_size, int &_leaf_node_cnt);
        public:
            void load_gbdt_model(const char *file_name);
            void save_gbdt_model(const char *file_name);
            void save_tree(const char *file_name, int *stat);
        private:
            int tree_size;
            float learning_rate;
            int tree_depth;
            int split_feature_num;
        private:
            std::vector<tree> forest;
    };

}

#endif
