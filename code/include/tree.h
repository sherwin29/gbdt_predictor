#ifndef __GBDT_H__
#define __GBDT_H__

#include <iostream>
#include <stdlib.h>

#include <utilis.h>

namespace shw{
	
    class tree{
        public:
            tree(int tree_depth, int split_feature_num, float learning_rate);
            tree();
            void build(std::vector<float> &origin_samples, std::vector<float> &cur_samples, cv::Mat_<float> &features, std::vector<int> &samples_ids);
            int predict(float &score, cv::Mat_<float> &feature);
            void save_tree_model(FILE *file);
            void load_tree_model(FILE *file);	
            void save_tree(FILE *file, int *stat);
        private:
            int left_child(int index);
            int right_child(int index);
	
            split_node split_tree(const std::vector<float> &origin_samples, const std::vector<float> &cur_samples, const cv::Mat_<float> &features, int start, int end, const float &parent_sum, float &left_child_sum, float &right_child_sum);

            int tree_depth;
            std::vector<split_node> tree_node_split_feature;
            std::vector<float> tree_leaf_regression_value;
            float learning_rate;
            int split_feature_num;
            
    };

}


#endif
