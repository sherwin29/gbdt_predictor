#include <predictor.h>
#include <utilis.h>
#include <lr.h>
#include <gbdt.h>

#define feature_size 25

std::string stat_features_name[feature_size] = // specify for datatype

Predictor::Predictor(const char *gbdt_model, const char *lr_model)
{
    forest = new shw::gbdt();
    log_reg = new shw::lr();
    ((shw::gbdt*)forest)->load_gbdt_model(gbdt_model);
    ((shw::lr*)log_reg)->load_lr_model(lr_model);
    ((shw::gbdt*)forest)->attributes(tree_size, leaf_node_cnt);
    std::cout << "tree size: " << tree_size << " leaf node cnt: " << leaf_node_cnt << std::endl;
}

Predictor::~Predictor()
{
    delete forest;
    delete log_reg;
}

void Predictor::predict(const float *feature, float &score, int length)
{
    cv::Mat_<float> _feature = cv::Mat_<float>(1, length);
    memcpy(_feature.data, feature, sizeof(float) * length);
    cv::Mat_<float> out_features = cv::Mat_<float>(1, tree_size).zeros(1, tree_size);
    ((shw::gbdt*)forest)->predict(_feature, out_features, score);
    ((shw::lr*)log_reg)->lr_predict(out_features, leaf_node_cnt, score);
}

void Predictor::save_model(const char *file_name){
    int *stat = new int[feature_size];
    memset(stat, 0, sizeof(int) * feature_size);
    ((shw::gbdt*)forest)->save_tree(file_name, stat);
    for (int i = 0; i < feature_size; i++)
        std::cout << stat_features_name[i] << ": " << stat[i] << std::endl;
    delete [] stat;
}
