#include <lr.h>

namespace shw
{
    lr::lr()
        :_model(NULL)
    {
    }

    lr::~lr()
    {
        free_and_destroy_model(&_model);
    }

    void lr::lr_train(cv::Mat_<float> &features, std::vector<int> &label, int interval)
    {
        struct feature_node **gbdt_features = new struct feature_node*[features.rows];
        for (int i = 0; i < features.rows; i++)
            gbdt_features[i] = new struct feature_node[(features.cols + 1)];
        for (int i = 0; i < features.rows; i++)
        {
            for (int j = 0; j < features.cols; j++)
            {
                gbdt_features[i][j].value = 1.0;
                gbdt_features[i][j].index = features(i, j) + interval * j + 1;
            }
            gbdt_features[i][features.cols].index = -1;
            gbdt_features[i][features.cols].value = -1.0;
        }
        //for (int i = 0; i < features.cols + 1; i++)
        //    std::cout << gbdt_features[0][i].value << ": " << gbdt_features[0][i].index << " ";
        //std::cout << std::endl;
        double *_label = new double[features.rows];

        for (int i = 0; i < features.rows; i++)
            _label[i] = label[i];
        struct problem prob;
        memset(&prob, 0, sizeof(struct problem));
        prob.l = features.rows;
        prob.bias = -1;
        prob.y = _label;
        prob.x = gbdt_features;
        prob.n = (features.cols) * interval;
        struct parameter param;
        memset(&param, 0, sizeof(struct parameter));
        param.solver_type = L2R_LR;
        param.C = 1.0;
        param.eps = 0.01;
        param.p = 0;

        _model = train(&prob, &param);
        std::cout << "finish generating lr model" << std::endl;
        return;
        for (int i = 0; i < features.rows; i++)
            delete [] gbdt_features[i];
        delete [] gbdt_features;
        delete [] _label;
        destroy_param(&param);
    }

    void lr::lr_predict(cv::Mat_<float> &features, std::vector<float> &result, int interval)
    {

        std::cout << features.rows << std::endl;
        // struct feature_node **gbdt_features;
        // gbdt_features = new struct feature_node*[features.rows];
        std::vector<struct feature_node> gbdt_features(features.rows * (features.cols + 1));
        // for (int i = 0; i < features.rows; i++)
        //    gbdt_features[i] = new struct feature_node[(features.cols + 1)];

        for (int i = 0; i < features.rows; i++)
        {
            for (int j = 0; j < features.cols; j++)
            {
                gbdt_features[i * (features.cols + 1) + j].value = 1.0;
                gbdt_features[i * (features.cols + 1) + j].index = features(i, j) + interval * j + 1;
            }
            gbdt_features[i * (features.cols + 1) + features.cols].index = -1;
            gbdt_features[i * (features.cols + 1) + features.cols].value = -1.0;
        }
        //double *proba = new double[2];
        std::vector<double> proba(2, 0.0);
        for (int i = 0; i < features.rows; i++)
        {
            predict_probability(_model, &gbdt_features[i * (features.cols + 1)], (double*)proba.data());
            result.push_back((float)proba[1]);
        }
        return;
        //for (int i = 0; i < features.rows; i++)
        //    delete [] gbdt_features[i];
        //delete [] gbdt_features;
        //delete [] proba;
    }

    void lr::lr_predict(cv::Mat_<float> &features, int interval, float &score)
    {
        std::vector<struct feature_node> gbdt_features(features.rows * (features.cols + 1));
        for (int i = 0; i < features.rows; i++)
        {
            for (int j = 0; j < features.cols; j++)
            {
                gbdt_features[i * (features.cols + 1) + j].value = 1.0;
                gbdt_features[i * (features.cols + 1) + j].index = features(i, j) + interval * j + 1;
            }
            gbdt_features[i * (features.cols + 1) + features.cols].index = -1;
            gbdt_features[i * (features.cols + 1) + features.cols].value = -1.0;
        }
        std::vector<double> proba(2, 0.0);
        for (int i = 0; i < features.rows; i++)
        {
            predict_probability(_model, &gbdt_features[i * (features.cols + 1)], (double*)proba.data());
            //result.push_back((float)proba[1]);
            score = (float)proba[1];
        }
        //return;
    }

    void lr::save_lr_model(const char *model_name)
    {
        save_model(model_name, _model);
    }

    void lr::load_lr_model(const char *model_name)
    {
        _model = load_model(model_name);
    }
}
