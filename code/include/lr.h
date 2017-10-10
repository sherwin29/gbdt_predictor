#ifndef __LR__H__
#define __LR__H__

#include <utilis.h>
#include <linear.h>

namespace shw{

    class lr{
        public:
            lr();
            ~lr();
        public:
            void lr_train(cv::Mat_<float> &features, std::vector<int> &label, int interval);

            void lr_predict(cv::Mat_<float> &features, std::vector<float> &result, int interval);

            //void evaluate(std::vector<float> &predict_val, std::vector<float> &labels);
            void lr_predict(cv::Mat_<float> &features, int interval, float &score);
        
            void load_lr_model(const char *model_name);
            void save_lr_model(const char *model_name);
        private:
            struct model* _model;
    };
}

#endif
