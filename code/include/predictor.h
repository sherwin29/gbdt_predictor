#ifndef __Predictor__H__
#define __Predictor__H__

class Predictor{
        public:
            Predictor(const char *gbdt_model, const char *lr_model);
            ~Predictor();
        public:
            void predict(const float *feature, float &score, int length);
            void save_model(const char *file_name);
        private:
            void *forest;
            void *log_reg;
        private:
            int tree_size;
            int leaf_node_cnt;
    };

#endif
