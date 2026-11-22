#include "booster.h"

#if defined(_WIN32)
#define OMNIGBDT_EXPORT __declspec(dllexport)
#elif defined(__GNUC__) || defined(__clang__)
#define OMNIGBDT_EXPORT __attribute__((visibility("default")))
#else
#define OMNIGBDT_EXPORT
#endif

extern "C" {

OMNIGBDT_EXPORT void SetGH(BoosterUtils *foo, double *x, double *y) {
    return foo->set_gh(x, y);
}

OMNIGBDT_EXPORT void SetBaseScore(BoosterUtils *foo, double *x, int n) {
    return foo->set_base_score(x, n);
}

OMNIGBDT_EXPORT int SetSubsample(BoosterUtils *foo, double fraction) {
    return foo ? foo->set_subsample(fraction) : -1;
}

OMNIGBDT_EXPORT int SetColsampleBytree(BoosterUtils *foo, double fraction) {
    return foo ? foo->set_colsample_bytree(fraction) : -1;
}

OMNIGBDT_EXPORT int SetClassificationMetric(BoosterUtils *foo, bool use_logloss) {
    return foo ? foo->set_classification_metric(use_logloss) : -1;
}

OMNIGBDT_EXPORT int SetProjectionDim(BoosterUtils *foo, int dim) {
    auto *multi = dynamic_cast<BoosterMulti *>(foo);
    return multi ? multi->set_projection_dim(dim) : -1;
}

OMNIGBDT_EXPORT int SetOutputMetric(BoosterMulti *foo, const double *factor, int rows, int columns) {
    try {
        return foo ? foo->set_output_metric(factor, rows, columns) : -1;
    } catch (const std::exception &) {
        return -1;
    }
}

OMNIGBDT_EXPORT int SetDiagonalOutputWeights(BoosterMulti *foo, const double *weights, int columns) {
    try {
        return foo ? foo->set_diagonal_output_weights(weights, columns) : -1;
    } catch (const std::exception &) {
        return -1;
    }
}

OMNIGBDT_EXPORT int GetBaseScoreSize(BoosterUtils *foo) {
    return foo->base_score_size();
}

OMNIGBDT_EXPORT void GetBaseScore(BoosterUtils *foo, double *x) {
    return foo->get_base_scores(x);
}

OMNIGBDT_EXPORT void SetData(BoosterUtils *foo, uint16_t *maps, double *features, double *preds, int n, bool is_train) {
    return foo->set_data(maps, features, preds, n, is_train);
}

OMNIGBDT_EXPORT void SetLabelDouble(BoosterUtils *foo, double *x, bool is_train) {
    return foo->set_label(x, is_train);
}

OMNIGBDT_EXPORT void SetLabelInt(BoosterUtils *foo, int32_t *x, bool is_train) {
    return foo->set_label(x, is_train);
}

OMNIGBDT_EXPORT void SetBin(BoosterUtils *foo, uint16_t *bins, double *values) {
    return foo->set_bin(bins, values);
}

OMNIGBDT_EXPORT void Boost(BoosterUtils *foo) {
    foo->boost();
}

OMNIGBDT_EXPORT void Train(BoosterUtils *foo, int n) {
    return foo->train(n);
}

OMNIGBDT_EXPORT void TrimTrees(BoosterUtils *foo, int n) {
    return foo->trim_trees(n);
}

OMNIGBDT_EXPORT void Predict(BoosterUtils *foo, double *features, double *preds, int n, int num_trees) {
    return foo->predict(features, preds, n, num_trees);
}

OMNIGBDT_EXPORT void Dump(BoosterUtils *foo, const char *path) {
    foo->dump(path);
}

OMNIGBDT_EXPORT void Load(BoosterUtils *foo, const char *path) {
    foo->load(path);
}

OMNIGBDT_EXPORT int GetTreeCount(BoosterUtils *foo) {
    return foo->tree_count();
}

OMNIGBDT_EXPORT int DumpChecked(BoosterUtils *foo, const char *path) {
    return foo->dump(path);
}

OMNIGBDT_EXPORT int LoadChecked(BoosterUtils *foo, const char *path, int round_width) {
    return foo->load(path, round_width);
}

// GBDT for single outputs
OMNIGBDT_EXPORT BoosterSingle *SingleNew(
        int inp_dim,
        const char *name = "mse",
        int max_depth = 5,
        int max_leaves = 32,
        int seed = 0,
        int min_samples = 5,
        int num_threads = 1,
        double lr = 0.2,
        double reg_l1 = 0.0,
        double reg_l2 = 1.0,
        double gamma = 1e-3,
        double base_score = 0.0f,
        int early_stop = 0,
        int verbosity = 2,
        int hist_cache = 16) {
    return new BoosterSingle(inp_dim, name, max_depth, max_leaves,
                             seed, min_samples, num_threads,
                             lr, reg_l1, reg_l2, gamma, base_score,
                             early_stop, verbosity, hist_cache);
}

OMNIGBDT_EXPORT void TrainMulti(BoosterSingle *foo, int num_rounds, int out_dim) {
    return foo->train_multi(num_rounds, out_dim);
}

OMNIGBDT_EXPORT void PredictMulti(BoosterSingle *foo, double *features, double *preds, int n, int out_dim, int num_trees) {
    return foo->predict_multi(features, preds, n, out_dim, num_trees);
}

OMNIGBDT_EXPORT void Reset(BoosterSingle *foo) {
    return foo->reset();
}

OMNIGBDT_EXPORT void SingleFree(BoosterSingle *foo) {
    delete foo;
}

// GBDT for multiple outputs
OMNIGBDT_EXPORT BoosterMulti *MultiNew(
        int inp_dim,
        int out_dim,
        int topk = 0,
        const char *name = "mse",
        int max_depth = 5,
        int max_leaves = 32,
        int seed = 0,
        int min_samples = 5,
        int num_threads = 1,
        double lr = 0.2,
        double reg_l1 = 0.0,
        double reg_l2 = 1.0,
        double gamma = 1e-3,
        double base_score = 0.0f,
        int early_stop = 0,
        bool one_side = true,
        int verbosity = 2,
        int hist_cache = 16) {
    return new BoosterMulti(inp_dim, out_dim, topk, name, max_depth, max_leaves,
                            seed, min_samples, num_threads,
                            lr, reg_l1, reg_l2, gamma, base_score,
                            early_stop, one_side, verbosity, hist_cache);
}

OMNIGBDT_EXPORT void MultiFree(BoosterMulti *foo) {
    delete foo;
}

}
