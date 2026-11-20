#ifndef MGBDT_BOOSTER_H
#define MGBDT_BOOSTER_H

#include "tree.h"
#include "mathFunc.h"
#include "dataStruct.h"
#include "loss.h"
#include "io.h"
#include <algorithm>

struct CacheInfo {
    int node;
    int depth;
    SplitInfo split;
    vector<int32_t> order;
    vector<Histogram> hist;

    bool operator>(const CacheInfo &x) const { return split.gain > x.split.gain; }
};


class BoosterUtils {
public:
    void set_bin(uint16_t *, double *);

    void set_gh(double *, double *);

    void set_base_score(double *, int);

    int base_score_size() const;

    void get_base_scores(double *) const;

    void trim_trees(int);

    void set_data(uint16_t *, double *, double *, int, bool);

    void set_label(double *, bool);

    void set_label(int32_t *, bool);

    void rebuild_order(vector<int32_t> &, vector<int32_t> &, vector<int32_t> &, uint16_t *, uint16_t);

    void showloss(double, double, int);

    void showloss(double, int);

    int tree_count() const { return static_cast<int>(trees.size()); }

    int load(const char *path, int round_width = 1);

    int dump(const char *path) { return DumpTrees(trees, hp.base_scores, path); }

    virtual void update() = 0;

    void growth();

    void boost();

    virtual void train(int) = 0;

    virtual void predict(double *, double *, int, int) = 0;

protected:
    Tree tree;
    vector<Tree> trees;
    vector<double> configured_base_scores;
    std::vector<uint16_t> bin_nums;
    std::vector<std::vector<double>> bin_values;
    SplitInfo meta;
    Dataset Train;
    Dataset Eval;
    double *G;
    double *H;
    HyperParameter hp;
    TopkDeque<CacheInfo> cache;
    Objective obj;

    void init_preds(double *, int) const;
    bool can_split(int depth, int rows, int leaves) const;
    vector<Histogram> build_histograms(vector<int32_t> &order, int columns);
    void queue_split(int leaf, int depth, vector<int32_t> &order, vector<Histogram> &hist);
    virtual LeafNode fit_leaf(Histogram &hist, int parent) = 0;
    virtual void hist_all(vector<int32_t> &, vector<Histogram> &) = 0;
    virtual void boost_all(vector<Histogram> &) = 0;

    void restore_best(int tree_count, int round_width = 1);
};


class BoosterSingle : public BoosterUtils {
public:
    BoosterSingle(int, const char *,
                  int, int, int, int, int,
                  double, double, double, double, double,
                  int, int, int);

    void update() override;

    void train(int) override;

    void predict(double *, double *, int, int) override;

    void reset();

    void train_multi(int, int);

    void predict_multi(double *, double *, int, int, int);

private:
    double Score_sum, Opt;

    void get_score_opt(Histogram &, double &, double &);

    void hist_all(vector<int32_t> &, vector<Histogram> &) override;

    SplitInfo boost_column(Histogram &, int);

    void boost_all(vector<Histogram> &) override;

    LeafNode fit_leaf(Histogram &hist, int parent) override;
};


class BoosterMulti : public BoosterUtils {
public:
    BoosterMulti(int, int, int, const char *,
                 int, int, int, int, int,
                 double, double, double, double, double,
                 int, bool, int, int);

    void update() override;

    void train(int) override;

    void predict(double *, double *, int, int) override;

private:
    double Score_sum;
    vector<double> Score;
    vector<double> Opt;
    vector<pair<double, int>> OptPair;

    void get_score_opt(Histogram &, vector<double> &, vector<double> &, double &);

    void get_score_opt(Histogram &, vector<pair<double, int>> &, vector<double> &, double &);

    void hist_all(vector<int32_t> &, vector<Histogram> &) override;

    SplitInfo boost_column_full(Histogram &, int);

    SplitInfo boost_column_topk_two_side(Histogram &, int);

    SplitInfo boost_column_topk_one_side(Histogram &, int);

    void boost_all(vector<Histogram> &) override;

    LeafNode fit_leaf(Histogram &hist, int parent) override;
};

#endif //MGBDT_BOOSTER_H
