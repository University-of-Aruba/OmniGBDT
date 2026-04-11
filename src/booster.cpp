#include "booster.h"

namespace {

constexpr double kSplitTieEpsilon = 1e-12;

bool is_valid_split(const SplitInfo &split) {
    return split.column >= 0;
}

bool gain_is_greater(double lhs, double rhs) {
    return lhs > rhs + kSplitTieEpsilon;
}

bool threshold_is_less(double lhs, double rhs) {
    return lhs + kSplitTieEpsilon < rhs;
}

bool is_better_split(const SplitInfo &candidate, const SplitInfo &best) {
    if (!is_valid_split(candidate)) {
        return false;
    }
    if (!is_valid_split(best)) {
        return true;
    }
    if (gain_is_greater(candidate.gain, best.gain)) {
        return true;
    }
    if (gain_is_greater(best.gain, candidate.gain)) {
        return false;
    }
    if (candidate.column != best.column) {
        return candidate.column < best.column;
    }
    if (candidate.bin != best.bin) {
        return candidate.bin < best.bin;
    }
    if (threshold_is_less(candidate.threshold, best.threshold)) {
        return true;
    }
    if (threshold_is_less(best.threshold, candidate.threshold)) {
        return false;
    }
    return false;
}

SplitInfo select_best_split(const vector<SplitInfo> &candidates) {
    SplitInfo best;
    best.reset();
    for (const auto &candidate : candidates) {
        if (is_better_split(candidate, best)) {
            best = candidate;
        }
    }
    return best;
}

}  // namespace

void BoosterUtils::set_bin(uint16_t *bins, double *values) {
    bin_nums.clear();
    bin_values.clear();
    int cnt = 0;
    for (int i = 0; i < hp.inp_dim; i++) {
        vector<double> tmp;
        bin_nums.push_back(bins[i] + 1);
        for (int j = 0; j < bins[i]; j++) {
            tmp.push_back(values[cnt + j]);
        }
        tmp.push_back(std::numeric_limits<double>::max());
        cnt += bins[i];
        bin_values.push_back(tmp);
    }
}

void BoosterUtils::set_gh(double *x, double *y) {
    G = x;
    H = y;
}

void BoosterUtils::set_base_score(double *x, int n) {
    if (x == nullptr || n <= 0) {
        return;
    }

    if (hp.out_dim > 1 && n == 1) {
        hp.base_scores.assign(hp.out_dim, x[0]);
    } else {
        hp.base_scores.assign(x, x + n);
    }

    if (!hp.base_scores.empty()) {
        hp.base_score = hp.base_scores[0];
    }
}

int BoosterUtils::base_score_size() const {
    if (!hp.base_scores.empty()) {
        return static_cast<int>(hp.base_scores.size());
    }
    return max(hp.out_dim, 1);
}

void BoosterUtils::get_base_scores(double *x) const {
    if (x == nullptr) {
        return;
    }

    if (!hp.base_scores.empty()) {
        copy(hp.base_scores.begin(), hp.base_scores.end(), x);
        return;
    }

    fill_n(x, base_score_size(), hp.base_score);
}

void BoosterUtils::trim_trees(int num_trees) {
    if (num_trees < 0) {
        num_trees = 0;
    }
    if (num_trees < static_cast<int>(trees.size())) {
        trees.resize(num_trees);
    }
}

void BoosterUtils::set_data(uint16_t *x, double *y, double *z, int n, bool is_train) {
    if (is_train) {
        Train.num = n;
        Train.Maps = x;
        Train.Features = y;
        Train.Preds = z;
        Train.Orders.resize(n);
        Train.LeafIndex.resize(n);
        for (int32_t i = 0; i < n; ++i) {
            Train.Orders[i] = i;
        }
    } else {
        Eval.num = n;
        Eval.Maps = x;
        Eval.Features = y;
        Eval.Preds = z;
        Eval.LeafIndex.resize(n);
    }
}

void BoosterUtils::set_label(double *x, bool is_train) {
    if (is_train) { Train.Label_double = x; }
    else { Eval.Label_double = x; }
}

void BoosterUtils::set_label(int32_t *x, bool is_train) {
    if (is_train) { Train.Label_int32 = x; }
    else { Eval.Label_int32 = x; }
}

void BoosterUtils::init_preds(double *preds, int n) const {
    if (preds == nullptr || n <= 0) {
        return;
    }

    if (hp.out_dim <= 1 || hp.base_scores.size() <= 1) {
        fill_n(preds, n * max(hp.out_dim, 1), hp.base_score);
        return;
    }

    for (int i = 0; i < n; ++i) {
        copy(hp.base_scores.begin(), hp.base_scores.end(), preds + i * hp.out_dim);
    }
}

void BoosterUtils::rebuild_order(vector<int32_t> &order, vector<int32_t> &order_l, vector<int32_t> &order_r,
                                 uint16_t *maps, uint16_t bin) {
    int cnt_l = 0, cnt_r = 0;
    for (auto i : order) {
        if (maps[i] <= bin) {
            order_l[cnt_l++] = i;
        } else {
            order_r[cnt_r++] = i;
        }
    }
}

void BoosterUtils::showloss(double score, double metric, int i) {
    cout << "[" << i << "] train->" << setprecision(5) << fixed << score << "\teval->" <<
         setprecision(5) << fixed << metric << endl;
}

void BoosterUtils::showloss(double metric, int i) {
    cout << "[" << i << "] score->" << setprecision(5) << fixed << metric << endl;
}


BoosterSingle::BoosterSingle(
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
    hp.inp_dim = inp_dim;
    hp.loss = name;
    hp.max_depth = max_depth;
    hp.max_leaves = max_leaves;
    hp.seed = seed;
    hp.min_samples = min_samples;
    hp.num_threads = min(num_threads, omp_get_max_threads());
    hp.lr = lr;
    hp.reg_l1 = reg_l1;
    hp.reg_l2 = reg_l2;
    hp.gamma = gamma;
    hp.base_score = base_score;
    hp.base_scores.assign(1, base_score);
    hp.early_stop = early_stop;
    hp.verbosity = max(0, min(verbosity, 2));
    hp.Max_caches = hist_cache;

    omp_set_num_threads(hp.num_threads);
    cache = TopkDeque<CacheInfo>(hp.Max_caches);
    obj = Objective(hp.loss);
}

void BoosterSingle::reset() {
    trees.clear();
    if (Train.num > 0) { init_preds(Train.Preds, Train.num); }
    if (Eval.num > 0) { init_preds(Eval.Preds, Eval.num); }

}

void BoosterSingle::get_score_opt(Histogram &Hist, double &opt, double &score_sum) {
    double gr = Hist.g[Hist.g.size() - 1];
    double hr = Hist.h[Hist.h.size() - 1];
    opt = CalWeight(gr, hr, hp.reg_l1, hp.reg_l2);
    score_sum = CalScore(gr, hr, hp.reg_l1, hp.reg_l2);
}

void BoosterSingle::hist_all(vector<int32_t> &order, vector<Histogram> &Hist) {
#pragma omp parallel for schedule(static)
    for (int i = 0; i < hp.inp_dim; i++) {
        histogram_single(order, Hist[i], Train.Maps + i * Train.num, G, H);
    }
}

SplitInfo BoosterSingle::boost_column(Histogram &Hist, int column) {
    int max_bins = Hist.cnt.size() - 1;
    int total_rows = Hist.cnt[max_bins];
    double gr, hr;
    double gain = 0.0f, tmp;
    int row_bins = -1;
    for (int i = 0; i < max_bins; i++) {
        int rows_l = Hist.cnt[i];
        int rows_r = total_rows - rows_l;
        if (rows_l < hp.min_samples || rows_r < hp.min_samples) {
            continue;
        }
        gr = Hist.g[max_bins] - Hist.g[i];
        hr = Hist.h[max_bins] - Hist.h[i];
        tmp = CalScore(Hist.g[i], Hist.h[i], hp.reg_l1, hp.reg_l2) + \
              CalScore(gr, hr, hp.reg_l1, hp.reg_l2);
        if (tmp > gain) {
            gain = tmp;
            row_bins = i;
        }
    }
    gain -= Score_sum;
    gain *= 0.5f;
    SplitInfo candidate;
    candidate.reset();
    if (row_bins > -1) {
        candidate.update(gain, column, row_bins, bin_values[column][row_bins]);
    }
    return candidate;
}

void BoosterSingle::boost_all(vector<Histogram> &Hist) {
    vector<SplitInfo> candidates(hp.inp_dim);
#pragma omp parallel for schedule(static)
    for (int i = 0; i < hp.inp_dim; ++i) {
        candidates[i] = boost_column(Hist[i], i);
    }
    meta = select_best_split(candidates);
}

void BoosterSingle::build_tree_best() {
    if (tree.leaf_num >= hp.max_leaves) { return; }

    auto info = &cache.data[0];
    int parent = info->node;
    int depth = info->depth;

    int rows_l = info->hist[info->split.column].cnt[info->split.bin];
    int rows_r = info->order.size() - rows_l;

    vector<int32_t> order_l(rows_l), order_r(rows_r);
    rebuild_order(info->order, order_l, order_r,
                  Train.Maps + Train.num * info->split.column, info->split.bin);

    vector<Histogram> Hist_l(hp.inp_dim), Hist_r(hp.inp_dim);

    if (rows_l >= rows_r) {
        for (int i = 0; i < hp.inp_dim; i++) { Hist_r[i] = Histogram(bin_nums[i], 1); }
        hist_all(order_r, Hist_r);
#pragma omp parallel for schedule(static)
        for (int i = 0; i < hp.inp_dim; ++i) { info->hist[i] - Hist_r[i]; }
        Hist_l.assign(info->hist.begin(), info->hist.end());
    } else {
        for (int i = 0; i < hp.inp_dim; i++) { Hist_l[i] = Histogram(bin_nums[i], 1); }
        hist_all(order_l, Hist_l);
#pragma omp parallel for schedule(static)
        for (int i = 0; i < hp.inp_dim; ++i) { info->hist[i] - Hist_l[i]; }
        Hist_r.assign(info->hist.begin(), info->hist.end());
    }
    cache.data.pop_front();

    bool split_left = false;
    if (rows_l > 0) {
        get_score_opt(Hist_l[0], Opt, Score_sum);
        if (rows_l >= hp.min_samples) {
            boost_all(Hist_l);

            if (depth + 1 < hp.max_depth && meta.column > -1 && meta.gain > hp.gamma) {
                auto node = NonLeafNode(parent, meta.column, meta.bin, meta.threshold);
                tree.add_nonleaf(node, true);
                cache.push(CacheInfo(tree.nonleaf_num, depth + 1, meta, order_l, Hist_l));
                split_left = true;
            }
        }
    }
    if (!split_left) {
        auto node = LeafNode(1);
        if (rows_l > 0) { node.Update(parent, Opt); }
        else { node.Update(parent, 0.0); }
        tree.add_leaf(node, true);
    }
    order_l.clear();
    Hist_l.clear();

    if (tree.leaf_num >= hp.max_leaves) { return; }
    bool split_right = false;
    if (rows_r > 0) {
        get_score_opt(Hist_r[0], Opt, Score_sum);
        if (rows_r >= hp.min_samples) {
            boost_all(Hist_r);

            if (depth + 1 < hp.max_depth && meta.column > -1 && meta.gain > hp.gamma) {
                auto node = NonLeafNode(parent, meta.column, meta.bin, meta.threshold);
                tree.add_nonleaf(node, false);
                cache.push(CacheInfo(tree.nonleaf_num, depth + 1, meta, order_r, Hist_r));
                split_right = true;
            }
        }
    }
    if (!split_right) {
        auto node = LeafNode(1);
        if (rows_r > 0) { node.Update(parent, Opt); }
        else { node.Update(parent, 0.0); }
        tree.add_leaf(node, false);
    }
    order_r.clear();
    Hist_r.clear();

    if (!cache.data.empty()) { build_tree_best(); }
}

void BoosterSingle::update() {
    tree.shrinkage(hp.lr);
    tree.pred_value_single(Train.Maps, Train.Preds, hp, Train.num);
    if (Eval.num > 0) {
        tree.pred_value_single(Eval.Features, Eval.Preds, hp, Eval.num);
    }
    trees.push_back(tree);
}

void BoosterSingle::growth() {
    tree.clear();
    cache.data.clear();

    vector<Histogram> Hist(hp.inp_dim);
    for (int i = 0; i < hp.inp_dim; i++) { Hist[i] = Histogram(bin_nums[i], 1); }
    hist_all(Train.Orders, Hist);
    get_score_opt(Hist[0], Opt, Score_sum);
    boost_all(Hist);
    if (meta.column > -1 & meta.gain > -10.0f) {
        auto node = NonLeafNode(-1, meta.column, meta.bin, meta.threshold);
        tree.add_nonleaf(node, true);
        cache.push(CacheInfo(-1, 0, meta, Train.Orders, Hist));
        build_tree_best();
    } else {
        auto node = LeafNode(1);
        node.Update(-1, Opt);
        tree.add_leaf(node, true);
    }
}

void BoosterSingle::train(int num_rounds) {
    // initialize gradient and hessian
    G = (double *) calloc(Train.num, sizeof(double));
    H = (double *) calloc(Train.num, sizeof(double));
    if (obj.constHessian) {
        double value = obj.hessian;
#pragma omp parallel for schedule(static)
        for (int i = 0; i < Train.num; ++i) {
            H[i] = value;
        }
    }

    int round = hp.early_stop;
    if (round == 0) { round = num_rounds; }
    auto early_stoper = EarlyStoper(round, obj.largerBetter);

    // start training
    for (int i = 0; i < num_rounds; ++i) {
        obj.f_grad(Train, G, H, Train.num, 1);
        growth();
        update();
        double score = obj.f_score(Train, Train.num, 1);
        if (Eval.num > 0) {
            double metric = obj.f_metric(Eval, Eval.num, 1);
            if (hp.show_round_metrics()) { showloss(score, metric, i); }
            early_stoper.push(make_pair(metric, i));
            if (!early_stoper.is_continue) {
                auto info = early_stoper.best();
                if (hp.show_summary()) { early_stoper.print_best(); }
                trees.resize(get<1>(info) + 1);
                break;
            }
        } else {
            if (hp.show_round_metrics()) { showloss(score, i); }
        }
    }
    if (early_stoper.is_continue && Eval.num > 0 && hp.show_summary()) { early_stoper.print_best(); }
    free(G);
    free(H);
}

void BoosterSingle::train_multi(int num_rounds, int out_dim) {
    out_dim = max(out_dim, 2);
    int elements = Train.num * out_dim;
    G = (double *) calloc(elements, sizeof(double));
    H = (double *) calloc(elements, sizeof(double));
    if (obj.constHessian) {
        double value = obj.hessian;
#pragma omp parallel for schedule(static)
        for (int i = 0; i < elements; ++i) {
            H[i] = value;
        }
    }
    auto early_stoper = EarlyStoper(hp.early_stop, obj.largerBetter);
    if (hp.early_stop == 0) { early_stoper = EarlyStoper(num_rounds, obj.largerBetter); }

    //start training
    for (int i = 0; i < num_rounds; ++i) {
        obj.f_grad(Train, G, H, Train.num, out_dim);
        for (int j = 0; j < out_dim; ++j) {
            growth();
            update();
            if (j < out_dim - 1) {
                G += Train.num;
                H += Train.num;
                Train.Preds += Train.num;
                if (Eval.num > 0) { Eval.Preds += Eval.num; }
            } else {
                int pos = (out_dim - 1) * Train.num;
                G -= pos;
                H -= pos;
                Train.Preds -= pos;
                if (Eval.num > 0) { Eval.Preds -= (out_dim - 1) * Eval.num; }
            }
        }
        double score = obj.f_score(Train, Train.num, out_dim);
        if (Eval.num > 0) {
            double metric = obj.f_metric(Eval, Eval.num, out_dim);
            if (hp.show_round_metrics()) { showloss(score, metric, i); }
            early_stoper.push(make_pair(metric, i));
            if (!early_stoper.is_continue) {
                auto info = early_stoper.best();
                if (hp.show_summary()) { early_stoper.print_best(); }
                trees.resize(get<1>(info) + 1);
                break;
            }
        } else {
            if (hp.show_round_metrics()) { showloss(score, i); }
        }
    }
    if (early_stoper.is_continue && Eval.num > 0 && hp.show_summary()) { early_stoper.print_best(); }
    free(G);
    free(H);
}

void BoosterSingle::predict(double *features, double *preds, int n, int num_trees = 0) {
    if (num_trees == 0) { num_trees = int(trees.size()); }
    else { num_trees = min(num_trees, int(trees.size())); }
    for (int i = 0; i < num_trees; ++i) {
        trees[i].pred_value_single(features, preds, hp, n);
    }
}

void BoosterSingle::predict_multi(double *features, double *preds, int n, int out_dim, int num_trees = 0) {
    if (num_trees == 0) { num_trees = int(trees.size()); }
    else { num_trees = min(num_trees, int(trees.size())); }
    int t = 0;
    for (int i = 0; i < num_trees; ++i) {
        int start = 0;
        for (int j = 0; j < out_dim; ++j) {
            trees[t + j].pred_value_single(features, preds + start, hp, n);
            start += n;
        }
        t += out_dim;
    }
}


BoosterMulti::BoosterMulti(
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
    hp.inp_dim = inp_dim;
    hp.out_dim = out_dim;
    hp.topk = min(topk, out_dim);
    hp.loss = name;
    hp.max_depth = max_depth;
    hp.max_leaves = max_leaves;
    hp.seed = seed;
    hp.min_samples = min_samples;
    hp.num_threads = min(num_threads, omp_get_max_threads());
    hp.lr = lr;
    hp.reg_l1 = reg_l1;
    hp.reg_l2 = reg_l2;
    hp.gamma = gamma;
    hp.base_score = base_score;
    hp.base_scores.assign(hp.out_dim, base_score);
    hp.early_stop = early_stop;
    hp.one_side = one_side;
    hp.verbosity = max(0, min(verbosity, 2));
    hp.Max_caches = hist_cache;

    Score.resize(hp.out_dim);
    if (hp.topk > 0) {
        OptPair.resize(hp.topk);
    } else {
        Opt.resize(hp.out_dim);
    }

    omp_set_num_threads(hp.num_threads);
    cache = TopkDeque<CacheInfo>(hp.Max_caches);
    obj = Objective(hp.loss);
}

void BoosterMulti::get_score_opt(Histogram &Hist, vector<double> &opt, vector<double> &score, double &score_sum) {
    double *gr = &Hist.g[Hist.g.size() - hp.out_dim];
    double *hr = &Hist.h[Hist.h.size() - hp.out_dim];
    CalWeight(opt, gr, hr, hp.reg_l1, hp.reg_l2);
    CalScore(score, gr, hr, hp.reg_l1, hp.reg_l2);
    score_sum = 0.0f;
    for (int i = 0; i < hp.out_dim; ++i) {
        score_sum += score[i];
    }
}

void
BoosterMulti::get_score_opt(Histogram &Hist, vector<pair<double, int>> &opt, vector<double> &score, double &score_sum) {
    double *gr = &Hist.g[Hist.g.size() - hp.out_dim];
    double *hr = &Hist.h[Hist.h.size() - hp.out_dim];
    TopkPriority<pair<double, int>> score_k(hp.topk);
    for (int i = 0; i < hp.out_dim; ++i) {
        score[i] = CalScore(gr[i], hr[i], hp.reg_l1, hp.reg_l2);
        score_k.push(make_pair(score[i], i));
    }
    opt.resize(0);
    score_sum = 0.0f;
    while (!score_k.data.empty()) {
        auto top = score_k.data.top();
        score_sum += get<0>(top);
        int k = get<1>(top);
        opt.push_back(make_pair(CalWeight(gr[k], hr[k], hp.reg_l1, hp.reg_l2), k));
        score_k.data.pop();
    }
}

void BoosterMulti::hist_all(vector<int32_t> &order, vector<Histogram> &Hist) {
#pragma omp parallel for schedule(static)
    for (int i = 0; i < hp.inp_dim; ++i) {
        histogram_multi(order, Hist[i], Train.Maps + i * Train.num, G, H, hp.out_dim);
    }
}

SplitInfo BoosterMulti::boost_column_full(Histogram &Hist, int column) {
    double *gr = &Hist.g[Hist.g.size() - hp.out_dim];
    double *hr = &Hist.h[Hist.h.size() - hp.out_dim];
    double gain = 0.0f, tmp;
    int ind_l = 0, ind_l_tmp = 0, row_bins = -1;
    int max_bins = Hist.cnt.size() - 1;
    int total_rows = Hist.cnt[max_bins];

    for (int i = 0; i < max_bins; ++i) {
        int rows_l = Hist.cnt[i];
        int rows_r = total_rows - rows_l;
        if (rows_l < hp.min_samples || rows_r < hp.min_samples) {
            ind_l += hp.out_dim;
            continue;
        }
        tmp = 0.0f;
        for (int j = 0; j < hp.out_dim; ++j) {
            ind_l_tmp = ind_l + j;
            tmp += CalScore(Hist.g[ind_l_tmp], Hist.h[ind_l_tmp], hp.reg_l1, hp.reg_l2) + \
                   CalScore(gr[j] - Hist.g[ind_l_tmp], hr[j] - Hist.h[ind_l_tmp], hp.reg_l1, hp.reg_l2);
        }
        ind_l += hp.out_dim;
        if (tmp > gain) {
            gain = tmp;
            row_bins = i;
        }
    }
    gain -= Score_sum;
    gain *= 0.5 / hp.out_dim;
    SplitInfo candidate;
    candidate.reset();
    if (row_bins > -1) {
        candidate.update(gain, column, row_bins, bin_values[column][row_bins]);
    }
    return candidate;
}

SplitInfo BoosterMulti::boost_column_topk_two_side(Histogram &Hist, int column) {
    double *gr = &Hist.g[Hist.g.size() - hp.out_dim];
    double *hr = &Hist.h[Hist.h.size() - hp.out_dim];

    double gain = 0.0f, tmp;
    double score_l, score_r;
    int ind_l = 0, ind_l_tmp = 0, row_bins = -1;
    int max_bins = Hist.cnt.size() - 1;
    int total_rows = Hist.cnt[max_bins];

    auto pq_l = TopkPriority<pair<double, int>>(hp.topk);
    auto pq_r = TopkPriority<pair<double, int>>(hp.topk);

    for (int i = 0; i < max_bins; ++i) {
        int rows_l = Hist.cnt[i];
        int rows_r = total_rows - rows_l;
        if (rows_l < hp.min_samples || rows_r < hp.min_samples) {
            ind_l += hp.out_dim;
            continue;
        }
        for (int j = 0; j < hp.out_dim; ++j) {
            ind_l_tmp = ind_l + j;
            score_l = CalScore(Hist.g[ind_l_tmp], Hist.h[ind_l_tmp], hp.reg_l1, hp.reg_l2);
            score_r = CalScore(gr[j] - Hist.g[ind_l_tmp], hr[j] - Hist.h[ind_l_tmp], hp.reg_l1, hp.reg_l2);
            pq_l.push(make_pair(score_l, j));
            pq_r.push(make_pair(score_r, j));
        }
        ind_l += hp.out_dim;
        tmp = 0.0f;
        while (!pq_l.data.empty()) {
            tmp += get<0>(pq_l.data.top());
            pq_l.data.pop();
        }
        while (!pq_r.data.empty()) {
            tmp += get<0>(pq_r.data.top());
            pq_r.data.pop();
        }
        if (tmp > gain) {
            gain = tmp;
            row_bins = i;
        }
    }
    gain -= Score_sum;
    gain *= 0.5 / hp.topk;
    SplitInfo candidate;
    candidate.reset();
    if (row_bins > -1) {
        candidate.update(gain, column, row_bins, bin_values[column][row_bins]);
    }
    return candidate;
}

SplitInfo BoosterMulti::boost_column_topk_one_side(Histogram &Hist, int column) {
    double *gr = &Hist.g[Hist.g.size() - hp.out_dim];
    double *hr = &Hist.h[Hist.h.size() - hp.out_dim];

    double gain = 0.0f, tmp;
    double score_l, score_r;
    int ind_l = 0, ind_l_tmp = 0, row_bins = -1;
    int max_bins = Hist.cnt.size() - 1;
    int total_rows = Hist.cnt[max_bins];

    auto pq = TopkPriority<pair<double, int>>(hp.topk);

    for (int i = 0; i < max_bins; ++i) {
        int rows_l = Hist.cnt[i];
        int rows_r = total_rows - rows_l;
        if (rows_l < hp.min_samples || rows_r < hp.min_samples) {
            ind_l += hp.out_dim;
            continue;
        }
        for (int j = 0; j < hp.out_dim; ++j) {
            ind_l_tmp = ind_l + j;
            score_l = CalScore(Hist.g[ind_l_tmp], Hist.h[ind_l_tmp], hp.reg_l1, hp.reg_l2);
            score_r = CalScore(gr[j] - Hist.g[ind_l_tmp], hr[j] - Hist.h[ind_l_tmp], hp.reg_l1, hp.reg_l2);
            pq.push(make_pair(score_l + score_r, j));
        }
        ind_l += hp.out_dim;
        tmp = 0.0f;
        while (!pq.data.empty()) {
            tmp += get<0>(pq.data.top());
            pq.data.pop();
        }
        if (tmp > gain) {
            gain = tmp;
            row_bins = i;
        }
    }
    gain -= Score_sum;
    gain *= 0.5 / hp.topk;
    SplitInfo candidate;
    candidate.reset();
    if (row_bins > -1) {
        candidate.update(gain, column, row_bins, bin_values[column][row_bins]);
    }
    return candidate;
}

void BoosterMulti::boost_all(vector<Histogram> &Hist) {
    vector<SplitInfo> candidates(hp.inp_dim);
    if (hp.topk == 0) {
#pragma omp parallel for schedule(static)
        for (int i = 0; i < hp.inp_dim; i++) {
            candidates[i] = boost_column_full(Hist[i], i);
        }
    } else {
        if (hp.one_side) {
#pragma omp parallel for schedule(static)
            for (int i = 0; i < hp.inp_dim; i++) {
                candidates[i] = boost_column_topk_one_side(Hist[i], i);
            }
        } else {
#pragma omp parallel for schedule(static)
            for (int i = 0; i < hp.inp_dim; i++) {
                candidates[i] = boost_column_topk_two_side(Hist[i], i);
            }
        }
    }
    meta = select_best_split(candidates);
}

void BoosterMulti::build_tree_best() {
    if (tree.leaf_num >= hp.max_leaves) { return; }

    auto info = &cache.data[0];
    int parent = info->node;
    int depth = info->depth;

    int rows_l = info->hist[info->split.column].cnt[info->split.bin];
    int rows_r = info->order.size() - rows_l;

    vector<int32_t> order_l(rows_l), order_r(rows_r);
    rebuild_order(info->order, order_l, order_r,
                  Train.Maps + Train.num * info->split.column, info->split.bin);

    vector<Histogram> Hist_l(hp.inp_dim), Hist_r(hp.inp_dim);

    if (rows_l >= rows_r) {
        for (int i = 0; i < hp.inp_dim; i++) { Hist_r[i] = Histogram(bin_nums[i], hp.out_dim); }
        hist_all(order_r, Hist_r);
#pragma omp parallel for schedule(static)
        for (int i = 0; i < hp.inp_dim; ++i) { info->hist[i] - Hist_r[i]; }
        Hist_l.assign(info->hist.begin(), info->hist.end());
    } else {
        for (int i = 0; i < hp.inp_dim; i++) { Hist_l[i] = Histogram(bin_nums[i], hp.out_dim); }
        hist_all(order_l, Hist_l);
#pragma omp parallel for schedule(static)
        for (int i = 0; i < hp.inp_dim; ++i) { info->hist[i] - Hist_l[i]; }
        Hist_r.assign(info->hist.begin(), info->hist.end());
    }
    cache.data.pop_front();

    bool split_left = false;
    if (rows_l > 0) {
        if (hp.topk == 0) {
            get_score_opt(Hist_l[0], Opt, Score, Score_sum);
        } else {
            get_score_opt(Hist_l[0], OptPair, Score, Score_sum);
        }
        if (rows_l >= hp.min_samples) {
            boost_all(Hist_l);

            if (depth + 1 < hp.max_depth && meta.column > -1 && meta.gain > hp.gamma) {
                auto node = NonLeafNode(parent, meta.column, meta.bin, meta.threshold);
                tree.add_nonleaf(node, true);
                cache.push(CacheInfo(tree.nonleaf_num, depth + 1, meta, order_l, Hist_l));
                split_left = true;
            }
        }
    }
    if (!split_left) {
        auto node = LeafNode(hp.out_dim);
        if (rows_l > 0) {
            if (hp.topk > 0) { node.Update(parent, OptPair); }
            else { node.Update(parent, Opt); }
        } else {
            node.parent = parent;
        }
        tree.add_leaf(node, true);
    }
    order_l.clear();
    Hist_l.clear();

    if (tree.leaf_num >= hp.max_leaves) { return; }
    bool split_right = false;
    if (rows_r > 0) {
        if (hp.topk == 0) {
            get_score_opt(Hist_r[0], Opt, Score, Score_sum);
        } else {
            get_score_opt(Hist_r[0], OptPair, Score, Score_sum);
        }
        if (rows_r >= hp.min_samples) {
            boost_all(Hist_r);
            if (depth + 1 < hp.max_depth && meta.column > -1 && meta.gain > hp.gamma) {
                auto node = NonLeafNode(parent, meta.column, meta.bin, meta.threshold);
                tree.add_nonleaf(node, false);
                cache.push(CacheInfo(tree.nonleaf_num, depth + 1, meta, order_r, Hist_r));
                split_right = true;
            }
        }
    }
    if (!split_right) {
        auto node = LeafNode(hp.out_dim);
        if (rows_r > 0) {
            if (hp.topk > 0) { node.Update(parent, OptPair); }
            else { node.Update(parent, Opt); }
        } else {
            node.parent = parent;
        }
        tree.add_leaf(node, false);
    }
    order_r.clear();
    Hist_r.clear();

    if (!cache.data.empty()) { build_tree_best(); }
}

void BoosterMulti::update() {
    tree.shrinkage(hp.lr);
    tree.pred_value_multi(Train.Maps, Train.Preds, hp, Train.num);
    if (Eval.num > 0) {
        tree.pred_value_multi(Eval.Features, Eval.Preds, hp, Eval.num);
    }
    trees.push_back(tree);
}

void BoosterMulti::growth() {
    tree.clear();
    cache.data.clear();

    vector<Histogram> Hist(hp.inp_dim);
    for (int i = 0; i < hp.inp_dim; i++) { Hist[i] = Histogram(bin_nums[i], hp.out_dim); }
    hist_all(Train.Orders, Hist);
    if (hp.topk == 0) {
        get_score_opt(Hist[0], Opt, Score, Score_sum);
    } else {
        get_score_opt(Hist[0], OptPair, Score, Score_sum);
    }
    boost_all(Hist);
    if (meta.column > -1 & meta.gain > -10.0f) {
        auto node = NonLeafNode(-1, meta.column, meta.bin, meta.threshold);
        tree.add_nonleaf(node, true);
        cache.push(CacheInfo(-1, 0, meta, Train.Orders, Hist));
        build_tree_best();
    } else {
        auto node = LeafNode(hp.out_dim);
        if (hp.topk > 0) { node.Update(-1, OptPair); }
        else { node.Update(-1, Opt); }
        tree.add_leaf(node, true);
    }
}

void BoosterMulti::train(int num_rounds) {
    // initialize gradient and hessian
    int elements = Train.num * hp.out_dim;
    G = (double *) calloc(elements, sizeof(double));
    H = (double *) calloc(elements, sizeof(double));
    if (obj.constHessian) {
        double value = obj.hessian;
#pragma omp parallel for schedule(static)
        for (int i = 0; i < elements; ++i) {
            H[i] = value;
        }
    }

    int round = hp.early_stop;
    if (round == 0) { round = num_rounds; }
    auto early_stoper = EarlyStoper(round, obj.largerBetter);

    // start training
    for (int i = 0; i < num_rounds; ++i) {
        obj.f_grad(Train, G, H, Train.num, hp.out_dim);
        growth();
        update();
        double score = obj.f_score(Train, Train.num, hp.out_dim);
        if (Eval.num > 0) {
            double metric = obj.f_metric(Eval, Eval.num, hp.out_dim);
            if (hp.show_round_metrics()) { showloss(score, metric, i); }
            early_stoper.push(make_pair(metric, i));
            if (!early_stoper.is_continue) {
                auto info = early_stoper.best();
                if (hp.show_summary()) { early_stoper.print_best(); }
                trees.resize(get<1>(info) + 1);
                break;
            }
        } else {
            if (hp.show_round_metrics()) { showloss(score, i); }
        }
    }
    if (early_stoper.is_continue && Eval.num > 0 && hp.show_summary()) { early_stoper.print_best(); }
    free(G);
    free(H);
}

void BoosterMulti::predict(double *features, double *preds, int n, int num_trees = 0) {
    if (num_trees == 0) { num_trees = int(trees.size()); }
    else { num_trees = min(num_trees, int(trees.size())); }
    for (int i = 0; i < num_trees; ++i) {
        trees[i].pred_value_multi(features, preds, hp, n);
    }
}
