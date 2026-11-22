#include "booster.h"

namespace {

constexpr double kSplitTieEpsilon = 1e-12;

// Loss and histogram kernels inherit their calling task's OpenMP team size.
class ScopedTeamSize {
    const int previous;
public:
    explicit ScopedTeamSize(int requested) : previous(omp_get_max_threads()) {
        omp_set_num_threads(requested);
    }
    ~ScopedTeamSize() { omp_set_num_threads(previous); }
};

bool is_valid_split(const SplitInfo &split) {
    return split.column >= 0;
}

bool gain_is_greater(double lhs, double rhs, bool relative) {
    if (relative) {
        if (!std::isfinite(lhs) || !std::isfinite(rhs)) { return lhs > rhs; }
        return lhs > rhs && lhs - rhs > kSplitTieEpsilon * max(std::abs(lhs), std::abs(rhs));
    }
    return lhs > rhs + kSplitTieEpsilon;
}

bool threshold_is_less(double lhs, double rhs) {
    return lhs + kSplitTieEpsilon < rhs;
}

bool is_better_split(const SplitInfo &candidate, const SplitInfo &best, bool relative) {
    if (!is_valid_split(candidate)) {
        return false;
    }
    if (!is_valid_split(best)) {
        return true;
    }
    if (gain_is_greater(candidate.gain, best.gain, relative)) {
        return true;
    }
    if (gain_is_greater(best.gain, candidate.gain, relative)) {
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

SplitInfo select_best_split(const vector<SplitInfo> &candidates, bool relative = false) {
    SplitInfo best;
    best.reset();
    for (const auto &candidate : candidates) {
        if (is_better_split(candidate, best, relative)) {
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

int BoosterUtils::set_subsample(double fraction) {
    if (!(fraction > 0.0 && fraction <= 1.0)) { return -1; }
    hp.subsample = fraction;
    return 0;
}

int BoosterUtils::set_classification_metric(bool use_logloss) {
    if (obj.accuracy_metric == nullptr) { return -1; }
    obj.f_metric = use_logloss ? obj.f_score : obj.accuracy_metric;
    obj.largerBetter = !use_logloss;
    return 0;
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

    hp.base_score = hp.base_scores.front();
    configured_base_scores = hp.base_scores;
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

void BoosterUtils::restore_best(int tree_count, int round_width) {
    trim_trees(tree_count);
    for (Dataset *data : {&Train, &Eval}) {
        if (data->num == 0) { continue; }
        init_preds(data->Preds, data->num * round_width);
        for (size_t i = 0; i < trees.size(); ++i) {
            if (hp.out_dim > 1) {
                trees[i].pred_value_multi(data->Features, data->Preds, hp, data->num);
            } else {
                double *column = data->Preds + (i % round_width) * data->num;
                trees[i].pred_value_single(data->Features, column, hp, data->num);
            }
        }
    }
}

int BoosterUtils::load(const char *path, int round_width) {
    auto scores = configured_base_scores;
    int status = LoadTrees(trees, scores, path, hp.inp_dim, hp.out_dim, round_width);
    if (status == 0) {
        hp.base_scores.swap(scores);
        hp.base_score = hp.base_scores.front();
    }
    return status;
}

void BoosterUtils::set_data(uint16_t *x, double *y, double *z, int n, bool is_train) {
    Dataset &data = is_train ? Train : Eval;
    data.Label_double = nullptr;
    data.Label_int32 = nullptr;
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
    Dataset &data = is_train ? Train : Eval;
    data.Label_double = x;
    data.Label_int32 = nullptr;
}

void BoosterUtils::set_label(int32_t *x, bool is_train) {
    Dataset &data = is_train ? Train : Eval;
    data.Label_int32 = x;
    data.Label_double = nullptr;
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
    set_base_score(&base_score, 1);
    hp.early_stop = early_stop;
    hp.verbosity = max(0, min(verbosity, 2));
    hp.Max_caches = hist_cache;

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
    for (int i = 0; i < static_cast<int>(Hist.size()); i++) {
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

LeafNode BoosterSingle::fit_leaf(Histogram &hist, int parent) {
    get_score_opt(hist, Opt, Score_sum);
    LeafNode leaf(1);
    leaf.Update(parent, Opt);
    return leaf;
}

bool BoosterUtils::can_split(int depth, int rows, int leaves) const {
    return leaves < hp.max_leaves && depth < hp.max_depth && hp.Max_caches > 0 &&
           rows / 2 >= hp.min_samples;
}

vector<Histogram> BoosterUtils::build_histograms(vector<int32_t> &order, int columns) {
    vector<Histogram> hist;
    hist.reserve(columns);
    for (int i = 0; i < columns; ++i) { hist.emplace_back(bin_nums[i], hp.out_dim); }
    hist_all(order, hist);
    return hist;
}

void BoosterUtils::queue_split(int leaf, int depth, vector<int32_t> &order, vector<Histogram> &hist) {
    // A binary tree with k splits already has k + 1 terminal leaves.
    if (!can_split(depth, static_cast<int>(order.size()), 1 - tree.nonleaf_num)) {
        return;
    }
    boost_all(hist);
    if (meta.column >= 0 && meta.gain > minimum_split_score()) {
        // The full training order is reused by subsequent trees. Sampled root
        // rows and all child rows belong to this candidate.
        auto rows = &order == &Train.Orders ? vector<int32_t>(order) : std::move(order);
        cache.push(CacheInfo{leaf, depth, meta, std::move(rows), std::move(hist)});
    }
}

void BoosterUtils::boost() {
    ScopedTeamSize team(hp.num_threads);
    growth();
    update();
}

void BoosterUtils::growth() {
    tree.clear();
    cache.data.clear();

    vector<int32_t> sampled_rows;
    if (hp.subsample < 1.0) {
        sampled_rows = Train.Orders;
        // Derive the stream from the retained ensemble so that continued fits
        // and training after early-stop trimming use the same next sample.
        std::mt19937 sampler(hp.seed + static_cast<unsigned int>(trees.size()));
        std::shuffle(sampled_rows.begin(), sampled_rows.end(), sampler);
        sampled_rows.resize(max(1, static_cast<int>(Train.num * hp.subsample)));
        std::sort(sampled_rows.begin(), sampled_rows.end());
    }
    auto &root_rows = hp.subsample < 1.0 ? sampled_rows : Train.Orders;
    auto root_hist = build_histograms(root_rows, can_split(0, static_cast<int>(root_rows.size()), 1) ? hp.inp_dim : 1);
    tree.add_leaf(fit_leaf(root_hist[0], -1), true);
    queue_split(tree.leaf_num, 0, root_rows, root_hist);

    // Pending candidates are fitted leaves. A full cache or leaf budget can
    // discard a candidate without leaving a branch without a prediction.
    while (!cache.data.empty() && tree.leaf_num < hp.max_leaves) {
        CacheInfo info = std::move(cache.data.front());
        cache.data.pop_front();
        const auto &split = info.split;
        int rows_l = info.hist[split.column].cnt[split.bin];
        int rows_r = static_cast<int>(info.order.size()) - rows_l;
        vector<int32_t> order_l(rows_l), order_r(rows_r);
        rebuild_order(info.order, order_l, order_r, Train.Maps + Train.num * split.column, split.bin);

        // A terminal leaf takes its totals from feature zero. Preserve its summation order
        // while skipping all other features if neither child can produce a split.
        const int columns = can_split(info.depth + 1, max(rows_l, rows_r), tree.leaf_num + 1)
                            ? hp.inp_dim : 1;
        bool right_is_smaller = rows_r <= rows_l;
        auto smaller_hist = build_histograms(right_is_smaller ? order_r : order_l, columns);
#pragma omp parallel for schedule(static)
        for (int i = 0; i < columns; ++i) { info.hist[i] - smaller_hist[i]; }
        auto &hist_l = right_is_smaller ? info.hist : smaller_hist;
        auto &hist_r = right_is_smaller ? smaller_hist : info.hist;

        // Reuse the selected leaf's index for the left child so leaf IDs stay dense.
        int leaf = info.node;
        int parent = tree.leaf.at(leaf).parent;
        bool left = tree.nonleaf.empty() || tree.nonleaf.at(parent).left == leaf;
        tree.add_nonleaf(NonLeafNode(parent, split.column, split.bin, split.threshold), left);
        int node = tree.nonleaf_num;
        tree.nonleaf.at(node).left = leaf;
        tree.leaf.at(leaf) = fit_leaf(hist_l[0], node);
        queue_split(leaf, info.depth + 1, order_l, hist_l);
        tree.add_leaf(fit_leaf(hist_r[0], node), false);
        queue_split(tree.leaf_num, info.depth + 1, order_r, hist_r);
    }
    cache.data.clear();
}

void BoosterSingle::update() {
    tree.shrinkage(hp.lr);
    tree.pred_value_single(Train.Maps, Train.Preds, hp, Train.num);
    if (Eval.num > 0) {
        tree.pred_value_single(Eval.Features, Eval.Preds, hp, Eval.num);
    }
    trees.push_back(std::move(tree));
}

void BoosterSingle::train(int num_rounds) {
    if (num_rounds <= 0) { return; }
    ScopedTeamSize team(hp.num_threads);
    const int initial_trees = tree_count();
    const bool evaluate = Eval.has_labels();
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
        const double score = hp.show_round_metrics() ? obj.f_score(Train, Train.num, 1) : 0.0;
        if (evaluate) {
            double metric = obj.f_metric(Eval, Eval.num, 1);
            if (hp.show_round_metrics()) { showloss(score, metric, i); }
            early_stoper.push(make_pair(metric, i));
            if (!early_stoper.is_continue) {
                auto info = early_stoper.best();
                if (hp.show_summary()) { early_stoper.print_best(); }
                restore_best(initial_trees + info.second + 1);
                break;
            }
        } else {
            if (hp.show_round_metrics()) { showloss(score, i); }
        }
    }
    if (early_stoper.is_continue && evaluate && hp.show_summary()) { early_stoper.print_best(); }
    free(G);
    free(H);
    G = H = nullptr;
}

void BoosterSingle::train_multi(int num_rounds, int out_dim) {
    if (num_rounds <= 0) { return; }
    ScopedTeamSize team(hp.num_threads);
    const int initial_trees = tree_count();
    const bool evaluate = Eval.has_labels();
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
        const double score = hp.show_round_metrics() ? obj.f_score(Train, Train.num, out_dim) : 0.0;
        if (evaluate) {
            double metric = obj.f_metric(Eval, Eval.num, out_dim);
            if (hp.show_round_metrics()) { showloss(score, metric, i); }
            early_stoper.push(make_pair(metric, i));
            if (!early_stoper.is_continue) {
                auto info = early_stoper.best();
                if (hp.show_summary()) { early_stoper.print_best(); }
                restore_best(initial_trees + (info.second + 1) * out_dim, out_dim);
                break;
            }
        } else {
            if (hp.show_round_metrics()) { showloss(score, i); }
        }
    }
    if (early_stoper.is_continue && evaluate && hp.show_summary()) { early_stoper.print_best(); }
    free(G);
    free(H);
    G = H = nullptr;
}

void BoosterSingle::predict(double *features, double *preds, int n, int num_trees = 0) {
    if (num_trees == 0) { num_trees = int(trees.size()); }
    else { num_trees = min(num_trees, int(trees.size())); }
    for (int i = 0; i < num_trees; ++i) {
        trees[i].pred_value_single(features, preds, hp, n);
    }
}

void BoosterSingle::predict_multi(double *features, double *preds, int n, int out_dim, int num_trees = 0) {
    if (out_dim < 1) { return; }
    const int rounds = tree_count() / out_dim;
    num_trees = num_trees == 0 ? rounds : min(num_trees, rounds);
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
    metric_rank = out_dim;
    lsq_gamma = gamma;
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
    set_base_score(&base_score, 1);
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

    cache = TopkDeque<CacheInfo>(hp.Max_caches);
    obj = Objective(hp.loss);
}

bool BoosterMulti::supports_output_metric(int columns) const {
    return columns >= 2 && columns == hp.out_dim && obj.f_grad == mse_grad &&
           hp.topk == 0 && hp.reg_l1 == 0 && std::isfinite(hp.reg_l2) && hp.reg_l2 >= 0;
}

int BoosterMulti::set_projection_dim(int dim) {
    if (dim < 0 || dim > hp.out_dim || (dim > 0 && !supports_output_metric(hp.out_dim))) {
        return -1;
    }
    projection_dim = dim;
    return 0;
}

int BoosterMulti::set_diagonal_output_weights(const double *weights, int columns) {
    if (!weights || !supports_output_metric(columns)) { return -1; }
    bool uniform = true;
    for (int j = 0; j < columns; ++j) {
        if (!std::isfinite(weights[j]) || weights[j] < 0) { return -1; }
        uniform = uniform && weights[j] == weights[0];
    }
    if (!uniform) {
        vector<double> factor(columns);
        for (int j = 0; j < columns; ++j) { factor[j] = sqrt(weights[j]); }
        metric_factor.swap(factor);
        metric_rows = 0;
        metric_rank = static_cast<int>(std::count_if(weights, weights + columns, [](double w) { return w > 0; }));
        rmse_scale = 1.0;
        return 0;
    }
    if (weights[0] == 0) { return -1; }
    use_uniform_weight(weights[0]);
    return 0;
}

void BoosterMulti::use_uniform_weight(double weight) {
    // For w*I the weighted gain is d*w times the ordinary gain. Keep the
    // entire LSQ tree builder (including tie-breaking) in its original units.
    const int columns = hp.out_dim;
    const double scale = weight == 1.0 / columns ? 1.0 : weight * columns;
    if (std::isfinite(scale)) {
        lsq_gamma = hp.gamma / scale;
        rmse_scale = sqrt(scale);
    } else {
        lsq_gamma = (hp.gamma / weight) / columns;
        rmse_scale = sqrt(weight) * sqrt(static_cast<double>(columns));
    }
    metric_factor.clear();
    metric_rank = columns;
}

int BoosterMulti::set_output_metric(const double *factor, int rows, int columns) {
    if (!factor || !supports_output_metric(columns) || rows < 0 || rows > columns) { return -1; }
    const size_t width = static_cast<size_t>(columns);
    const size_t height = static_cast<size_t>(max(rows, 1));
    if (height > metric_factor.max_size() / width) { return -1; }
    const size_t size = height * width;
    bool nonzero = false;
    for (size_t i = 0; i < size; ++i) {
        if (!std::isfinite(factor[i]) || (rows == 0 && factor[i] < 0)) { return -1; }
        nonzero = nonzero || factor[i] != 0;
    }
    if (!nonzero) { return -1; }
    // Older callers supply square-root diagonal weights to this entry point.
    if (rows == 0 && std::all_of(factor + 1, factor + size, [&](double v) { return v == factor[0]; })) {
        const double canonical = 1.0 / columns;
        const double weight = factor[0] == sqrt(canonical) ? canonical : factor[0] * factor[0];
        if (weight > 0 && std::isfinite(weight)) {
            use_uniform_weight(weight);
            return 0;
        }
    }
    metric_factor.assign(factor, factor + size);
    metric_rows = rows;
    metric_rank = rows ? rows : static_cast<int>(std::count_if(factor, factor + columns, [](double w) { return w > 0; }));
    rmse_scale = 1.0;
    return 0;
}

void BoosterMulti::project_residual(const double *residual, double *projected) const {
    if (metric_rows == 0) {
        for (int j = 0; j < hp.out_dim; ++j) { projected[j] = metric_factor[j] * residual[j]; }
        return;
    }
    for (int i = 0; i < metric_rows; ++i) {
        const double *row = metric_factor.data() + static_cast<size_t>(i) * hp.out_dim;
        double value = 0;
        for (int j = 0; j < hp.out_dim; ++j) { value += row[j] * residual[j]; }
        projected[i] = value;
    }
}

double BoosterMulti::quadratic_rmse(const Dataset &data) const {
    const int block_size = 256;
    const int blocks = 1 + (data.num - 1) / block_size;
    const double divisor = sqrt(static_cast<double>(data.num));
    const int width = projected_width();
    vector<double> norms(blocks);
    // Fixed blocks give a deterministic reduction. hypot avoids squaring large errors.
#pragma omp parallel for schedule(static) if (blocks > 1)
    for (int block = 0; block < blocks; ++block) {
        vector<double> residual(hp.out_dim), projected(width);
        const int start = block * block_size;
        const int stop = start + min(block_size, data.num - start);
        double norm = 0;
        for (int i = start; i < stop; ++i) {
            const size_t offset = static_cast<size_t>(i) * hp.out_dim;
            for (int j = 0; j < hp.out_dim; ++j) {
                const double pred = data.Preds[offset + j], label = data.Label_double[offset + j];
                const double error = pred - label;
                residual[j] = std::isfinite(error) ? error / divisor : pred / divisor - label / divisor;
            }
            project_residual(residual.data(), projected.data());
            for (double value : projected) { norm = std::hypot(norm, value); }
        }
        norms[block] = norm;
    }
    double norm = 0;
    for (double part : norms) { norm = std::hypot(norm, part); }
    return norm;
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
    for (int i = 0; i < static_cast<int>(Hist.size()); ++i) {
        histogram_multi(order, Hist[i], Train.Maps + i * Train.num, G, H, hp.out_dim);
    }
}

vector<Histogram> BoosterMulti::build_histograms(vector<int32_t> &order, int columns) {
    if (!uses_projection()) { return BoosterUtils::build_histograms(order, columns); }

    // Every feature needs k split channels. Only feature zero needs raw residual
    // sums to fit the leaves, including leaves obtained by sibling subtraction.
    vector<Histogram> hist;
    hist.reserve(columns);
    for (int col = 0; col < columns; ++col) {
        hist.emplace_back(bin_nums[col], col == 0 ? hp.out_dim : 0, projection_dim);
    }
#pragma omp parallel for schedule(static)
    for (int col = 0; col < columns; ++col) {
        auto &bins = hist[col];
        const uint16_t *maps = Train.Maps + static_cast<size_t>(col) * Train.num;
        const int raw_width = col == 0 ? hp.out_dim : 0;
        for (int row : order) {
            const size_t bin = maps[row];
            ++bins.cnt[bin];
            for (int j = 0; j < projection_dim; ++j) {
                bins.h[bin * projection_dim + j] += H[static_cast<size_t>(row) * projection_dim + j];
            }
            for (int j = 0; j < raw_width; ++j) {
                bins.g[bin * raw_width + j] += G[static_cast<size_t>(row) * raw_width + j];
            }
        }
        for (size_t bin = 1; bin < bins.cnt.size(); ++bin) { bins.cnt[bin] += bins.cnt[bin - 1]; }
        for (size_t i = projection_dim; i < bins.h.size(); ++i) { bins.h[i] += bins.h[i - projection_dim]; }
        for (size_t i = raw_width; i < bins.g.size(); ++i) { bins.g[i] += bins.g[i - raw_width]; }
    }
    return hist;
}

SplitInfo BoosterMulti::boost_column_full(Histogram &Hist, int column) {
    const bool projected = uses_projection();
    const int width = projected ? projection_dim : hp.out_dim;
    const auto &grad = projected ? Hist.h : Hist.g;
    const double *gr = grad.data() + grad.size() - width;
    const double *hr = Hist.h.data() + Hist.h.size() - width;
    double gain = 0.0f, tmp;
    int ind_l = 0, ind_l_tmp = 0, row_bins = -1;
    int max_bins = Hist.cnt.size() - 1;
    int total_rows = Hist.cnt[max_bins];

    for (int i = 0; i < max_bins; ++i) {
        int rows_l = Hist.cnt[i];
        int rows_r = total_rows - rows_l;
        if (rows_l < hp.min_samples || rows_r < hp.min_samples) {
            ind_l += width;
            continue;
        }
        tmp = 0.0f;
        for (int j = 0; j < width; ++j) {
            ind_l_tmp = ind_l + j;
            const double left_h = projected ? rows_l : Hist.h[ind_l_tmp];
            const double right_h = projected ? rows_r : hr[j] - left_h;
            tmp += CalScore(grad[ind_l_tmp], left_h, hp.reg_l1, hp.reg_l2) +
                   CalScore(gr[j] - grad[ind_l_tmp], right_h, hp.reg_l1, hp.reg_l2);
        }
        ind_l += width;
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

double BoosterMulti::minimum_split_score() const {
    if (metric_factor.empty()) { return lsq_gamma; }
    // Compare thresholds in the same signed square-root units as quadratic gains.
    return std::copysign(sqrt(std::abs(hp.gamma)), hp.gamma);
}

SplitInfo BoosterMulti::boost_column_quadratic(const Histogram &hist, int column) const {
    const int bins = static_cast<int>(hist.cnt.size()) - 1;
    const int n = hist.cnt.back();
    const bool projected = uses_projection();
    const int width = projected ? projection_dim : projected_width();
    const int stride = projected ? projection_dim : hp.out_dim;
    const double *total = hist.h.data() + static_cast<size_t>(bins) * stride;
    const double parent_weight = n + hp.reg_l2;
    const double half_child_weight = 0.5 * n + hp.reg_l2;

    // For a node the half-score is ||sum(R e)||² / (2 (n + lambda)).
    // Children minus parent simplifies to a weighted squared distance between
    // child means, less the cost of a second ridge penalty on their common mean.
    const double penalty_scale = sqrt((0.25 * parent_weight / half_child_weight) * hp.reg_l2);
    double penalty = 0;
    for (int j = 0; j < width; ++j) {
        penalty = std::hypot(penalty, penalty_scale * (total[j] / parent_weight));
    }

    SplitInfo candidate;
    candidate.reset();
    double best_distance = 0;
    int best_bin = -1;
    for (int i = 0; i < bins; ++i) {
        const int left_rows = hist.cnt[i], right_rows = n - left_rows;
        if (left_rows < hp.min_samples || right_rows < hp.min_samples ||
            (i > 0 && left_rows == hist.cnt[i - 1])) { continue; }
        const double left_weight = left_rows + hp.reg_l2, right_weight = right_rows + hp.reg_l2;
        const double scale = sqrt((0.25 * left_weight / half_child_weight) * right_weight);
        const double *left = hist.h.data() + static_cast<size_t>(i) * stride;
        double distance = 0;
        for (int j = 0; j < width; ++j) {
            const double left_mean = left[j] / left_weight;
            const double right_mean = (total[j] - left[j]) / right_weight;
            distance = std::hypot(distance, scale * left_mean - scale * right_mean);
        }
        if (distance > best_distance) {
            best_distance = distance;
            best_bin = i;
        }
    }
    if (best_bin >= 0) {
        // The gain is distance² - penalty². Its signed square root preserves
        // ordering across nodes even when the squared norms exceed double range.
        const double high = max(best_distance, penalty), low = min(best_distance, penalty);
        const double magnitude = high * sqrt(((high - low) / high) * (1.0 + low / high));
        const double gain = std::copysign(magnitude, best_distance - penalty);
        if (gain > minimum_split_score()) {
            candidate.update(gain, column, best_bin, bin_values[column][best_bin]);
        }
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
            candidates[i] = metric_factor.empty() ? boost_column_full(Hist[i], i)
                                                  : boost_column_quadratic(Hist[i], i);
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
    meta = select_best_split(candidates, !metric_factor.empty());
}

LeafNode BoosterMulti::fit_leaf(Histogram &hist, int parent) {
    LeafNode leaf(hp.out_dim);
    if (uses_projection() || !metric_factor.empty()) {
        // The same metric on the residuals and L2 penalty leaves the optimal
        // coefficient update equal to the ordinary shrunken mean residual.
        const double *sum = hist.g.data() + hist.g.size() - hp.out_dim;
        const double weight = hist.cnt.back() + hp.reg_l2;
        for (int j = 0; j < hp.out_dim; ++j) { Opt[j] = -sum[j] / weight; }
        if (metric_factor.empty()) {
            const double *total = hist.h.data() + hist.h.size() - projection_dim;
            Score_sum = 0.0;
            for (int j = 0; j < projection_dim; ++j) {
                Score_sum += CalScore(total[j], hist.cnt.back(), 0.0, hp.reg_l2);
            }
        }
        leaf.Update(parent, Opt);
    } else if (hp.topk > 0) {
        get_score_opt(hist, OptPair, Score, Score_sum);
        leaf.Update(parent, OptPair);
    } else {
        get_score_opt(hist, Opt, Score, Score_sum);
        leaf.Update(parent, Opt);
    }
    return leaf;
}

void BoosterMulti::update() {
    tree.shrinkage(hp.lr);
    tree.pred_value_multi(Train.Maps, Train.Preds, hp, Train.num);
    if (Eval.num > 0) {
        tree.pred_value_multi(Eval.Features, Eval.Preds, hp, Eval.num);
    }
    trees.push_back(std::move(tree));
}

void BoosterMulti::boost() {
    // Manual gradients and Hessians have no built-in MSE/metric contract.
    if (projection_dim == 0) { BoosterUtils::boost(); }
}

void BoosterMulti::project_split_gradients() {
    const int input_width = projected_width();
    const double divisor = sqrt(static_cast<double>(projection_dim));
    vector<double> directions(static_cast<size_t>(projection_dim) * input_width);
    // A fresh stream per retained tree makes continued fits and early-stop
    // trimming reproducible. Row sampling has its own independent generator.
    std::seed_seq seed{hp.seed, static_cast<unsigned>(trees.size())};
    std::mt19937 rng(seed);
    std::normal_distribution<double> normal;
    for (auto &value : directions) { value = normal(rng) / divisor; }

#pragma omp parallel
    {
        vector<double> metric_residual(metric_factor.empty() ? 0 : input_width);
#pragma omp for schedule(static)
        for (int row = 0; row < Train.num; ++row) {
            const double *residual = G + static_cast<size_t>(row) * hp.out_dim;
            if (!metric_factor.empty()) {
                project_residual(residual, metric_residual.data());
                residual = metric_residual.data();
            }
            for (int j = 0; j < projection_dim; ++j) {
                const double *direction = directions.data() + static_cast<size_t>(j) * input_width;
                double sum = 0.0;
                for (int t = 0; t < input_width; ++t) { sum += residual[t] * direction[t]; }
                H[static_cast<size_t>(row) * projection_dim + j] = sum;
            }
        }
    }
}

void BoosterMulti::train(int num_rounds) {
    if (num_rounds <= 0) { return; }
    ScopedTeamSize team(hp.num_threads);
    const int initial_trees = tree_count();
    const bool evaluate = Eval.has_labels();
    const bool quadratic = !metric_factor.empty();
    const bool projected = uses_projection();
    // Compare uniform-weight fits in ordinary RMSE units, while displaying
    // the error in the units of the supplied loss.
    auto score_data = [&](Dataset &data, func_score ordinary, double &comparison) {
        if (rmse_scale != 1.0) {
            return scaled_mse_score(data, data.num, hp.out_dim, rmse_scale, comparison);
        }
        comparison = quadratic ? quadratic_rmse(data) : ordinary(data, data.num, hp.out_dim);
        return comparison;
    };
    // initialize gradient and hessian
    const int64_t elements = static_cast<int64_t>(Train.num) * hp.out_dim;
    G = (double *) calloc(elements, sizeof(double));
    H = (double *) calloc(projected ? static_cast<size_t>(Train.num) * projection_dim : elements, sizeof(double));
    if (obj.constHessian && !quadratic && !projected) {
        double value = obj.hessian;
#pragma omp parallel for schedule(static)
        for (int64_t i = 0; i < elements; ++i) {
            H[i] = value;
        }
    }

    int round = hp.early_stop;
    if (round == 0) { round = num_rounds; }
    auto early_stoper = EarlyStoper(round, obj.largerBetter);

    // start training
    for (int i = 0; i < num_rounds; ++i) {
        obj.f_grad(Train, G, H, Train.num, hp.out_dim);
        if (projected) {
            project_split_gradients();
        } else if (quadratic) {
            // MSE has unit Hessians, already represented by histogram counts.
            // Keep raw residual sums in g for leaves and projected sums in h for splits.
#pragma omp parallel for schedule(static)
            for (int row = 0; row < Train.num; ++row) {
                const size_t offset = static_cast<size_t>(row) * hp.out_dim;
                project_residual(G + offset, H + offset);
            }
        }
        growth();
        update();
        double comparison, score = 0;
        if (hp.show_round_metrics()) {
            score = score_data(Train, obj.f_score, comparison);
        }
        if (evaluate) {
            double metric = score_data(Eval, obj.f_metric, comparison);
            if (hp.show_round_metrics()) { showloss(score, metric, i); }
            early_stoper.push(make_pair(comparison, i), metric);
            if (!early_stoper.is_continue) {
                auto info = early_stoper.best();
                if (hp.show_summary()) { early_stoper.print_best(); }
                restore_best(initial_trees + info.second + 1);
                break;
            }
        } else {
            if (hp.show_round_metrics()) { showloss(score, i); }
        }
    }
    if (early_stoper.is_continue && evaluate && hp.show_summary()) { early_stoper.print_best(); }
    free(G);
    free(H);
    G = H = nullptr;
}

void BoosterMulti::predict(double *features, double *preds, int n, int num_trees = 0) {
    if (num_trees == 0) { num_trees = int(trees.size()); }
    else { num_trees = min(num_trees, int(trees.size())); }
    for (int i = 0; i < num_trees; ++i) {
        trees[i].pred_value_multi(features, preds, hp, n);
    }
}
