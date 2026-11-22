#ifndef MGBDT_DATASTRUCTURE_H
#define MGBDT_DATASTRUCTURE_H

#include <math.h>
#include <queue>
#include <iostream>
#include <random>
#include <utility>
#include <omp.h>
#include <limits.h>

using namespace std;

struct Histogram {
    Histogram(int n = 1, int d = 1) : Histogram(n, d, d) {}
    Histogram(int bins, int g_width, int h_width) :
        cnt(bins), g(static_cast<size_t>(bins) * g_width), h(static_cast<size_t>(bins) * h_width) {}

    vector<int> cnt;
    vector<double> g;
    vector<double> h;

    inline void operator-(const Histogram &x) {
        for (size_t i = 0; i < cnt.size(); ++i) cnt[i] -= x.cnt[i];
        for (size_t i = 0; i < g.size(); ++i) g[i] -= x.g[i];
        for (size_t i = 0; i < h.size(); ++i) h[i] -= x.h[i];
    }
};

void histogram_single(vector<int32_t> &, Histogram &, uint16_t *, double *, double *);

void histogram_multi(vector<int32_t> &, Histogram &, uint16_t *, double *, double *, int);

struct Dataset {
    int num = 0;
    double *Features;
    uint16_t *Maps;
    double *Preds;
    std::vector<int32_t> Orders;
    std::vector<int32_t> LeafIndex;

    double *Label_double = nullptr;
    int32_t *Label_int32 = nullptr;

    bool has_labels() const { return num > 0 && (Label_double || Label_int32); }
};

struct HyperParameter {
    bool hist = true;
    bool one_side = true;
    bool best_first = true;
    const char *loss = "mse";
    int max_depth = 5;
    int max_leaves = int(pow(2, max_depth));
    int inp_dim;
    int out_dim = 1;
    int min_samples = 2;
    int num_threads = 1;
    double lr = 0.2;
    double reg_l1 = 0.0;
    double reg_l2 = 1.0;
    double gamma = 1e-3;
    double subsample = 1.0;
    double max_delta = 10.0;
    unsigned int seed = 0;
    double base_score = 0.0f;
    vector<double> base_scores;
    int topk = 0;
    int Max_caches = 16;
    int early_stop = 0;
    int verbosity = 2;

    inline bool show_summary() const { return verbosity >= 1; }
    inline bool show_round_metrics() const { return verbosity >= 2; }
};

// access to all top-k elements.
template<class T>
class TopkDeque {
private:
    int max_size;

public:
    deque<T> data;

    TopkDeque(int n = 8) : max_size(n) {};

    //binary search
    inline int search(T const &x) {
        int l = 0, r = data.size();
        while (l < r) {
            int m = (r + l - 1) / 2;
            if (x > data[m]) {
                r = m;
            } else {
                l = m + 1;
            }
        }
        return l;
    }

    inline void push(T &&x) {
        if (data.size() == max_size) {
            if (x > data[max_size - 1]) {
                int i = search(x);
                data.insert(data.begin() + i, std::move(x));
                data.pop_back();
            }
        } else {
            int i = search(x);
            data.insert(data.begin() + i, std::move(x));
        }
    }

};


// access only the largest element.
template<class T>
class TopkPriority {
private:
    int k = 8;

public:
    priority_queue<T, vector<T>, greater<T>> data;

    TopkPriority(int n) : k(n) {};

    inline void push(T const &x) {
        data.push(x);
        if (data.size() > k) {
            data.pop();
        }
    }
};

class EarlyStoper {
private:
    int k = 10;
    bool larger_better = true;
    pair<double, int> info{0.0, -1};
    double reported_best = 0;

public:
    bool is_continue = true;

    EarlyStoper() {};

    EarlyStoper(int n, bool state) : k(n), larger_better(state) {}

    inline pair<double, int> best() const {
        return info;
    }

    inline void print_best() const {
        cout << "Best score " << reported_best << " at round " << get<1>(info) << endl;
    }

    inline void push(const pair<double, int> &x) {
        push(x, x.first);
    }

    inline void push(const pair<double, int> &x, double reported_score) {
        if (info.second < 0 || (larger_better ? x.first > info.first : x.first < info.first)) {
            info = x;
            reported_best = reported_score;
        }
        is_continue = x.second - info.second < k;
    }
};


#endif //MGBDT_DATASTRUCTURE_H
