#include "loss.h"
#include <cmath>
#include <limits>

namespace {

// Fixed blocks keep early-stopping comparisons independent of OpenMP scheduling.
template<typename Term>
double ordered_sum(int n, Term term) {
    const int block_size = 256;
    const int blocks = 1 + (n - 1) / block_size;
    vector<double> partial(blocks);
#pragma omp parallel for schedule(static) if (blocks > 1)
    for (int block = 0; block < blocks; ++block) {
        const int begin = block * block_size;
        const int end = begin + min(block_size, n - begin);
        double sum = 0.0;
        for (int i = begin; i < end; ++i) { sum += term(i); }
        partial[block] = sum;
    }
    double sum = 0.0;
    for (double value : partial) { sum += value; }
    return sum;
}

template<typename ScaledLoss>
double mean_log_loss(int n, ScaledLoss loss) {
    const double mean = ordered_sum(n, [&](int i) { return loss(i, n); });
    if (!std::isinf(mean)) { return mean; }

    // Rounding each loss / n can overflow the final sum even when the mean
    // is exactly DBL_MAX. Recenter near that boundary in the rare overflow case.
    // Halving first accommodates a finite logit gap up to twice DBL_MAX.
    const double limit = std::numeric_limits<double>::max();
    const double adjustment = ordered_sum(n, [&](int i) {
        return (loss(i, 2) - limit * 0.5) / n;
    });
    return limit + adjustment * 2;
}

double multiclass_log_loss(Dataset &data, int n, int classes, size_t row_stride, size_t class_stride) {
    return mean_log_loss(n, [&](int i, int divisor) {
        const double *row = data.Preds + i * row_stride;
        int largest = 0;
        for (int j = 1; j < classes; ++j) {
            if (row[j * class_stride] > row[largest * class_stride]) { largest = j; }
        }
        const double peak = row[largest * class_stride];
        double rest = 0.0;
        for (int j = 0; j < classes; ++j) {
            if (j != largest) { rest += exp(row[j * class_stride] - peak); }
        }
        // Center before adding the log normalizer to retain the loss when all
        // logits share a large offset. Scale extreme losses before reduction.
        const double target = row[data.Label_int32[i] * class_stride];
        const double gap = peak - target;
        const double scaled_gap = std::isfinite(gap) ? gap / divisor : peak / divisor - target / divisor;
        return scaled_gap + log1p(rest) / divisor;
    });
}

}  // namespace

void mse_grad(Dataset &data, double *g, double *h, int n, int out_dim = 1) {
    auto preds = data.Preds;
    auto labels = data.Label_double;
    int N = n * out_dim;
#pragma omp parallel for schedule(static)
    for (int i = 0; i < N; ++i) {
        g[i] = preds[i] - labels[i];
    }
}

double mse_score(Dataset &data, int n, int out_dim = 1) {
    double rmse;
    return scaled_mse_score(data, n, out_dim, 1.0, rmse);
}

double scaled_mse_score(Dataset &data, int n, int out_dim, double score_scale, double &rmse) {
    auto preds = data.Preds;
    auto labels = data.Label_double;
    int N = n * out_dim;
    const double mean_square = ordered_sum(N, [&](int i) { return Sqr(preds[i] - labels[i]); }) / N;
    if (std::isnormal(mean_square) || std::isnan(mean_square)) {
        rmse = sqrt(mean_square);
        return rmse * score_scale;
    }

    // A representable RMSE can have a squared sum outside the double range.
    // Rescale only at the extremes so ordinary fits keep their reduction order.
    double scale = 0;
    for (int i = 0; i < N; ++i) {
        const double error = preds[i] - labels[i];
        const double magnitude = std::isfinite(error) ? std::abs(error) : max(std::abs(preds[i]), std::abs(labels[i]));
        scale = max(scale, magnitude);
    }
    if (scale == 0 || !std::isfinite(scale)) {
        rmse = sqrt(mean_square);
        return rmse * score_scale;
    }
    const double normalized = ordered_sum(N, [&](int i) {
        const double error = preds[i] - labels[i];
        return Sqr(std::isfinite(error) ? error / scale : preds[i] / scale - labels[i] / scale);
    });
    const double normalized_rmse = sqrt(normalized / N);
    rmse = scale * normalized_rmse;
    return scale * (normalized_rmse * score_scale);
}

void bce_grad(Dataset &data, double *g, double *h, int n, int out_dim) {
    auto preds = data.Preds;
    auto labels = data.Label_int32;
    int N = n * out_dim;
#pragma omp parallel for schedule(static)
    for (int i = 0; i < N; ++i) {
        const double t = 1.0f / (1.0f + exp(-preds[i]));
        g[i] = t - labels[i];
        h[i] = t * (1 - t);
    }
}

double bce_score(Dataset &data, int n, int out_dim) {
    auto preds = data.Preds;
    auto labels = data.Label_int32;
    int N = n * out_dim;
    return mean_log_loss(N, [&](int i, int divisor) {
        const double margin = labels[i] == 1 ? -preds[i] : preds[i];
        return (max(margin, 0.0) + log1p(exp(-abs(margin)))) / divisor;
    });
}

void ce_grad_thread(Dataset &data, double *g, double *h, int n, int out_dim, int start) {
    auto preds = data.Preds + start * out_dim;
    auto labels = data.Label_int32 + start;
    vector<double> rec(out_dim);
    int i, j, idx = 0;
    for (i = 0; i < n; ++i) {
        for (j = 0; j < out_dim; ++j) { rec[j] = preds[idx + j]; }
        Softmax(rec);
        for (j = 0; j < out_dim; ++j) {
            g[idx + j] = rec[j];
            h[idx + j] = rec[j] * (1.0f - rec[j]);
        }
        g[idx + labels[i]] -= 1.0f;
        idx += out_dim;
    }
}

void ce_grad(Dataset &data, double *g, double *h, int n, int out_dim) {
    if (n < 32) {
        ce_grad_thread(data, g, h, n, out_dim, 0);
    } else {
#pragma omp parallel
        {
            int threads = omp_get_num_threads();
            int id = omp_get_thread_num();
            int start = n / threads * id;
            int sample;
            if (id == threads - 1) {
                sample = n - n / threads * id;
            } else {
                sample = n / threads;
            }
            ce_grad_thread(data, g + start * out_dim, h + start * out_dim, sample, out_dim, start);
        };
    }
}

double ce_score(Dataset &data, int n, int out_dim) {
    return multiclass_log_loss(data, n, out_dim, out_dim, 1);
}

void ce_grad_column_thread(Dataset &data, double *g, double *h, int n, int out_dim, int threads, int thread) {
    auto preds = data.Preds;
    auto labels = data.Label_int32;
    int i, j;
    vector<int> idx(out_dim);
    vector<double> rec(out_dim);
    for (i = thread; i < n; i += threads) {
        idx[0] = i;
        rec[0] = preds[i];
        for (j = 1; j < out_dim; ++j) {
            idx[j] = idx[j - 1] + n;
            rec[j] = preds[idx[j]];
        }
        Softmax(rec);
        for (j = 0; j < out_dim; ++j) {
            g[idx[j]] = rec[j];
            h[idx[j]] = rec[j] * (1.0f - rec[j]);
        }
        g[idx[labels[i]]] -= 1.0f;
    }
}

void ce_grad_column(Dataset &data, double *g, double *h, int n, int out_dim) {
    if (n < 32) {
        ce_grad_column_thread(data, g, h, n, out_dim, 1, 0);
    } else {
#pragma omp parallel
        {
            int threads = omp_get_num_threads();
            int id = omp_get_thread_num();
            ce_grad_column_thread(data, g, h, n, out_dim, threads, id);
        };
    }
}

double ce_score_column(Dataset &data, int n, int out_dim) {
    return multiclass_log_loss(data, n, out_dim, 1, n);
}

double acc_multiclass(Dataset &data, int n, int out_dim) {
    auto preds = data.Preds;
    auto labels = data.Label_int32;
    int acc = 0;
#pragma omp parallel for schedule(static) reduction(+:acc)
    for (int i = 0; i < n; ++i) {
        int idx = i * out_dim;
        double score = preds[idx];
        int ind = 0;
        for (int j = 1; j < out_dim; j++) {
            ++idx;
            if (preds[idx] > score) {
                score = preds[idx];
                ind = j;
            }
        }
        if (ind == labels[i]) { ++acc; }
    }
    return static_cast<double> (acc) / n;
}

double acc_multiclass_column(Dataset &data, int n, int out_dim) {
    auto preds = data.Preds;
    auto labels = data.Label_int32;
    int acc = 0;
#pragma omp parallel for schedule(static) reduction(+:acc)
    for (int i = 0; i < n; ++i) {
        int idx = i;
        double score = preds[idx];
        int ind = 0;
        for (int j = 1; j < out_dim; j++) {
            idx += n;
            if (preds[idx] > score) {
                score = preds[idx];
                ind = j;
            }
        }
        if (ind == labels[i]) { ++acc; }
    }
    return static_cast<double> (acc) / n;
}

double acc_binary(Dataset &data, int n, int out_dim) {
    auto preds = data.Preds;
    auto labels = data.Label_int32;
    int acc = 0, N = n * out_dim;
#pragma omp parallel for schedule(static) reduction(+:acc)
    for (int i = 0; i < N; ++i) {
        if (labels[i] == 1) {
            if (preds[i] >= 0.0f) { ++acc; }
        } else {
            if (preds[i] < 0.0f) { ++acc; }
        }
    }
    return static_cast<double> (acc) / N;
}


