#ifndef MGBDT_MATH_H
#define MGBDT_MATH_H

#include <vector>
#include <math.h>
#include <limits>
#include <unordered_set>

using namespace std;

template<typename T>
inline static T Sum(vector<T> x) {
    T s = 0;
    for (auto &i : x) { s += i; }
    return s;
}

template<typename T>
inline static T Sqr(T x) { return x * x; }

inline static double ThresholdL1(double w, double reg_l1) {
    if (w > +reg_l1) return w - reg_l1;
    if (w < -reg_l1) return w + reg_l1;
    return 0.0;
}

// Nonnegative binary observation curvatures satisfy H >= (epsilon/4) * |G|
// under bce_grad, a bound inherited by exact sums and by L1 shrinkage.
// Sibling or prefix subtraction can lose a tiny Hessian. In that case use
// a finite corrective step of at most four log-odds, as for saturated rows.
inline static double NewtonCurvature(double grad, double curvature, bool binary) {
    if (binary && (curvature <= 0.0 ||
                   curvature < std::abs(grad) * (std::numeric_limits<double>::epsilon() / 4))) {
        return 0.25 * std::max(1.0, std::abs(grad));
    }
    return curvature;
}

inline static double CalWeight(double g_sum, double h_sum, double reg_l1, double reg_l2, bool binary = false) {
    const double grad = reg_l1 == 0 ? g_sum : ThresholdL1(g_sum, reg_l1);
    const double curvature = NewtonCurvature(grad, h_sum + reg_l2, binary);
    // A locally flat objective admits a zero update, including unregularized leaves.
    if (grad == 0.0 && curvature == 0.0) { return 0.0; }
    return -grad / curvature;
}

inline static void CalWeight(vector<double> &value, double *g_sum, double *h_sum,
                             double reg_l1, double reg_l2, bool binary = false) {
    for (size_t i = 0; i < value.size(); ++i) {
        value[i] = CalWeight(g_sum[i], h_sum[i], reg_l1, reg_l2, binary);
    }
}

inline static double CalScore(double g_sum, double h_sum, double reg_l1, double reg_l2, bool binary = false) {
    const double grad = reg_l1 == 0 ? g_sum : ThresholdL1(g_sum, reg_l1);
    const double curvature = NewtonCurvature(grad, h_sum + reg_l2, binary);
    if (grad == 0.0 && curvature == 0.0) { return 0.0; }
    return Sqr(grad) / curvature;
}

inline static void CalScore(vector<double> &value, double *g_sum, double *h_sum,
                            double reg_l1, double reg_l2, bool binary = false) {
    for (size_t i = 0; i < value.size(); ++i) {
        value[i] = CalScore(g_sum[i], h_sum[i], reg_l1, reg_l2, binary);
    }
}

inline static void Softmax(vector<double> &rec) {
    double wmax = rec[0];
    for (size_t i = 1; i < rec.size(); ++i) {
        wmax = std::max(rec[i], wmax);
    }
    double wsum = 0.0f;
    for (size_t i = 0; i < rec.size(); ++i) {
        rec[i] = exp(rec[i] - wmax);
        wsum += rec[i];
    }
    for (size_t i = 0; i < rec.size(); ++i) {
        rec[i] /= wsum;
    }
}

inline static double Log_sum_exp(vector<double> &rec) {
    double wmax = rec[0];
    for (size_t i = 1; i < rec.size(); ++i) {
        wmax = std::max(rec[i], wmax);
    }
    double wsum = 0.0f;
    for (size_t i = 0; i < rec.size(); ++i) {
        wsum += exp(rec[i] - wmax);
    }
    return log(wsum) + wmax;
}

#endif //MGBDT_MATH_H
