#include "line_width.h"
#include "constants.h"
#include "voigt.h"
#include <cmath>
#include <vector>
#include <algorithm>
#include <limits>
#include <omp.h>

namespace cgm {
namespace analysis {

namespace internal {

inline double voigt_model(double v, double tau_0, double b, double v_center, double damping) {
    return compute_voigt_optical_depth(v, tau_0, b, v_center, damping);
}

struct FitParams {
    double tau_0;
    double b;
    double v_center;
};

double compute_residual(
    const FitParams& params,
    const double* tau_data,
    const double* v_values,
    int n_points,
    double damping) {
    
    if (params.b <= 0.1 || params.b > 200.0 || params.tau_0 <= 0) {
        return 1e10;
    }
    
    double residual = 0.0;
    for (int i = 0; i < n_points; ++i) {
        double tau_model = voigt_model(v_values[i], params.tau_0, params.b, params.v_center, damping);
        double diff = tau_data[i] - tau_model;
        residual += diff * diff;
    }
    
    return residual / n_points;
}

// A line railing against the width limit is not a measurement: the old code clamped it to
// 80 km/s instead, which piled roughly a tenth of all simulated lines into a single bin and
// left a hard edge no real b distribution has. The search now runs to B_SEARCH_MAX and the
// caller rejects fits that end up on either boundary.
constexpr double B_SEARCH_MIN = 2.0;
constexpr double B_SEARCH_MAX = 200.0;
constexpr double B_GRID_MIN = 5.0;
constexpr double B_GRID_STEP = 2.0;
// Largest grid point at or below B_SEARCH_MAX: the search is B_GRID_MIN + k * B_GRID_STEP,
// so it ends at 199, not 200. Testing against 200 let every railed fit through.
constexpr double B_GRID_MAX = B_GRID_MIN
    + B_GRID_STEP * static_cast<int>((B_SEARCH_MAX - B_GRID_MIN) / B_GRID_STEP);

// Railed within one grid step of either end: the minimiser had nowhere further to go, so the
// width is a bound, not a measurement.
bool b_is_railed(double b) {
    return b <= B_GRID_MIN + B_GRID_STEP || b >= B_GRID_MAX - B_GRID_STEP;
}

bool fit_voigt_profile(
    const double* tau_data,
    const double* v_values,
    int n_points,
    double tau_peak,
    double v_peak,
    double fwhm,
    double damping,
    FitParams& result) {
    
    if (n_points < 10) return false;
    
    double b_guess = fwhm / (2.0 * std::sqrt(std::log(2.0)));
    b_guess = std::clamp(b_guess, B_GRID_MIN, B_GRID_MAX);
    
    double best_error = 1e20;
    double best_b = b_guess;
    double best_tau_0 = tau_peak;
    double best_v = v_peak;
    
    double a = damping * 1215.67 / b_guess;
    double a_scaled = a * 2.0 * SQRT_LN2;
    double voigt_at_center = cgm::analysis::internal::voigt(0.0, a_scaled);
    double V_0 = voigt_at_center * (2.0 * SQRT_LN2) / b_guess;
    double tau_0_guess = tau_peak / V_0;
    
    for (double b_test = B_GRID_MIN; b_test <= B_GRID_MAX; b_test += B_GRID_STEP) {
        double a_test = damping * 1215.67 / b_test;
        double a_scaled_test = a_test * 2.0 * SQRT_LN2;
        
        double V_0_test = 0.0;
        for (int i = 0; i < n_points; ++i) {
            double u = (v_values[i] - v_peak) / b_test;
            double voigt_val = cgm::analysis::internal::voigt(u, a_scaled_test);
            V_0_test = std::max(V_0_test, voigt_val * (2.0 * SQRT_LN2) / b_test);
        }
        
        double tau_0_test = tau_peak / V_0_test;
        
        FitParams params = {tau_0_test, b_test, v_peak};
        double err = compute_residual(params, tau_data, v_values, n_points, damping);
        
        if (err < best_error) {
            best_error = err;
            best_b = b_test;
            best_tau_0 = tau_0_test;
        }
    }
    
    FitParams params = {best_tau_0, best_b, best_v};
    
    const int max_iter = 30;
    const double eps = 1e-6;
    
    double current_error = compute_residual(params, tau_data, v_values, n_points, damping);
    
    for (int iter = 0; iter < max_iter; ++iter) {
        double h_tau = 0.1 * std::max(params.tau_0, 0.1);
        double h_b = 0.1 * std::max(params.b, 1.0);
        double h_v = 0.5;
        
        double r0 = current_error;
        
        double r_tau = compute_residual({params.tau_0 + h_tau, params.b, params.v_center}, 
                                        tau_data, v_values, n_points, damping);
        double r_b = compute_residual({params.tau_0, params.b + h_b, params.v_center}, 
                                      tau_data, v_values, n_points, damping);
        double r_v = compute_residual({params.tau_0, params.b, params.v_center + h_v}, 
                                      tau_data, v_values, n_points, damping);
        
        if (r_tau >= 1e9 || r_b >= 1e9 || r_v >= 1e9) break;
        
        double grad_tau = (r_tau - r0) / h_tau;
        double grad_b = (r_b - r0) / h_b;
        double grad_v = (r_v - r0) / h_v;
        
        double alpha = 1.0;
        FitParams params_new = params;
        
        for (int ls = 0; ls < 20; ++ls) {
            params_new.tau_0 = std::max(params.tau_0 - alpha * grad_tau * params.tau_0, 0.01);
            params_new.b = std::clamp(params.b - alpha * grad_b * params.b * 0.1,
                                      B_SEARCH_MIN, B_SEARCH_MAX);
            params_new.v_center = params.v_center - alpha * grad_v;
            
            double new_error = compute_residual(params_new, tau_data, v_values, n_points, damping);
            
            if (new_error < r0) {
                params = params_new;
                current_error = new_error;
                break;
            }
            alpha *= 0.5;
        }
        
        if (alpha < 1e-6) break;
        
        if (current_error < eps) break;
    }
    
    double final_error = compute_residual(params, tau_data, v_values, n_points, damping);
    
    if (final_error > 0.1 || params.b < B_SEARCH_MIN || params.b > B_SEARCH_MAX
        || params.tau_0 < 0.01) {
        return false;
    }
    
    result = params;
    return true;
}

double estimate_b_from_fwhm(double fwhm) {
    if (fwhm < 1.0) return 10.0;
    return std::clamp(fwhm / (2.0 * std::sqrt(std::log(2.0))),
                      B_SEARCH_MIN, B_SEARCH_MAX);
}

}

LineWidthResult compute_line_width_distribution(
    const Eigen::Ref<const Eigen::ArrayXXf>& tau,
    double velocity_spacing,
    float threshold,
    const Eigen::Ref<const Eigen::ArrayXXf>* colden) {
    
    const int n_sightlines = tau.rows();
    const int n_pixels = tau.cols();
    
    const double damping = VOIGT_DAMPING;
    
    std::vector<double> column_densities;
    std::vector<double> b_params;
    column_densities.reserve(n_sightlines * 10);
    b_params.reserve(n_sightlines * 10);
    
    std::vector<double> v_buffer(n_pixels);
    for (int j = 0; j < n_pixels; ++j) {
        v_buffer[j] = j * velocity_spacing;
    }

    // Per-sightline peak finding + Voigt fitting is independent: parallelize
    // across sightlines with thread-local accumulators, then concatenate
    // (keeping the N_HI/b pairing). Distribution stats are order-independent.
    const int n_threads = omp_get_max_threads();
    std::vector<std::vector<double>> tl_cd(n_threads), tl_b(n_threads);

    #pragma omp parallel
    {
        std::vector<double>& local_cd = tl_cd[omp_get_thread_num()];
        std::vector<double>& local_b = tl_b[omp_get_thread_num()];
        std::vector<double> tau_buffer(n_pixels);  // thread-private scratch

        #pragma omp for schedule(dynamic, 32)
        for (int i = 0; i < n_sightlines; ++i) {
        const auto& tau_line = tau.row(i);
        const float* colden_line = nullptr;
        if (colden) {
            colden_line = colden->row(i).data();
        }
        
        std::vector<int> peak_indices;
        std::vector<double> peak_values;
        
        for (int j = 2; j < n_pixels - 2; ++j) {
            if (tau_line(j) > threshold && 
                tau_line(j) >= tau_line(j-1) && 
                tau_line(j) >= tau_line(j-2) &&
                tau_line(j) >= tau_line(j+1) && 
                tau_line(j) >= tau_line(j+2)) {
                peak_indices.push_back(j);
                peak_values.push_back(tau_line(j));
            }
        }
        
        if (peak_indices.empty()) continue;
        
        std::vector<int> merged_peaks;
        std::vector<double> merged_values;
        int min_separation = 20;
        
        merged_peaks.push_back(peak_indices[0]);
        merged_values.push_back(peak_values[0]);
        
        for (size_t k = 1; k < peak_indices.size(); ++k) {
            if (peak_indices[k] - merged_peaks.back() < min_separation) {
                if (peak_values[k] > merged_values.back()) {
                    merged_peaks.back() = peak_indices[k];
                    merged_values.back() = peak_values[k];
                }
            } else {
                merged_peaks.push_back(peak_indices[k]);
                merged_values.push_back(peak_values[k]);
            }
        }
        
        peak_indices = merged_peaks;
        peak_values = merged_values;
        
        for (size_t pi = 0; pi < peak_indices.size(); ++pi) {
            int peak_idx = peak_indices[pi];
            double tau_peak = peak_values[pi];
            
            int left = peak_idx;
            while (left > 0 && tau_line(left) > threshold * 0.5) {
                --left;
            }
            
            int right = peak_idx;
            while (right < n_pixels - 1 && tau_line(right) > threshold * 0.5) {
                ++right;
            }
            
            int feature_width = right - left + 1;
            if (feature_width < 5) continue;
            
            double v_peak = peak_idx * velocity_spacing;
            double fwhm = feature_width * velocity_spacing;
            
            internal::FitParams fit_result;
            bool fit_success = false;
            
            if (feature_width >= 10) {
                int n_feature = right - left + 1;
                for (int k = 0; k < n_feature; ++k) {
                    tau_buffer[k] = tau_line(left + k);
                }
                
                fit_success = internal::fit_voigt_profile(
                    tau_buffer.data(),
                    v_buffer.data(),
                    n_feature,
                    tau_peak,
                    v_peak,
                    fwhm,
                    damping,
                    fit_result
                );
            }
            
            double b;
            if (fit_success) {
                b = fit_result.b;
            } else {
                b = internal::estimate_b_from_fwhm(fwhm);
            }
            
            // Unconstrained width: drop the absorber rather than record the boundary.
            if (internal::b_is_railed(b)) {
                continue;
            }
            
            double N_HI;
            if (colden_line) {
                float max_colden = 0;
                for (int k = left; k <= right; ++k) {
                    max_colden = std::max(max_colden, colden_line[k]);
                }
                N_HI = max_colden;
            } else {
                double tau_sum = 0;
                for (int k = left; k <= right; ++k) {
                    tau_sum += tau_line(k);
                }
                N_HI = constants::TAU_TO_COLDEN_CONSTANT * tau_sum * velocity_spacing;
            }
            
            if (N_HI > constants::COLUMN_DENSITY_MIN) {
                local_cd.push_back(N_HI);
                local_b.push_back(b);
            }
        }
    }
    
    }  // end #pragma omp parallel

    for (int t = 0; t < n_threads; ++t) {
        column_densities.insert(column_densities.end(), tl_cd[t].begin(), tl_cd[t].end());
        b_params.insert(b_params.end(), tl_b[t].begin(), tl_b[t].end());
    }

    LineWidthResult result;
    if (b_params.empty()) {
        result.N_HI = Eigen::VectorXd(0);
        result.b_params = Eigen::VectorXd(0);
        result.temperatures = Eigen::VectorXd(0);
        result.b_median = std::nan("");
        result.b_mean = std::nan("");
        result.b_std = std::nan("");
        result.n_absorbers = 0;
        return result;
    }
    
    result.N_HI = Eigen::VectorXd::Map(column_densities.data(), column_densities.size());
    result.b_params = Eigen::VectorXd::Map(b_params.data(), b_params.size());
    result.temperatures = result.b_params.array().square() * constants::B_TO_T_FACTOR;
    result.n_absorbers = b_params.size();
    
    Eigen::VectorXd sorted_b = result.b_params;
    std::sort(sorted_b.data(), sorted_b.data() + sorted_b.size());
    result.b_median = sorted_b(sorted_b.size() / 2);
    result.b_mean = result.b_params.mean();
    
    double b_mean = result.b_mean;
    double variance = 0.0;
    for (int i = 0; i < result.b_params.size(); ++i) {
        double diff = result.b_params(i) - b_mean;
        variance += diff * diff;
    }
    variance /= result.b_params.size();
    result.b_std = std::sqrt(variance);
    
    return result;
}

}
}
