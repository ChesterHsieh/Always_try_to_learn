#include "pickleball_perception/ball_filter.hpp"

#include <algorithm>
#include <cmath>

namespace pickleball_perception
{

namespace
{
constexpr std::size_t kDim = 6;

/// P = F P F^T + Q，其中 F 是等加速度模型的狀態轉移矩陣。
/// F 極度稀疏（每列最多 3 個非零項），所以手寫展開比通用矩陣乘法快，
/// 也省掉一個 Eigen 相依。
void covariancePredict(std::array<double, kDim * kDim> & P, double dt, double q)
{
  const double h = 0.5 * dt * dt;

  // FP = F * P：位置列吸收速度與加速度列，速度列吸收加速度列。
  std::array<double, kDim * kDim> FP{};
  for (std::size_t c = 0; c < kDim; ++c) {
    FP[0 * kDim + c] = P[0 * kDim + c] + dt * P[2 * kDim + c] + h * P[4 * kDim + c];
    FP[1 * kDim + c] = P[1 * kDim + c] + dt * P[3 * kDim + c] + h * P[5 * kDim + c];
    FP[2 * kDim + c] = P[2 * kDim + c] + dt * P[4 * kDim + c];
    FP[3 * kDim + c] = P[3 * kDim + c] + dt * P[5 * kDim + c];
    FP[4 * kDim + c] = P[4 * kDim + c];
    FP[5 * kDim + c] = P[5 * kDim + c];
  }

  // P = FP * F^T。F^T 的第 c 行就是 F 的第 c 列，
  // 所以要對 FP 的「行」做線性組合：先取出整行再寫回，
  // 否則同一列內先被覆寫的元素會污染後面的計算（P 會失去對稱性）。
  for (std::size_t r = 0; r < kDim; ++r) {
    const double f0 = FP[r * kDim + 0];
    const double f1 = FP[r * kDim + 1];
    const double f2 = FP[r * kDim + 2];
    const double f3 = FP[r * kDim + 3];
    const double f4 = FP[r * kDim + 4];
    const double f5 = FP[r * kDim + 5];

    P[r * kDim + 0] = f0 + dt * f2 + h * f4;
    P[r * kDim + 1] = f1 + dt * f3 + h * f5;
    P[r * kDim + 2] = f2 + dt * f4;
    P[r * kDim + 3] = f3 + dt * f5;
    P[r * kDim + 4] = f4;
    P[r * kDim + 5] = f5;
  }

  const double q_pos = q * h * h;
  const double q_vel = q * dt * dt;
  P[0 * kDim + 0] += q_pos;
  P[1 * kDim + 1] += q_pos;
  P[2 * kDim + 2] += q_vel;
  P[3 * kDim + 3] += q_vel;
  P[4 * kDim + 4] += q;
  P[5 * kDim + 5] += q;
}
}  // namespace

double BallState::speed() const
{
  return std::sqrt(vx * vx + vy * vy);
}

BallFilter::BallFilter(const BallFilterParams & params)
: params_(params)
{
}

void BallFilter::reset()
{
  has_track_ = false;
  coast_frames_ = 0;
  state_ = BallState{};
  P_.fill(0.0);
}

void BallFilter::startTrack(const Observation & obs)
{
  state_ = BallState{};
  state_.px = obs.u;
  state_.py = obs.v;

  // 速度與加速度未知，給大的初始不確定性讓前幾筆觀測快速收斂。
  P_.fill(0.0);
  for (std::size_t i = 0; i < kDim; ++i) {
    P_[i * kDim + i] = params_.initial_covariance;
  }
  // 位置直接由觀測給定，不確定性就是觀測雜訊。
  const double r = params_.measurement_noise * params_.measurement_noise;
  P_[0 * kDim + 0] = r;
  P_[1 * kDim + 1] = r;

  has_track_ = true;
  coast_frames_ = 0;
  ++track_id_;
}

void BallFilter::predictStep(double dt)
{
  const double h = 0.5 * dt * dt;
  state_.px += state_.vx * dt + state_.ax * h;
  state_.py += state_.vy * dt + state_.ay * h;
  state_.vx += state_.ax * dt;
  state_.vy += state_.ay * dt;
  covariancePredict(P_, dt, params_.process_noise);
}

void BallFilter::predictOnly(double dt)
{
  if (!has_track_) {
    return;
  }

  ++coast_frames_;
  if (coast_frames_ > params_.max_coast_frames) {
    reset();
    return;
  }

  predictStep(dt);
  state_.is_predicted = true;
}

bool BallFilter::update(const Observation & obs, double dt)
{
  if (!has_track_) {
    startTrack(obs);
    state_.is_predicted = false;
    return true;
  }

  predictStep(dt);

  // 離群值閘門：預測位置與觀測差太遠就不信這筆觀測。
  const double du = obs.u - state_.px;
  const double dv = obs.v - state_.py;
  if (std::sqrt(du * du + dv * dv) > params_.outlier_gate_px) {
    ++coast_frames_;
    if (coast_frames_ > params_.max_coast_frames) {
      // 連續離群 = 舊軌跡已經不成立，用這筆觀測重新起始。
      reset();
      startTrack(obs);
      state_.is_predicted = false;
      return true;
    }
    state_.is_predicted = true;
    return false;
  }

  // 標準 Kalman 更新。觀測矩陣 H 只取位置兩維，
  // 所以 S、K 都能手寫展開成 2x2 與 6x2，不需要通用矩陣求逆。
  const double r = params_.measurement_noise * params_.measurement_noise;
  const double s00 = P_[0 * kDim + 0] + r;
  const double s01 = P_[0 * kDim + 1];
  const double s10 = P_[1 * kDim + 0];
  const double s11 = P_[1 * kDim + 1] + r;

  const double det = s00 * s11 - s01 * s10;
  if (std::abs(det) < 1e-12) {
    // 數值退化，這一格當作沒有觀測比較安全。
    state_.is_predicted = true;
    return false;
  }
  const double inv00 = s11 / det;
  const double inv01 = -s01 / det;
  const double inv10 = -s10 / det;
  const double inv11 = s00 / det;

  // K = P H^T S^-1，H^T 取的就是 P 的前兩行。
  std::array<double, kDim * 2> K{};
  for (std::size_t i = 0; i < kDim; ++i) {
    const double p0 = P_[i * kDim + 0];
    const double p1 = P_[i * kDim + 1];
    K[i * 2 + 0] = p0 * inv00 + p1 * inv10;
    K[i * 2 + 1] = p0 * inv01 + p1 * inv11;
  }

  double * x[kDim] = {&state_.px, &state_.py, &state_.vx, &state_.vy, &state_.ax, &state_.ay};
  for (std::size_t i = 0; i < kDim; ++i) {
    *x[i] += K[i * 2 + 0] * du + K[i * 2 + 1] * dv;
  }

  // P = (I - K H) P
  std::array<double, kDim * kDim> newP{};
  for (std::size_t i = 0; i < kDim; ++i) {
    for (std::size_t c = 0; c < kDim; ++c) {
      newP[i * kDim + c] = P_[i * kDim + c] -
        K[i * 2 + 0] * P_[0 * kDim + c] -
        K[i * 2 + 1] * P_[1 * kDim + c];
    }
  }
  P_ = newP;

  coast_frames_ = 0;
  state_.is_predicted = false;
  return true;
}

}  // namespace pickleball_perception
