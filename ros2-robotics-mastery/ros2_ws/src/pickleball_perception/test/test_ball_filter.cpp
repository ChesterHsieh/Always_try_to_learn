// BallFilter 的行為測試。
//
// 每個測試的一句話說明（對應 README 第 8 節第 4 點）：
//   NoTrackBeforeFirstObservation — 沒餵資料前不該宣稱有軌跡。
//   FirstObservationStartsTrack   — 第一筆觀測直接成為位置，不做平滑。
//   ConvergesToConstantVelocity   — 等速直線運動下，速度估計要收斂到真值。
//   PredictsForwardWhenBlind      — 沒有觀測時，預測位置要沿著既有速度往前走。
//   DropsTrackAfterMaxCoast       — 連續遺失超過上限就放棄軌跡。
//   RejectsOutlierObservation     — 離群觀測不該把軌跡拉走。
//   RestartsAfterPersistentOutlier— 但持續離群代表舊軌跡已死，要換新軌跡。
//   SurvivesNoisyObservations     — 有雜訊時位置估計仍該貼近真值。

#include <gtest/gtest.h>

#include <cmath>
#include <random>

#include "pickleball_perception/ball_filter.hpp"

using pickleball_perception::BallFilter;
using pickleball_perception::BallFilterParams;
using pickleball_perception::Observation;

namespace
{
constexpr double kDt = 1.0 / 30.0;  // 素材是 30 fps
}

TEST(BallFilterTest, NoTrackBeforeFirstObservation)
{
  BallFilter f;
  EXPECT_FALSE(f.hasTrack());

  // 沒有軌跡時 predictOnly 應該是無害的 no-op。
  f.predictOnly(kDt);
  EXPECT_FALSE(f.hasTrack());
}

TEST(BallFilterTest, FirstObservationStartsTrack)
{
  BallFilter f;
  ASSERT_TRUE(f.update({100.0, 200.0}, kDt));

  EXPECT_TRUE(f.hasTrack());
  EXPECT_FALSE(f.state().is_predicted);
  EXPECT_DOUBLE_EQ(f.state().px, 100.0);
  EXPECT_DOUBLE_EQ(f.state().py, 200.0);
  EXPECT_EQ(f.coastFrames(), 0u);
}

TEST(BallFilterTest, ConvergesToConstantVelocity)
{
  BallFilter f;
  const double vx = 300.0;   // px/s
  const double vy = -120.0;

  for (int i = 0; i < 40; ++i) {
    const double t = i * kDt;
    ASSERT_TRUE(f.update({100.0 + vx * t, 200.0 + vy * t}, kDt));
  }

  ASSERT_TRUE(f.hasTrack());
  EXPECT_NEAR(f.state().vx, vx, 15.0);
  EXPECT_NEAR(f.state().vy, vy, 15.0);
  EXPECT_NEAR(f.state().speed(), std::hypot(vx, vy), 20.0);
}

TEST(BallFilterTest, PredictsForwardWhenBlind)
{
  BallFilter f;
  const double vx = 300.0;
  for (int i = 0; i < 40; ++i) {
    ASSERT_TRUE(f.update({100.0 + vx * i * kDt, 200.0}, kDt));
  }
  const double last_x = f.state().px;

  f.predictOnly(kDt);

  EXPECT_TRUE(f.hasTrack());
  EXPECT_TRUE(f.state().is_predicted);
  EXPECT_EQ(f.coastFrames(), 1u);
  // 應該往前走了大約 vx*dt，而不是停在原地。
  EXPECT_NEAR(f.state().px - last_x, vx * kDt, 3.0);
}

TEST(BallFilterTest, DropsTrackAfterMaxCoast)
{
  BallFilterParams p;
  p.max_coast_frames = 3;
  BallFilter f(p);

  ASSERT_TRUE(f.update({100.0, 200.0}, kDt));
  for (std::size_t i = 0; i < p.max_coast_frames; ++i) {
    f.predictOnly(kDt);
    EXPECT_TRUE(f.hasTrack()) << "第 " << i + 1 << " 格不該放棄";
  }

  f.predictOnly(kDt);
  EXPECT_FALSE(f.hasTrack());
}

TEST(BallFilterTest, RejectsOutlierObservation)
{
  BallFilterParams p;
  p.outlier_gate_px = 50.0;
  BallFilter f(p);

  for (int i = 0; i < 20; ++i) {
    ASSERT_TRUE(f.update({100.0 + 5.0 * i, 200.0}, kDt));
  }
  const double before = f.state().px;

  // 記分板被誤判成球：位置暴衝 800 px。
  EXPECT_FALSE(f.update({1000.0, 50.0}, kDt));

  // 軌跡還在，而且沒有被拉到 1000。
  EXPECT_TRUE(f.hasTrack());
  EXPECT_LT(std::abs(f.state().px - before), p.outlier_gate_px);
}

TEST(BallFilterTest, RestartsAfterPersistentOutlier)
{
  BallFilterParams p;
  p.outlier_gate_px = 50.0;
  p.max_coast_frames = 3;
  BallFilter f(p);

  ASSERT_TRUE(f.update({100.0, 200.0}, kDt));
  const auto first_id = f.trackId();

  // 球真的跳到別的地方（例如換發球），持續離群。
  bool accepted = false;
  for (int i = 0; i < 10 && !accepted; ++i) {
    accepted = f.update({1000.0, 50.0}, kDt);
  }

  EXPECT_TRUE(accepted) << "持續離群後應該要重新起始軌跡";
  EXPECT_TRUE(f.hasTrack());
  EXPECT_NE(f.trackId(), first_id) << "重新起始要換新的 track_id";
  EXPECT_NEAR(f.state().px, 1000.0, 1.0);
}

TEST(BallFilterTest, SurvivesNoisyObservations)
{
  BallFilter f;
  std::mt19937 rng(42);            // 固定種子，測試必須可重現
  std::normal_distribution<double> noise(0.0, 3.0);

  const double vx = 250.0;
  double max_err = 0.0;

  for (int i = 0; i < 60; ++i) {
    const double t = i * kDt;
    const double truth_x = 100.0 + vx * t;
    f.update({truth_x + noise(rng), 200.0 + noise(rng)}, kDt);
    if (i > 10) {  // 給前幾格收斂時間
      max_err = std::max(max_err, std::abs(f.state().px - truth_x));
    }
  }

  // 濾波後的誤差應該小於原始雜訊的量級。
  EXPECT_LT(max_err, 8.0);
}
