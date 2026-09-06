// ScoreMachine 的行為測試。
//
// 每個測試的一句話說明（對應 README 第 8 節第 4 點）：
//   OpensAtZeroZeroTwo        — 開局唸分必須是 0-0-2。
//   ServingTeamScoresOnWin    — 只有發球方贏球才會加分。
//   ReceivingWinDoesNotScore  — 接發方贏球不加分，只轉移發球權。
//   SecondServerThenSideOut   — 第一發球員失分換第二發球員，再失分才 side out。
//   SideOutSwapsPerspective   — side out 後唸分視角換到另一隊。
//   WinsAtElevenByTwo         — 11 分且領先 2 分才算贏。
//   RequiresWinByTwo          — 10-10 後必須連拿 2 分。
//   ConfidenceMultiplies      — 信心值是各回合的乘積。
//   IgnoresRalliesAfterFinish — 比賽結束後的回合不再改變比分。

#include <gtest/gtest.h>

#include "pickleball_analysis/score_machine.hpp"

using pickleball_analysis::RallyOutcome;
using pickleball_analysis::ScoreMachine;
using pickleball_analysis::Team;

namespace
{
/// 讓發球方連得 n 分。
void serverScores(ScoreMachine & m, int n)
{
  for (int i = 0; i < n; ++i) {
    m.applyRally({m.servingTeam(), 1.0});
  }
}
}  // namespace

TEST(ScoreMachineTest, OpensAtZeroZeroTwo)
{
  ScoreMachine m;
  EXPECT_EQ(m.callScore(), "0-0-2");
  EXPECT_EQ(m.servingTeam(), Team::kA);
  EXPECT_FALSE(m.isFinished());
}

TEST(ScoreMachineTest, ServingTeamScoresOnWin)
{
  ScoreMachine m;
  m.applyRally({Team::kA, 1.0});

  EXPECT_EQ(m.points(Team::kA), 1);
  EXPECT_EQ(m.points(Team::kB), 0);
  EXPECT_EQ(m.servingTeam(), Team::kA) << "發球方得分後發球權不變";
  EXPECT_EQ(m.callScore(), "1-0-2");
}

TEST(ScoreMachineTest, ReceivingWinDoesNotScore)
{
  ScoreMachine m;
  m.applyRally({Team::kB, 1.0});   // 接發方贏

  EXPECT_EQ(m.points(Team::kB), 0) << "接發方贏球不得分";
  EXPECT_EQ(m.points(Team::kA), 0);
}

TEST(ScoreMachineTest, SecondServerThenSideOut)
{
  ScoreMachine m;
  // 開局是第 2 發球員，失分即直接 side out。
  m.applyRally({Team::kB, 1.0});
  EXPECT_EQ(m.servingTeam(), Team::kB);
  EXPECT_EQ(m.serverNumber(), 1);

  // B 隊第 1 發球員失分 → 換 B 隊第 2 發球員，發球權仍在 B。
  m.applyRally({Team::kA, 1.0});
  EXPECT_EQ(m.servingTeam(), Team::kB);
  EXPECT_EQ(m.serverNumber(), 2);

  // B 隊第 2 發球員再失分 → side out 回 A 隊。
  m.applyRally({Team::kA, 1.0});
  EXPECT_EQ(m.servingTeam(), Team::kA);
  EXPECT_EQ(m.serverNumber(), 1);
}

TEST(ScoreMachineTest, SideOutSwapsPerspective)
{
  ScoreMachine m;
  serverScores(m, 3);              // A: 3-0-2
  ASSERT_EQ(m.callScore(), "3-0-2");

  m.applyRally({Team::kB, 1.0});   // side out（開局第 2 發球員）
  // 現在換 B 發球，唸分以 B 為發球方視角：B 0 分、A 3 分。
  EXPECT_EQ(m.callScore(), "0-3-1");
  EXPECT_EQ(m.points(Team::kA), 3) << "絕對分數不因發球權改變";
}

TEST(ScoreMachineTest, WinsAtElevenByTwo)
{
  ScoreMachine m;
  serverScores(m, 11);

  EXPECT_TRUE(m.isFinished());
  EXPECT_EQ(m.winner(), Team::kA);
  EXPECT_EQ(m.points(Team::kA), 11);
}

TEST(ScoreMachineTest, RequiresWinByTwo)
{
  ScoreMachine m;
  // 做出 10-10：A 拿 10 分後 side out，B 也拿 10 分。
  serverScores(m, 10);
  m.applyRally({Team::kB, 1.0});
  serverScores(m, 10);
  ASSERT_EQ(m.points(Team::kA), 10);
  ASSERT_EQ(m.points(Team::kB), 10);
  EXPECT_FALSE(m.isFinished()) << "10-10 不算結束";

  serverScores(m, 1);              // 11-10
  EXPECT_FALSE(m.isFinished()) << "只領先 1 分不算贏";

  serverScores(m, 1);              // 12-10
  EXPECT_TRUE(m.isFinished());
}

TEST(ScoreMachineTest, ConfidenceMultiplies)
{
  ScoreMachine m;
  m.applyRally({Team::kA, 0.9});
  m.applyRally({Team::kA, 0.8});

  EXPECT_NEAR(m.confidence(), 0.72, 1e-9);
}

TEST(ScoreMachineTest, IgnoresRalliesAfterFinish)
{
  ScoreMachine m;
  serverScores(m, 11);
  ASSERT_TRUE(m.isFinished());

  serverScores(m, 3);
  EXPECT_EQ(m.points(Team::kA), 11) << "結束後不該再加分";
}
