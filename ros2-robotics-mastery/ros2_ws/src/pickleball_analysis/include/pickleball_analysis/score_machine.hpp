// Pickleball 計分狀態機（side-out scoring）。
//
// 這是 README 第 11 節列為高風險的部分：「AI 計分在雙打 side-out 規則下錯誤累積」。
// 因此刻意抽成純邏輯、無 ROS 相依的類別，讓規則錯誤能被單元測試直接逼出來，
// 而不必等到跑完整段影片才發現比分歪掉。
//
// 規則（雙打 side-out）：
//   * 比分唸法為三碼：發球方分數 - 接發方分數 - 第幾發球員，開局 0-0-2。
//   * 只有發球方得分；接發方贏球則發球權在隊內移轉或 side out。
//   * 開局特例：第一局第一輪只有一位發球員（故從 "2" 起算）。
//   * 先到 11 分且領先 2 分獲勝。
#ifndef PICKLEBALL_ANALYSIS__SCORE_MACHINE_HPP_
#define PICKLEBALL_ANALYSIS__SCORE_MACHINE_HPP_

#include <cstdint>
#include <string>

namespace pickleball_analysis
{

enum class Team : std::uint8_t { kA = 0, kB = 1 };

/// 一次回合的結果：哪一隊贏得該回合。
struct RallyOutcome
{
  Team winner{Team::kA};

  /// 該回合關鍵判定的信心值乘積，會累積進 MatchState.score_confidence。
  double confidence{1.0};
};

/// 比分快照。
struct Score
{
  std::uint8_t serving_team_points{0};
  std::uint8_t receiving_team_points{0};
  std::uint8_t server_number{2};   ///< 1 或 2；開局為 2
  Team serving_team{Team::kA};

  bool operator==(const Score & o) const;
};

class ScoreMachine
{
public:
  ScoreMachine() = default;

  /// 設定獲勝分數與必須領先的分數（預設 11 分、領先 2 分）。
  ScoreMachine(std::uint8_t win_points, std::uint8_t win_by);

  /// 餵入一個回合的結果，推進狀態機。
  void applyRally(const RallyOutcome & outcome);

  /// 目前比分（以發球方視角）。
  [[nodiscard]] Score score() const;

  /// 各隊的絕對分數，不隨發球權改變。
  [[nodiscard]] std::uint8_t points(Team t) const;

  [[nodiscard]] Team servingTeam() const {return serving_team_;}
  [[nodiscard]] std::uint8_t serverNumber() const {return server_number_;}

  /// 比賽是否結束。
  [[nodiscard]] bool isFinished() const;

  /// 勝方；未結束時回傳值無意義。
  [[nodiscard]] Team winner() const;

  /// 所有回合信心值的乘積。
  [[nodiscard]] double confidence() const {return confidence_;}

  /// 標準唸分，例如 "0-0-2"。
  [[nodiscard]] std::string callScore() const;

private:
  std::uint8_t points_[2]{0, 0};
  Team serving_team_{Team::kA};
  std::uint8_t server_number_{2};   ///< 開局特例：第一輪只有第二發球員
  double confidence_{1.0};

  std::uint8_t win_points_{11};
  std::uint8_t win_by_{2};
};

}  // namespace pickleball_analysis

#endif  // PICKLEBALL_ANALYSIS__SCORE_MACHINE_HPP_
