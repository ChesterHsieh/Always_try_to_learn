#include "pickleball_analysis/score_machine.hpp"

#include <algorithm>
#include <cstdlib>

namespace pickleball_analysis
{

namespace
{
inline std::size_t idx(Team t) {return static_cast<std::size_t>(t);}
inline Team other(Team t) {return t == Team::kA ? Team::kB : Team::kA;}
}  // namespace

bool Score::operator==(const Score & o) const
{
  return serving_team_points == o.serving_team_points &&
         receiving_team_points == o.receiving_team_points &&
         server_number == o.server_number &&
         serving_team == o.serving_team;
}

ScoreMachine::ScoreMachine(std::uint8_t win_points, std::uint8_t win_by)
: win_points_(win_points), win_by_(win_by)
{
}

void ScoreMachine::applyRally(const RallyOutcome & outcome)
{
  if (isFinished()) {
    return;   // 比賽已結束，忽略後續回合
  }

  confidence_ *= outcome.confidence;

  if (outcome.winner == serving_team_) {
    // 發球方得分，發球權留在同一位球員（換邊由 rally_engine 處理）。
    ++points_[idx(serving_team_)];
    return;
  }

  // 接發方贏球：發球方不得分，發球權在隊內移轉，或 side out 給對手。
  if (server_number_ == 1) {
    server_number_ = 2;
  } else {
    serving_team_ = other(serving_team_);
    server_number_ = 1;
  }
}

std::uint8_t ScoreMachine::points(Team t) const
{
  return points_[idx(t)];
}

Score ScoreMachine::score() const
{
  Score s;
  s.serving_team = serving_team_;
  s.serving_team_points = points_[idx(serving_team_)];
  s.receiving_team_points = points_[idx(other(serving_team_))];
  s.server_number = server_number_;
  return s;
}

bool ScoreMachine::isFinished() const
{
  const int a = points_[0];
  const int b = points_[1];
  const int hi = std::max(a, b);
  return hi >= win_points_ && std::abs(a - b) >= win_by_;
}

Team ScoreMachine::winner() const
{
  return points_[0] >= points_[1] ? Team::kA : Team::kB;
}

std::string ScoreMachine::callScore() const
{
  const Score s = score();
  return std::to_string(s.serving_team_points) + "-" +
         std::to_string(s.receiving_team_points) + "-" +
         std::to_string(s.server_number);
}

}  // namespace pickleball_analysis
