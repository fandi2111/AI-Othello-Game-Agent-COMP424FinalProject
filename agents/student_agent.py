# Student agent: Add your own agent here
from agents.agent import Agent
from store import register_agent
import sys
import numpy as np
from copy import deepcopy
import time
from helpers import random_move, count_capture, execute_move, check_endgame, get_valid_moves, get_directions

@register_agent("student_agent")
class StudentAgent(Agent):
  """
  A class for your implementation. Feel free to use this class to
  add any helper functionalities needed for your agent.
  """

  def __init__(self):
    super(StudentAgent, self).__init__()
    self.name = "AlphaBetaAgent"
    #self.max_depth = 4  # Limit the search depth
    self.time_limit = 1.9  # Time constraint in seconds
    self.start_time = time.time()

  def step(self, chess_board, player, opponent):
    """
    Implement the step function of your agent here.
    You can use the following variables to access the chess board:
    - chess_board: a numpy array of shape (board_size, board_size)
      where 0 represents an empty spot, 1 represents Player 1's discs (Blue),
      and 2 represents Player 2's discs (Brown).
    - player: 1 if this agent is playing as Player 1 (Blue), or 2 if playing as Player 2 (Brown).
    - opponent: 1 if the opponent is Player 1 (Blue), or 2 if the opponent is Player 2 (Brown).

    You should return a tuple (r,c), where (r,c) is the position where your agent
    wants to place the next disc. Use functions in helpers to determine valid moves
    and more helpful tools.

    Please check the sample implementation in agents/random_agent.py or agents/human_agent.py for more details.
    """
    # Some simple code to help you with timing. Consider checking 
    # time_taken during your search and breaking with the best answer
    # so far when it nears 2 seconds.

    start_time = time.time()
    self.start_time = start_time

    best_move = None
    best_score = float('-inf')

    depth = 1  # Start with shallow search
    while True:
      for move in get_valid_moves(chess_board, player):
        simulated_board = deepcopy(chess_board)
        execute_move(simulated_board, move, player)


        score = self.min_value(simulated_board, depth, float('-inf'), float('inf'), opponent, player)

        if score > best_score:
          best_score = score
          best_move = move

      if time.time() - start_time > self.time_limit - 0.1:  # Buffer for safety
        break
      depth += 1  # Increase depth
    time_taken = time.time() - start_time
    print("My AI's turn took ", time_taken, "seconds.")

    print(f"Best move: {best_move}, Best score: {best_score}")

    return best_move

  def max_value(self, chess_board, depth, alpha, beta, player, opponent):
    """
    Maximize the player's score.
    """
    if self.cutoff_test(chess_board, depth, player, opponent):
      return self.evaluate(chess_board, player, opponent)

    value = float('-inf')
    legal_moves = get_valid_moves(chess_board, player)

    for move in legal_moves:
      simulated_board = np.copy(chess_board)
      execute_move(simulated_board, move, player)

      value = max(value, self.min_value(simulated_board, depth - 1, alpha, beta, opponent, player))
      if value >= beta:
        return value
      alpha = max(alpha, value)

    return value

  def min_value(self, chess_board, depth, alpha, beta, player, opponent):
    """
    Minimize the opponent's score.
    """
    if self.cutoff_test(chess_board, depth, player, opponent):
      return self.evaluate(chess_board, opponent, player)

    value = float('inf')
    legal_moves = get_valid_moves(chess_board, player)

    for move in legal_moves:
      simulated_board = np.copy(chess_board)
      execute_move(simulated_board, move, player)

      value = min(value, self.max_value(simulated_board, depth - 1, alpha, beta, opponent, player))
      if value <= alpha:
        return value
      beta = min(beta, value)

    return value

  def cutoff_test(self, chess_board, depth, player, opponent):
    is_endgame, _, _ = check_endgame(chess_board, player, opponent)
    return depth == 0 or is_endgame or (time.time() - self.start_time >= self.time_limit)

  def evaluate(self, chess_board, player, opponent):

    n = chess_board.shape[0]  # Board size
    weights = np.zeros((n, n))

    # Corner positions
    corners = [(0, 0), (0, n - 1), (n - 1, 0), (n - 1, n - 1)]
    for r, c in corners:
      weights[r, c] = 100000  # High value for corners
    #
    ajd = [(1,1), (1, n-2), (n-2, 1), (n-2, n-2)]
    #

    # Corner-adjacent penalty
    for r, c in corners:
      adjacent = [
        (r + dr, c + dc)
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]
        if 0 <= r + dr < n and 0 <= c + dc < n
      ]
      if chess_board[r, c] == player:
        for x, y in adjacent:
          weights[x, y] = 20
        for nr, nc in adjacent:
          weights[nr, nc] = 100000
          if chess_board[nr, nc] == player:
            if (nr == n-1 or nr == 0) and nc == 1:
              weights[nr, nc+1] = 100000
              if chess_board[nr, nc+1] == player:#
                weights[nr, nc+2] = 100000#
                if chess_board[nr, nc+2] == player:
                  weights[nr, nc+3] = 100000
                  if chess_board[nr, nc+3] == player:
                    weights[nr, nc+4] = 100000

            if (nr == n-1 or nr == 0) and nc == n-2:
              weights[nr, nc-1] = 100000
              #
              if chess_board[nr, nc-1] == player:#
                weights[nr, nc-2] = 100000#
                if chess_board[nr, nc-2] == player:
                  weights[nr, nc-3] = 100000
                  if chess_board[nr, nc-3] == player:
                    weights[nr, nc-4] = 100000
#             #
            if (nc == n-1 or nc == 0) and nr == 1:
              weights[nr+1, nc] = 100000
              if chess_board[nr+1, nc] == player:#
                weights[nr+2, nc] = 100000#
                if chess_board[nr+2, nc] == player:
                  weights[nr+3, nc] = 100000
                  if chess_board[nr+3, nc] == player:
                    weights[nr+4, nc] = 100000

            if (nc == n-1 or nc == 0) and nr == n-2:
              weights[nr-1, nc] = 100000
#             #
              if chess_board[nr-1, nc] == player:#
                weights[nr-2, nc] = 100000#
                if chess_board[nr-2, nc] == player:
                  weights[nr-3, nc] = 100000
                  if chess_board[nr-3, nc] == player:
                    weights[nr-4, nc] = 100000
#             #


      else:
        for nr, nc in adjacent:
          weights[nr, nc] = -500
        for x, y in adjacent:
          weights[x, y] = -500

    # Edge weights (non-corner)
    for i in range(2, n - 1):#
      if weights[0, i] != 100000:
        weights[0, i] = 20  # Top edge
        weights[0, 2] = 25
        weights[0, n-3] = 25

      if weights[n - 1, i] != 100000:
        weights[n - 1, i] = 20  # Bottom edge
        weights[n-1, 2] = 25
        weights[n-1, n-3] = 25

      if weights[i, 0] != 100000:
        weights[i, 0] = 20  # Left edge
        weights[2, 0] = 25
        weights[n-3, 0] = 25


      if weights[i, n - 1] != 100000:
        weights[i, n - 1] = 20  # Right edge
        weights[2, n-1] = 25
        weights[n-3, n-1] = 25

    # Inner grid weights
    inner_start = 1
    inner_end = n - 2
    for r in range(inner_start, inner_end):
      for c in range(inner_start, inner_end):
        weights[r, c] = -5

    # Frontier stability heuristic: count stable pieces
    frontier_score = 0
    for r in range(n):
      for c in range(n):
        if chess_board[r, c] == player:
          if self.is_stable(chess_board, r, c, player):
            frontier_score += 10
        elif chess_board[r, c] == opponent:
          if self.is_stable(chess_board, r, c, opponent):
            frontier_score -= 10

    # Mobility heuristic
    player_moves = len(get_valid_moves(chess_board, player))
    opponent_moves = len(get_valid_moves(chess_board, opponent))
    mobility_score = player_moves - opponent_moves

    # Piece count heuristic
    player_count = np.sum(chess_board == player)
    opponent_count = np.sum(chess_board == opponent)

    # Game stage analysis
    total_pieces = player_count + opponent_count
    total_tiles = n * n
    game_stage = total_pieces / total_tiles  # Percentage of board filled

    # Dynamic weighting based on game stage
    if game_stage < 0.3:  # Early game
      mobility_weight = 15
      stability_weight = 2
      piece_weight = 1
    elif game_stage < 0.7:  # Mid game
      mobility_weight = 10
      stability_weight = 5
      piece_weight = 3
    else:  # Late game
      mobility_weight = 5
      stability_weight = 10
      piece_weight = 10

    # Final score
    positional_score = sum(
      weights[r, c] if chess_board[r, c] == player else -weights[r, c]
      for r in range(n)
      for c in range(n)
      if chess_board[r, c] != 0
    )
    total_score = (
            positional_score
            + mobility_weight * mobility_score
            + stability_weight * frontier_score
            + piece_weight * (player_count - opponent_count)
    )
    return total_score

  def is_stable(self, chess_board, r, c, player):
    """
    Determine if a piece at (r, c) is stable (cannot be flipped).
    """
    directions = get_directions()
    n = chess_board.shape[0]
    for dr, dc in directions:
      nr, nc = r, c
      stable = True
      while 0 <= nr < n and 0 <= nc < n:
        if chess_board[nr, nc] != player:
          stable = False
          break
        nr += dr
        nc += dc
      if not stable:
        return False
    return True