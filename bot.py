import chess
from board import ChessBoard
import time
import heapq

from constants import DEPTH, CHECKING_MOVE_ARROW

class ChessBot:
    def __init__(self, game=None):
        self.moves_checked = 0
        self.game = game # Reference to the game object
    
    def evaluate_position(self, board: ChessBoard):
        """
        Evaluate the current position.
        Positive values favor white, negative values favor black.
        """
        chess_board = board.get_board_state()

        if board.is_game_over():
            if chess_board.is_checkmate():
                return -10000 if chess_board.turn else 10000
            return 0  # Draw

        score = 0
        piece_values = {
            chess.PAWN: 100,
            chess.KNIGHT: 320,
            chess.BISHOP: 330,
            chess.ROOK: 500,
            chess.QUEEN: 900,
            chess.KING: 20000
        }

        # Count material
        for piece in piece_values:
            score += len(chess_board.pieces(piece, True)) * piece_values[piece]
            score -= len(chess_board.pieces(piece, False)) * piece_values[piece]
           
        return score
    
    # def is_threat(self, board: ChessBoard, move: chess.Move) -> bool:
    #     """
    #     Check if the move is a threat to the opponent's pieces.
    #     """
    #     board.make_move(move)
    #     is_threat = board.get_board_state().is_check() or board.get_board_state().is_checkmate() or board.get_board_state()
    #     board.undo_move()
    #     return is_threat

    def get_pseduo_sorted_moves(self, board: ChessBoard) -> list:
        """
        Score all legal moves in the current position.
            - Check captures first
            - Then threats
            - Then positionally strong moves
        Returns a deque of moves sorted by importance and a dictionary of scores.
        """
        chess_board = board.get_board_state()
        important_moves, other_moves = [], []
        for move in board.get_legal_moves():
            if chess_board.is_capture(move): # Capture
                heapq.heappush(important_moves, (1, id(move), move))
            elif chess_board.gives_check(move): # Check
                heapq.heappush(important_moves, (2, id(move), move))
            # elif self.is_threat(board, move): # Threat
            #     scores[move] = 3
            else: # Other moves
                other_moves.append(move)

        sorted_important_moves = [item[2] for item in important_moves]
        return sorted_important_moves + other_moves

    def minimax_alpha_beta(self, board: ChessBoard, remaining_depth: int, alpha: int, beta: int, maximizing_player: bool):
        """
        Minimax algorithm with alpha-beta pruning.
        Returns (best_value, best_move) tuple.
        """
        if remaining_depth == 0 or board.is_game_over():
            return self.evaluate_position(board), None

        best_move = None
        pseduo_sorted_moves = self.get_pseduo_sorted_moves(board)
        if maximizing_player:
            best_value = float('-inf')
            for move in pseduo_sorted_moves:
                self.moves_checked += 1
                if CHECKING_MOVE_ARROW and remaining_depth == DEPTH: # Display the root move
                    self.game.checking_move = move
                    self.game.display_board(self.game.last_move)  # Update display

                # Make a move and evaluate the position
                board.make_move(move)
                value = self.minimax_alpha_beta(board, remaining_depth - 1, alpha, beta, False)[0]
                board.undo_move()

                if value > best_value:
                    best_value = value
                    best_move = move
                alpha = max(alpha, best_value)
                if beta <= alpha:
                    break
        else: # Minimizing player
            best_value = float('inf')
            for move in pseduo_sorted_moves:
                self.moves_checked += 1
                if CHECKING_MOVE_ARROW and remaining_depth == DEPTH: # Display the root move
                    self.game.checking_move = move
                    self.game.display_board(self.game.last_move)  # Update display

                # Make a move and evaluate the position
                board.make_move(move)
                value = self.minimax_alpha_beta(board, remaining_depth - 1, alpha, beta, True)[0]
                board.undo_move()

                if value < best_value:
                    best_value = value
                    best_move = move
                beta = min(beta, best_value)
                if beta <= alpha:
                    break

        return best_value, best_move

    def alpha_beta_max(self, board: ChessBoard, alpha: int, beta: int, remaining_depth: int):
        """
        Alpha-beta pruning implementation for maximizing player.
        Returns (score, best_move) tuple.
        """
        if remaining_depth == 0 or board.is_game_over():
            return self.evaluate_position(board), None
        
        best_value = float('-inf')
        best_move = None
        pseduo_sorted_moves = self.get_pseduo_sorted_moves(board)
        for move in pseduo_sorted_moves:
            self.moves_checked += 1
            if CHECKING_MOVE_ARROW and remaining_depth == DEPTH: # Display the root move
                self.game.checking_move = move
                self.game.display_board(self.game.last_move)  # Update display
            
            # Make a move and evaluate the position
            board.make_move(move)
            score = self.alpha_beta_min(board, alpha, beta, remaining_depth - 1)[0]
            board.undo_move()

            if (score > best_value):
                best_value = score
                best_move = move
                alpha = max(alpha, best_value) # alpha acts like max in MiniMax

            if best_value >= beta:
                return best_value, best_move # fail hard beta-cutoff
    
        return best_value, best_move
    
    def alpha_beta_min(self, board: ChessBoard, alpha: int, beta: int, remaining_depth: int ):
        """
        Alpha-beta pruning implementation for minimizing player.
        Returns (score, best_move) tuple.
        """
        if remaining_depth == 0 or board.is_game_over():
            return self.evaluate_position(board), None
        
        best_value = float('inf')
        best_move = None
        pseduo_sorted_moves = self.get_pseduo_sorted_moves(board)
        for move in pseduo_sorted_moves:
            self.moves_checked += 1
            if CHECKING_MOVE_ARROW and remaining_depth == DEPTH: # Display the root move
                self.game.display_board(self.game.last_move)  # Update display

            # Make a move and evaluate the position
            board.make_move(move)
            score = self.alpha_beta_max(board, alpha, beta, remaining_depth - 1)[0]
            board.undo_move()

            if (score < best_value):
                best_value = score
                best_move = move
                beta = min(beta, best_value) # beta acts like min in MiniMax

            if best_value <= alpha:
                return best_value, best_move # fail soft alpha-cutoff

        return best_value, best_move

    def minimax(self, board, depth: int, maximizing_player):
        """
        Minimax implementation.
        Returns (best_score, best_move)
        """
        if depth == 0 or board.is_game_over(): # Base case
            return self.evaluate_position(board), None

        best_move = None
        if maximizing_player:
            max_eval = float('-inf')
            for move in board.get_legal_moves(): # Try all possible moves and find best one according to the highest score
                self.moves_checked += 1
                board.make_move(move)
                eval, _ = self.minimax(board, depth - 1, False)
                board.undo_move()
                if eval > max_eval:
                    max_eval = eval
                    best_move = move
            return max_eval, best_move
        else: # Minimizing player (opponent)
            min_eval = float('inf')
            for move in board.get_legal_moves():
                self.moves_checked += 1
                board.make_move(move)
                eval, _ = self.minimax(board, depth - 1, True)
                board.undo_move()
                if eval < min_eval:
                    min_eval = eval
                    best_move = move
            return min_eval, best_move

    def get_move(self, board):
        """
        Main method to select the best move.
        """
        self.moves_checked = 0
        start_time = time.time()
        # best_move = self.minimax(board, depth=DEPTH, maximizing_player=board.get_board_state().turn)[1]

        # 21,025, 13,900, 20,633, 25,691, 17,216
        best_value, best_move = self.minimax_alpha_beta(board, DEPTH, float('-inf'), float('inf'), board.get_board_state().turn)
        # best_value, best_move = self.alpha_beta_max(board, float('-inf'), float('inf'), DEPTH)
        end_time = time.time()
        print(f"Moves checked: {self.moves_checked}") # Number of moves checked
        print(f"Time taken: {end_time - start_time:.2f}s") # Time taken to calculate best move
        # print(f"Eval: {self.evaluate_position(board)}") # The current position evaluation
        # print(f"Best value: {best_value}") # The best move evaluation
        return best_move
