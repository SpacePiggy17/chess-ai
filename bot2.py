from board import ChessBoard
import chess.polyglot # Zobrist hashing

from enum import IntEnum # For flags
from dataclasses import dataclass # For TT entries
from typing import Optional, TYPE_CHECKING
if TYPE_CHECKING:
    from game import ChessGame # Only import while type checking

from lru import LRU # For TT and history tables 

from constants import DEPTH, PIECE_VALUES_STOCKFISH, MAX_VALUE, MIN_VALUE, CHECKING_MOVE_ARROW

import colors # Debug log colors
import timeit # Debug timing

class Flag(IntEnum):
    EXACT = 0
    LOWERBOUND = 1
    UPPERBOUND = 2

@dataclass
class TTEntry:
    depth: int
    value: int
    flag: Flag
    best_move: Optional[chess.Move]

class ChessBot:
    def __init__(self, game):
        self.game: "ChessGame" = game
        self.moves_checked: int = 0

    def display_checking_move_arrow(self, move):
        self.game.checking_move = move
        self.game.display_board(self.game.last_move)  # Update display

    def evaluate_position(self, chess_board: chess.Board, key: Optional[int] = None, has_legal_moves = False) -> float:
        """
        Evaluate the current position.
        Uses the evaluation cache if available.
        Positive values favor white, negative values favor black.
        """
        # Check expensive operations once
        if not has_legal_moves:
            has_legal_moves = bool(chess_board.legal_moves) # ! SLOW
        is_check: bool = chess_board.is_check()

        # Evaluate game-ending conditions
        if not has_legal_moves: # No legal moves
            if is_check: # Checkmate
                return -10_000 if chess_board.turn else 10_000
            return 0 # Stalemate
        elif chess_board.is_insufficient_material(): # Insufficient material for either side to win
            return 0
        elif chess_board.can_claim_fifty_moves(): # Avoid fifty move rule
            return 0
        
        # Evaluate the position
        score = self.evaluate_material(chess_board) # Material evaluation
       
        return score

    def evaluate_material(self, chess_board: chess.Board):
        """
        Basic piece counting with standard values.
        Additional bonuses for piece combinations.
        Can be extended with phase-dependent values.
        """    
        # Count all pieces at once with direct bitboard access (much more optimized than provided version)
        wp = chess_board.occupied_co[chess.WHITE]
        bp = chess_board.occupied_co[chess.BLACK]
        
        # Pawns, Knights, Rooks, Queens
        score = (
            PIECE_VALUES_STOCKFISH[chess.PAWN] * (chess.popcount(wp & chess_board.pawns) - chess.popcount(bp & chess_board.pawns)) + 
            PIECE_VALUES_STOCKFISH[chess.KNIGHT] * (chess.popcount(wp & chess_board.knights) - chess.popcount(bp & chess_board.knights)) +
            PIECE_VALUES_STOCKFISH[chess.ROOK] * (chess.popcount(wp & chess_board.rooks) - chess.popcount(bp & chess_board.rooks)) +
            PIECE_VALUES_STOCKFISH[chess.QUEEN] * (chess.popcount(wp & chess_board.queens) - chess.popcount(bp & chess_board.queens))
        )

        # Bishops
        white_bishop_count = chess.popcount(wp & chess_board.bishops)
        black_bishop_count = chess.popcount( bp & chess_board.bishops)
        score += PIECE_VALUES_STOCKFISH[chess.BISHOP] * (white_bishop_count - black_bishop_count)
        
        # Bishop pair bonus worth half a pawn
        return score + ((white_bishop_count >= 2) - (black_bishop_count >= 2)) * (PIECE_VALUES_STOCKFISH[chess.PAWN] >> 1)
        
    # def evaluate_piece_position(self, chess_board: chess.Board):


    def alpha_beta(self, chess_board: chess.Board, depth: int, alpha, beta, maximizing_player: bool):
        # Terminal node check
        if depth == 0:
            return self.evaluate_position(chess_board), None
        legal_moves = list(chess_board.legal_moves)
        if not legal_moves:
            return self.evaluate_position(chess_board, has_legal_moves=False), None
        
        best_move = None
        if maximizing_player:
            best_value = MIN_VALUE
            for move in legal_moves:
                self.moves_checked += 1
                if CHECKING_MOVE_ARROW and depth == DEPTH: # Display the root move
                    self.display_checking_move_arrow(move)

                chess_board.push(move)
                value = self.alpha_beta(chess_board, depth - 1, alpha, beta, False)[0]
                chess_board.pop()

                if value > best_value:
                    best_value = value
                    best_move = move
                alpha = max(alpha, value)
                if value >= beta:
                    break # Beta cutoff

        else: # Minimizing player
            best_value = MAX_VALUE
            for move in legal_moves:
                self.moves_checked += 1
                if CHECKING_MOVE_ARROW and depth == DEPTH: # Display the root move
                    self.display_checking_move_arrow(move)

                chess_board.push(move)
                value = self.alpha_beta(chess_board, depth - 1, alpha, beta, True)[0]
                chess_board.pop()
                if value < best_value:
                    best_value = value
                    best_move = move
                beta = min(beta, value)
                if value <= alpha:
                    break # Alpha cutoff

        return best_value, best_move


    # TODO ---------------------------------------------
    # def next_guess(self, alpha, beta, subtree_count):
    #     return alpha + (beta - alpha) * (subtree_count - 1) / subtree_count

    # def best_node_search(self, chess_board: chess.Board, alpha, beta, maximizing_player: bool):
    #     return 0
    
    
    def get_move(self, board: ChessBoard):
        """
        Main method to get the best move for the current player.
        """        
        chess_board = board.get_board_state()
    
        self.moves_checked = 0

        # Run minimax once with manual timing
        start_time = timeit.default_timer()
        best_value, best_move = self.alpha_beta(chess_board, DEPTH, MIN_VALUE, MAX_VALUE, chess_board.turn)
        time_taken = timeit.default_timer() - start_time

        # TODO move print stuff into function
        # Moves checked over time taken
        time_per_move = time_taken / self.moves_checked if self.moves_checked > 0 else 0
        print(f"Moves/Time: {colors.BOLD}{colors.get_moves_color(self.moves_checked)}{self.moves_checked:,}{colors.RESET} / "
            f"{colors.BOLD}{colors.get_move_time_color(time_taken)}{time_taken:.2f}{colors.RESET} s = "
            f"{colors.BOLD}{colors.CYAN}{time_per_move * 1000:.4f}{colors.RESET} ms/M")

        # # Calculate memory usage more accurately
        # tt_entry_size = sys.getsizeof(TranspositionEntry(0, 0.0, "", None))
        # tt_size_mb = len(self.transposition_table) * tt_entry_size / (1024 * 1024)
        # eval_size_mb = sum(sys.getsizeof(k) + sys.getsizeof(v) for k, v in list(self.evaluation_cache.items())[:10]) / 10
        # eval_size_mb = eval_size_mb * len(self.evaluation_cache) / (1024 * 1024)
        # transposition_table_entries = len(self.transposition_table)

        # # Print cache statistics
        # print(f"Transposition table: {colors.BOLD}{colors.MAGENTA}{len(self.transposition_table):,}{colors.RESET} entries, "
        #     f"{colors.BOLD}{colors.CYAN}{tt_size_mb:.4f}{colors.RESET} MB")
        # print(f"Evaluation cache: {colors.BOLD}{colors.MAGENTA}{len(self.evaluation_cache):,}{colors.RESET} entries, "
        #       f"{colors.BOLD}{colors.CYAN}{eval_size_mb:.4f}{colors.RESET} MB")

        # Print the FEN
        print(f"FEN: {chess_board.fen()}")

        return best_move
