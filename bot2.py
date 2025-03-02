from board import ChessBoard
import chess.polyglot # Zobrist hashing

from enum import IntEnum # For flags
from dataclasses import dataclass # For TT entries
from typing import Optional

from lru import LRU # For TT and history tables 

from constants import DEPTH, PIECE_VALUES

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
        self.moves_checked = 0
        self.game = game

    def evaluate_position(self, chess_board: chess.Board, key: Optional[int] = None, has_legal_moves = False) -> float:
        """
        Evaluate the current position.
        Uses the evaluation cache if available.
        Positive values favor white, negative values favor black.
        """
        # Check expensive operations once
        if not has_legal_moves:
            has_legal_moves = bool(chess_board.legal_moves) # ! SLOW
        is_check = chess_board.is_check()

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
        score = 0
    
        # Count all pieces at once with direct bitboard access (much more optimized than provided version)
        wp = chess_board.occupied_co[chess.WHITE]
        bp = chess_board.occupied_co[chess.BLACK]
        
        # Pawns
        score += PIECE_VALUES[chess.PAWN] * chess.popcount(wp & chess_board.pawns)
        score -= PIECE_VALUES[chess.PAWN] * chess.popcount(bp & chess_board.pawns)
        
        # Knights
        score += PIECE_VALUES[chess.KNIGHT] * chess.popcount(wp & chess_board.knights)
        score -= PIECE_VALUES[chess.KNIGHT] * chess.popcount(bp & chess_board.knights)
        
        # Bishops
        white_bishop_count = chess.popcount(wp & chess_board.bishops)
        black_bishop_count = chess.popcount( bp & chess_board.bishops)
        score += PIECE_VALUES[chess.BISHOP] * white_bishop_count
        score -= PIECE_VALUES[chess.BISHOP] * black_bishop_count
        
        # Bishop pair bonus
        if white_bishop_count >= 2:
            score += 50
        if black_bishop_count >= 2:
            score -= 50
        
        # Rooks
        score += PIECE_VALUES[chess.ROOK] * chess.popcount(wp & chess_board.rooks)
        score -= PIECE_VALUES[chess.ROOK] * chess.popcount(bp & chess_board.rooks)
        
        # Queens
        score += PIECE_VALUES[chess.QUEEN] * chess.popcount(wp & chess_board.queens)
        score -= PIECE_VALUES[chess.QUEEN] * chess.popcount(bp & chess_board.queens)

        return score
        


    def alpha_beta(self, chess_board: chess.Board, depth: int, alpha, beta, maximizing_player: bool):
        legal_moves = chess_board.legal_moves

        # Ternimal node check
        if depth == 0:
            return self.evaluate_position(chess_board)
        elif not legal_moves:
            return self.evaluate_position(chess_board)
        
        if maximizing_player:
            value = float('-inf')
            for move in legal_moves:
                chess_board.push(move)
                temp_value = self.alpha_beta(chess_board, depth - 1, alpha, beta, False)
                value = max(value, temp_value)
                chess_board.pop()

                alpha = max(alpha, value)
                if value >= beta:
                    break # Beta cutoff
            return value
        else: # Minimizing player
            value = float('inf')
            for move in legal_moves:
                chess_board.push(move)
                temp_value = self.alpha_beta(chess_board, depth - 1, alpha, beta, True)
                value = min(value, temp_value)
                chess_board.pop()
                beta = min(beta, value)
                if value <= alpha:
                    break # Alpha cutoff
            return value

    def next_guess(self, alpha, beta, subtree_count):
        return alpha + (beta - alpha) * (subtree_count - 1) / subtree_count

    def best_node_search(self, chess_board: chess.Board, alpha, beta, maximizing_player: bool):
        # Get the number of subtrees to search
        legal_moves = list(chess_board.legal_moves)
        subtree_count = len(legal_moves)

        if subtree_count == 0:
            print("No legal moves")
            return None

        best_move = None
        while beta - alpha >= 2:
            gamma = self.next_guess(alpha, beta, subtree_count)
            better_count = 0
            candidate_move = None
            
            for move in legal_moves:
                chess_board.push(move)
                best_value = -self.alpha_beta(chess_board, 1, -gamma, -(gamma-1), not maximizing_player)
                chess_board.pop()
                
                if best_value >= gamma:
                    better_count += 1
                    best_move = move
            
            # Update alpha-beta range based on the search results
            if better_count > 1: # Too many nodes passed the test, raise the bar
                alpha = gamma
            elif better_count == 0: # No nodes passed the test, lower the bar
                beta = gamma
            else: # Found exactly one best node
                break
        
        return best_move
    
    
    def get_move(self, board: ChessBoard):
        """
        Main method to get the best move for the current player.
        """        
        chess_board = board.get_board_state()
    
        self.moves_checked = 0

        # Define the code to time
        def timed_minimax():
            best_move = self.best_node_search(chess_board, -1_000_000, 1_000_000, chess_board.turn)
            return best_move

        # Run minimax with timing
        number = 1  # Number of executions
        time_taken = timeit.timeit(timed_minimax, number=number) / number
        best_move = timed_minimax()


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
