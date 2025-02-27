from board import ChessBoard
import time
import heapq

from constants import PIECE_VALUES, CENTER_SQUARES, DEPTH, CHECKING_MOVE_ARROW
import colors

from dataclasses import dataclass
from typing import Optional
import chess

import sys

@dataclass
class TranspositionEntry:
    depth: int
    value: float
    flag: str  # 'EXACT', 'LOWERBOUND', or 'UPPERBOUND'
    best_move: Optional[chess.Move]

class ChessBot:
    def __init__(self, game=None):
        self.moves_checked = 0
        self.game = game # Reference to the game object
        self.transposition_table = {}  # Dictionary to store positions

    def store_position(self, board: ChessBoard, depth: int, value: float, flag: str, best_move: Optional[chess.Move]):
        """Store a position in the transposition table using the fen as the key."""
        key = board.get_board_state().fen()
        existing = self.transposition_table.get(key)
        
        # Replace if new position is searched deeper
        if not existing or existing.depth <= depth:
            self.transposition_table[key] = TranspositionEntry(
                depth=depth,
                value=value,
                flag=flag,
                best_move=best_move
            )

    def lookup_position(self, board: ChessBoard) -> Optional[TranspositionEntry]:
        """Lookup a position in the transposition table using the fen as the key."""
        key = board.get_board_state().fen()
        return self.transposition_table.get(key)

    def evaluate_position(self, board: ChessBoard):
        """
        Evaluate the current position.
        Positive values favor white, negative values favor black.
        """
        chess_board = board.get_board_state()

        if board.is_game_over():
            if chess_board.is_checkmate():
                return -10_000 if chess_board.turn else 10_000
            return 0  # Draw

        score = 0

        # Count material
        for piece in PIECE_VALUES:
            score += len(chess_board.pieces(piece, True)) * PIECE_VALUES[piece]
            score -= len(chess_board.pieces(piece, False)) * PIECE_VALUES[piece]
           
        return score

    def get_sorted_moves(self, board: ChessBoard) -> list:
        """
        Score all legal moves in the current position using the following data:
            - Checkmate
            - Check
            - Pinned pieces
            - Captures weighted by piece values
            - Center control
            - Promotion
            - Threats
        Avoid making a move and undoing it to evaluate the position.
        Returns a list of moves sorted by importance.
        """
        chess_board = board.get_board_state()
        good_moves, bad_moves, other_moves = [], [], []
        for move in board.get_legal_moves():
            score = 0
                
            # if chess_board.is_checkmate(): # In checkmate
            #     score -= 100_000

            # Piece is pinned to the king
            # if chess_board.is_pinned(chess_board.turn, move.from_square):
            #     score -= 50_000

            # if chess_board.is_check(): # Is in check
            #     score -= 10_000

            if chess_board.is_capture(move): # Capturing a piece
                victim = chess_board.piece_at(move.to_square)
                attacker = chess_board.piece_at(move.from_square)

                # Handle en passant captures
                if victim is None and move.to_square == chess_board.ep_square:
                    victim_square = move.to_square + (8 if chess_board.turn else -8)
                    victim = chess_board.piece_at(victim_square)

                if victim and attacker:
                    # Prioritize capturing higher value pieces using lower value pieces
                    score += 10_000 + PIECE_VALUES[victim.piece_type] - PIECE_VALUES[attacker.piece_type]/100

            if move.promotion: # Move results in promotion
                score += 9_000

            if move.to_square in CENTER_SQUARES: # Center control
                score += 100

            # if score == 0: # Move is not positive or negative



            # board.make_move(move)
            # chess_board = board.get_board_state()

            # if chess_board.is_checkmate(): # Results in checkmate
            #     score += 100_000

            # if chess_board.is_check(): # Results in check
            #     score += 10_000

            # # Find any pieces that were just pinned by the move
            # for square in chess_board.piece_map():  # Check all pieces
            #     if chess_board.color_at(square) == chess_board.turn: # Opponent pieces
            #         if chess_board.is_pinned(chess_board.turn, square): # Opponent pinned
            #             score += 9_000
            #     else: # Our pieces
            #         if chess_board.is_pinned(not chess_board.turn, square): # Us pinned
            #             score -= 20_000

            # for square in chess_board.attacks(move.to_square): # Move is a threat to capture a piece
            #     if chess_board.is_attacked_by(not chess_board.turn, square): # Opponent threatened
            #         victim = chess_board.piece_at(square)
            #         attacker = chess_board.piece_at(move.to_square)
                
            #         if victim and attacker and victim.color != attacker.color:
            #             # Prioritize endangering higher value pieces using lower value pieces
            #             score += 1_000 + PIECE_VALUES[victim.piece_type] - PIECE_VALUES[attacker.piece_type]/100

            # board.undo_move()
                
            if score > 0:
                heapq.heappush(good_moves, (-score, id(move), move))  # Negative score for max-heap
            elif score < 0:
                heapq.heappush(bad_moves, (-score, id(move), move))  # Negative score for max-heap
            else:
                other_moves.append(move)



        good_moves = [item[2] for item in good_moves]
        bad_moves = [item[2] for item in bad_moves]
        return good_moves + other_moves + bad_moves

    def minimax_alpha_beta(self, board: ChessBoard, remaining_depth: int, alpha: int, beta: int, maximizing_player: bool):
        """
        Minimax algorithm with alpha-beta pruning.
        Returns (best_value, best_move) tuple.
        """
        if remaining_depth == 0 or board.is_game_over():
            return self.evaluate_position(board), None

        # Lookup position in transposition table
        transposition = self.lookup_position(board)
        if transposition and transposition.depth >= remaining_depth:
            stored_value = transposition.value if maximizing_player else -transposition.value
            if transposition.flag == 'EXACT':
                return stored_value, transposition.best_move
            elif transposition.flag == 'LOWERBOUND':
                alpha = max(alpha, stored_value)
            elif transposition.flag == 'UPPERBOUND':
                beta = min(beta, stored_value)
            if beta <= alpha:
                return stored_value, transposition.best_move

        best_move = None
        original_alpha = alpha
        pseduo_sorted_moves = self.get_sorted_moves(board)

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
                    break # Black's best response is worse than White's guarenteed value

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
                    break # White guarenteed value is better than Black's best option

        # Store position in transposition table with normalized value
        store_value = best_value if maximizing_player else -best_value
        if best_value <= original_alpha:
            flag = 'UPPERBOUND'
        elif best_value >= beta:
            flag = 'LOWERBOUND'
        else:
            flag = 'EXACT'
        
        self.store_position(board, remaining_depth, best_value, flag, best_move)

        return best_value, best_move

    def get_move(self, board):
        """
        Main method to select the best move.
        """
        if board.get_board_state().fullmove_number == 1 and board.get_board_state().turn:  # First move of the game
            self.current_run_row = 0  # Reset row counter for new game

        self.moves_checked = 0
        start_time = time.time()

        best_value, best_move = self.minimax_alpha_beta(board, DEPTH, float('-inf'), float('inf'), board.get_board_state().turn)

        end_time = time.time()
        time_taken = end_time - start_time

        # Moves checked over time taken
        print(f"Moves/Time: {colors.BOLD}{colors.get_moves_color(self.moves_checked)}{self.moves_checked:,}{colors.RESET} / "
            f"{colors.BOLD}{colors.get_move_time_color(time_taken)}{time_taken:.2f}{colors.RESET} s = "
            f"{colors.BOLD}{colors.CYAN}{time_taken/self.moves_checked * 1000:.4f}{colors.RESET} ms/M")
        # Size of transposition table
        print(f"Transposition table: {colors.BOLD}{colors.MAGENTA}{len(self.transposition_table):,}{colors.RESET} entries, "
            f"{colors.BOLD}{colors.CYAN}{sys.getsizeof(self.transposition_table)/ (1024 * 1024):.4f}{colors.RESET} MB")

        return best_move
