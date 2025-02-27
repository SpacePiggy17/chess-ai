from board import ChessBoard
import chess.polyglot # Built-in Zobrist hashing
import time
import heapq # For priority queue

from constants import PIECE_VALUES, CENTER_SQUARES, DEPTH, CHECKING_MOVE_ARROW
import colors # Print logging colors

from dataclasses import dataclass
from typing import Optional

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

    # Built-in Zobrist hashing
    def store_position(self, board: ChessBoard, depth: int, value: float, flag: str, best_move: Optional[chess.Move]):
        """Store a position in the transposition table using chess.polyglot Zobrist hashing."""
        key = chess.polyglot.zobrist_hash(board.get_board_state())  # Use built-in hashing
        existing = self.transposition_table.get(key)
        
        if not existing or existing.depth <= depth:
            self.transposition_table[key] = TranspositionEntry(
                depth=depth,
                value=value,
                flag=flag,
                best_move=best_move,
            )

    def lookup_position(self, board: ChessBoard) -> Optional[TranspositionEntry]:
        """Lookup a position in the transposition table using chess.polyglot Zobrist hashing."""
        key = chess.polyglot.zobrist_hash(board.get_board_state())
        return self.transposition_table.get(key)


    def evaluate_position(self, board: ChessBoard):
        """
        Evaluate the current position.
        Positive values favor white, negative values favor black.
        """
        chess_board = board.get_board_state()
        if chess_board.is_checkmate():
            return -10_000 if chess_board.turn else 10_000    

        # elif chess_board.is_stalemate():
        #     return 0
        # elif chess_board.is_insufficient_material():
        #     return 0
        # elif chess_board.is_seventyfive_moves():
        #     return 0
        # elif chess_board.is_fivefold_repetition():
        #     return 0

        score = 0
        
        score += self.evaluate_material(chess_board) # Material and piece position evaluation
        
        # score += self.evaluate_piece_position(chess_board)

        # score += self.evaluate_pawn_structure(chess_board) # Pawn structure # WARNING REALLY SLOW
        # score += self.evaluate_king_safety(chess_board) # King safety # ~2s slower

        # score += self.evaluate_mobility(chess_board) # Mobility
       
        return score

    def evaluate_material(self, chess_board: chess.Board):
        """
        Basic piece counting with standard values.
        Additional bonuses for piece combinations.
        Can be extended with phase-dependent values.
        """
        score = 0
    
        # Basic material count
        for piece in PIECE_VALUES:
            score += len(chess_board.pieces(piece, True)) * PIECE_VALUES[piece]
            score -= len(chess_board.pieces(piece, False)) * PIECE_VALUES[piece]
    
        # Bishop pair bonus
        if len(chess_board.pieces(chess.BISHOP, True)) >= 2:
            score += 50
        if len(chess_board.pieces(chess.BISHOP, False)) >= 2:
            score -= 50

        return score

    def evaluate_pawn_structure(self, chess_board: chess.Board):
        """
        Checks for common pawn weaknesses.
        Evaluates pawn chains and islands.
        Identifies passed pawns and their value.
        """
        score = 0
    
        # Evaluate for both colors
        for color in [True, False]:
            multiplier = 1 if color else -1
            pawns = chess_board.pieces(chess.PAWN, color)
        
            # Check each file for isolated pawns
            for file in range(8):
                pawns_in_file = sum(1 for pawn in pawns
                                if chess.square_file(pawn) == file)
                if pawns_in_file > 0:
                    # Check adjacent files
                    adjacent_pawns = sum(1 for pawn in pawns
                        if chess.square_file(pawn) in [file-1, file+1])
                    if adjacent_pawns == 0:
                        score -= 20 * multiplier  # Isolated pawn penalty
                    
                if pawns_in_file > 1:
                    score -= 10 * multiplier  # Doubled pawn penalty
                
        return score

    def evaluate_king_safety(self, chess_board: chess.Board):
        """
        Analyze pawn shield and open files in near king.
        Can be extended with attack pattern recognition.
        """
        score = 0
    
        # Evaluate pawn shield for both kings
        for color in [True, False]:
            multiplier = 1 if color else -1
            king_square = chess_board.king(color)
            if king_square is None:
                continue
            
            king_file = chess.square_file(king_square)
            king_rank = chess.square_rank(king_square)
        
            # Check pawn shield
            shield_score = 0
            for file in range(max(0, king_file - 1), min(8, king_file + 2)):
                shield_rank = king_rank + (1 if color else -1)
                shield_square = chess.square(file, shield_rank)
                if chess_board.piece_at(shield_square) == chess.Piece(chess.PAWN, color):
                    shield_score += 10
                
            score += shield_score * multiplier
        
        return score

    def get_game_phase(chess_board: chess.Board):
        """
        Returns a value between 0 (endgame) and 256 (opening)
        based on remaining material
        """
        npm = 0  # Non-pawn material
        for piece_type in [chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN]:
            npm += len(chess_board.pieces(piece_type, True)) * PIECE_VALUES[piece_type]
            npm += len(chess_board.pieces(piece_type, False)) * PIECE_VALUES[piece_type]
    
        return min(npm, 256)

    def interpolate(mg_score, eg_score, phase):
        """
        Interpolate between middlegame and endgame scores
        based on game phase
        """
        return ((mg_score * phase) + (eg_score * (256 - phase))) // 256

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
        Avoids making a move and undoing it to evaluate the position.
        Returns a list of moves sorted by importance.
        TODO Turn into generator
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

    def sorted_move_generator(self, chess_board: chess.Board):
        """
        Generator that yields moves in order of importance:
        1. Captures by material gain
        2. Promotions
        3. Center control moves
        4. All other moves
        """
        good_moves = []
        other_moves = []
        
        for move in chess_board.legal_moves:
            score = 0
                
            # Score captures
            if chess_board.is_capture(move):
                victim = chess_board.piece_at(move.to_square)
                attacker = chess_board.piece_at(move.from_square)

                # Handle en passant captures
                if victim is None and move.to_square == chess_board.ep_square:
                    victim_square = move.to_square + (8 if chess_board.turn else -8)
                    victim = chess_board.piece_at(victim_square)

                if victim and attacker:
                    # Material gain/loss from capture
                    score += PIECE_VALUES[victim.piece_type] - PIECE_VALUES[attacker.piece_type]/100

            # Bonus for promotions
            if move.promotion:
                score += PIECE_VALUES[move.promotion] - PIECE_VALUES[chess.PAWN]

            # Small bonus for center control
            if move.to_square in CENTER_SQUARES:
                score += 0.1
                
            # Use a single heap for all moves
            heapq.heappush(good_moves, (-score, id(move), move))

        # Yield all moves in order of score
        while good_moves:
            yield heapq.heappop(good_moves)[2]

    def minimax_alpha_beta(self, board: ChessBoard, remaining_depth: int, alpha: int, beta: int, maximizing_player: bool):
        """
        Minimax algorithm with alpha-beta pruning.
        Returns (best_value, best_move) tuple.
        """
        if remaining_depth <= 0 or board.get_board_state().is_checkmate() or board.get_board_state().is_stalemate(): # Simple game over check
            return self.evaluate_position(board), None

        # if remaining_depth <= 0: # Implement quiescence search
        #     return self.quiescence_search(board, alpha, beta), None

        # Lookup position in transposition table
        transposition = self.lookup_position(board)
        if transposition and transposition.depth >= remaining_depth:
            stored_value: float = transposition.value
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

        if maximizing_player:
            # pseduo_sorted_moves = self.get_sorted_moves(board)
            best_value = float('-inf')
            for move in self.get_sorted_moves(board):
            # for move in self.sorted_move_generator(board.get_board_state()):
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
                if beta <= alpha: # Big performance improvement
                    break # Black's best response is worse than White's guarenteed value

        else: # Minimizing player
            # pseduo_sorted_moves = self.get_sorted_moves(board)
            best_value = float('inf')
            for move in self.get_sorted_moves(board):
            # for move in self.sorted_move_generator(board.get_board_state()):
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
                if beta <= alpha: # Big performance improvement
                    break # White guarenteed value is better than Black's best option

        # Store position in transposition table with normalized value
        if best_value <= original_alpha:
            flag = 'UPPERBOUND'
        if best_value >= beta:
            flag = 'LOWERBOUND'
        else:
            flag = 'EXACT'
        
        self.store_position(board, remaining_depth, best_value, flag, best_move)

        return best_value, best_move


    def get_move(self, board):
        """
        Main method to select the best move.
        """
        self.moves_checked = 0
        start_time = time.time()

        best_value, best_move = self.minimax_alpha_beta(board, DEPTH, float('-inf'), float('inf'), board.get_board_state().turn)

        end_time = time.time()
        time_taken = end_time - start_time

        # Moves checked over time taken
        time_per_move = time_taken / self.moves_checked if self.moves_checked > 0 else 0
        print(f"Moves/Time: {colors.BOLD}{colors.get_moves_color(self.moves_checked)}{self.moves_checked:,}{colors.RESET} / "
            f"{colors.BOLD}{colors.get_move_time_color(time_taken)}{time_taken:.2f}{colors.RESET} s = "
            f"{colors.BOLD}{colors.CYAN}{time_per_move * 1000:.4f}{colors.RESET} ms/M")
        # Size of transposition table
        print(f"Transposition table: {colors.BOLD}{colors.MAGENTA}{len(self.transposition_table):,}{colors.RESET} entries, "
            f"{colors.BOLD}{colors.CYAN}{sys.getsizeof(self.transposition_table)/ (1024 * 1024):.4f}{colors.RESET} MB")

        return best_move

# How to improve speed and efficiency more?
