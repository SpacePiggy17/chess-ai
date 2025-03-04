import chess
# from board import ChessBoard
from chess.polyglot import zobrist_hash # Built-in Zobrist hashing  TODO implement incremental hashing

from dataclasses import dataclass  # For TT entries and scores
from typing_extensions import TypeAlias  # For flags
import numpy as np
from typing import Optional, TYPE_CHECKING
if TYPE_CHECKING:
    from game import ChessGame  # Only import while type checking

from lru import LRU  # For TT and history tables
from sys import getsizeof # For memory usage

from constants import DEPTH, MAX_VALUE, MIN_VALUE, CHECKING_MOVE_ARROW, RENDER_DEPTH, TT_SIZE, \
    PIECE_VALUES_STOCKFISH, BISHOP_PAIR_BONUS, FLIP, MIDGAME, ENDGAME, PSQT, CASTLING_UPDATES, NPM_SCALAR

import colors  # Debug log colors
from timeit import default_timer # For debugging timing


# Transposition table entry flags
Flag: TypeAlias = np.int8
EXACT: Flag = 1
UPPERBOUND: Flag = 2 # Alhpa (fail-low)
LOWERBOUND: Flag = 3 # Beta (fail-high)

# Transposition table entry
@dataclass
class TTEntry:
    __slots__ = ["depth", "value", "flag", "best_move"] # Optimization for faster lookups

    depth: np.int8
    value: np.int16
    flag: Flag
    best_move: Optional[chess.Move]

# Score class to store scores and update them
@dataclass
class Score: # Positive values favor white, negative values favor black
    __slots__ = ["material", "mg", "eg", "npm"] # Optimization for faster lookups

    material: np.int16 # Material score
    mg: np.int16 # Midgame score
    eg: np.int16 # Endgame score
    npm: np.int16 # Non-pawn material (for phase calculation)

    def update(self, chess_board: chess.Board, move: chess.Move):
        """
        Updates the material, midgame, endgame, and non-pawn material scores based on the move.
        Much faster than re-evaluating the entire board, even if only the leaf nodes are re-evaluated.
        """
        from_square = move.from_square
        to_square = move.to_square
        promotion_piece_type: chess.PieceType = move.promotion

        piece_type = chess_board.piece_type_at(from_square)
        piece_color = chess_board.turn
        color_multiplier = 1 if piece_color else -1

        # Cache tables for faster lookups
        piece_values = PIECE_VALUES_STOCKFISH
        mg_tables = PSQT[MIDGAME]
        eg_tables = PSQT[ENDGAME]
        flip = FLIP

        # Update rook scores for castling
        castling = False
        if piece_type == chess.KING:
            castle_info = CASTLING_UPDATES.get((from_square, to_square, piece_color))
            if castle_info:
                castling = True
                mg_rook_table = mg_tables[chess.ROOK]
                eg_rook_table = eg_tables[chess.ROOK]

                rook_from, rook_to = castle_info
                if piece_color: # Flip rook square for white
                    rook_from, rook_to = flip[rook_from], flip[rook_to]
                
                self.mg += color_multiplier * (mg_rook_table[rook_to] - mg_rook_table[rook_from])
                self.eg += color_multiplier * (eg_rook_table[rook_to] - eg_rook_table[rook_from])
                
        # Flip squares for white
        new_from_square, new_to_square = from_square, to_square
        if piece_color:
            new_from_square, new_to_square = flip[from_square], flip[to_square]

        # Update position scores for moving piece
        if promotion_piece_type: # Promotion
            # Update bishop pair bonus if pawn promoted to bishop
            if promotion_piece_type == chess.BISHOP:
                bishop_count_before = chess_board.pieces_mask(chess.BISHOP, piece_color).bit_count()
                if bishop_count_before == 1: # If 2 bishops now, add bonus
                    self.material += color_multiplier * BISHOP_PAIR_BONUS

            self.npm += piece_values[promotion_piece_type]
            self.material += color_multiplier * (piece_values[promotion_piece_type] - piece_values[chess.PAWN])
            self.mg += color_multiplier * (mg_tables[promotion_piece_type][new_to_square] - mg_tables[chess.PAWN][new_from_square])
            self.eg += color_multiplier * (eg_tables[promotion_piece_type][new_to_square] - eg_tables[chess.PAWN][new_from_square])

        else: # Normal move
            mg_table = mg_tables[piece_type]
            eg_table = eg_tables[piece_type]
            self.mg += color_multiplier * (mg_table[new_to_square] - mg_table[new_from_square])
            self.eg += color_multiplier * (eg_table[new_to_square] - eg_table[new_from_square])

        if castling: # Done if castling
            return
        
        # Handle captures
        captured_piece: chess.Piece = chess_board.piece_at(to_square)

        # Get en passant capture piece if applicable
        if not captured_piece and piece_type == chess.PAWN and chess_board.is_en_passant(move):
            to_square -= color_multiplier * 8
            captured_piece = chess_board.piece_at(to_square)

        if captured_piece: # Capture
            captured_piece_type = captured_piece.piece_type
            captured_piece_color = captured_piece.color
            if captured_piece_type != chess.PAWN:
                # Update npm score
                self.npm -= piece_values[captured_piece_type]

                # Update bishop pair bonus if bishop captured
                if captured_piece_type == chess.BISHOP and chess_board.pieces_mask(captured_piece_type, captured_piece_color).bit_count() == 2:
                    self.material -= -color_multiplier * BISHOP_PAIR_BONUS # If 2 bishops before, remove bonus

            if captured_piece_color: # Flip squares for white
                to_square = flip[to_square]

            # Remove captured piece from material and position scores
            self.material -= -color_multiplier * piece_values[captured_piece_type]
            self.mg -= -color_multiplier * mg_tables[captured_piece_type][to_square]
            self.eg -= -color_multiplier * eg_tables[captured_piece_type][to_square]

    def initialize_scores(self, chess_board: chess.Board):
        """
        Initialize values for starting position.
        Calculates material score, npm score, and evaluates piece positions.
        Evaluates piece positions using PSQT with interpolation between middlegame and endgame.
        Runs only once so not optimized for clarity.
        """
        self.material = 0
        self.mg = 0
        self.eg = 0
        self.npm = 0

        white_bishop_count = 0
        black_bishop_count = 0

        mg_tables = PSQT[MIDGAME]
        eg_tables = PSQT[ENDGAME]

        # Evaluate each piece type
        for square in chess.SQUARES:
            piece = chess_board.piece_at(square)
            if piece:
                # Update npm score
                if piece.piece_type != chess.PAWN and piece.piece_type != chess.KING:
                    self.npm += PIECE_VALUES_STOCKFISH[piece.piece_type]

                # Update material and position scores
                if piece.color: # White piece
                    self.material += PIECE_VALUES_STOCKFISH[piece.piece_type]
                    self.mg += mg_tables[piece.piece_type][FLIP[square]]
                    self.eg += eg_tables[piece.piece_type][FLIP[square]]
                    if piece.piece_type == chess.BISHOP:
                        white_bishop_count += 1
                else: # Black piece
                    self.material -= PIECE_VALUES_STOCKFISH[piece.piece_type]
                    self.mg -= mg_tables[piece.piece_type][square]
                    self.eg -= eg_tables[piece.piece_type][square]
                    if piece.piece_type == chess.BISHOP:
                        black_bishop_count += 1

        # Bishop pair bonus worth half a pawn
        if white_bishop_count >= 2:
            self.material += BISHOP_PAIR_BONUS
        if black_bishop_count >= 2:
            self.material -= BISHOP_PAIR_BONUS

class ChessBot:
    __slots__ = ["game", "moves_checked", "transposition_table"]  # Optimization for fast lookups

    def __init__(self, game):
        self.game: "ChessGame" = game
        self.moves_checked: int = 0

        # Initialize transposition table with size in MB
        tt_entry_size = getsizeof(TTEntry(0, 0, EXACT, chess.Move.from_uci("e2e4")))
        self.transposition_table = LRU(int(TT_SIZE * 1024 * 1024 / tt_entry_size))  # Initialize TT with size in MB

    def display_checking_move_arrow(self, move):
        """Display an arrow on the board for the move being checked."""
        self.game.checking_move = move
        self.game.display_board(self.game.last_move)  # Update display

    def evaluate_position(self, chess_board: chess.Board, score: Score, tt_entry=None, has_legal_moves=True) -> float:
        """
        Evaluate the current position.
        Positive values favor white, negative values favor black.
        """
        if tt_entry:
            return tt_entry.value

        # Check expensive operations once
        if has_legal_moves:
            has_legal_moves = any(chess_board.legal_moves) # ! REALLY SLOW

        # Evaluate game-ending conditions
        if not has_legal_moves:  # No legal moves
            if chess_board.is_check():  # Checkmate
                return -10_000 if chess_board.turn else 10_000
            return 0  # Stalemate
        elif chess_board.is_insufficient_material():  # Insufficient material for either side to win
            return 0
        elif chess_board.can_claim_fifty_moves():  # Avoid fifty move rule
            return 0

        # Return score (material + interpolated mg/eg score)
        phase = min(score.npm // NPM_SCALAR, 256) # Phase value between 0 and 256 (0 = endgame, 256 = opening)
        interpolated_score = ((int(score.mg) * phase) + (int(score.eg) * (256 - phase))) >> 8 # Int division by 256
        return score.material + interpolated_score

    # def quiescence(self, chess_board: chess.Board, alpha, beta, depth):

    def ordered_moves_generator(self, chess_board: chess.Board, tt_move: Optional[chess.Move]):
        """Generate ordered moves for the current position."""
        # Yield transposition table move first
        if tt_move:
            yield tt_move

        # Sort remaining moves
        ordered_moves = []
        for move in chess_board.legal_moves:
            if not tt_move or move != tt_move: # Skip TT move since already yielded
                score = 0

                # Capturing a piece bonus (MVV/LVA - Most Valuable Victim/Least Valuable Attacker)
                if chess_board.is_capture(move):
                    victim = chess_board.piece_at(move.to_square)
                    attacker = chess_board.piece_at(move.from_square)

                    # Handle en passant captures
                    if victim is None and move.to_square == chess_board.ep_square:
                        victim = chess_board.piece_at(move.to_square + (-8 if attacker.color else 8))
                        score += 5 # Small bonus for en passant captures

                    if victim and attacker:
                        # Prioritize capturing higher value pieces using lower value pieces
                        score += 10_000 + PIECE_VALUES_STOCKFISH[victim.piece_type] - PIECE_VALUES_STOCKFISH[attacker.piece_type]

                if move.promotion: # Promotion bonus
                    score += 1_000 + PIECE_VALUES_STOCKFISH[move.promotion] - PIECE_VALUES_STOCKFISH[chess.PAWN]

                # if chess_board.gives_check(move): # Check bonus
                #     score += 100

                # # Center control bonus
                # if move.to_square in CENTER_SQUARES:
                #     score += 100

                ordered_moves.append((move, score))

        ordered_moves.sort(key=lambda x: x[1], reverse=True)

        yield from (move for move, _ in ordered_moves)


    def alpha_beta(self, chess_board: chess.Board, depth: np.int8, alpha, beta, maximizing_player: bool, score: Score):
        # Lookup position in transposition table
        key = chess_board._transposition_key() # ? Much faster
        # key = zobrist_hash(chess_board) # ! REALLY SLOW
        tt_entry = self.transposition_table.get(key)

        # If position is in transposition table and depth is sufficient
        if tt_entry and tt_entry.depth >= depth:
            if (tt_entry.flag == EXACT) or (tt_entry.flag == LOWERBOUND and tt_entry.value >= beta) or (tt_entry.flag == UPPERBOUND and tt_entry.value <= alpha):
                return tt_entry.value, tt_entry.best_move

        # Terminal node check
        if depth == 0:
            return self.evaluate_position(chess_board, score, tt_entry), None

        tt_move = tt_entry.best_move if tt_entry else None
        original_alpha = alpha
        best_move = None
        if maximizing_player:
            original_score = score
            best_value = MIN_VALUE
            for move in self.ordered_moves_generator(chess_board, tt_move):
                has_legal_moves = True
                self.moves_checked += 1
                if CHECKING_MOVE_ARROW and depth >= RENDER_DEPTH:  # Display the root move
                    self.display_checking_move_arrow(move)

                score.update(chess_board, move)

                chess_board.push(move)
                value = self.alpha_beta(chess_board, depth - 1, alpha, beta, False, score)[0]
                chess_board.pop()

                score = original_score # Restore score

                if value > best_value:
                    best_value = value
                    best_move = move
                alpha = max(alpha, value)
                if value >= beta:
                    break  # Beta cutoff

        else:  # Minimizing player
            original_score = score
            best_value = MAX_VALUE
            for move in self.ordered_moves_generator(chess_board, tt_move):
                self.moves_checked += 1
                if CHECKING_MOVE_ARROW and depth >= RENDER_DEPTH:  # Display the root move
                    self.display_checking_move_arrow(move)

                score.update(chess_board, move)

                chess_board.push(move)
                value = self.alpha_beta(chess_board, depth - 1, alpha, beta, True, score)[0]
                chess_board.pop()

                score = original_score

                if value < best_value:
                    best_value = value
                    best_move = move
                beta = min(beta, value)
                if value <= alpha:
                    break  # Alpha cutoff

        
        if best_move is None: # If no legal moves, evaluate position
            return self.evaluate_position(chess_board, score, tt_entry, has_legal_moves=False), None

        # Store position in transposition table
        flag = EXACT
        if best_value <= original_alpha:
            flag = UPPERBOUND
        elif best_value >= beta:
            flag = LOWERBOUND
        self.transposition_table[key] = TTEntry(depth, best_value, flag, best_move)

        return best_value, best_move

    # TODO ---------------------------------------------
    # def next_guess(self, alpha, beta, subtree_count):
    #     return alpha + (beta - alpha) * (subtree_count - 1) / subtree_count

    # def best_node_search(self, chess_board: chess.Board, alpha, beta, maximizing_player: bool):
    #     return 0

    def get_move(self, chess_board: chess.Board):
        """
        Main method to get the best move for the current player.
        """
        self.moves_checked = 0


        # Run minimax once with manual timing
        start_time = default_timer()

        temp_score = Score(
            self.game.score.material,
            self.game.score.mg,
            self.game.score.eg,
            self.game.score.npm)
        best_value, best_move = self.alpha_beta(
            chess_board,
            DEPTH,
            MIN_VALUE,
            MAX_VALUE,
            chess_board.turn,
            temp_score)
        self.game.score.update(chess_board, best_move)

        time_taken = default_timer() - start_time

        # TODO move print stuff into function
        # Moves checked over time taken
        time_per_move = time_taken / self.moves_checked if self.moves_checked > 0 else 0
        moves_per_second = 1 / time_per_move if time_per_move > 0 else 0
        print(f"Moves/Time: {colors.BOLD}{colors.get_moves_color(self.moves_checked)}{self.moves_checked:,}{colors.RESET} / "
              f"{colors.BOLD}{colors.get_move_time_color(time_taken)}{time_taken:.2f}{colors.RESET} s = "
              f"{colors.BOLD}{colors.CYAN}{time_per_move * 1000:.4f}{colors.RESET} ms/M, "
              f"{colors.BOLD}{colors.CYAN}{moves_per_second:,.0f}{colors.RESET} M/s")

        # Calculate memory usage more accurately
        tt_entry_size = getsizeof(TTEntry(0, 0, EXACT, chess.Move.from_uci("e2e4")))
        tt_size_mb = len(self.transposition_table) * tt_entry_size / (1024 * 1024)
        transposition_table_entries = len(self.transposition_table)
        # eval_size_mb = sum(getsizeof(k) + getsizeof(v) for k, v in list(self.evaluation_cache.items())[:10]) / 10
        # eval_size_mb = eval_size_mb * len(self.evaluation_cache) / (1024 * 1024)

        # # Print cache statistics
        print(f"Transposition table: {colors.BOLD}{colors.MAGENTA}{len(self.transposition_table):,}{colors.RESET} entries, "
            f"{colors.BOLD}{colors.CYAN}{tt_size_mb:.4f}{colors.RESET} MB")
        # print(f"Evaluation cache: {colors.BOLD}{colors.MAGENTA}{len(self.evaluation_cache):,}{colors.RESET} entries, "
        #       f"{colors.BOLD}{colors.CYAN}{eval_size_mb:.4f}{colors.RESET} MB")

        # Print the FEN
        print(f"FEN: {chess_board.fen()}")

        return best_move
