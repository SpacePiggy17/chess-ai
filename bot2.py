from board import ChessBoard
import chess.polyglot  # Zobrist hashing

from dataclasses import dataclass  # For TT entries
from typing_extensions import TypeAlias  # For flags
from typing import Optional, TYPE_CHECKING
if TYPE_CHECKING:
    from game import ChessGame  # Only import while type checking

from lru import LRU  # For TT and history tables

from constants import DEPTH, MAX_VALUE, MIN_VALUE, CHECKING_MOVE_ARROW, RENDER_DEPTH, \
    PIECE_VALUES_STOCKFISH, FLIP, MIDGAME, ENDGAME, PSQT, START_NPM, NPM_SCALAR

import colors  # Debug log colors
import timeit  # Debug timing


# Transposition table entry flags
Flag: TypeAlias = int
EXACT: Flag = 1
LOWERBOUND: Flag = 2
UPPERBOUND: Flag = 3

# Transposition table entry
@dataclass
class TTEntry:
    depth: int
    value: int
    flag: Flag
    best_move: Optional[chess.Move]

# Score class to store scores and update them
@dataclass
class Score: # Positive values favor white, negative values favor black
    material: int # Material score
    mg: int # Midgame score
    eg: int # Endgame score
    npm: int # Non-pawn material (for phase calculation)

    def update(self, chess_board: chess.Board, move: chess.Move):
        """
        Updates the material, midgame, endgame, and non-pawn material scores based on the move.
        """
        from_square = move.from_square
        to_square = move.to_square
        promotion = move.promotion

        piece = chess_board.piece_at(from_square)
        captured_piece = chess_board.piece_at(to_square)

        is_en_passant = chess_board.is_en_passant(move)

        if piece.piece_type == chess.KING:
            # Update rook scores for castling
            if from_square == chess.E1: # White king
                if to_square == chess.G1: # White kingside castle
                    self.mg += PSQT[MIDGAME][chess.ROOK][FLIP[chess.F1]] - PSQT[MIDGAME][chess.ROOK][FLIP[chess.H1]]
                    self.eg += PSQT[ENDGAME][chess.ROOK][FLIP[chess.F1]] - PSQT[ENDGAME][chess.ROOK][FLIP[chess.H1]]
                elif to_square == chess.C1: # White queenside castle
                    self.mg += PSQT[MIDGAME][chess.ROOK][FLIP[chess.D1]] - PSQT[MIDGAME][chess.ROOK][FLIP[chess.A1]]
                    self.eg += PSQT[ENDGAME][chess.ROOK][FLIP[chess.D1]] - PSQT[ENDGAME][chess.ROOK][FLIP[chess.A1]]
            elif from_square == chess.E8: # Black king
                if to_square == chess.G8: # Black kingside castle
                    self.mg -= PSQT[MIDGAME][chess.ROOK][chess.F8] - PSQT[MIDGAME][chess.ROOK][chess.H8]
                    self.eg -= PSQT[ENDGAME][chess.ROOK][chess.F8] - PSQT[ENDGAME][chess.ROOK][chess.H8]
                elif to_square == chess.C8: # Black queenside castle
                    self.mg -= PSQT[MIDGAME][chess.ROOK][chess.D8] - PSQT[MIDGAME][chess.ROOK][chess.A8]
                    self.eg -= PSQT[ENDGAME][chess.ROOK][chess.D8] - PSQT[ENDGAME][chess.ROOK][chess.A8]

        # Update position scores for moving piece
        if promotion: # Promotion
            self.npm += PIECE_VALUES_STOCKFISH[promotion]
            if piece.color: # White promotion
                material += PIECE_VALUES_STOCKFISH[promotion] - PIECE_VALUES_STOCKFISH[chess.PAWN]
                self.mg += PSQT[MIDGAME][promotion][FLIP[to_square]] - PSQT[MIDGAME][chess.PAWN][FLIP[from_square]]
                self.eg += PSQT[ENDGAME][promotion][FLIP[to_square]] - PSQT[ENDGAME][chess.PAWN][FLIP[from_square]]
            else: # Black promotion
                material -= PIECE_VALUES_STOCKFISH[promotion] - PIECE_VALUES_STOCKFISH[chess.PAWN]
                self.mg -= PSQT[MIDGAME][promotion][to_square] - PSQT[MIDGAME][chess.PAWN][from_square]
                self.eg -= PSQT[ENDGAME][promotion][to_square] - PSQT[ENDGAME][chess.PAWN][from_square]
        else: # Normal move
            if piece.color: # White move
                self.mg += PSQT[MIDGAME][piece.piece_type][FLIP[to_square]] - PSQT[MIDGAME][piece.piece_type][FLIP[from_square]]
                self.eg += PSQT[ENDGAME][piece.piece_type][FLIP[to_square]] - PSQT[ENDGAME][piece.piece_type][FLIP[from_square]]
            else: # Black move
                self.mg -= PSQT[MIDGAME][piece.piece_type][to_square] - PSQT[MIDGAME][piece.piece_type][from_square]
                self.eg -= PSQT[ENDGAME][piece.piece_type][to_square] - PSQT[ENDGAME][piece.piece_type][from_square]

        if captured_piece: # Capture
            if is_en_passant: # Get the captured pawn from en passant
                # Update captured piece and square
                to_square += 8 if piece.color else -8
                captured_piece = chess_board.piece_at(to_square)

            if captured_piece.color: # White piece captured
                self.material -= PIECE_VALUES_STOCKFISH[captured_piece.piece_type]
                self.mg -= PSQT[MIDGAME][captured_piece.piece_type][FLIP[to_square]]
                self.eg -= PSQT[ENDGAME][captured_piece.piece_type][FLIP[to_square]]
            else: # Black piece captured
                self.material += PIECE_VALUES_STOCKFISH[captured_piece.piece_type]
                self.mg += PSQT[MIDGAME][captured_piece.piece_type][to_square]
                self.eg += PSQT[ENDGAME][captured_piece.piece_type][to_square]

            # Update npm score
            if captured_piece.piece_type != chess.PAWN:
                self.npm -= PIECE_VALUES_STOCKFISH[captured_piece.piece_type]

    def initialize_scores(self, chess_board: chess.Board):
        """
        Initialize values for starting position.
        Calculates material score, npm score, and evaluates piece positions.
        Evaluates piece positions using PSQT with interpolation between middlegame and endgame.
        Runs only once so not optimized for clarity.
        """
        white_bishop_count = 0
        black_bishop_count = 0

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
                    self.mg += PSQT[MIDGAME][piece.piece_type][FLIP[square]]
                    self.eg += PSQT[ENDGAME][piece.piece_type][FLIP[square]]
                    if piece.piece_type == chess.BISHOP:
                        white_bishop_count += 1
                else: # Black piece
                    self.material -= PIECE_VALUES_STOCKFISH[piece.piece_type]
                    self.mg -= PSQT[MIDGAME][piece.piece_type][square]
                    self.eg -= PSQT[ENDGAME][piece.piece_type][square]
                    if piece.piece_type == chess.BISHOP:
                        black_bishop_count += 1

        # Bishop pair bonus worth half a pawn
        if white_bishop_count >= 2:
            self.material += PIECE_VALUES_STOCKFISH[chess.PAWN] >> 1
        if black_bishop_count >= 2:
            self.material -= PIECE_VALUES_STOCKFISH[chess.PAWN] >> 1

    def get_interpolated_score(self) -> int:
        """Interpolates between midgame and endgame scores based on phase."""
        phase = self.npm // NPM_SCALAR # Phase value between 0 and 256 (0 = endgame, 256 = opening)
        return ((self.mg * phase) + (self.eg * (256 - phase))) // 256

class ChessBot:
    def __init__(self, game):
        self.game: "ChessGame" = game
        self.moves_checked: int = 0

        self.score = Score(0, 0, 0, 0)  # Score object to store material, mg, eg, and npm scores
        self.score.initialize_scores(self.game.board.get_board_state()) # Initialize scores once and update from there

        # self.transposition_table = LRU(1_000_000)  # Transposition table

    def display_checking_move_arrow(self, move):
        """Display an arrow on the board for the move being checked."""
        self.game.checking_move = move
        self.game.display_board(self.game.last_move)  # Update display

    def evaluate_position(self, chess_board: chess.Board, score: Score, key: Optional[int] = None, has_legal_moves=True) -> float:
        """
        Evaluate the current position.
        Positive values favor white, negative values favor black.
        """
        # Check expensive operations once
        if has_legal_moves:
            has_legal_moves = any(chess_board.legal_moves) # ! SLOW
        is_check: bool = chess_board.is_check()

        # Evaluate game-ending conditions
        if not has_legal_moves:  # No legal moves
            if is_check:  # Checkmate
                return -10_000 if chess_board.turn else 10_000
            return 0  # Stalemate
        elif chess_board.is_insufficient_material():  # Insufficient material for either side to win
            return 0
        elif chess_board.can_claim_fifty_moves():  # Avoid fifty move rule
            return 0

        # Return score (material + interpolated mg/eg score)
        return score.material + score.get_interpolated_score()

    # def quiescence(self, chess_board: chess.Board, alpha, beta, depth):

    def alpha_beta(self, chess_board: chess.Board, depth: int, alpha, beta, maximizing_player: bool, score: Score):
        # Terminal node check
        if depth == 0:
            return self.evaluate_position(chess_board, score), None
        legal_moves = list(chess_board.legal_moves)
        if not legal_moves:
            return self.evaluate_position(chess_board, score, has_legal_moves=False), None

        best_move = None
        if maximizing_player:
            best_value = MIN_VALUE
            for move in legal_moves:
                self.moves_checked += 1
                if CHECKING_MOVE_ARROW and depth >= RENDER_DEPTH:  # Display the root move
                    self.display_checking_move_arrow(move)

                old_material, old_mg, old_eg, old_npm = score.material, score.mg, score.eg, score.npm

                score.update(chess_board, move)
                chess_board.push(move)
                value = self.alpha_beta(chess_board, depth - 1, alpha, beta, False, score)[0]
                chess_board.pop()

                score.material, score.mg, score.eg, score.npm = old_material, old_mg, old_eg, old_npm

                if value > best_value:
                    best_value = value
                    best_move = move
                alpha = max(alpha, value)
                if value >= beta:
                    break  # Beta cutoff

        else:  # Minimizing player
            best_value = MAX_VALUE
            for move in legal_moves:
                self.moves_checked += 1
                if CHECKING_MOVE_ARROW and depth >= RENDER_DEPTH:  # Display the root move
                    self.display_checking_move_arrow(move)

                old_material, old_mg, old_eg, old_npm = score.material, score.mg, score.eg, score.npm

                score.update(chess_board, move)
                chess_board.push(move)
                value = self.alpha_beta(chess_board, depth - 1, alpha, beta, True, score)[0]
                chess_board.pop()

                score.material, score.mg, score.eg, score.npm = old_material, old_mg, old_eg, old_npm

                if value < best_value:
                    best_value = value
                    best_move = move
                beta = min(beta, value)
                if value <= alpha:
                    break  # Alpha cutoff
        
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
        start_time = timeit.default_timer()
        best_value, best_move = self.alpha_beta(
            chess_board, DEPTH, MIN_VALUE, MAX_VALUE, chess_board.turn, self.score)
        self.score.update(chess_board, best_move)
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
