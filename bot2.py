from board import ChessBoard
import chess.polyglot  # Zobrist hashing

from dataclasses import dataclass  # For TT entries
from typing_extensions import TypeAlias  # For flags
from typing import Optional, TYPE_CHECKING
if TYPE_CHECKING:
    from game import ChessGame  # Only import while type checking

from lru import LRU  # For TT and history tables

from constants import DEPTH, MAX_VALUE, MIN_VALUE, CHECKING_MOVE_ARROW, \
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

class ChessBot:

    def __init__(self, game):
        self.game: "ChessGame" = game
        self.moves_checked: int = 0

        self.transposition_table = LRU(1_000_000)  # Transposition table

        self.material_score: int = 0 # Material score
        self.mg_score: int = 0 # Midgame score
        self.eg_score: int = 0 # Endgame score
        self.npm_score: int = 0 # Non-pawn material score

        self.initialize_scores() # Initialize scores

    @staticmethod
    def get_phase(npm_score: int) -> int:
        """Returns a value between 0 (endgame) and 256 (opening) based on remaining material."""
        return npm_score // NPM_SCALAR
    
    @staticmethod
    def interpolate(mg_score, eg_score, phase: int) -> int:
        """Interpolates between midgame and endgame scores based on phase."""
        return ((mg_score * phase) + (eg_score * (256 - phase))) // 256

    def initialize_scores(self):
        """
        Initialize values for starting position.
        Calculates material score, npm score, and evaluates piece positions.
        Evaluates piece positions using PSQT with interpolation between middlegame and endgame.
        Runs only once so not optimized for clarity.
        """
        chess_board = self.game.board.get_board_state()

        white_bishop_count = 0
        black_bishop_count = 0

        # Evaluate each piece type
        for piece_type in chess.PIECE_TYPES:
            # White pieces (flip square indices since tables are oriented for black)
            for square in chess_board.pieces(piece_type, chess.WHITE):
                # Update material score
                self.material_score += PIECE_VALUES_STOCKFISH[piece_type]

                # Update npm score
                if piece_type != chess.PAWN and piece_type != chess.KING:
                    self.npm_score += PIECE_VALUES_STOCKFISH[piece_type]

                    # Update bishop count for bishop pair bonus
                    if piece_type == chess.BISHOP:
                        white_bishop_count += 1

                # Update piece position scores
                flipped_square = FLIP[square]
                self.mg_score += PSQT[MIDGAME][piece_type][flipped_square]
                self.eg_score += PSQT[ENDGAME][piece_type][flipped_square]
            
            # Black pieces (use square indices directly)
            for square in chess_board.pieces(piece_type, chess.BLACK):
                # Update material score
                self.material_score -= PIECE_VALUES_STOCKFISH[piece_type]

                # Update npm score
                if piece_type != chess.PAWN and piece_type != chess.KING:
                    self.npm_score += PIECE_VALUES_STOCKFISH[piece_type]

                    # Update bishop count for bishop pair bonus
                    if piece_type == chess.BISHOP:
                        black_bishop_count += 1
                
                # Update piece position scores
                self.mg_score -= PSQT[MIDGAME][piece_type][square]
                self.eg_score -= PSQT[ENDGAME][piece_type][square]
        
        # Bishop pair bonus worth half a pawn
        if white_bishop_count >= 2:
            self.material_score += PIECE_VALUES_STOCKFISH[chess.PAWN] >> 1
        if black_bishop_count >= 2:
            self.material_score -= PIECE_VALUES_STOCKFISH[chess.PAWN] >> 1

        # Interpolate between middlegame and endgame scores based on phase
        phase = self.get_phase(self.npm_score)
        self.position_score = self.interpolate(self.mg_score, self.eg_score, phase)
        

    def display_checking_move_arrow(self, move):
        self.game.checking_move = move
        self.game.display_board(self.game.last_move)  # Update display

    def update_scores_for_moves(self, board: chess.Board, move: chess.Move):


    def evaluate_position(self, chess_board: chess.Board, key: Optional[int] = None, has_legal_moves=False) -> float:
        """
        Evaluate the current position.
        Uses the evaluation cache if available.
        Positive values favor white, negative values favor black.
        """
        # Check expensive operations once
        if not has_legal_moves:
            has_legal_moves = bool(chess_board.legal_moves)  # ! SLOW
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

        # # Get white and black bitboards
        # wp = chess_board.occupied_co[chess.WHITE]
        # bp = chess_board.occupied_co[chess.BLACK]

        # # Material evaluation
        # score, npm = self.evaluate_material(chess_board, wp, bp)
        
        # # Phase-based piece position evaluation
        # score += self.evaluate_piece_position(chess_board, wp, bp, phase)

        # Return the score (material + position)
        phase = self.get_phase(self.npm_score)
        return self.material_score + self.interpolate(self.mg_score, self.eg_score, phase)

    # def evaluate_material(self, chess_board: chess.Board, wp: chess.Bitboard, bp: chess.Bitboard) -> tuple[int, int]:
        """
        Basic piece counting with standard values.
        Additional bonuses for piece combinations.
        Can be extended with phase-dependent values.
        """
        white_pawn_count = (wp & chess_board.pawns).bit_count()
        black_pawn_count = (bp & chess_board.pawns).bit_count()
        white_knight_count = (wp & chess_board.knights).bit_count()
        black_knight_count = (bp & chess_board.knights).bit_count()
        white_bishop_count = (wp & chess_board.bishops).bit_count()
        black_bishop_count = (bp & chess_board.bishops).bit_count()
        white_rook_count = (wp & chess_board.rooks).bit_count()
        black_rook_count = (bp & chess_board.rooks).bit_count()
        white_queen_count = (wp & chess_board.queens).bit_count()
        black_queen_count = (bp & chess_board.queens).bit_count()

        score = PIECE_VALUES_STOCKFISH[chess.PAWN] * (white_pawn_count - black_pawn_count) + \
            PIECE_VALUES_STOCKFISH[chess.KNIGHT] * (white_knight_count - black_knight_count) + \
            PIECE_VALUES_STOCKFISH[chess.BISHOP] * (white_bishop_count - black_bishop_count) + \
            PIECE_VALUES_STOCKFISH[chess.ROOK] * (white_rook_count - black_rook_count) + \
            PIECE_VALUES_STOCKFISH[chess.QUEEN] * (white_queen_count - black_queen_count)

        # Non-pawn material (NPM)
        npm = PIECE_VALUES_STOCKFISH[chess.KNIGHT] * (white_knight_count + black_knight_count) + \
            PIECE_VALUES_STOCKFISH[chess.BISHOP] * (white_bishop_count + black_bishop_count) + \
            PIECE_VALUES_STOCKFISH[chess.ROOK] * (white_rook_count + black_rook_count) + \
            PIECE_VALUES_STOCKFISH[chess.QUEEN] * (white_queen_count + black_queen_count)

        # Return all socres (bishop pair bonus worth half a pawn)
        return score + ((white_bishop_count >= 2) - (black_bishop_count >= 2)) * (PIECE_VALUES_STOCKFISH[chess.PAWN] >> 1), npm

    # def evaluate_piece_position(self, chess_board: chess.Board, wp: chess.Bitboard, bp: chess.Bitboard, phase):
        """
        Evaluates piece positions using PSQT tables with interpolation between middlegame and endgame.
        Returns a score where positive values favor white and negative.
        """
        mg_score = 0
        eg_score = 0
        
        # Cache these lookups to avoid repeated dictionary access
        mg_pawn = PSQT[MIDGAME][chess.PAWN]
        mg_knight = PSQT[MIDGAME][chess.KNIGHT]
        mg_bishop = PSQT[MIDGAME][chess.BISHOP]
        mg_rook = PSQT[MIDGAME][chess.ROOK]
        mg_queen = PSQT[MIDGAME][chess.QUEEN]
        mg_king = PSQT[MIDGAME][chess.KING]
        
        eg_pawn = PSQT[ENDGAME][chess.PAWN]
        eg_knight = PSQT[ENDGAME][chess.KNIGHT]
        eg_bishop = PSQT[ENDGAME][chess.BISHOP]
        eg_rook = PSQT[ENDGAME][chess.ROOK]
        eg_queen = PSQT[ENDGAME][chess.QUEEN]
        eg_king = PSQT[ENDGAME][chess.KING]
        
        # White pawns
        w_bb = wp & chess_board.pawns
        while w_bb:
            square = chess.lsb(w_bb)
            flip_square = FLIP[square]
            mg_score += mg_pawn[flip_square]
            eg_score += eg_pawn[flip_square]
            w_bb &= w_bb - 1
        
        # White knights
        w_bb = wp & chess_board.knights
        while w_bb:
            square = chess.lsb(w_bb)
            flip_square = FLIP[square]
            mg_score += mg_knight[flip_square]
            eg_score += eg_knight[flip_square]
            w_bb &= w_bb - 1
        
        # White bishops
        w_bb = wp & chess_board.bishops
        while w_bb:
            square = chess.lsb(w_bb)
            flip_square = FLIP[square]
            mg_score += mg_bishop[flip_square]
            eg_score += eg_bishop[flip_square]
            w_bb &= w_bb - 1
        
        # White rooks
        w_bb = wp & chess_board.rooks
        while w_bb:
            square = chess.lsb(w_bb)
            flip_square = FLIP[square]
            mg_score += mg_rook[flip_square]
            eg_score += eg_rook[flip_square]
            w_bb &= w_bb - 1
        
        # White queens
        w_bb = wp & chess_board.queens
        while w_bb:
            square = chess.lsb(w_bb)
            flip_square = FLIP[square]
            mg_score += mg_queen[flip_square]
            eg_score += eg_queen[flip_square]
            w_bb &= w_bb - 1
        
        # White king
        w_bb = wp & chess_board.kings
        if w_bb:
            square = chess.lsb(w_bb)
            flip_square = FLIP[square]
            mg_score += mg_king[flip_square]
            eg_score += eg_king[flip_square]
        
        # Black pawns
        b_bb = bp & chess_board.pawns
        while b_bb:
            square = chess.lsb(b_bb)
            mg_score -= mg_pawn[square]
            eg_score -= eg_pawn[square]
            b_bb &= b_bb - 1
        
        # Black knights
        b_bb = bp & chess_board.knights
        while b_bb:
            square = chess.lsb(b_bb)
            mg_score -= mg_knight[square]
            eg_score -= eg_knight[square]
            b_bb &= b_bb - 1
        
        # Black bishops
        b_bb = bp & chess_board.bishops
        while b_bb:
            square = chess.lsb(b_bb)
            mg_score -= mg_bishop[square]
            eg_score -= eg_bishop[square]
            b_bb &= b_bb - 1
        
        # Black rooks
        b_bb = bp & chess_board.rooks
        while b_bb:
            square = chess.lsb(b_bb)
            mg_score -= mg_rook[square]
            eg_score -= eg_rook[square]
            b_bb &= b_bb - 1
        
        # Black queens
        b_bb = bp & chess_board.queens
        while b_bb:
            square = chess.lsb(b_bb)
            mg_score -= mg_queen[square]
            eg_score -= eg_queen[square]
            b_bb &= b_bb - 1
        
        # Black king
        b_bb = bp & chess_board.kings
        if b_bb:
            square = chess.lsb(b_bb)
            mg_score -= mg_king[square]
            eg_score -= eg_king[square]

        # Inline interpolate function to save a function call
        return

    # def quiescence(self, chess_board: chess.Board, alpha, beta, depth):

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
                if CHECKING_MOVE_ARROW and depth == DEPTH:  # Display the root move
                    self.display_checking_move_arrow(move)

                chess_board.push(move)
                value = self.alpha_beta(chess_board, depth - 1, alpha, beta, False)[0]
                chess_board.pop()

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
                if CHECKING_MOVE_ARROW and depth == DEPTH:  # Display the root move
                    self.display_checking_move_arrow(move)

                chess_board.push(move)
                value = self.alpha_beta(chess_board, depth - 1, alpha, beta, True)[0]
                chess_board.pop()
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

    def get_move(self, board: ChessBoard):
        """
        Main method to get the best move for the current player.
        """
        chess_board = board.get_board_state()

        self.moves_checked = 0

        # Run minimax once with manual timing
        start_time = timeit.default_timer()
        best_value, best_move = self.alpha_beta(
            chess_board, DEPTH, MIN_VALUE, MAX_VALUE, chess_board.turn)
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
