from board import ChessBoard
import chess.polyglot # Built-in Zobrist hashing
# import time
import timeit
import heapq # For priority queue

from constants import PIECE_VALUES, CENTER_SQUARES, DEPTH, CHECKING_MOVE_ARROW
import colors # Print logging colors

from dataclasses import dataclass
from typing import Optional

import sys

from cachetools import LRUCache

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
        self.transposition_table = LRUCache(maxsize=1_000_000)  # LRU cache for transposition table
        self.position_history = {} # Dictionary to store position history
        self.evaluation_cache = LRUCache(maxsize=300_000)  # LRU cache for evaluations

    # Built-in Zobrist hashing
    def store_position(self, chess_board: chess.Board, depth: int, value: float, flag: str, best_move: Optional[chess.Move], key: int = None):
        """Store a position in the transposition table using chess.polyglot Zobrist hashing."""
        existing = self.lookup_position(chess_board, key)
        
        # Store if not existing
        if not existing:
            self.transposition_table[key] = TranspositionEntry(
                depth=depth,
                value=value,
                flag=flag,
                best_move=best_move,
            )
        elif existing.depth <= depth: # Replace if new depth is greater
            self.transposition_table[key].depth = depth
            self.transposition_table[key].value = value
            self.transposition_table[key].flag = flag
            self.transposition_table[key].best_move = best_move

    def lookup_position(self, chess_board: chess.Board, key: int) -> Optional[TranspositionEntry]:
        """
        Lookup a position in the transposition table using chess.polyglot Zobrist hashing.
        Returns the entry if found, otherwise None.
        """
        return self.transposition_table.get(key)

    def update_position_history(self, key: int):
        """Update position history with the current position and the number of times it has occurred."""
        self.position_history[key] = self.position_history.get(key, 0) + 1

    def check_for_threefold_repetition(self, key: int):
        """
        Check for threefold repetition in the position history.
        If there is 3 or more of the same position (the key counts as a repetition), return True.
        """
        return self.position_history.get(key, 0) >= 3 # Check for threefold repetition

    def evaluate_position(self, chess_board: chess.Board, key: Optional[int] = None, has_legal_moves = False) -> float:
        """
        Evaluate the current position.
        Uses the evaluation cache if available.
        Positive values favor white, negative values favor black.
        """
        # Check evaluation cache
        cached_value = self.evaluation_cache.get(key)
        if cached_value is not None:
            return cached_value
        elif key in self.transposition_table: # If no cached value, check transposition table since we already checked this position
            return 0
    
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
        
        # Get game phase (0 to 256)
        phase = self.get_game_phase(chess_board)
        
        # Material evaluation
        score = self.evaluate_material(chess_board) # Material evaluation
        
        # score += self.evaluate_piece_position(chess_board) TODO
    
        # Phase-dependent evaluations with smooth interpolation
        # Middlegame and endgame evaluation weights
        mg_pawn_structure = self.evaluate_pawn_structure(chess_board)
        eg_pawn_structure = mg_pawn_structure // 2
        
        mg_king_safety = self.evaluate_king_safety(chess_board)
        eg_king_safety = mg_king_safety // 4
        
        # King centralization (only meaningful in endgame)
        mg_king_centralization = 0
        eg_king_centralization = self.evaluate_king_centralization(chess_board, phase)
        
        # Interpolate between middlegame and endgame values
        score += self.interpolate(mg_pawn_structure, eg_pawn_structure, phase)
        score += self.interpolate(mg_king_safety, eg_king_safety, phase)
        score += self.interpolate(mg_king_centralization, eg_king_centralization, phase)

        # score += self.evaluate_mobility(chess_board) # Mobility TODO

        # Cache the evaluation
        if key and score != 0: # Only store scores other than 0
            self.evaluation_cache[key] = score
       
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

    def evaluate_pawn_structure(self, chess_board: chess.Board):
        """
        Checks for common pawn weaknesses.
        Evaluates pawn chains and islands.
        Optimized version using bitboard operations.
        """
        score = 0

        file_masks = chess.BB_FILES

        # Process both colors at once with direct bitboard manipulation
        white_pawns = chess_board.pieces_mask(chess.PAWN, chess.WHITE)
        black_pawns = chess_board.pieces_mask(chess.PAWN, chess.BLACK)
        
        # Evaluate isolated pawns for both colors in one pass
        for file in range(8):
            # Pawns in this file
            white_pawns_in_file = chess.popcount(white_pawns & file_masks[file])
            black_pawns_in_file = chess.popcount(black_pawns & file_masks[file])
            
            # Calculate adjacent files mask once per iteration
            adjacent_mask = 0
            if file > 0:
                adjacent_mask |= file_masks[file - 1]
            if file < 7:
                adjacent_mask |= file_masks[file + 1]
            
            # Check for isolated pawns
            if white_pawns_in_file > 0 and chess.popcount(white_pawns & adjacent_mask) == 0:
                score -= 20  # Isolated white pawn penalty
            if black_pawns_in_file > 0 and chess.popcount(black_pawns & adjacent_mask) == 0:
                score += 20  # Isolated black pawn penalty
                
            # Check for doubled pawns
            if white_pawns_in_file > 1:
                score -= 10  # Doubled white pawn penalty
            if black_pawns_in_file > 1:
                score += 10  # Doubled black pawn penalty

        return score

    def evaluate_king_safety(self, chess_board: chess.Board):
        """
        Analyze pawn shield and open files in near king.
        Can be extended with attack pattern recognition.
        """
        score = 0
        
        # Precompute bitboards for pawns
        white_pawns = chess_board.pieces_mask(chess.PAWN, chess.WHITE)
        black_pawns = chess_board.pieces_mask(chess.PAWN, chess.BLACK)
        
        # Evaluate pawn shield for both kings
        for color in chess.COLORS:
            king_square = chess_board.king(color)
            if king_square is None:
                continue
            
            # Create king area mask - includes squares in front of the king
            king_file = chess.square_file(king_square)
            king_rank = chess.square_rank(king_square)
            
            # Direction for pawn shield (ahead of king based on color)
            shield_rank = king_rank + (1 if color else -1)
            
            # Only create shield mask if the rank is valid (not at edge of board)
            if 0 <= shield_rank < 8:
                shield_mask = 0
                
                # Add the three squares in front of the king to the shield mask
                for file_offset in [-1, 0, 1]:
                    if 0 <= king_file + file_offset < 8:  # Check if file is valid
                        shield_square = chess.square(king_file + file_offset, shield_rank)
                        shield_mask |= (1 << shield_square)
                
                # Count pawns in the shield area
                if color:  # White
                    shield_pawns_count = chess.popcount(white_pawns & shield_mask)
                    score += shield_pawns_count * 10
                else:      # Black
                    shield_pawns_count = chess.popcount(black_pawns & shield_mask)
                    score -= shield_pawns_count * 10
        
        return score

    def evaluate_king_centralization(self, chess_board: chess.Board, phase: int):
        """
        In endgames, kings should move to the center.
        Returns a score based on king proximity to center.
        Only meaningful in endgames, so weight by phase.
        """
        score = 0
        
        # Only apply when we're in endgame territory
        endgame_weight = (256 - phase) // 16
        if endgame_weight <= 0:
            return 0
            
        # Center distance calculations
        center_files = [3, 4]  # d and e files
        center_ranks = [3, 4]  # 4th and 5th ranks
        
        # White king centralization
        wk_square = chess_board.king(chess.WHITE)
        if wk_square is not None:
            wk_file = chess.square_file(wk_square)
            wk_rank = chess.square_rank(wk_square)
            wk_file_dist = min(abs(wk_file - f) for f in center_files)
            wk_rank_dist = min(abs(wk_rank - r) for r in center_ranks)
            wk_center_dist = wk_file_dist + wk_rank_dist
            # Maximum distance = 6, so 6 - dist = 0-6 points, weighted by endgame_weight
            score += (6 - wk_center_dist) * endgame_weight  
        
        # Black king centralization (subtract from score)
        bk_square = chess_board.king(chess.BLACK)
        if bk_square is not None:
            bk_file = chess.square_file(bk_square)
            bk_rank = chess.square_rank(bk_square)
            bk_file_dist = min(abs(bk_file - f) for f in center_files)
            bk_rank_dist = min(abs(bk_rank - r) for r in center_ranks)
            bk_center_dist = bk_file_dist + bk_rank_dist
            score -= (6 - bk_center_dist) * endgame_weight
        
        return score

    def get_game_phase(self, chess_board: chess.Board):
        """
        Returns a value between 0 (endgame) and 256 (opening)
        based on remaining material
        """
        # npm = 0  # Non-pawn material
        # for piece_type in [chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN]:
        #     npm += len(chess_board.pieces(piece_type, True)) * PIECE_VALUES[piece_type]
        #     npm += len(chess_board.pieces(piece_type, False)) * PIECE_VALUES[piece_type]
    
        # return min(npm, 256)

        npm = 0  # Non-pawn material
        
        # Use bitboards for faster counting
        wp = chess_board.occupied_co[chess.WHITE]
        bp = chess_board.occupied_co[chess.BLACK]
        
        # Knights (320 points each)
        npm += chess.popcount(wp & chess_board.knights) * PIECE_VALUES[chess.KNIGHT]
        npm += chess.popcount(bp & chess_board.knights) * PIECE_VALUES[chess.KNIGHT]
        
        # Bishops (330 points each)
        npm += chess.popcount(wp & chess_board.bishops) * PIECE_VALUES[chess.BISHOP]
        npm += chess.popcount(bp & chess_board.bishops) * PIECE_VALUES[chess.BISHOP]
        
        # Rooks (500 points each)
        npm += chess.popcount(wp & chess_board.rooks) * PIECE_VALUES[chess.ROOK]
        npm += chess.popcount(bp & chess_board.rooks) * PIECE_VALUES[chess.ROOK]
        
        # Queens (900 points each)
        npm += chess.popcount(wp & chess_board.queens) * PIECE_VALUES[chess.QUEEN]
        npm += chess.popcount(bp & chess_board.queens) * PIECE_VALUES[chess.QUEEN]
        
        return min(npm, 256)

    def interpolate(self, mg_score, eg_score, phase):
        """
        Interpolate between middlegame and endgame scores
        based on game phase
        """
        return ((mg_score * phase) + (eg_score * (256 - phase))) // 256

    def get_sorted_moves(self, chess_board: chess.Board) -> list:
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
        Returns a max heap of scored moves (-score, i, move).
        TODO Implement killer moves
        """
        # Instead of creating a new list for all moves, consider:
        # 1. Pre-allocate the array based on typical move count
        # 2. Use move ordering heuristics (killer moves, history heuristic)
        # 3. For shallow depths, consider simplified sorting
        moves_heap = []

        # Calculate once
        ep_square = chess_board.ep_square
        turn = chess_board.turn
        ep_pawn_delta = 8 if turn else -8

        for i, move in enumerate(chess_board.legal_moves): # ! REDUCE TIME
            score = 0

            # Capturing a piece bonus
            if chess_board.is_capture(move):
                victim = chess_board.piece_at(move.to_square)
                attacker = chess_board.piece_at(move.from_square)

                # Handle en passant captures
                if victim is None and move.to_square == ep_square:
                    victim = chess_board.piece_at(move.to_square + ep_pawn_delta)

                if victim and attacker:
                    # Prioritize capturing higher value pieces using lower value pieces
                    score += 10 * PIECE_VALUES[victim.piece_type] - PIECE_VALUES[attacker.piece_type]

            # Promotion bonus
            if move.promotion:
                score += 9_000

            # Center control bonus
            if move.to_square in CENTER_SQUARES:
                score += 100

            # Negative score for max-heap
            moves_heap.append((-score, i, move)) # Add i for comparision if scores are equal

        # Heapify to sort by score once
        heapq.heapify(moves_heap) # Turn into max heap
        return moves_heap


    def has_sufficient_material(self, chess_board: chess.Board, side: bool):
        """
        Check if a side has sufficient material to avoid zugzwang positions.
        Returns True if the side has any piece other than king and pawns.
        """
        pieces = chess_board.occupied_co[side]
        non_pawn_pieces = pieces & ~chess_board.pawns & ~chess_board.kings
        return chess.popcount(non_pawn_pieces) > 0

    def minimax_alpha_beta(self, board: ChessBoard, remaining_depth: int, alpha: int, beta: int, maximizing_player: bool, allow_null_move: bool = True):
        """
        Minimax algorithm with alpha-beta pruning.
        Returns (best_value, best_move) tuple.
        """
        chess_board = board.get_board_state()
        key = chess.polyglot.zobrist_hash(chess_board) # Calculate once and reuse through code
        
        if remaining_depth <= 0: # Leaf node
            return self.evaluate_position(chess_board, key), None
        elif not chess_board.legal_moves: # No legal moves
            return self.evaluate_position(chess_board, key, has_legal_moves=False), None

        # if remaining_depth <= 0: # TODO Implement quiescence search
            # return self.quiescence_search(board, alpha, beta), None

        # Check for threefold repetition
        if self.check_for_threefold_repetition(key):
            return 0, None
        
        # Lookup position in transposition table (skip if threefold repetition and continue evaluation)
        transposition = self.lookup_position(chess_board, key)
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

        # Null-Move Pruning
        # Only try null move if:
        # 1. We're not in check (can't skip a turn when in check)
        # 2. We have sufficient material (avoiding zugzwang positions in endgames)
        # 3. We have depth > 2 (avoid incorrect evaluations near leaf nodes)
        # 4. allow_null_move flag is True (to prevent consecutive null moves)
        R = 2  # Null-move reduction factor (can be adjusted)
        if (allow_null_move and 
            remaining_depth >= 3 and 
            not chess_board.is_check() and
            self.has_sufficient_material(chess_board, chess_board.turn)):
            
            # Skip turn and do a reduced depth search
            board.make_null_move()
            null_value = -self.minimax_alpha_beta(
                board, remaining_depth - 1 - R, -beta, -beta + 1, 
                not maximizing_player, False)[0]
            board.undo_null_move()
            
            # If the reduced depth search exceeds beta, we can prune this branch
            if null_value >= beta:
                return null_value, None

        best_move = None
        original_alpha = alpha

        moves_heap = self.get_sorted_moves(chess_board)

        if maximizing_player:
            best_value = float('-inf')
            while moves_heap:
                move = heapq.heappop(moves_heap)[2] # Get move from heap
                self.moves_checked += 1
                if CHECKING_MOVE_ARROW and remaining_depth == DEPTH: # Display the root move
                    self.game.checking_move = move
                    self.game.display_board(self.game.last_move)  # Update display

                # Make a move and evaluate the position
                board.make_move(move) # ! VERY SLOW
                value = self.minimax_alpha_beta(board, remaining_depth - 1, alpha, beta, False)[0]
                board.undo_move()

                if value > best_value:
                    best_value = value
                    best_move = move
                alpha = max(alpha, best_value)
                if beta <= alpha: # Big performance improvement
                    break # Black's best response is worse than White's guarenteed value

        else: # Minimizing player
            best_value = float('inf')
            while moves_heap:
                move = heapq.heappop(moves_heap)[2] # Get move from heap
                self.moves_checked += 1
                if CHECKING_MOVE_ARROW and remaining_depth == DEPTH: # Display the root move
                    self.game.checking_move = move
                    self.game.display_board(self.game.last_move)  # Update display

                # Make a move and evaluate the position
                board.make_move(move) # ! VERY SLOW
                value = self.minimax_alpha_beta(board, remaining_depth - 1, alpha, beta, True)[0]
                board.undo_move()

                if value < best_value:
                    best_value = value
                    best_move = move
                beta = min(beta, best_value)
                if beta <= alpha: # Big performance improvement
                    break # White guarenteed value is better than Black's best option

        # Store position in transposition table
        if best_value <= original_alpha:
            flag = 'UPPERBOUND'
        if best_value >= beta:
            flag = 'LOWERBOUND'
        else:
            flag = 'EXACT'
        
        # Store in transposition table with previously calculated key
        self.store_position(chess_board, remaining_depth, best_value, flag, best_move, key)

        return best_value, best_move


    def get_move(self, board):
        """
        Main method to select the best move.
        """        
        self.moves_checked = 0

        # Update position history with other player's move
        self.update_position_history(chess.polyglot.zobrist_hash(board.get_board_state())) # Update position history

        # Define the code to time
        def timed_minimax():# -> tuple[float, None] | tuple[Literal[-1000, 1000], None] | tuple[float, Move | None]:
            return self.minimax_alpha_beta(board, DEPTH, float('-inf'), float('inf'), board.get_board_state().turn)

        # Run minimax with timing
        number = 1  # Number of executions
        time_taken = timeit.timeit(timed_minimax, number=number) / number
        best_value, best_move = timed_minimax()  # Run once more to get the actual result

        # Update position history with the real move made
        self.update_position_history(chess.polyglot.zobrist_hash(board.get_board_state())) # Update position history

        # Moves checked over time taken
        time_per_move = time_taken / self.moves_checked if self.moves_checked > 0 else 0
        print(f"Moves/Time: {colors.BOLD}{colors.get_moves_color(self.moves_checked)}{self.moves_checked:,}{colors.RESET} / "
            f"{colors.BOLD}{colors.get_move_time_color(time_taken)}{time_taken:.2f}{colors.RESET} s = "
            f"{colors.BOLD}{colors.CYAN}{time_per_move * 1000:.4f}{colors.RESET} ms/M")

        # Calculate memory usage more accurately
        tt_entry_size = sys.getsizeof(TranspositionEntry(0, 0.0, "", None))
        tt_size_mb = len(self.transposition_table) * tt_entry_size / (1024 * 1024)
        eval_size_mb = sum(sys.getsizeof(k) + sys.getsizeof(v) for k, v in list(self.evaluation_cache.items())[:10]) / 10
        eval_size_mb = eval_size_mb * len(self.evaluation_cache) / (1024 * 1024)
        transposition_table_entries = len(self.transposition_table)

        # Print cache statistics
        print(f"Transposition table: {colors.BOLD}{colors.MAGENTA}{len(self.transposition_table):,}{colors.RESET} entries, "
            f"{colors.BOLD}{colors.CYAN}{tt_size_mb:.4f}{colors.RESET} MB")
        print(f"Evaluation cache: {colors.BOLD}{colors.MAGENTA}{len(self.evaluation_cache):,}{colors.RESET} entries, "
              f"{colors.BOLD}{colors.CYAN}{eval_size_mb:.4f}{colors.RESET} MB")

        # Print the FEN
        print(f"FEN: {board.get_board_state().fen()}")

        return best_move

# How to improve speed and efficiency more?
