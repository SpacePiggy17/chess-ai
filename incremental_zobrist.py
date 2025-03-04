import chess
from chess.polyglot import POLYGLOT_RANDOM_ARRAY, zobrist_hash

class IncrementalZobrist:
    def __init__(self, chess_board: chess.Board):
        """Initialize hash with a board."""
        self.hash_value = zobrist_hash(chess_board)
        
    def update_for_move(self, chess_board: chess.Board, move: chess.Move):
        """Update hash incrementally based on the move."""
        if not move:  # Handle null moves
            self.hash_value ^= POLYGLOT_RANDOM_ARRAY[780]  # Just toggle side to move
            return
            
        # Get piece information
        from_square = move.from_square
        to_square = move.to_square
        piece = chess_board.piece_at(from_square)
        piece_type = piece.piece_type
        piece_color = piece.color
        
        # Handle en passant square changes - must be done before moving pieces
        old_ep_square = chess_board.ep_square
        if old_ep_square is not None:
            # Only hash en passant if a pawn can capture
            if chess_board.turn == chess.WHITE:
                ep_mask = chess.shift_down(chess.BB_SQUARES[old_ep_square])
            else:
                ep_mask = chess.shift_up(chess.BB_SQUARES[old_ep_square])
            ep_mask = chess.shift_left(ep_mask) | chess.shift_right(ep_mask)
            
            if ep_mask & chess_board.pawns & chess_board.occupied_co[chess_board.turn]:
                self.hash_value ^= POLYGLOT_RANDOM_ARRAY[772 + chess.square_file(old_ep_square)]
        
        # Handle castling rights changes
        old_castling = chess_board.castling_rights
        new_castling = old_castling
        
        # If king or rook moves from its original square, remove castling rights
        if piece_type == chess.KING:
            if piece_color == chess.WHITE and from_square == chess.E1:
                new_castling &= ~chess.BB_H1  # Remove white kingside
                new_castling &= ~chess.BB_A1  # Remove white queenside
            elif piece_color == chess.BLACK and from_square == chess.E8:
                new_castling &= ~chess.BB_H8  # Remove black kingside
                new_castling &= ~chess.BB_A8  # Remove black queenside
        elif piece_type == chess.ROOK:
            if from_square == chess.H1 and piece_color == chess.WHITE:
                new_castling &= ~chess.BB_H1  # Remove white kingside
            elif from_square == chess.A1 and piece_color == chess.WHITE:
                new_castling &= ~chess.BB_A1  # Remove white queenside
            elif from_square == chess.H8 and piece_color == chess.BLACK:
                new_castling &= ~chess.BB_H8  # Remove black kingside
            elif from_square == chess.A8 and piece_color == chess.BLACK:
                new_castling &= ~chess.BB_A8  # Remove black queenside
        
        # If a rook is captured on its original square, remove castling rights
        captured_piece = chess_board.piece_at(to_square)
        if captured_piece and captured_piece.piece_type == chess.ROOK:
            if to_square == chess.H1:
                new_castling &= ~chess.BB_H1  # Remove white kingside
            elif to_square == chess.A1:
                new_castling &= ~chess.BB_A1  # Remove white queenside
            elif to_square == chess.H8:
                new_castling &= ~chess.BB_H8  # Remove black kingside
            elif to_square == chess.A8:
                new_castling &= ~chess.BB_A8  # Remove black queenside
        
        # Update castling rights hash
        if old_castling != new_castling:
            # XOR out the old castling rights
            if old_castling & chess.BB_H1:
                self.hash_value ^= POLYGLOT_RANDOM_ARRAY[768]      # White kingside
            if old_castling & chess.BB_A1:
                self.hash_value ^= POLYGLOT_RANDOM_ARRAY[768 + 1]  # White queenside
            if old_castling & chess.BB_H8:
                self.hash_value ^= POLYGLOT_RANDOM_ARRAY[768 + 2]  # Black kingside
            if old_castling & chess.BB_A8:
                self.hash_value ^= POLYGLOT_RANDOM_ARRAY[768 + 3]  # Black queenside
            
            # XOR in the new castling rights
            if new_castling & chess.BB_H1:
                self.hash_value ^= POLYGLOT_RANDOM_ARRAY[768]      # White kingside
            if new_castling & chess.BB_A1:
                self.hash_value ^= POLYGLOT_RANDOM_ARRAY[768 + 1]  # White queenside
            if new_castling & chess.BB_H8:
                self.hash_value ^= POLYGLOT_RANDOM_ARRAY[768 + 2]  # Black kingside
            if new_castling & chess.BB_A8:
                self.hash_value ^= POLYGLOT_RANDOM_ARRAY[768 + 3]  # Black queenside
        
        # Remove piece from source square
        # Match python-chess's indexing: (piece_type-1)*2 + color
        piece_index = (piece_type - 1) * 2 + (1 if piece_color else 0)
        self.hash_value ^= POLYGLOT_RANDOM_ARRAY[64 * piece_index + from_square]
        
        # Handle captures
        if captured_piece:
            captured_index = (captured_piece.piece_type - 1) * 2 + (1 if captured_piece.color else 0)
            self.hash_value ^= POLYGLOT_RANDOM_ARRAY[64 * captured_index + to_square]
        
        # Handle en passant captures
        if piece_type == chess.PAWN and to_square == chess_board.ep_square:
            # Calculate the square of the captured pawn
            if piece_color == chess.WHITE:
                ep_capture_square = to_square - 8
            else:
                ep_capture_square = to_square + 8
                
            # Remove the captured pawn from the hash
            # Opposite color from the moving piece
            captured_pawn_index = 0 + (1 if not piece_color else 0)  # PAWN=0, color is opposite
            self.hash_value ^= POLYGLOT_RANDOM_ARRAY[64 * captured_pawn_index + ep_capture_square]
        
        # Add moved piece to destination square
        if move.promotion:
            # Add the promotion piece instead of the pawn
            promotion_index = (move.promotion - 1) * 2 + (1 if piece_color else 0)
            self.hash_value ^= POLYGLOT_RANDOM_ARRAY[64 * promotion_index + to_square]
        else:
            # Add the original piece
            self.hash_value ^= POLYGLOT_RANDOM_ARRAY[64 * piece_index + to_square]
        
        # Handle castling (rook movement)
        if piece_type == chess.KING and abs(from_square - to_square) == 2:
            # Determine rook squares using proper chess square constants
            if to_square > from_square:  # Kingside
                if piece_color == chess.WHITE:
                    rook_from = chess.H1
                    rook_to = chess.F1
                else:
                    rook_from = chess.H8
                    rook_to = chess.F8
            else:  # Queenside
                if piece_color == chess.WHITE:
                    rook_from = chess.A1
                    rook_to = chess.D1
                else:
                    rook_from = chess.A8
                    rook_to = chess.D8
            
            # Update hash for rook movement
            rook_index = (chess.ROOK - 1) * 2 + (1 if piece_color else 0)
            self.hash_value ^= POLYGLOT_RANDOM_ARRAY[64 * rook_index + rook_from]
            self.hash_value ^= POLYGLOT_RANDOM_ARRAY[64 * rook_index + rook_to]
        
        # Calculate new en passant square
        new_ep_square = None
        if piece_type == chess.PAWN and abs(from_square - to_square) == 16:
            new_ep_square = (from_square + to_square) // 2
            
            # Only hash if a pawn can actually capture
            # Check for opponent pawns on adjacent files that could capture
            if chess_board.turn == chess.BLACK:  # White just moved, black to move
                ep_mask = chess.shift_down(chess.BB_SQUARES[new_ep_square])
            else:  # Black just moved, white to move
                ep_mask = chess.shift_up(chess.BB_SQUARES[new_ep_square])
                
            ep_mask = chess.shift_left(ep_mask) | chess.shift_right(ep_mask)
            
            if ep_mask & chess_board.pawns & chess_board.occupied_co[not piece_color]:
                self.hash_value ^= POLYGLOT_RANDOM_ARRAY[772 + chess.square_file(new_ep_square)]
        
        # Toggle side to move
        self.hash_value ^= POLYGLOT_RANDOM_ARRAY[780]
    
    def get_hash(self):
        """Return the current hash value."""
        return self.hash_value
