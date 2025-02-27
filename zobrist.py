import random
import chess

# Initialize random numbers for pieces on squares
PIECE_SQUARE = {}
for piece in chess.PIECE_TYPES:
    for color in [True, False]:  # True = White, False = Black
        for square in chess.SQUARES:
            PIECE_SQUARE[(piece, color, square)] = random.getrandbits(64)

# Random numbers for castling rights - using correct chess module constants
CASTLING = {
    'K': random.getrandbits(64),  # White kingside
    'Q': random.getrandbits(64),  # White queenside
    'k': random.getrandbits(64),  # Black kingside
    'q': random.getrandbits(64),  # Black queenside
}

# Random numbers for en passant files
EN_PASSANT = {file: random.getrandbits(64) for file in range(8)}

# Random number for side to move
SIDE_TO_MOVE = random.getrandbits(64)

def calculate_hash(board: chess.Board) -> int:
    """Calculate the Zobrist hash for a given board position"""
    h = 0
    
    # Hash pieces
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            h ^= PIECE_SQUARE[(piece.piece_type, piece.color, square)]
    
    # Hash castling rights using correct chess module methods
    if board.has_kingside_castling_rights(chess.WHITE):
        h ^= CASTLING['K']
    if board.has_queenside_castling_rights(chess.WHITE):
        h ^= CASTLING['Q']
    if board.has_kingside_castling_rights(chess.BLACK):
        h ^= CASTLING['k']
    if board.has_queenside_castling_rights(chess.BLACK):
        h ^= CASTLING['q']
    
    # Hash en passant
    if board.ep_square:
        h ^= EN_PASSANT[chess.square_file(board.ep_square)]
    
    # Hash side to move
    if board.turn:
        h ^= SIDE_TO_MOVE
        
    return h
