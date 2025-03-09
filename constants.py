import chess
import numpy as np  # For piece square tables
from typing_extensions import TypeAlias  # For GameStage

# Set to None for standard starting position, or FEN string for custom starting position
STARTING_FEN: str = None
# STARTING_FEN: str = "2kr1r2/p1p2p1p/4N1p1/1p2B3/5R2/1B4P1/P1P3KP/2q5 b - - 0 26" # Checkmate soon

# STARTING_FEN: str = "r1b1k2r/pppp1p1p/4p1pB/4P3/3q4/6P1/PPP2K1P/RN3BNR w kq - 0 13"
# STARTING_FEN: str = "r1b1k2r/pppp1p1p/4p1pB/4P3/8/6P1/PqP3KP/RN3BNR w kq - 0 14" # Big jump in evaluation

MAX_VALUE: np.int16 = 32767 # 2**(16-1) - 1 (max value for 16 bit integer)
MIN_VALUE: np.int16 = -32768 # -2**(16-1) (min value for 16 bit integer)

# CENTER_SQUARES = {chess.D4, chess.D5, chess.E4, chess.E5}  # Chess already has this built in

# Game settings
IS_BOT: bool = True  # Set to False for human vs bot, True for bot vs bot
LAST_MOVE_ARROW: bool = True  # Set to True to display last move arrow
TT_SIZE: np.int8 = 64 # Size of the transposition table (in MB)

# Debug settings
CHECKING_MOVE_ARROW: bool = False  # Set to True to display checking move arrow (switches the mode to svg rendering)
UPDATE_DELAY_MS: np.int8 = 30  # Delay between visual updates in milliseconds
RENDER_DEPTH: np.int8 = 5 # Depth to render checking moves (set to DEPTH to render root moves)

BREAK_TURN: np.int8 = None # Number of turns to break after (for debugging)
# BREAK_TURN: np.int8 = 5 # Number of turns to break after (for debugging)

# Search settings
DEPTH: np.int8 = 5  # Search depth for the minimax algorithm
# RENDER_DEPTH: np.int8 = 0 # Depth to render checking moves (set to DEPTH to render root moves)


'''
Board and piece values
'''
PIECE_VALUES: dict[int, int] = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 20000
}
PIECE_VALUES_STOCKFISH: dict[int, int] = {
    chess.PAWN: 208,
    chess.KNIGHT: 781,
    chess.BISHOP: 825,
    chess.ROOK: 1_276,
    chess.QUEEN: 2_538,
    chess.KING: 32_000
} # TODO MG: 198, 817, 836, 1_270, 2_521, EG: 258, 846, 857, 1_278, 2_558

BISHOP_PAIR_BONUS: int = PIECE_VALUES_STOCKFISH[chess.PAWN] >> 1 # Half the value of a pawn

# Total npm at start (16604 with stockfish values)
START_NPM: np.int16 = PIECE_VALUES_STOCKFISH[chess.KNIGHT] * 4 + \
    PIECE_VALUES_STOCKFISH[chess.BISHOP] * 4 + \
    PIECE_VALUES_STOCKFISH[chess.ROOK] * 4 + \
    PIECE_VALUES_STOCKFISH[chess.QUEEN] * 2
# NPM scalar for evaluation (65 with stockfish values)
NPM_SCALAR: np.int8 = (START_NPM // 256) + 1

# Game stages
GameStage: TypeAlias = np.int8
MIDGAME: GameStage = 0
ENDGAME: GameStage = 1

# TODO Autotune these values
# Piece square tables from Rofchade
mg_pawn_table = np.array([
      0,   0,   0,   0,   0,   0,  0,   0,
     98, 134,  61,  95,  68, 126, 34, -11,
     -6,   7,  26,  31,  65,  56, 25, -20,
    -14,  13,   6,  21,  23,  12, 17, -23,
    -27,  -2,  -5,  12,  17,   6, 10, -25,
    -26,  -4,  -4, -10,   3,   3, 33, -12,
    -35,  -1, -20, -23, -15,  24, 38, -22,
      0,   0,   0,   0,   0,   0,  0,   0,
], dtype=np.int32)

eg_pawn_table = np.array([
      0,   0,   0,   0,   0,   0,   0,   0,
    178, 173, 158, 134, 147, 132, 165, 187,
     94, 100,  85,  67,  56,  53,  82,  84,
     32,  24,  13,   5,  -2,   4,  17,  17,
     13,   9,  -3,  -7,  -7,  -8,   3,  -1,
      4,   7,  -6,   1,   0,  -5,  -1,  -8,
     13,   8,   8,  10,  13,   0,   2,  -7,
      0,   0,   0,   0,   0,   0,   0,   0,
], dtype=np.int32)

mg_knight_table = np.array([
    -167, -89, -34, -49,  61, -97, -15, -107,
     -73, -41,  72,  36,  23,  62,   7,  -17,
     -47,  60,  37,  65,  84, 129,  73,   44,
      -9,  17,  19,  53,  37,  69,  18,   22,
     -13,   4,  16,  13,  28,  19,  21,   -8,
     -23,  -9,  12,  10,  19,  17,  25,  -16,
     -29, -53, -12,  -3,  -1,  18, -14,  -19,
    -105, -21, -58, -33, -17, -28, -19,  -23,
], dtype=np.int32)

eg_knight_table = np.array([
    -58, -38, -13, -28, -31, -27, -63, -99,
    -25,  -8, -25,  -2,  -9, -25, -24, -52,
    -24, -20,  10,   9,  -1,  -9, -19, -41,
    -17,   3,  22,  22,  22,  11,   8, -18,
    -18,  -6,  16,  25,  16,  17,   4, -18,
    -23,  -3,  -1,  15,  10,  -3, -20, -22,
    -42, -20, -10,  -5,  -2, -20, -23, -44,
    -29, -51, -23, -15, -22, -18, -50, -64,
], dtype=np.int32)

mg_bishop_table = np.array([
    -29,   4, -82, -37, -25, -42,   7,  -8,
    -26,  16, -18, -13,  30,  59,  18, -47,
    -16,  37,  43,  40,  35,  50,  37,  -2,
     -4,   5,  19,  50,  37,  37,   7,  -2,
     -6,  13,  13,  26,  34,  12,  10,   4,
      0,  15,  15,  15,  14,  27,  18,  10,
      4,  15,  16,   0,   7,  21,  33,   1,
    -33,  -3, -14, -21, -13, -12, -39, -21,
], dtype=np.int32)

eg_bishop_table = np.array([
    -14, -21, -11,  -8, -7,  -9, -17, -24,
     -8,  -4,   7, -12, -3, -13,  -4, -14,
      2,  -8,   0,  -1, -2,   6,   0,   4,
     -3,   9,  12,   9, 14,  10,   3,   2,
     -6,   3,  13,  19,  7,  10,  -3,  -9,
    -12,  -3,   8,  10, 13,   3,  -7, -15,
    -14, -18,  -7,  -1,  4,  -9, -15, -27,
    -23,  -9, -23,  -5, -9, -16,  -5, -17,
], dtype=np.int32)

mg_rook_table = np.array([
     32,  42,  32,  51, 63,  9,  31,  43,
     27,  32,  58,  62, 80, 67,  26,  44,
     -5,  19,  26,  36, 17, 45,  61,  16,
    -24, -11,   7,  26, 24, 35,  -8, -20,
    -36, -26, -12,  -1,  9, -7,   6, -23,
    -45, -25, -16, -17,  3,  0,  -5, -33,
    -44, -16, -20,  -9, -1, 11,  -6, -71,
    -19, -13,   1,  17, 16,  7, -37, -26,
], dtype=np.int32)

eg_rook_table = np.array([
    13, 10, 18, 15, 12,  12,   8,   5,
    11, 13, 13, 11, -3,   3,   8,   3,
     7,  7,  7,  5,  4,  -3,  -5,  -3,
     4,  3, 13,  1,  2,   1,  -1,   2,
     3,  5,  8,  4, -5,  -6,  -8, -11,
    -4,  0, -5, -1, -7, -12,  -8, -16,
    -6, -6,  0,  2, -9,  -9, -11,  -3,
    -9,  2,  3, -1, -5, -13,   4, -20,
], dtype=np.int32)

mg_queen_table = np.array([
    -28,   0,  29,  12,  59,  44,  43,  45,
    -24, -39,  -5,   1, -16,  57,  28,  54,
    -13, -17,   7,   8,  29,  56,  47,  57,
    -27, -27, -16, -16,  -1,  17,  -2,   1,
     -9, -26,  -9, -10,  -2,  -4,   3,  -3,
    -14,   2, -11,  -2,  -5,   2,  14,   5,
    -35,  -8,  11,   2,   8,  15,  -3,   1,
     -1, -18,  -9,  10, -15, -25, -31, -50,
], dtype=np.int32)

eg_queen_table = np.array([
     -9,  22,  22,  27,  27,  19,  10,  20,
    -17,  20,  32,  41,  58,  25,  30,   0,
    -20,   6,   9,  49,  47,  35,  19,   9,
      3,  22,  24,  45,  57,  40,  57,  36,
    -18,  28,  19,  47,  31,  34,  39,  23,
    -16, -27,  15,   6,   9,  17,  10,   5,
    -22, -23, -30, -16, -16, -23, -36, -32,
    -33, -28, -22, -43,  -5, -32, -20, -41,
], dtype=np.int32)

mg_king_table = np.array([
    -65,  23,  16, -15, -56, -34,   2,  13,
     29,  -1, -20,  -7,  -8,  -4, -38, -29,
     -9,  24,   2, -16, -20,   6,  22, -22,
    -17, -20, -12, -27, -30, -25, -14, -36,
    -49,  -1, -27, -39, -46, -44, -33, -51,
    -14, -14, -22, -46, -44, -30, -15, -27,
      1,   7,  -8, -64, -43, -16,   9,   8,
    -15,  36,  12, -54,   8, -28,  24,  14,
], dtype=np.int32)

eg_king_table = np.array([
    -74, -35, -18, -18, -11,  15,   4, -17,
    -12,  17,  14,  17,  17,  38,  23,  11,
     10,  17,  23,  15,  20,  45,  44,  13,
     -8,  22,  24,  27,  26,  33,  26,   3,
    -18,  -4,  21,  24,  27,  23,   9, -11,
    -19,  -3,  11,  21,  23,  16,   7,  -9,
    -27, -11,   4,  13,  14,   4,  -5, -17,
    -53, -34, -21, -11, -28, -14, -24, -43
], dtype=np.int32)

# Flip piece square tables for white (PSQT[game_stage][piece_type][FLIP[square]])
FLIP = np.array([
    56, 57, 58, 59, 60, 61, 62, 63,
    48, 49, 50, 51, 52, 53, 54, 55,
    40, 41, 42, 43, 44, 45, 46, 47,
    32, 33, 34, 35, 36, 37, 38, 39,
    24, 25, 26, 27, 28, 29, 30, 31,
    16, 17, 18, 19, 20, 21, 22, 23,
     8,  9, 10, 11, 12, 13, 14, 15,
     0,  1,  2,  3,  4,  5,  6,  7,
], dtype=np.int32)

PSQT: list[list[chess.PieceType, np.ndarray]] = [ # Using a list instead of a dict for less overhead (PSQT[MIDGAME/ENDGAME][piece_type][square])
    [
        None, # Python chess starts piece indexing at 1, so we add empty slots
        mg_pawn_table,
        mg_knight_table,
        mg_bishop_table,
        mg_rook_table,
        mg_queen_table,
        mg_king_table
    ],
    [
        None,
        eg_pawn_table,
        eg_knight_table,
        eg_bishop_table,
        eg_rook_table,
        eg_queen_table,
        eg_king_table
    ]
]

# Use a dict for faster lookup of castling updates
CASTLING_UPDATES: dict[tuple[int, int], tuple[int, int, bool]] = {
    # (from_square, to_square, color): (rook_from, rook_to)
    (chess.E1, chess.G1, chess.WHITE): (chess.H1, chess.F1), # White kingside
    (chess.E1, chess.C1, chess.WHITE): (chess.A1, chess.D1), # White queenside
    (chess.E8, chess.G8, chess.BLACK): (chess.H8, chess.F8), # Black kingside
    (chess.E8, chess.C8, chess.BLACK): (chess.A8, chess.D8), # Black queenside
}
