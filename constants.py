import chess

# Board and piece settings
PIECE_VALUES = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 20000
}

CENTER_SQUARES = {chess.D4, chess.D5, chess.E4, chess.E5}

# Game settings
IS_BOT = True  # Set to False for human vs bot, True for bot vs bot
UPDATE_DELAY_MS = 160  # Delay between visual updates in milliseconds
LAST_MOVE_ARROW = True  # Set to True to display last move arrow
CHECKING_MOVE_ARROW = True  # Set to True to display checking move arrow

# Search settings
DEPTH = 5 # Search depth for the minimax algorithm
