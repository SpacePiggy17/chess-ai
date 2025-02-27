# Log colors
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
CYAN = "\033[96m"
MAGENTA = "\033[95m"
RESET = "\033[0m"
BOLD = "\033[1m"

def get_moves_color(moves: int):
    """Returns color based on number of moves."""
    if moves < 20_000:
        return GREEN
    elif moves < 50_000:
        return YELLOW
    return RED

def get_move_time_color(move_time):
    """Returns color based on move time."""
    if move_time < 6:
        return GREEN
    elif move_time < 10:
        return YELLOW
    return RED
