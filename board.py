import chess

from constants import STARTING_FEN

class ChessBoard:
    def __init__(self):
        if STARTING_FEN:
            self.board = chess.Board(STARTING_FEN)
        else:
            self.board = chess.Board()
    
    def get_legal_moves(self):
        """Returns a list of legal moves in the current position."""
        return list(self.board.legal_moves)
    
    def make_move(self, move):
        """
        Attempts to make a move on the board.
        Returns True if successful, False if illegal.
        """
        try:
            if move in self.board.legal_moves:
                self.board.push(move)
                return True
            return False
        except:
            return False
        
    def undo_move(self):
        """Undoes the last move made on the board."""
        # Pop if board is not empty
        if self.board.move_stack:
            self.board.pop()
    
    def is_game_over(self):
        """Returns True if the game is over."""
        return self.board.is_game_over()
    
    def get_board_state(self):
        """Returns the current board state."""
        return self.board

    def get_result(self):
        """Returns the game result if the game is over."""
        if self.is_game_over():
            return self.board.outcome()
        return None
