import chess
import chess.svg
from board import ChessBoard
from bot import ChessBot
from human import HumanPlayer
import pygame
import cairosvg
import io
from PIL import Image

from constants import IS_BOT, UPDATE_DELAY_MS, LAST_MOVE_ARROW, CHECKING_MOVE_ARROW

class ChessGame:
    def __init__(self):
        self.board = ChessBoard()
        self.update_delay = UPDATE_DELAY_MS  # Millisecond delay between visual updates 

        self.checking_move = None # Currently checked move
        self.last_move = None     # Last move played
        
        # Initialize players based on IS_BOT flag
        if IS_BOT:
            self.white_player = ChessBot(self)
            self.black_player = ChessBot(self)
        else:
            self.white_player = HumanPlayer(chess.WHITE, self)
            self.black_player = ChessBot(self)
        
        # Initialize Pygame
        pygame.init()
        self.WINDOW_SIZE = 600
        self.screen = pygame.display.set_mode((self.WINDOW_SIZE, self.WINDOW_SIZE))
        pygame.display.set_caption("Chess Game")
        
    def svg_to_pygame_surface(self, svg_string):
        """Convert SVG string to Pygame surface"""        
        # Convert SVG to surface
        png_data = cairosvg.svg2png(bytestring=svg_string.encode('utf-8'))
        image = Image.open(io.BytesIO(png_data))
        image = image.resize((self.WINDOW_SIZE, self.WINDOW_SIZE))
        mode = image.mode
        size = image.size
        data = image.tobytes()

        return pygame.image.fromstring(data, size, mode)

    def display_board(self, last_move=None, selected_square=None, force_update=False):
        """Display the current board state"""
        current_time = pygame.time.get_ticks()
        # Skip update if too soon (unless forced)
        if not force_update and hasattr(self, 'last_update_time'):
            if current_time - self.last_update_time < self.update_delay:
                return

        # Build highlight dictionary for the selected square
        highlight_squares = None
        if selected_square is not None:
            highlight_squares = {
                selected_square: {"fill": "#FFFF00", "stroke": "none"}
            }

        arrows = []
        if (LAST_MOVE_ARROW and last_move): # Last move arrow
            # Create arrow for the last move
            arrows.append(chess.svg.Arrow(
                last_move.from_square,
                last_move.to_square,
                color="#0000FF80"  # Blue color with 50% transparency
            ))
        if (not force_update and CHECKING_MOVE_ARROW and self.checking_move): # Bot checking move arrow
            arrows.append(chess.svg.Arrow(
                self.checking_move.from_square,
                self.checking_move.to_square,
                color="#FF000080"  # Red for checked move
            ))

        # Create SVG with highlighted last move and selected square
        svg = chess.svg.board(
            board=self.board.get_board_state(),
            lastmove=last_move,
            squares=highlight_squares,     # colored square highlight
            arrows=arrows,                 # arrow for last move
            size=self.WINDOW_SIZE
        )
        
        # Convert SVG to Pygame surface and display
        py_image = self.svg_to_pygame_surface(svg)
        self.screen.blit(py_image, (0, 0))
        pygame.display.flip()
        self.last_update_time = current_time

    def play_game(self):
        """Main game loop"""
        print("-------------------")
        
        while not self.board.is_game_over():
            print(f"Player: {'White' if self.board.get_board_state().turn else 'Black'} - {self.board.get_board_state().fullmove_number}")

            # Get current player for selected square highlighting
            current_player = self.white_player if self.board.get_board_state().turn else self.black_player
            selected_square = getattr(current_player, 'selected_square', None)
            
            # Display current board with highlights
            self.display_board(self.last_move, selected_square, force_update=True)
            
            # Determine current player
            current_player = self.white_player if self.board.get_board_state().turn else self.black_player
            
            # Get player's move
            move = current_player.get_move(self.board)
            
            if move is None:
                print("Game ended by player")
                break
                
            # Make the move
            if not self.board.make_move(move):
                print(f"Illegal move attempted: {move}")
                break
                
            print(f"Eval: {self.black_player.evaluate_position(self.board)}")
            # print(f"Move played: {move}")
            print("-------------------")
            self.last_move = move            
            
        # Display final position
        self.display_board(self.last_move, force_update=True)
        print(f"Number of turns: {self.board.get_board_state().fullmove_number}") # Print number of turns
        result = self.board.get_result()
        print(f"Game Over! Result: {result}")
        
        # Keep window open until closed
        while True:
            event = pygame.event.wait()
            if event.type == pygame.QUIT:
                break
        
        pygame.quit()

if __name__ == "__main__":
    game = ChessGame()
    game.play_game()
