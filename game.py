import chess
import chess.svg
from board import ChessBoard
from bot2 import ChessBot
from human import HumanPlayer
import pygame
import cairosvg
import io
from PIL import Image

from constants import IS_BOT, UPDATE_DELAY_MS, LAST_MOVE_ARROW, CHECKING_MOVE_ARROW

class ChessGame:
    def __init__(self):
        self.board = ChessBoard()

        self.checking_move = None # Currently checked move
        self.last_move = None     # Last move played
        
        # Initialize players based on IS_BOT flag
        if IS_BOT:
            self.white_player = ChessBot(self)
            self.black_player = ChessBot(self)
        else:
            self.white_player = HumanPlayer(chess.WHITE, self)
            self.black_player = ChessBot(self)

        # Cache for piece images and board squares
        self.piece_images = {}
        self.square_colors = {
            'light': pygame.Color('#FFFFFF'),  # White squares
            'dark': pygame.Color('#769656')    # Green squares
        }
        self.highlighted_square_color = pygame.Color(255, 255, 0, 128)  # Semi-transparent yellow
        
        # Initialize Pygame
        pygame.init()
        self.WINDOW_SIZE = 600
        self.screen = pygame.display.set_mode((self.WINDOW_SIZE, self.WINDOW_SIZE))
        pygame.display.set_caption("Chess Game")

        # Pre-render all piece images
        self.prerender_pieces()
        
        # Initialize last board state for optimized rendering
        self.last_board_state = None

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

    def prerender_pieces(self):
        """Pre-render all chess piece images as surfaces"""
        piece_symbols = ['p', 'n', 'b', 'r', 'q', 'k', 'P', 'N', 'B', 'R', 'Q', 'K']
        
        # Create a small SVG for each piece and convert to surface
        for symbol in piece_symbols:
            # Create SVG for single piece
            color = 'white' if symbol.isupper() else 'black'
            piece_svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="300" height="300">
                {chess.svg.piece(chess.Piece.from_symbol(symbol), size=300)}
            </svg>"""
            
            # Convert to Pygame surface
            self.piece_images[symbol] = self.svg_to_pygame_surface(piece_svg)

    def render_board(self, last_move=None, selected_square=None):
        """Render chess board using cached pieces and direct drawing"""
        board_state = self.board.get_board_state()
        square_size = self.WINDOW_SIZE // 8
        
        # Create a new surface for this frame
        surface = pygame.Surface((self.WINDOW_SIZE, self.WINDOW_SIZE))
        
        # Draw board squares
        for rank in range(8):
            for file in range(8):
                square = chess.square(file, 7-rank)  # Flip rank for correct orientation
                square_color = self.square_colors['light'] if (file + rank) % 2 == 0 else self.square_colors['dark']
                
                # Draw the square
                rect = pygame.Rect(file * square_size, rank * square_size, square_size, square_size)
                surface.fill(square_color, rect)
                
                # Highlight selected square if any
                if selected_square is not None and square == selected_square:
                    highlight_surf = pygame.Surface((square_size, square_size), pygame.SRCALPHA)
                    highlight_surf.fill(self.highlighted_square_color)
                    surface.blit(highlight_surf, rect)
                
                # Highlight last move if any
                if last_move and (square == last_move.from_square or square == last_move.to_square):
                    highlight_color = pygame.Color(0, 0, 255, 80)  # Semi-transparent blue
                    highlight_surf = pygame.Surface((square_size, square_size), pygame.SRCALPHA)
                    highlight_surf.fill(highlight_color)
                    surface.blit(highlight_surf, rect)
        
        # Draw pieces
        for rank in range(8):
            for file in range(8):
                square = chess.square(file, 7-rank)
                piece = board_state.piece_at(square)
                
                if piece:
                    piece_img = self.piece_images[piece.symbol()]
                    # Scale piece to fit square
                    scaled_img = pygame.transform.scale(piece_img, (square_size, square_size))
                    surface.blit(scaled_img, (file * square_size, rank * square_size))
        
        # Draw arrows if needed
        if LAST_MOVE_ARROW and last_move:
            self.draw_arrow(surface, last_move.from_square, last_move.to_square, pygame.Color(0, 0, 255, 128))
        
        if CHECKING_MOVE_ARROW and self.checking_move:
            self.draw_arrow(surface, self.checking_move.from_square, self.checking_move.to_square, pygame.Color(255, 0, 0, 128))
        
        return surface

    def draw_arrow(self, surface, from_square, to_square, color):
        """Draw an arrow from one square to another"""
        square_size = self.WINDOW_SIZE // 8
        
        # Calculate start and end positions
        from_file, from_rank = chess.square_file(from_square), 7 - chess.square_rank(from_square)
        to_file, to_rank = chess.square_file(to_square), 7 - chess.square_rank(to_square)
        
        start_pos = ((from_file + 0.5) * square_size, (from_rank + 0.5) * square_size)
        end_pos = ((to_file + 0.5) * square_size, (to_rank + 0.5) * square_size)
        
        # Draw arrow line
        pygame.draw.line(surface, color, start_pos, end_pos, width=5)
        
        # Draw arrow head (simplified)
        pygame.draw.circle(surface, color, end_pos, 8)

    def display_board(self, last_move=None, selected_square=None, force_update=False):
        """Display the current board state with dynamic rendering selection"""
        current_time = pygame.time.get_ticks()
        # Skip update if too soon (unless forced)
        if not force_update and hasattr(self, 'last_update_time'):
            if current_time - self.last_update_time < UPDATE_DELAY_MS:
                return
        
        # REVERSE THE LOGIC: Use optimized rendering when CHECKING_MOVE_ARROW is True,
        # otherwise use pretty SVG rendering
        if CHECKING_MOVE_ARROW and self.checking_move:
            # Use fast direct rendering during AI analysis
            board_surface = self.render_board(last_move, selected_square)
            self.screen.blit(board_surface, (0, 0))
        else:
            # Use pretty SVG rendering during normal gameplay
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
                size=self.WINDOW_SIZE,
                colors={
                    "square light": "#FFFFFF",  # White
                    "square dark": "#769656",   # Green
                }
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
            move = current_player.get_move(self.board.get_board_state())
            
            if move is None:
                print("Game ended by player")
                break
                
            # Make the move
            if not self.board.make_move(move):
                print(f"Illegal move attempted: {move}")
                break
                
            # TODO Create generic evaluation function
            if current_player == self.white_player and type(self.white_player) == ChessBot:
                print(f"Eval: {self.white_player.evaluate_position(self.board.get_board_state(), self.white_player.score)}")
            else:
                print(f"Eval: {self.black_player.evaluate_position(self.board.get_board_state(), self.black_player.score)}")
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
