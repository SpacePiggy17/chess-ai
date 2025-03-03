import chess
import chess.svg
from board import ChessBoard
from bot2 import ChessBot
from human import HumanPlayer
import pygame
import cairosvg
import io
from PIL import Image
import math # For fast render arrows

from constants import IS_BOT, UPDATE_DELAY_MS, LAST_MOVE_ARROW, CHECKING_MOVE_ARROW, BREAK_TURN

class ChessGame:
    __slots__ = ['board', 'checking_move', 'last_move', 'last_update_time', 'white_player', 'black_player', 'piece_images', 'square_colors', 'highlighted_square_color', 'WINDOW_SIZE', 'screen', 'empty_board_surface', 'last_board_state']
    
    def __init__(self):
        self.board = ChessBoard()

        self.checking_move = None # Current move to check
        self.last_move = None # Last move played
        self.last_update_time: int = pygame.time.get_ticks()
        
        # Initialize players based on IS_BOT flag
        if IS_BOT:
            self.white_player = ChessBot(self)
            self.black_player = ChessBot(self)
        else:
            self.white_player = HumanPlayer(chess.WHITE, self)
            self.black_player = ChessBot(self)

        # Cache for piece images and board squares
        self.piece_images = {}

                # Use the default chess.svg colors (tan/brown)
        self.square_colors = {
            'light': pygame.Color('#f0d9b5'),  # Tan/cream squares (default chess.svg)
            'dark': pygame.Color('#b58863'),   # Brown squares (default chess.svg)
            'light_lastmove': pygame.Color('#cdd16a'),  # Highlighted light square for last move
            'dark_lastmove': pygame.Color('#aaa23b'),   # Highlighted dark square for last move
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

        # Pre-render the empty board
        self.empty_board_surface = self.create_empty_board()

    def svg_to_pygame_surface(self, svg_string):
        """Convert SVG string to Pygame surface with optimized parameters"""
        # Reduce the resolution if it's just for pieces (they'll be scaled anyway)
        png_data = cairosvg.svg2png(
            bytestring=svg_string.encode('utf-8'),
            output_width=self.WINDOW_SIZE,  # Directly specify final size
            output_height=self.WINDOW_SIZE
        )
        # Skip resizing step since we specified size in cairosvg
        image = Image.open(io.BytesIO(png_data))
        mode = image.mode
        size = image.size
        data = image.tobytes()
        
        return pygame.image.fromstring(data, size, mode)

    def create_empty_board(self):
        """Create and cache the empty chess board with squares"""
        square_size = self.WINDOW_SIZE // 8
        surface = pygame.Surface((self.WINDOW_SIZE, self.WINDOW_SIZE))
        
        # Draw board squares
        for rank in range(8):
            for file in range(8):
                is_light = (file + rank) % 2 == 0
                square_color = self.square_colors['light' if is_light else 'dark']
                rect = pygame.Rect(file * square_size, rank * square_size, square_size, square_size)
                surface.fill(square_color, rect)
        
        return surface

    def prerender_pieces(self):
        """Pre-render all chess piece images at the correct size"""
        piece_symbols = ['p', 'n', 'b', 'r', 'q', 'k', 'P', 'N', 'B', 'R', 'Q', 'K']
        square_size = self.WINDOW_SIZE // 8
        
        # Create high resolution pieces then scale them once
        for symbol in piece_symbols:
            piece_svg = f"""<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" width="500" height="500" viewBox="0 0 45 45">
                {chess.svg.PIECES[symbol]}
            </svg>"""
            
            # Convert to Pygame surface and pre-scale to square size
            piece_img = self.svg_to_pygame_surface(piece_svg)
            self.piece_images[symbol] = pygame.transform.scale(piece_img, (square_size, square_size))

    def fast_render_board(self, last_move=None, selected_square=None):
        """Render chess board using cached empty board and pieces"""
        board_state = self.board.get_board_state()
        square_size = self.WINDOW_SIZE // 8
        
        # Start with a copy of the empty board
        surface = self.empty_board_surface.copy()
        
        # Only highlight last move squares and selected square
        if last_move:
            for square in [last_move.from_square, last_move.to_square]:
                file = chess.square_file(square)
                rank = 7 - chess.square_rank(square)
                is_light = (file + rank) % 2 == 0
                square_color = self.square_colors['light_lastmove' if is_light else 'dark_lastmove']
                rect = pygame.Rect(file * square_size, rank * square_size, square_size, square_size)
                surface.fill(square_color, rect)
        
        # Highlight selected square
        if selected_square is not None:
            file = chess.square_file(selected_square)
            rank = 7 - chess.square_rank(selected_square)
            highlight_surf = pygame.Surface((square_size, square_size), pygame.SRCALPHA)
            highlight_surf.fill(self.highlighted_square_color)
            surface.blit(highlight_surf, (file * square_size, rank * square_size))
        
        # Draw pieces (now using pre-scaled pieces)
        for rank in range(8):
            for file in range(8):
                square = chess.square(file, 7-rank)
                piece = board_state.piece_at(square)
                if piece:
                    # Directly use the pre-scaled piece image
                    surface.blit(self.piece_images[piece.symbol()], (file * square_size, rank * square_size))
        
        # Draw arrows if needed
        if LAST_MOVE_ARROW and last_move:
            # Using solid blue to match SVG arrow color
            self.draw_arrow(surface, last_move.from_square, last_move.to_square, pygame.Color("#0000FF"))
        
        if CHECKING_MOVE_ARROW and self.checking_move:
            # Using solid red to match SVG arrow color
            self.draw_arrow(surface, self.checking_move.from_square, self.checking_move.to_square, pygame.Color("#FF0000"))
        
            return surface

    def draw_arrow(self, surface, from_square, to_square, color):
        """Draw an arrow that matches the SVG implementation"""
        square_size = self.WINDOW_SIZE // 8
        
        # Calculate start and end positions (centered in squares)
        from_file, from_rank = chess.square_file(from_square), 7 - chess.square_rank(from_square)
        to_file, to_rank = chess.square_file(to_square), 7 - chess.square_rank(to_square)
        
        # Match the SVG coordinate calculations
        xtail = (from_file + 0.5) * square_size
        ytail = (from_rank + 0.5) * square_size
        xhead = (to_file + 0.5) * square_size
        yhead = (to_rank + 0.5) * square_size
        
        # Calculate the direction vector
        dx, dy = xhead - xtail, yhead - ytail
        hypot = math.hypot(dx, dy)
        
        if hypot == 0:
            return  # Can't draw an arrow with zero length
        
        # Create semitransparent surface for entire arrow
        arrow_surface = pygame.Surface((self.WINDOW_SIZE, self.WINDOW_SIZE), pygame.SRCALPHA)
        
        # Use exact parameters from the SVG implementation
        marker_size = 0.75 * square_size
        marker_margin = 0.1 * square_size
        
        # Calculate shaft endpoint (where head begins)
        shaft_x = xhead - dx * (marker_size + marker_margin) / hypot
        shaft_y = yhead - dy * (marker_size + marker_margin) / hypot
        
        # Calculate arrowhead tip position (slightly inset from square edge)
        xtip = xhead - dx * marker_margin / hypot
        ytip = yhead - dy * marker_margin / hypot
        
        # Draw thick shaft - match SVG stroke width
        pygame.draw.line(
            arrow_surface,
            color,
            (xtail, ytail),
            (shaft_x, shaft_y),
            width=int(square_size * 0.2)  # Match the SVG stroke-width
        )
        
        # Calculate arrowhead points using SVG algorithm
        marker_points = [
            (xtip, ytip),  # Tip
            (shaft_x + dy * 0.5 * marker_size / hypot, shaft_y - dx * 0.5 * marker_size / hypot),
            (shaft_x - dy * 0.5 * marker_size / hypot, shaft_y + dx * 0.5 * marker_size / hypot)
        ]
        
        # Draw arrowhead
        pygame.draw.polygon(arrow_surface, color, marker_points)
        
        # Blit the arrow onto the main surface
        surface.blit(arrow_surface, (0, 0))

    def display_board(self, last_move=None, selected_square=None, force_update=False):
        """Display the current board state with dynamic rendering selection"""
        current_time = pygame.time.get_ticks()
        # Skip update if too soon (unless forced)
        if not force_update and hasattr(self, 'last_update_time'):
            if current_time - self.last_update_time < UPDATE_DELAY_MS:
                return
                
        if CHECKING_MOVE_ARROW and self.checking_move:
            # Use fast direct rendering during AI analysis
            board_surface = self.fast_render_board(last_move, selected_square)
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
            if LAST_MOVE_ARROW and last_move:
                arrows.append(chess.svg.Arrow(
                    last_move.from_square,
                    last_move.to_square,
                    color="#0000FF"  # Blue color, solid
                ))
            if CHECKING_MOVE_ARROW and self.checking_move:
                arrows.append(chess.svg.Arrow(
                    self.checking_move.from_square,
                    self.checking_move.to_square,
                    color="#FF0000"  # Red for checked move, solid
                ))

            # Create SVG with highlighted last move and selected square
            svg = chess.svg.board(
                board=self.board.get_board_state(),
                lastmove=last_move,
                squares=highlight_squares,
                arrows=arrows,
                size=self.WINDOW_SIZE,
                colors={
                    "square light": "#f0d9b5",  # Tan/cream
                    "square dark": "#b58863",   # Brown
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

            if self.board.get_board_state().fullmove_number > BREAK_TURN:
                pygame.quit()
                return
            
        # Display final position
        self.display_board(self.last_move, force_update=True)
        print(f"Number of turns: {self.board.get_board_state().fullmove_number}") # Print number of turns
        result = self.board.get_result()
        print(f"Game Over! Result: {result}")
        
        # # Keep window open until closed
        # while True:
        #     event = pygame.event.wait()
        #     if event.type == pygame.QUIT:
        #         break
        # 
        # pygame.quit()

        while 1:
            # Process all pending events at once rather than waiting
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return

if __name__ == "__main__":
    game = ChessGame()
    game.play_game()
