import chess
import pygame

class HumanPlayer:
    def __init__(self, color, game):
        self.color = color
        self.game = game  # Store reference to game for redrawing
        self.selected_square = None
        
    def get_square_from_coords(self, x, y, flipped=False):
        """Convert screen coordinates to chess square."""
        file_idx = x * 8 // 600
        rank_idx = y * 8 // 600
        if flipped:
            file_idx = 7 - file_idx
            rank_idx = 7 - rank_idx
        else:
            rank_idx = 7 - rank_idx
        return chess.square(file_idx, rank_idx)

    def is_promotion_move(self, board, from_square, to_square):
        """Check if the move would be a pawn promotion."""
        piece = board.get_board_state().piece_at(from_square)
        if piece and piece.piece_type == chess.PAWN:
            rank = chess.square_rank(to_square)
            return (self.color == chess.WHITE and rank == 7) or \
                   (self.color == chess.BLACK and rank == 0)
        return False

    def get_promotion_choice(self):
        """Get the promotion piece choice from the player through clickable buttons with piece icons."""
        import chess.svg
        import cairosvg
        import io
        from PIL import Image
        
        pygame.font.init()
        font = pygame.font.Font(None, 36)
        screen = pygame.display.get_surface()
        
        # Define piece options
        pieces = [
            (chess.QUEEN, "Queen"),
            (chess.ROOK, "Rook"),
            (chess.BISHOP, "Bishop"),
            (chess.KNIGHT, "Knight")
        ]
        
        # Calculate button dimensions and positions
        button_width = 200
        button_height = 80
        button_margin = 10
        total_height = (button_height + button_margin) * len(pieces)
        start_y = (600 - total_height) // 2  # Center vertically in 600x600 window
        
        # Create semi-transparent overlay
        overlay = pygame.Surface((600, 600))
        overlay.fill((0, 0, 0))
        overlay.set_alpha(128)
        screen.blit(overlay, (0, 0))
        
        # Function to convert SVG to Pygame surface
        def svg_to_pygame_surface(svg_string, size):
            png_data = cairosvg.svg2png(bytestring=svg_string.encode('utf-8'))
            image = Image.open(io.BytesIO(png_data))
            image = image.resize((size, size))
            mode = image.mode
            size = image.size
            data = image.tobytes()
            return pygame.image.fromstring(data, size, mode)
        
        # Draw buttons and store their rects
        buttons = []
        current_y = start_y
        
        for piece_type, piece_name in pieces:
            # Create button rectangle
            button_rect = pygame.Rect(
                (600 - button_width) // 2,  # Center horizontally
                current_y,
                button_width,
                button_height
            )
            
            # Draw button background
            pygame.draw.rect(screen, (240, 240, 240), button_rect)
            pygame.draw.rect(screen, (100, 100, 100), button_rect, 2)  # Border
            
            # Generate piece SVG
            piece_svg = chess.svg.piece(chess.Piece(piece_type, self.color), size=button_height-20)
            piece_surface = svg_to_pygame_surface(piece_svg, button_height-20)
            
            # Calculate positions for piece icon and text
            piece_x = button_rect.left + 20
            piece_y = button_rect.top + 10
            text_x = piece_x + button_height  # Position text after the piece icon
            
            # Draw piece icon
            screen.blit(piece_surface, (piece_x, piece_y))
            
            # Draw text
            text = font.render(piece_name, True, (0, 0, 0))
            text_rect = text.get_rect(midleft=(text_x, button_rect.centery))
            screen.blit(text, text_rect)
            
            buttons.append((button_rect, piece_type))
            current_y += button_height + button_margin
        
        pygame.display.flip()
        
        # Wait for valid choice
        while True:
            event = pygame.event.wait()
            if event.type == pygame.QUIT:
                return None
            if event.type == pygame.MOUSEBUTTONDOWN:
                mouse_pos = event.pos
                for button_rect, piece_type in buttons:
                    if button_rect.collidepoint(mouse_pos):
                        return piece_type
        
    def get_move(self, board):
        """Get move from human player through GUI interaction."""
        # Removed pygame.event.clear() to avoid discarding important events
        
        while True:
            event = pygame.event.wait()
            
            if event.type == pygame.QUIT:
                return None
                
            if event.type == pygame.MOUSEBUTTONDOWN:
                x, y = event.pos
                square = self.get_square_from_coords(x, y, self.color == chess.BLACK)
                
                if self.selected_square is None:
                    # First click - select piece
                    piece = board.get_board_state().piece_at(square)
                    if piece and piece.color == self.color:
                        self.selected_square = square
                        # Immediately redraw board with highlighted square
                        self.game.display_board(last_move=self.game.last_move, selected_square=self.selected_square, force_update=True)
                else:
                    # Second click - try to make move
                    from_square = self.selected_square
                    to_square = square
                    
                    # Check if this is a promotion move
                    if self.is_promotion_move(board, from_square, to_square):
                        promotion_piece = self.get_promotion_choice()
                        if promotion_piece is None:
                            self.selected_square = None
                            continue
                        move = chess.Move(from_square, to_square, promotion=promotion_piece)
                    else:
                        move = chess.Move(from_square, to_square)
                    
                    # Check if move is legal
                    if move in board.get_legal_moves():
                        self.selected_square = None
                        return move
                    
                    # If illegal move, clear selection
                    self.selected_square = None
        
        return None
