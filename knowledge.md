# Chess AI Project Knowledge

## Project Overview
A chess game implementation with human vs bot play using Python, pygame, and python-chess.

## Key Components
- `game.py`: Main game loop and board display
- `board.py`: Chess board state and move validation
- `human.py`: Human player interaction and UI
- `bot.py`: AI/bot player implementation

## UI Guidelines
- Use chess.svg for piece rendering
- Convert SVG to pygame surfaces using cairosvg
- Board size is 600x600 pixels
- Promotion UI should show actual piece icons next to text
- Buttons should have:
  - Light background (#F0F0F0)
  - Clear borders
  - Both icon and text for clarity

## Display Updates
- Board display happens in game loop for consistency
- Use chess.svg.board() for rendering with options:
  - lastmove: Highlight the last move made (brown)
  - squares: Pass dict with {"fill": color, "stroke": "none"} for colored highlights (not SquareSet which shows X marks)
  - fill: Show legal moves (semi-transparent blue)
  - orientation: Match the player's color
- Highlight selected piece with yellow square for better UX

## Move Handling
- Validate moves against legal_moves list
- Check for promotions when pawns reach the back rank
- Clear selections after illegal moves
- Support both human and bot players

## Dependencies
- pygame: Game UI and interaction
- python-chess: Chess logic and SVG rendering
- cairosvg: SVG to PNG conversion
- PIL: Image processing

## Style Preferences
- Use clear docstrings
- Keep UI code modular and separated from game logic
- Handle all pygame events explicitly
- Clean up resources properly on exit

