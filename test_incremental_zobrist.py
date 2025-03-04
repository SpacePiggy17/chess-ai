import chess
from chess.polyglot import zobrist_hash
from incremental_zobrist import IncrementalZobrist
import unittest

class TestIncrementalZobrist(unittest.TestCase):
    
    def verify_hash(self, board, incremental_hash):
        """Verify that incremental hash matches the standard zobrist_hash."""
        expected_hash = zobrist_hash(board)
        self.assertEqual(incremental_hash.get_hash(), expected_hash, 
                        f"Hash mismatch at position {board.fen()}")
        
    def test_simple_moves(self):
        """Test simple piece movements."""
        board = chess.Board()
        incremental_hash = IncrementalZobrist(board)
        
        # Test initial position
        self.verify_hash(board, incremental_hash)
        
        # Test a few regular moves
        moves = ["e2e4", "e7e5", "g1f3", "b8c6", "f1b5"]
        for uci in moves:
            move = chess.Move.from_uci(uci)
            incremental_hash.update_for_move(board, move)
            board.push(move)
            self.verify_hash(board, incremental_hash)
            
    def test_captures(self):
        """Test capturing moves."""
        board = chess.Board("r1bqkbnr/ppp2ppp/2np4/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 0 4")
        incremental_hash = IncrementalZobrist(board)
        
        # Capture with bishop
        moves = ["c4f7", "e8f7", "f3e5", "d6e5", "d1h5", "g7g6", "h5e5"]
        for uci in moves:
            move = chess.Move.from_uci(uci)
            incremental_hash.update_for_move(board, move)
            board.push(move)
            self.verify_hash(board, incremental_hash)
            
    def test_castling(self):
        """Test castling and castling rights changes."""
        # Test kingside castling
        board = chess.Board("r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/2NP1N2/PPP2PPP/R1BQK2R w KQkq - 0 5")
        incremental_hash = IncrementalZobrist(board)
        
        # Kingside castle
        move = chess.Move.from_uci("e1g1")
        incremental_hash.update_for_move(board, move)
        board.push(move)
        self.verify_hash(board, incremental_hash)
        
        # Black kingside castle
        move = chess.Move.from_uci("e8g8")
        incremental_hash.update_for_move(board, move)
        board.push(move)
        self.verify_hash(board, incremental_hash)
        
        # Test queenside castling
        board = chess.Board("r3k2r/pppq1ppp/2np1n2/2b1p1B1/2B1P3/2NP1N2/PPP2PPP/R2QK2R b KQkq - 2 8")
        incremental_hash = IncrementalZobrist(board)
        
        # Queenside castle
        move = chess.Move.from_uci("e8c8")
        incremental_hash.update_for_move(board, move)
        board.push(move)
        self.verify_hash(board, incremental_hash)
        
    def test_castling_rights_changes(self):
        """Test changes to castling rights."""
        board = chess.Board()
        incremental_hash = IncrementalZobrist(board)
        
        # Move rook, losing castling rights
        move = chess.Move.from_uci("h1h4")
        incremental_hash.update_for_move(board, move)
        board.push(move)
        self.verify_hash(board, incremental_hash)
        
        # Capture a rook, affecting castling rights
        board = chess.Board("r3k2r/ppp2ppp/2n5/3pp3/8/2N5/PPPPQPPP/R3K2R b KQkq - 0 1")
        incremental_hash = IncrementalZobrist(board)
        
        move = chess.Move.from_uci("a8a1")
        incremental_hash.update_for_move(board, move)
        board.push(move)
        self.verify_hash(board, incremental_hash)
        
    def test_en_passant(self):
        """Test en passant captures and hash updates."""
        board = chess.Board("rnbqkbnr/ppp1p1pp/8/3pPp2/8/8/PPPP1PPP/RNBQKBNR w KQkq f6 0 3")
        incremental_hash = IncrementalZobrist(board)
        
        # En passant capture
        move = chess.Move.from_uci("e5f6")
        incremental_hash.update_for_move(board, move)
        board.push(move)
        self.verify_hash(board, incremental_hash)
        
        # Move creating an en passant opportunity
        board = chess.Board("rnbqkbnr/pppp1ppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1")
        incremental_hash = IncrementalZobrist(board)
        
        move = chess.Move.from_uci("f7f5")
        incremental_hash.update_for_move(board, move)
        board.push(move)
        self.verify_hash(board, incremental_hash)
        
    def test_promotion(self):
        """Test pawn promotions."""
        board = chess.Board("8/1P6/8/8/8/8/k7/K7 w - - 0 1")
        incremental_hash = IncrementalZobrist(board)
        
        # Promote to queen
        move = chess.Move.from_uci("b7b8q")
        incremental_hash.update_for_move(board, move)
        board.push(move)
        self.verify_hash(board, incremental_hash)
        
        # Promote to knight
        board = chess.Board("8/7P/8/8/8/8/k7/K7 w - - 0 1")
        incremental_hash = IncrementalZobrist(board)
        
        move = chess.Move.from_uci("h7h8n")
        incremental_hash.update_for_move(board, move)
        board.push(move)
        self.verify_hash(board, incremental_hash)
        
    def test_complex_sequence(self):
        """Test a complex sequence of moves with different edge cases."""
        board = chess.Board()
        incremental_hash = IncrementalZobrist(board)
        
        # Full game sequence with various move types
        moves = [
            "e2e4", "c7c5",          # Open Sicilian
            "g1f3", "d7d6",
            "d2d4", "c5d4",          # Capture
            "f3d4", "g8f6",
            "b1c3", "g7g6", 
            "c1e3", "f8g7",
            "f2f3", "e8g8",          # Castling
            "d1d2", "b8c6",
            "e1c1",                  # Queenside castling
            "f6e4", "f3e4",          # Capture sequence
            "c6d4", "e3d4",
            "d8b6", "d2d3",          
            "b6b2", "c1b2",          # Complex capture
            "g7b2", "h1c1",          
            "b2e5", "d3d1",          
            "a8b8", "a2a4",
            "b8b4", "a4a5",          
            "b4b5", "a5a6",          
            "b7a6",                  # Pawn capture
            "d1d6", "f8d8",
            "d6e7", "d8d7",          
            "c3e4", "b5e5",
            "e4c5", "e5e1",          
            "c1e1", "d7d1",          # Rook exchanges
            "e1d1", "e7d7",          
            "d1e1", "a6a5"           # Final move
        ]
        
        for i, uci in enumerate(moves):
            move = chess.Move.from_uci(uci)
            incremental_hash.update_for_move(board, move)
            board.push(move)
            self.verify_hash(board, incremental_hash)
            
    def test_full_game(self):
        """Test with a full real game."""
        # Famous game: Kasparov vs Topalov, Wijk aan Zee 1999 (abridged)
        moves = [
            "e2e4", "d7d6", "d2d4", "g8f6", "b1c3", "g7g6", "c1e3", "f8g7",
            "d1d2", "c7c6", "f2f3", "b7b5", "g1e2", "b8d7", "e3h6", "g7h6",
            "d2h6", "c8b7", "a2a3", "e7e5", "e1c1", "d8e7", "c1b1", "a7a6",
            "e2c1", "e8c8", "c1b3", "e5d4", "d1d4", "c6c5", "d4d1", "d7b6",
            "g2g3", "c8b8", "b3a5", "b7a8", "f1h3", "d6d5", "h6f4", "b8a7",
            "h1e1", "d5e4", "c3e4", "f6e4", "f4e4", "e7e4", "f3e4", "a8e4",
            "e1e4", "a7b6", "a5c6", "b6a5", "e4e7", "h8g8"
        ]
        
        board = chess.Board()
        incremental_hash = IncrementalZobrist(board)
        
        for uci in moves:
            move = chess.Move.from_uci(uci)
            incremental_hash.update_for_move(board, move)
            board.push(move)
            self.verify_hash(board, incremental_hash)

if __name__ == "__main__":
    unittest.main()
