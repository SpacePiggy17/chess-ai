import unittest
import chess
import time
import numpy as np
from timeit import default_timer
from typing import List, Optional
import sys
import os
from bot2 import ChessBot, Score

# Import the ChessBot and Score classes from bot2.py

# Mock ChessGame class for testing
class MockChessGame:
    def __init__(self, board=None):
        self.board = MockChessBoard(board)
        self.checking_move = None
        self.last_move = None
    
    def display_board(self, last_move):
        # Do nothing, this is just a mock
        pass

class MockChessBoard:
    def __init__(self, board=None):
        self.board = board if board else chess.Board()
    
    def get_board_state(self):
        return self.board

class TestAlphaBeta(unittest.TestCase):
    def setUp(self):
        # Create a fresh board and bot for each test
        self.board = chess.Board()
        self.game = MockChessGame(self.board)
        self.bot = ChessBot(self.game)
    
    def test_checkmate_in_one(self):
        """Test if alpha_beta can find checkmate in one move."""
        # Scholar's mate position - Black can checkmate in one with Qd8-h4#
        self.board.set_fen("rnb1kbnr/pppp1ppp/8/4p3/5PPq/8/PPPPP2P/RNBQKBNR w KQkq - 0 3")
        self.bot.score.initialize_scores(self.board)
        
        # Run alpha-beta for White, who should recognize the checkmate threat
        depth = np.int8(2)  # Depth 2 should be enough to find checkmate
        value, _ = self.bot.alpha_beta(
            self.board, depth, -float('inf'), float('inf'), True, self.bot.score
        )
        
        # Value should be very negative (Black has checkmate)
        self.assertLess(value, -9000)  # Checkmate value should be less than -9000
    
    def test_material_advantage_evaluation(self):
        """Test if alpha_beta correctly evaluates positions with material advantage."""
        # Position where white is up a queen (black's queen is missing)
        self.board.set_fen("rnb1kbnr/pppp1ppp/8/4p3/4P3/8/PPPPQPPP/RNB1KBNR w KQkq - 0 1")
        self.bot.score.initialize_scores(self.board)
        
        # Get evaluation for white's advantage
        value, _ = self.bot.alpha_beta(
            self.board, np.int8(1), -float('inf'), float('inf'), True, self.bot.score
        )
        
        # Value should be positive (white advantage)
        self.assertGreater(value, 500)  # Queen is worth more than 5 pawns
    
    def test_depth_effect(self):
        """Test that deeper searches potentially find different moves."""
        # Start with a position that requires deeper search to find best move
        self.board.set_fen("r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4")
        self.bot.score.initialize_scores(self.board)
        
        # Get move at depth 1
        _, move_depth1 = self.bot.alpha_beta(
            self.board, np.int8(1), -float('inf'), float('inf'), True, self.bot.score
        )
        
        # Get move at depth 3
        _, move_depth3 = self.bot.alpha_beta(
            self.board, np.int8(3), -float('inf'), float('inf'), True, self.bot.score
        )
        
        # Note moves at different depths - they may or may not be different
        # This is more a validation test than an assertion test
        print(f"Depth 1 move: {move_depth1}")
        print(f"Depth 3 move: {move_depth3}")
    
    def test_transposition_table(self):
        """Test that transposition table is used."""
        self.board.set_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
        self.bot.score.initialize_scores(self.board)
        
        # Clear the transposition table
        self.bot.transposition_table.clear()
        
        # Run alpha-beta once
        self.bot.alpha_beta(
            self.board, np.int8(3), -float('inf'), float('inf'), True, self.bot.score
        )
        
        # Check that transposition table has entries
        self.assertGreater(len(self.bot.transposition_table), 0)
        
        # Count entries before second run
        entries_before = len(self.bot.transposition_table)
        
        # Run alpha-beta on same position again
        self.bot.alpha_beta(
            self.board, np.int8(3), -float('inf'), float('inf'), True, self.bot.score
        )
        
        # For the second run, there should be fewer nodes evaluated due to TT hits
        self.assertGreaterEqual(len(self.bot.transposition_table), entries_before)

    def test_position_evaluation(self):
        """Test evaluation of different positions."""
        positions = [
            # Position, expected score range (min, max)
            (chess.Board(), (-50, 50)),  # Starting position should be roughly equal
            (chess.Board("rnb1kbnr/pppp1ppp/8/4p3/4P3/8/PPPPQPPP/RNB1KBNR w KQkq - 0 1"), (800, 1100)),  # White up a queen
            (chess.Board("rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 1"), (-50, 50)),  # Equal after e4 e5
            (chess.Board("rnbqkbnr/pppp1ppp/8/4p3/4PP2/8/PPPP2PP/RNBQKBNR b KQkq - 0 2"), (50, 150)),  # White slightly better after f4
        ]
        
        for pos, (min_score, max_score) in positions:
            self.bot.score.initialize_scores(pos)
            score = self.bot.evaluate_position(pos, self.bot.score)
            
            print(f"Position: {pos.fen()[:30]}...")
            print(f"Evaluation: {score} (Expected range: {min_score} to {max_score})")
            
            self.assertGreaterEqual(score, min_score)
            self.assertLessEqual(score, max_score)

class TestLegalMoveGeneration(unittest.TestCase):
    """Test the speed of legal move generation in the Python chess library."""
    
    def test_move_generation_speed(self):
        """Benchmark legal move generation for different positions."""
        # Create different positions for testing
        positions = [
            # Opening position
            chess.Board(),
            # Early middlegame position
            chess.Board("r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4"),
            # Complex middlegame position
            chess.Board("r4rk1/pp2ppbp/2n3p1/q1Bp4/3P4/2P1PN2/P2Q1PPP/R3K2R w KQ - 0 12"),
            # Endgame position with few pieces
            chess.Board("4k3/8/8/8/8/8/4P3/4K3 w - - 0 1")
        ]
        
        position_names = ["Opening", "Early middlegame", "Complex middlegame", "Simple endgame"]
        
        # Number of iterations for each test
        iterations = 10000
        
        results = []
        
        print("\nLegal Move Generation Speed Test:")
        print("-" * 70)
        print(f"{'Position':<20} {'Avg Time (μs)':<15} {'Moves per Second':<20} {'Avg # of Moves':<15}")
        print("-" * 70)
        
        for pos, name in zip(positions, position_names):
            start_time = default_timer()
            total_moves = 0
            
            for _ in range(iterations):
                moves = list(pos.legal_moves)
                total_moves += len(moves)
            
            end_time = default_timer()
            time_taken = end_time - start_time
            avg_time_us = (time_taken / iterations) * 1_000_000  # Convert to microseconds
            moves_per_sec = iterations / time_taken
            avg_moves = total_moves / iterations
            
            results.append((name, avg_time_us, moves_per_sec, avg_moves))
            print(f"{name:<20} {avg_time_us:.2f} μs      {moves_per_sec:,.0f}/sec        {avg_moves:.1f}")
        
        return results

    def test_move_generation_accuracy_and_speed(self):
        """Test both accuracy and speed of move generation."""
        # Define test positions with known move counts at different depths
        perft_tests = [
            # Position, depth, expected node count
            (chess.Board(), 1, 20),  # Starting position, depth 1: 20 moves
            (chess.Board(), 2, 400),  # Starting position, depth 2: 400 nodes
            (chess.Board("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1"), 1, 48),
            # Kiwipete position - complex middlegame
        ]
        
        print("\nMove Generation Accuracy Test:")
        print("-" * 70)
        print(f"{'Position':<20} {'Depth':<8} {'Expected':<12} {'Actual':<12} {'Time (ms)':<10} {'Status'}")
        print("-" * 70)
        
        for position, depth, expected in perft_tests:
            start_time = default_timer()
            count = self.perft(position, depth)
            end_time = default_timer()
            time_ms = (end_time - start_time) * 1000
            
            status = "PASS" if count == expected else f"FAIL (off by {count - expected})"
            
            print(f"{position.fen()[:20]}... {depth:<8} {expected:<12} {count:<12} {time_ms:.2f} ms    {status}")
            
            # Verify the accuracy
            self.assertEqual(count, expected)
        
        # Now run the original speed test
        return self.test_move_generation_speed()

    def perft(self, board, depth):
        """Perft function to count nodes at a given depth."""
        if depth == 0:
            return 1
        
        nodes = 0
        for move in board.legal_moves:
            board.push(move)
            nodes += self.perft(board, depth - 1)
            board.pop()
        
        return nodes

if __name__ == '__main__':
    unittest.main()
