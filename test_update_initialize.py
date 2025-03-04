import unittest
import chess
import numpy as np
import random
from timeit import default_timer as timer
import sys
from bot2 import Score, BISHOP_PAIR_BONUS, PIECE_VALUES_STOCKFISH

# ! This test concludes that the initialize approach is faster than the update approach but this is incorrect in practice.

MAX_NODES = 20_000  # Maximum nodes to visit in simulated search
MAX_DEPTH = 7  # Maximum depth to search

class TestScoreUpdate(unittest.TestCase):
    """Test cases for the Score.update method."""
    
    def setUp(self):
        """Initialize test environment."""
        self.board = chess.Board()
        self.score = Score(material=0, mg=0, eg=0, npm=0)
        self.score.initialize_scores(self.board)
        
    def test_pawn_move(self):
        """Test updating score for a simple pawn move."""
        # Store original values
        original_material = self.score.material
        original_mg = self.score.mg
        original_eg = self.score.eg
        original_npm = self.score.npm
        
        # Make a simple pawn move
        move = chess.Move.from_uci("e2e4")
        original_values = self.score.update(self.board, move)
        
        # Verify original values were returned
        self.assertEqual(original_values, (original_material, original_mg, original_eg, original_npm))
        
        # Verify material hasn't changed (no capture)
        self.assertEqual(self.score.material, original_material)
        
        # Verify positional scores have changed 
        self.assertNotEqual(self.score.mg, original_mg)
        self.assertNotEqual(self.score.eg, original_eg)
        
        # Verify npm unchanged (no non-pawn material change)
        self.assertEqual(self.score.npm, original_npm)
        
    def test_capture(self):
        """Test updating score for a capture move."""
        # Setup board with a capture possibility
        self.board = chess.Board("rnbqkbnr/ppp1pppp/8/3p4/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2")
        self.score = Score(material=0, mg=0, eg=0, npm=0)
        self.score.initialize_scores(self.board)
        
        # Store original values
        original_material = self.score.material
        original_npm = self.score.npm
        
        # Make a capture move: white pawn captures black pawn
        move = chess.Move.from_uci("e4d5")
        self.score.update(self.board, move)
        
        # Material should be equal (equal exchange)
        self.assertEqual(self.score.material, original_material)
        
        # NPM should remain unchanged (pawns don't count)
        self.assertEqual(self.score.npm, original_npm)
        
    def test_bishop_pair_bonus(self):
        """Test updating score when bishop pair changes."""
        # Setup board with two bishops for black
        self.board = chess.Board("r1bqk1nr/pppp1ppp/2n5/2b1p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4")
        self.score = Score(material=0, mg=0, eg=0, npm=0)
        self.score.initialize_scores(self.board)
        
        original_material = self.score.material
        
        # Capture one of black's bishops
        move = chess.Move.from_uci("c4e6")  # White bishop captures pawn at e6
        self.board.push(move)
        move = chess.Move.from_uci("c5e3")  # Black bishop gets captured
        self.score.update(self.board, move)
        
        # Material should change - bishop value plus bishop pair bonus change
        self.assertNotEqual(self.score.material, original_material)
        
        # Verify bishop pair bonus was removed
        expected_material_change = PIECE_VALUES_STOCKFISH[chess.BISHOP] - BISHOP_PAIR_BONUS
        self.assertEqual(self.score.material, original_material + expected_material_change)
        
    def test_castling(self):
        """Test updating score for castling moves."""
        # Setup board ready for castling
        self.board = chess.Board("r3k2r/pppqbppp/2np1n2/2b1p3/2B1P3/2NP1N2/PPPQBPPP/R3K2R w KQkq - 6 8")
        self.score = Score(material=0, mg=0, eg=0, npm=0)
        self.score.initialize_scores(self.board)
        
        # Store original values
        original_mg = self.score.mg
        original_eg = self.score.eg
        
        # Castle kingside
        move = chess.Move.from_uci("e1g1")  # White kingside castle
        self.score.update(self.board, move)
        
        # Positional scores should change
        self.assertNotEqual(self.score.mg, original_mg)
        self.assertNotEqual(self.score.eg, original_eg)
        
        # Both king and rook position scores should be updated

    def test_promotion(self):
        """Test updating score for pawn promotion."""
        # Setup board with a pawn ready to promote
        self.board = chess.Board("8/P7/8/8/8/8/8/k6K w - - 0 1")
        self.score = Score(material=0, mg=0, eg=0, npm=0)
        self.score.initialize_scores(self.board)
        
        original_material = self.score.material
        original_npm = self.score.npm
        
        # Promote to queen
        move = chess.Move.from_uci("a7a8q")
        self.score.update(self.board, move)
        
        # Material should increase by queen - pawn
        expected_material_change = PIECE_VALUES_STOCKFISH[chess.QUEEN] - PIECE_VALUES_STOCKFISH[chess.PAWN]
        self.assertEqual(self.score.material, original_material + expected_material_change)
        
        # NPM should increase by queen value
        expected_npm_change = PIECE_VALUES_STOCKFISH[chess.QUEEN]
        self.assertEqual(self.score.npm, original_npm + expected_npm_change)


class BenchmarkTests(unittest.TestCase):
    """Benchmark tests comparing update vs. initialize approaches."""
    
    def test_update_vs_initialize(self):
        """Compare performance of incremental updates vs. reinitializing scores."""
        # Parameters for the benchmark
        max_depth = MAX_DEPTH  # Maximum depth to search
        branching_factor = 35  # Average number of legal moves
        sample_size = MAX_NODES  # Number of positions to evaluate
        
        update_time, init_time = self._run_benchmark(max_depth, branching_factor, sample_size)
        
        print(f"\nBenchmark Results (Depth {max_depth}, {sample_size} positions):")
        print(f"Update approach: {update_time:.6f}s")
        print(f"Initialize approach: {init_time:.6f}s")
        
        if update_time > 0:
            if init_time < update_time:
                speed_ratio = update_time / init_time
                print(f"Speed ratio: Initialize is {speed_ratio:.2f}x faster")
            else:
                speed_ratio = init_time / update_time
                print(f"Speed ratio: Update is {speed_ratio:.2f}x faster")
        
        self.assertTrue(True)  # Not a real assertion
        
    def _run_benchmark(self, max_depth, branching_factor, sample_size):
        """Run the benchmark comparing update vs. initialize."""
        # First, generate a set of realistic positions by simulating moves
        positions = []
        initial_board = chess.Board()
        
        # Generate random positions at different depths
        for _ in range(sample_size):
            depth = random.randint(1, max_depth)
            board = initial_board.copy()
            moves = []
            
            # Make random moves to reach a position
            for _ in range(depth):
                legal_moves = list(board.legal_moves)
                if not legal_moves:
                    break
                # Limit the branching factor to simulate pruning
                move_sample = random.sample(legal_moves, min(len(legal_moves), 3))
                move = random.choice(move_sample)
                moves.append(move)
                board.push(move)
            
            if moves:  # Only add if we made at least one move
                positions.append((board.copy(), moves))
        
        # Benchmark update approach (incremental updates)
        start_time = timer()
        for _, move_path in positions:
            board = chess.Board()
            score = Score(material=0, mg=0, eg=0, npm=0)
            score.initialize_scores(board)
            
            # Apply each move and update score
            for move in move_path:
                original_values = score.update(board, move)
                board.push(move)
            
            # In a real search, we'd evaluate the position here
        update_time = timer() - start_time
        
        # Benchmark initialize approach
        start_time = timer()
        for final_board, _ in positions:
            # Initialize from scratch for each position
            score = Score(material=0, mg=0, eg=0, npm=0)
            score.initialize_scores(final_board)
            
            # In a real search, we'd evaluate the position here
        init_time = timer() - start_time
        
        return update_time, init_time
    
    def test_simulated_alphabeta_search(self):
        """
        Simulate a realistic alpha-beta search to compare approaches.
        This simulates a more realistic scenario where we evaluate
        positions in a depth-first manner.
        """
        max_depth = MAX_DEPTH
        max_nodes = MAX_NODES
        
        # Run the simulated search
        update_time, init_time, nodes_visited = self._simulate_alphabeta_search(max_depth, max_nodes)
        
        print(f"\nAlpha-Beta Search Simulation (Depth {max_depth}, {nodes_visited} nodes):")
        print(f"Update approach: {update_time:.6f}s")
        print(f"Initialize approach: {init_time:.6f}s")

        if update_time > 0:
            if init_time < update_time:
                speed_ratio = update_time / init_time
                print(f"  Result: Initialize is {speed_ratio:.2f}x faster at depth {max_depth}")
            else:
                speed_ratio = init_time / update_time
                print(f"  Result: Update is {speed_ratio:.2f}x faster at depth {max_depth}")
        
        self.assertTrue(True)  # Not a real assertion
    
    def _simulate_alphabeta_search(self, max_depth, max_nodes):
        """
        Simulate an alpha-beta search and compare both approaches.
        Returns (update_time, init_time, nodes_visited)
        """
        # First generate a search tree path to ensure both methods visit identical nodes
        paths_to_evaluate = []
        
        def generate_search_paths(board, depth, path):
            """Generate paths through the search tree."""
            if depth == 0 or len(paths_to_evaluate) >= max_nodes:
                paths_to_evaluate.append((board.copy(), path.copy()))
                return
            
            legal_moves = list(board.legal_moves)
            if not legal_moves:
                paths_to_evaluate.append((board.copy(), path.copy()))
                return
            
            # Use higher branching factors, especially at shallower depths
            # This creates a more realistic chess search tree
            if depth == max_depth:  # Root
                branch_factor = min(len(legal_moves), 15)  # Try more moves at root
            elif depth >= max_depth - 1:  # Near root
                branch_factor = min(len(legal_moves), 10)
            elif depth >= max_depth - 2:  # Mid-depth
                branch_factor = min(len(legal_moves), 7)
            else:  # Deep
                branch_factor = min(len(legal_moves), 5)
                
            if len(paths_to_evaluate) > max_nodes / 2:
                branch_factor = min(branch_factor, 3)  # Reduce if we're using too many nodes
                
            sampled_moves = random.sample(legal_moves, branch_factor)
            
            for move in sampled_moves:
                board.push(move)
                path.append(move)
                generate_search_paths(board, depth - 1, path)
                path.pop()
                board.pop()
        
        # Generate paths
        initial_board = chess.Board()
        generate_search_paths(initial_board, max_depth, [])
        nodes_visited = len(paths_to_evaluate)
        
        # Simulate update approach (incremental updates at each node)
        start_time = timer()
        for final_board, path in paths_to_evaluate:
            board = chess.Board()
            score = Score(material=0, mg=0, eg=0, npm=0)
            score.initialize_scores(board)  # Initialize once at the root
            
            # Apply each move and update score incrementally
            for move in path:
                original_values = score.update(board, move)
                board.push(move)
            
            # Evaluate at leaf node (simulated)
            evaluation = score.material + score.mg + score.eg
        update_time = timer() - start_time
        
        # Simulate initialize approach (reinitialize at each leaf)
        start_time = timer()
        for final_board, _ in paths_to_evaluate:
            # Initialize from scratch at each leaf node
            score = Score(material=0, mg=0, eg=0, npm=0)
            score.initialize_scores(final_board)
            
            # Evaluate at leaf node (simulated)
            evaluation = score.material + score.mg + score.eg
        init_time = timer() - start_time
        
        return update_time, init_time, nodes_visited


if __name__ == "__main__":
    # Run specific benchmark tests with enhanced output
    benchmark = BenchmarkTests()
    
    print("=" * 60)
    print("BENCHMARK: UPDATE VS INITIALIZE APPROACH")
    print("=" * 60)
    print("This test compares two approaches for position evaluation:")
    print("1. UPDATE: Incrementally update scores at each move")
    print("2. INITIALIZE: Reinitialize scores from scratch at leaf nodes only")
    print("\nRunning simple benchmarks...")
    benchmark.test_update_vs_initialize()
    
    print("\nRunning simulated alpha-beta search (more realistic scenario)...")
    benchmark.test_simulated_alphabeta_search()  # Use the test method which prints results

    # Additional tests with varied parameters
    depths = [i for i in range(3, MAX_DEPTH + 1)]
    print("\n" + "=" * 60)
    print("DETAILED COMPARISON BY SEARCH DEPTH")
    print("=" * 60)
    
    for depth in depths:
        print(f"\nTesting at depth {depth}:")
        update_time, init_time, nodes = benchmark._simulate_alphabeta_search(depth, MAX_NODES)
        
        print(f"  Nodes visited: {nodes}")
        print(f"  Update approach: {update_time:.6f}s")
        print(f"  Initialize approach: {init_time:.6f}s")
        
        if update_time > 0:
            if init_time < update_time:
                speed_ratio = update_time / init_time
                print(f"  Result: Initialize is {speed_ratio:.2f}x faster at depth {depth}")
            else:
                speed_ratio = init_time / update_time
                print(f"  Result: Update is {speed_ratio:.2f}x faster at depth {depth}")
    
    print("\nConclusion:")
    print("The most efficient approach depends on the search depth and")
    print("branching factor. At higher depths, the results above show")
    print("which approach is most suitable for your chess engine.")
    
    # Run all tests
    # unittest.main()
