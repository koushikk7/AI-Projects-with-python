import matplotlib.pyplot as plt
import numpy as np

class CSP:
    def __init__(self, puzzle):
        self.puzzle = puzzle
        self.variables = [(i, j) for i in range(9) for j in range(9) if puzzle[i][j] == 0]
        self.domains = {var: set(range(1, 10)) for var in self.variables}
        self.assignment = {}

    def print_sudoku(self):
        for i in range(9):
            if i % 3 == 0 and i != 0:
                print("-" * 21)
            for j in range(9):
                if j % 3 == 0 and j != 0:
                    print("| ", end="")
                print(self.puzzle[i][j] if self.puzzle[i][j] != 0 else ".", end=" ")
            print()

    def visualize(self, final=False):
        # This function creates a graphical representation of the Sudoku board.
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.matshow(np.ones((9, 9)), cmap="Wistia")

        for i in range(9):
            for j in range(9):
                c = self.puzzle[j][i]
                ax.text(i, j, str(c), va='center', ha='center')

        for i in range(1, 9):
            if i % 3 == 0:
                lw = 2
            else:
                lw = 0.5
            ax.axhline(i-0.5, color='black', linewidth=lw)
            ax.axvline(i-0.5, color='black', linewidth=lw)

        plt.axis('off')
        if final:
            plt.show()

    def is_consistent(self, var, value):
        i, j = var
        for x in range(9):
            if self.puzzle[x][j] == value or self.puzzle[i][x] == value:
                return False
        box_start_i, box_start_j = 3 * (i // 3), 3 * (j // 3)
        for x in range(box_start_i, box_start_i + 3):
            for y in range(box_start_j, box_start_j + 3):
                if self.puzzle[x][y] == value:
                    return False
        return True

    def forward_checking(self, var, value):
        removed_values = {}
        i, j = var
        for x in range(9):
            if (x, j) in self.variables and value in self.domains[(x, j)]:
                self.domains[(x, j)].remove(value)
                removed_values[(x, j)] = value
            if (i, x) in self.variables and value in self.domains[(i, x)]:
                self.domains[(i, x)].remove(value)
                removed_values[(i, x)] = value
        box_start_i, box_start_j = 3 * (i // 3), 3 * (j // 3)
        for x in range(box_start_i, box_start_i + 3):
            for y in range(box_start_j, box_start_j + 3):
                if (x, y) in self.variables and value in self.domains[(x, y)]:
                    self.domains[(x, y)].remove(value)
                    removed_values[(x, y)] = value
        return removed_values

    def undo_forward_checking(self, removed_values):
        for var, value in removed_values.items():
            self.domains[var].add(value)

    def select_unassigned_variable(self):
        return min([v for v in self.variables if v not in self.assignment], key=lambda var: len(self.domains[var]), default=None)

    def order_domain_values(self, var):
        if var is None:
            return []
        return list(self.domains[var])

    def assign(self, var, value):
        self.assignment[var] = value
        self.puzzle[var[0]][var[1]] = value

    def unassign(self, var):
        if var in self.assignment:
            del self.assignment[var]
            self.puzzle[var[0]][var[1]] = 0

    def backtrack(self):
        if len(self.assignment) == len(self.variables):
            return True

        var = self.select_unassigned_variable()
        for value in self.order_domain_values(var):
            if self.is_consistent(var, value):
                self.assign(var, value)
                removed_values = self.forward_checking(var, value)
                result = self.backtrack()
                if result:
                    return result
                self.unassign(var)
                self.undo_forward_checking(removed_values)

        return False

    def solve(self):
        if self.backtrack():
            self.visualize(final=True)
            return self.puzzle
        else:
            return None


# Easy Puzzle
easy_puzzle = [
    [5, 3, 0, 0, 7, 0, 0, 0, 0],
    [6, 0, 0, 1, 9, 5, 0, 0, 0],
    [0, 9, 8, 0, 0, 0, 0, 6, 0],
    [8, 0, 0, 0, 6, 0, 0, 0, 3],
    [4, 0, 0, 8, 0, 3, 0, 0, 1],
    [7, 0, 0, 0, 2, 0, 0, 0, 6],
    [0, 6, 0, 0, 0, 0, 2, 8, 0],
    [0, 0, 0, 4, 1, 9, 0, 0, 5],
    [0, 0, 0, 0, 8, 0, 0, 7, 9]
]

# Hard Puzzle
hard_puzzle = [
    [8, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 3, 6, 0, 0, 0, 0, 0],
    [0, 7, 0, 0, 9, 0, 2, 0, 0],
    [0, 5, 0, 0, 0, 7, 0, 0, 0],
    [0, 0, 0, 0, 4, 5, 7, 0, 0],
    [0, 0, 0, 1, 0, 0, 0, 3, 0],
    [0, 0, 1, 0, 0, 0, 0, 6, 8],
    [0, 0, 8, 5, 0, 0, 0, 1, 0],
    [0, 9, 0, 0, 0, 0, 4, 0, 0]
]

# Corner Case: Unsolvable Puzzle
unsolvable_puzzle = [
    [5, 1, 6, 8, 4, 9, 7, 3, 2],
    [3, 0, 7, 6, 0, 5, 0, 0, 0],
    [8, 0, 9, 7, 0, 0, 0, 6, 5],
    [1, 3, 5, 0, 6, 0, 9, 0, 7],
    [4, 7, 2, 5, 9, 1, 0, 0, 6],
    [9, 6, 8, 3, 7, 0, 0, 5, 0],
    [2, 5, 3, 1, 8, 6, 0, 7, 4],
    [6, 8, 4, 2, 0, 7, 5, 0, 0],
    [7, 9, 1, 0, 3, 4, 6, 0, 0]
]


# Testing the Solver
for index, puzzle in enumerate([easy_puzzle, hard_puzzle, unsolvable_puzzle]):
    csp = CSP(puzzle)
    print(f"Testing puzzle {index + 1}")
    print("Original Sudoku:")
    csp.print_sudoku()

    solution = csp.solve()
    if solution:
        print("\nSolved Sudoku:")
        csp.print_sudoku()
    else:
        print("No solution found.")

    print("\n" + "=" * 40 + "\n")
