namespace RlDemo;

/// <summary>
/// A cell of the grid-world. Only <see cref="Empty"/> is walkable;
/// walls block movement, and goal/pit end the episode.
/// </summary>
public enum Cell
{
    Empty,
    Wall,
    Goal,
    Pit,
}

/// <summary>
/// The four actions the agent can take.
/// </summary>
public enum Move
{
    Up = 0,
    Down = 1,
    Left = 2,
    Right = 3,
}

/// <summary>
/// The "environment" side of the reinforcement-learning loop.
///
/// It owns the grid, the agent's current position, and the transition
/// dynamics (what happens when an action is taken). It deliberately knows
/// nothing about learning: it just answers "here is the state, here is the
/// reward you got, here is the next state".
/// </summary>
public sealed class GridWorld
{
    private readonly Cell[,] _grid;
    private readonly Random _rng;

    public GridWorld(Cell[,] grid, int startRow, int startCol, double stepReward, double slipChance, Random? rng = null)
    {
        _grid = grid;
        Rows = grid.GetLength(0);
        Cols = grid.GetLength(1);
        StartRow = startRow;
        StartCol = startCol;
        StepReward = stepReward;
        SlipChance = slipChance;
        _rng = rng ?? Random.Shared;

        Row = startRow;
        Col = startCol;
    }

    public int Rows { get; }
    public int Cols { get; }
    public int StartRow { get; }
    public int StartCol { get; }

    /// <summary>Current row of the agent.</summary>
    public int Row { get; private set; }

    /// <summary>Current column of the agent.</summary>
    public int Col { get; private set; }

    /// <summary>Reward received for every non-terminal step (keeps paths short).</summary>
    public double StepReward { get; }

    /// <summary>Probability that a random action is executed instead of the chosen one.</summary>
    public double SlipChance { get; }

    public int ActionCount => 4;

    /// <summary>Number of distinct states = one per cell (walls included, they are just never visited).</summary>
    public int StateCount => Rows * Cols;

    /// <summary>Builds a world from a readable ASCII map: S = start, G = goal, P = pit, # = wall, . = empty.</summary>
    public static GridWorld FromMap(string[] map, double stepReward, double slipChance, Random? rng = null)
    {
        int rows = map.Length;
        int cols = map[0].Length;
        var grid = new Cell[rows, cols];

        int startRow = 0, startCol = 0;
        for (int r = 0; r < rows; r++)
        {
            for (int c = 0; c < cols; c++)
            {
                grid[r, c] = map[r][c] switch
                {
                    '#' => Cell.Wall,
                    'G' => Cell.Goal,
                    'P' => Cell.Pit,
                    _ => Cell.Empty,
                };

                if (map[r][c] == 'S')
                {
                    startRow = r;
                    startCol = c;
                }
            }
        }

        return new GridWorld(grid, startRow, startCol, stepReward, slipChance, rng);
    }

    /// <summary>Puts the agent back at the start and returns the initial state.</summary>
    public int Reset()
    {
        Row = StartRow;
        Col = StartCol;
        return State(Row, Col);
    }

    /// <summary>
    /// Executes one action and returns <c>(nextState, reward, done)</c>.
    /// Walls and the map border are solid: the agent simply stays put.
    /// </summary>
    public (int NextState, double Reward, bool Done) Step(int action)
    {
        // Exploration noise in the *environment* itself (slippery floor).
        int actualAction = action;
        if (_rng.NextDouble() < SlipChance)
        {
            actualAction = _rng.Next(ActionCount);
        }

        (int dr, int dc) = actualAction switch
        {
            (int)Move.Up => (-1, 0),
            (int)Move.Down => (1, 0),
            (int)Move.Left => (0, -1),
            (int)Move.Right => (0, 1),
            _ => (0, 0),
        };

        int nextRow = Row + dr;
        int nextCol = Col + dc;

        if (!IsInside(nextRow, nextCol) || this[nextRow, nextCol] == Cell.Wall)
        {
            // Bumped into something solid: don't move.
            nextRow = Row;
            nextCol = Col;
        }

        Row = nextRow;
        Col = nextCol;

        return this[Row, Col] switch
        {
            Cell.Goal => (State(Row, Col), +1.0, true),
            Cell.Pit => (State(Row, Col), -1.0, true),
            _ => (State(Row, Col), StepReward, false),
        };
    }

    /// <summary>Row/column -> a single integer state id used as the Q-table key.</summary>
    public int State(int row, int col) => row * Cols + col;

    public Cell this[int row, int col] => _grid[row, col];

    private bool IsInside(int row, int col) => row >= 0 && row < Rows && col >= 0 && col < Cols;
}
