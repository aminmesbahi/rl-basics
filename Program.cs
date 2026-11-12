using RlDemo;

// ---------------------------------------------------------------------------
// A tiny reinforcement-learning demo: Q-learning on a grid-world.
//
// The agent (a "rat") starts at S and must learn, purely from trial and
// error, to reach the goal G while avoiding the pit P and the walls. It gets
// +1 for the goal, -1 for the pit, and a small -0.04 penalty per step so it
// prefers short paths.
//
// There is no supervision: nobody tells it which way to go. It just explores,
// receives rewards, and gradually credits good actions. Run and watch it learn.
// ---------------------------------------------------------------------------

// Each character is one cell: S = start, G = goal, P = pit, # = wall, . = empty.
string[] map =
[
    "S....",   // Start top-left
    ".....",
    ".###.",   // Wall segment the agent must go around
    ".....",
    "P...G",   // Pit bottom-left, goal bottom-right
];

const int episodes = 1500;
const int maxStepsPerEpisode = 100;
const double stepReward = -0.04;
const double slipChance = 0.1;    // 10% of moves the floor makes the move random

const double alpha = 0.1;        // learning rate
const double gamma = 0.95;        // discount factor
const double epsilonStart = 1.0;  // start 100% exploratory...
const double epsilonMin = 0.05;   // ...and settle at 5%

var rng = new Random(42); // fixed seed so results are reproducible

var world = GridWorld.FromMap(map, stepReward, slipChance, rng);
var agent = new QLearningAgent(world.StateCount, world.ActionCount, alpha, gamma, epsilonStart, rng);

// --- Training loop -------------------------------------------------------
var rewards = new double[episodes];
int successes = 0;

for (int episode = 0; episode < episodes; episode++)
{
    int state = world.Reset();
    double total = 0.0;
    bool reachedGoal = false;

    for (int step = 0; step < maxStepsPerEpisode; step++)
    {
        int action = agent.ChooseAction(state);
        (int nextState, double reward, bool done) = world.Step(action);

        agent.Update(state, action, nextState, reward, done); // the learning step

        total += reward;
        state = nextState;

        if (done)
        {
            reachedGoal = world[world.Row, world.Col] == Cell.Goal;
            break;
        }
    }

    rewards[episode] = total;
    if (reachedGoal)
    {
        successes++;
    }

    // Shrink exploration over time: explore a lot early, exploit mostly later.
    agent.Epsilon = Math.Max(epsilonMin, agent.Epsilon * 0.997);
}

// --- Report --------------------------------------------------------------
double Avg(IEnumerable<double> xs) => xs.Average();
double averageFirst = Avg(rewards.Take(100));
double averageLast = Avg(rewards.TakeLast(100));

Console.WriteLine("=== Q-learning on a grid-world ===");
Console.WriteLine($"Episodes: {episodes}, Alpha: {alpha}, Gamma: {gamma}, Slip: {slipChance}");
Console.WriteLine();
Console.WriteLine($"Average reward, first 100 episodes : {averageFirst,6:F2}");
Console.WriteLine($"Average reward, last  100 episodes : {averageLast,6:F2}");
Console.WriteLine($"Goal reached in {successes} / {episodes} episodes ({100.0 * successes / episodes:F1}%)");
Console.WriteLine();
Console.WriteLine("Learned policy (greedy action in each cell):");
Console.WriteLine();
PrintPolicy(map, world, agent);
Console.WriteLine();
Console.WriteLine("Greedy path from start:");
Console.WriteLine();
PrintPath(map, world, agent);
Console.WriteLine();
Console.WriteLine("Q-values for each non-wall cell (Up / Down / Left / Right):");
Console.WriteLine();
PrintQTable(map, world, agent);

// ---------------------------------------------------------------------------
static char Arrow(int action) => action switch
{
    (int)Move.Up => '^',
    (int)Move.Down => 'v',
    (int)Move.Left => '<',
    _ => '>',
};

static char CellChar(string[] map, int r, int c)
{
    char ch = map[r][c];
    return ch == 'S' || ch == '.' ? '.' : ch;
}

static void PrintPolicy(string[] map, GridWorld world, QLearningAgent agent)
{
    for (int r = 0; r < world.Rows; r++)
    {
        for (int c = 0; c < world.Cols; c++)
        {
            char cell = CellChar(map, r, c);
            if (cell is '#' or 'G' or 'P')
            {
                Console.Write($" {cell} ");
                continue;
            }

            int state = world.State(r, c);
            Console.Write($" {Arrow(agent.BestAction(state))} ");
        }

        Console.WriteLine();
    }
}

static void PrintPath(string[] map, GridWorld world, QLearningAgent agent)
{
    // Follow the greedy policy from the start until goal, or a step/detour cap.
    var path = new bool[world.Rows, world.Cols];
    int state = world.State(world.StartRow, world.StartCol);
    (int row, int col) = (world.StartRow, world.StartCol);

    int visited = 0;
    while (visited++ < world.Rows * world.Cols * 2)
    {
        path[row, col] = true;
        if (world[row, col] == Cell.Goal)
        {
            break;
        }

        int action = agent.BestAction(state);
        (int dr, int dc) = action switch
        {
            (int)Move.Up => (-1, 0),
            (int)Move.Down => (1, 0),
            (int)Move.Left => (0, -1),
            _ => (0, 1),
        };

        int nr = row + dr;
        int nc = col + dc;
        if (nr < 0 || nr >= world.Rows || nc < 0 || nc >= world.Cols || world[nr, nc] == Cell.Wall)
        {
            break; // policy pointed into a wall/border — stop tracing
        }

        (row, col) = (nr, nc);
        state = world.State(row, col);
    }

    for (int r = 0; r < world.Rows; r++)
    {
        for (int c = 0; c < world.Cols; c++)
        {
            if (r == world.StartRow && c == world.StartCol)
            {
                Console.Write(" S ");
            }
            else if (world[r, c] == Cell.Goal)
            {
                Console.Write(" G ");
            }
            else if (world[r, c] == Cell.Pit)
            {
                Console.Write(" P ");
            }
            else if (world[r, c] == Cell.Wall)
            {
                Console.Write(" # ");
            }
            else
            {
                Console.Write(path[r, c] ? " * " : " . ");
            }
        }

        Console.WriteLine();
    }
}

static void PrintQTable(string[] map, GridWorld world, QLearningAgent agent)
{
    for (int r = 0; r < world.Rows; r++)
    {
        for (int c = 0; c < world.Cols; c++)
        {
            if (world[r, c] == Cell.Wall)
            {
                continue;
            }

            int state = world.State(r, c);
            Console.Write(
                $"  ({r},{c}) = " +
                $"{agent.GetQ(state, (int)Move.Up),6:F2}  " +
                $"{agent.GetQ(state, (int)Move.Down),6:F2}  " +
                $"{agent.GetQ(state, (int)Move.Left),6:F2}  " +
                $"{agent.GetQ(state, (int)Move.Right),6:F2}");
            Console.WriteLine();
        }
    }
}
