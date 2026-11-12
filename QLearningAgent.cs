namespace RlDemo;

/// <summary>
/// The "agent" side: a tabular Q-learning controller.
///
/// Q-learning keeps one number per (state, action) pair — the "Q-value" — that
/// estimates how much future reward the agent expects if it takes that action
/// in that state and then behaves greedily afterwards. Training repeatedly
/// nudges Q-values toward the observed reward plus the best next value:
///
///     Q(s, a) += alpha * [ r + gamma * max_a' Q(s', a')  -  Q(s, a) ]
///
/// </summary>
public sealed class QLearningAgent
{
    private readonly double[,] _q;   // Q-table: [state, action]
    private readonly Random _rng;

    public QLearningAgent(int stateCount, int actionCount, double alpha, double gamma, double epsilon, Random? rng = null)
    {
        StateCount = stateCount;
        ActionCount = actionCount;
        Alpha = alpha;
        Gamma = gamma;
        Epsilon = epsilon;
        _q = new double[stateCount, actionCount];
        _rng = rng ?? Random.Shared;
    }

    public int StateCount { get; }
    public int ActionCount { get; }

    /// <summary>Learning rate: how much each update moves the Q-value.</summary>
    public double Alpha { get; }

    /// <summary>Discount factor: weight placed on future vs immediate reward.</summary>
    public double Gamma { get; }

    /// <summary>Exploration rate: probability of a random (exploratory) action.</summary>
    public double Epsilon { get; set; }

    /// <summary>Epsilon-greedy: explore with probability Epsilon, otherwise exploit.</summary>
    public int ChooseAction(int state)
    {
        if (_rng.NextDouble() < Epsilon)
        {
            return _rng.Next(ActionCount);
        }

        return BestAction(state);
    }

    /// <summary>Greedy action for a state (the one with the highest Q-value).</summary>
    public int BestAction(int state)
    {
        int best = 0;
        for (int a = 1; a < ActionCount; a++)
        {
            if (_q[state, a] > _q[state, best])
            {
                best = a;
            }
        }

        return best;
    }

    /// <summary>The Q-learning update rule (see the class summary).</summary>
    public void Update(int state, int action, int nextState, double reward, bool done)
    {
        double target = reward + (done ? 0.0 : Gamma * MaxValue(nextState));
        double tdError = target - _q[state, action];   // "temporal-difference" error
        _q[state, action] += Alpha * tdError;
    }

    public double MaxValue(int state)
    {
        double max = _q[state, 0];
        for (int a = 1; a < ActionCount; a++)
        {
            if (_q[state, a] > max)
            {
                max = _q[state, a];
            }
        }

        return max;
    }

    public double GetQ(int state, int action) => _q[state, action];
}
