# Learn Q-learning in C# by guiding a rat through a maze

How can a program learn which way to move when all it gets is a score after each step? In this demo, a rat starts in the corner of a small maze and learns a route to the goal by trying moves and keeping track of how well they work. This is **reinforcement learning**. The method it uses to update those estimates is called **Q-learning**.

You can follow the demo with basic C# knowledge of arrays, loops, and methods. It uses a two-dimensional array to store what it learns.

## 1. Run the demo

Install the **.NET 11 SDK**. Open a terminal in the folder that contains `RlDemo.csproj` and run:

```sh
dotnet run -c Release
```

The program trains the rat automatically and prints a report in the terminal. If the build says it cannot target .NET 11, run `dotnet --list-sdks` to check that an 11.x SDK is installed.

## 2. Understand the maze and its rules

```text
S....
.....
.###.
.....
P...G
```

The rat starts at `S` and can move up, down, left, or right through empty cells (`.`). A wall (`#`) or the edge of the board blocks the move and leaves the rat in the same cell.

| What happens on a step | Reward |
| --- | ---: |
| Reach the goal (`G`) | +1 |
| Fall into the pit (`P`) | -1 |
| Take any other step, including bumping into a wall | -0.04 |

The small cost for each extra step makes shorter trips to the goal more rewarding. One trip from the start is called an **episode**. It ends at the goal, at the pit, or after 100 steps. The program runs 1,500 episodes, putting the rat back at the start each time and keeping everything it learned.

## 3. Follow one step of learning

In reinforcement learning, the **agent** is the decision-maker and the **environment** is the world it acts in. Here the rat's current cell is its **state**, and the direction it chooses is its **action**.

Imagine the rat is at `S` and chooses Right. If that move takes it into the next cell, the maze reports the new position and a reward of -0.04. The agent uses that reward and its best estimate for the new cell to revise the value of choosing Right at `S`. If the new cell later turns out to lead toward the goal, future updates can make Right look more promising.

That estimate is a **Q-value**. The **Q-table** starts at zero and stores four estimates per cell, one per direction. Each step updates the entry for the cell and action the agent just tried.

The agent starts by choosing randomly so it can discover routes. As training progresses, it more often chooses the direction with the highest Q-value. The chance of choosing randomly is called **epsilon**. It starts at 100% and drops to 5%. Separately, the floor has a 10% chance of using a randomly selected direction instead of the chosen one. That random draw can still select the intended direction.

## 4. Read the training report

The default run produced these results:

```text
Average reward, first 100 episodes :  -1.47
Average reward, last  100 episodes :   0.67
Goal reached in 1454 / 1500 episodes (96.9%)
```

Each episode's total is the sum of its step rewards. Compare the first two lines to see how that total changed with practice. The goal count covers all training episodes, including the early ones with more random moves.

- **Learned policy** shows the direction with the highest Q-value in each playable cell. A **policy** is a rule for choosing what to do in each state.
- **Greedy path** marks the route from `S` with `*` by following those arrows without exploration or slipping. **Greedy** means choosing the highest estimated value at each step.
- **Q-values** lists the four estimates in Up, Down, Left, Right order. At the starting cell `(0,0)`, the run printed `0.34  0.38  0.35  0.40`. Right has the highest value, so the top-left arrow is `>`.

## 5. Find the learning step in the code

Start in [Program.cs](Program.cs) and find the training loop. It calls `ChooseAction` to pick a move, `world.Step` to get the outcome, and `agent.Update` to revise one Q-value. Then it uses the returned position as the state for the next step.

- [GridWorld.cs](GridWorld.cs) applies the maze rules. Read `Step` to see how it handles slipping, blocked moves, and rewards.
- [QLearningAgent.cs](QLearningAgent.cs) chooses moves and updates the table. Read `Update` to see how the agent learns from one step.

The update moves the old estimate partway toward a new estimate called the **target**:

```text
target = reward + gamma × best Q-value in the next cell
new Q-value = old Q-value + alpha × (target - old Q-value)
```

`gamma` is the **discount factor**. It makes future rewards count a little less than rewards received now. `alpha` is the **learning rate**. At the default of 0.1, one update closes 10% of the gap between the old estimate and the target. When a move reaches the goal or pit, the target is just the reward for that move because the trip is over.

## 6. Try one change and compare

In [Program.cs](Program.cs), change `slipChance` from `0.1` to `0.0` and run the same command again. Before you run it, predict what will happen to the final average reward when the maze always uses the direction the agent chooses. Compare the two reports and check whether the arrows or the traced route changed too.
