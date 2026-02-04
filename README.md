# BoardGameRL

[![CI](https://github.com/Algieba-dean/OinkGameRL/actions/workflows/ci.yml/badge.svg)](https://github.com/Algieba-dean/OinkGameRL/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue)
[![codecov](https://codecov.io/gh/Algieba-dean/OinkGameRL/graph/badge.svg?token=YOUR_TOKEN_HERE)](https://codecov.io/gh/Algieba-dean/OinkGameRL)
![GitHub last commit](https://img.shields.io/github/last-commit/Algieba-dean/OinkGameRL)
![GitHub repo size](https://img.shields.io/github/repo-size/Algieba-dean/OinkGameRL)

A collection of board game and card game environments for reinforcement learning, built on `gymnasium`. Includes both Oink Games (Japanese tabletop games) and traditional Chinese card games. Follows modern Python engineering practices with **uv** for dependency management, **Ruff** for linting, and **Pre-commit** for workflow safety.

## Supported Games

### Oink Games (Japanese Tabletop)

| Game           | Status      | Players | Description                                      |
| -------------- | ----------- | ------- | ------------------------------------------------ |
| **Scout**      | ✅ Implemented | 2-5     | Card game where players build sets and sequences |
| **Maskmen**    | ✅ Implemented | 2-6     | Wrestling card game with hidden roles            |
| **Kobayakawa** | ✅ Implemented | 3-6     | Minimalist betting card game                     |
| **Startups**   | ✅ Implemented | 3-7     | Investment card game                             |
| **In a Grove** | ✅ Implemented | 2-4     | Deduction card game                              |

### Chinese Card Games (中式棋牌)

| Game           | Status      | Players | Description                                      |
| -------------- | ----------- | ------- | ------------------------------------------------ |
| **Doudizhu (斗地主)** | ✅ Implemented | 3     | Classic Chinese card game with landlord vs peasants |
| **Guandan (掼蛋)**   | ✅ Implemented | 4     | Team-based card game with level progression      |
| **Mahjong (麻将)**   | ✅ Implemented | 4     | Traditional tile-based game with chi/pong/gang   |

## Prerequisites

Before you begin, ensure you have the following installed:

- **Python 3.10+** (3.12 Recommended)
- **uv** (An extremely fast Python package installer and resolver)

### Installing uv

MacOS / Linux:

```bash
curl -LsSf [https://astral.sh/uv/install.sh](https://astral.sh/uv/install.sh) | sh
```

Windows:

```powershell
powershell -c "irm [https://astral.sh/uv/install.ps1](https://astral.sh/uv/install.ps1) | iex"
```

---

## Installation & Setup

1. **Clone the repository**

   ```bash
   git clone [https://github.com/Algieba-dean/OinkGameRL.git](https://github.com/Algieba-dean/OinkGameRL.git)
   cd OinkGameRL
   ```

2. **Sync the environment**
   This command creates a virtual environment (`.venv`) and installs all dependencies (including dev tools like `pytest` and `ruff`) defined in `pyproject.toml`.

   ```bash
   uv sync --all-extras --dev
   ```

3. **Install Git Hooks (Crucial)**
   We use `pre-commit` to ensure code quality and security before every commit.

   ```bash
   uv run pre-commit install
   ```

   _(Optional) If you encounter issues with `detect-secrets`, initialize the baseline:_

   ```bash
   uv tool run detect-secrets scan > .secrets.baseline
   ```

---

## Development Workflow

We use `uv run` to execute commands within the project's virtual environment. You generally **do not** need to manually activate the venv.

### 1. Running Tests

Run all unit tests using `pytest`:

```bash
uv run pytest
```

Generate a coverage report (HTML report will be in `htmlcov/`):

```bash
uv run pytest --cov --cov-report=html
```

### 2. Linting & Formatting (Ruff)

We use **Ruff** for both linting and formatting.

Check for code issues:

```bash
uv run ruff check .
```

Auto-fix issues and format code:

```bash
uv run ruff check --fix .
uv run ruff format .
```

### 3. Pre-commit Checks

These checks run automatically when you `git commit`. You can also trigger them manually:

```bash
uv run pre-commit run --all-files
```

---

## Project Rules & Best Practices

### Git Workflow

- **Main Branch Protection:** Direct pushes to `main` (or `master`) are **blocked**.
- **Feature Branches:** Always create a new branch for your changes:

  ```bash
  git checkout -b feature/my-new-feature
  ```

- **Pull Requests:** Submit a PR to merge your changes. CI checks must pass before merging.

### Secrets Detection

We use `detect-secrets` to prevent committing API keys or passwords.

- If the hook blocks your commit due to a "false positive" (a random string that looks like a secret), you can update the baseline:

  ```bash
  uv run detect-secrets scan --update .secrets.baseline
  git add .secrets.baseline
  ```

### Code Style

- **Type Hints:** Use type hints for function arguments and return values.
- **Imports:** Ruff handles import sorting automatically.
- **Testing:** New features must include unit tests.

---

## Project Structure

```text
BoardGameRL/
├── games/                      # Source code for environments and agents
│   ├── board_game.py           # Abstract base class for all game environments
│   ├── game_agent.py           # Abstract base class for AI agents
│   ├── registry.py             # Game registry for dynamic game loading
│   ├── auto_opponent_wrapper.py # Wrapper for multi-agent environments
│   │
│   ├── core/                   # Shared utilities for RL training
│   │   ├── base_player.py      # Generic base class for player management
│   │   ├── reward_shaping.py   # Reward utilities (ranking, relative scores)
│   │   └── observation_space.py # Observation space builder
│   │
│   │── # Oink Games
│   ├── scout/                  # Scout game implementation
│   ├── maskmen/                # Maskmen game implementation
│   ├── kobayakawa/             # Kobayakawa game implementation
│   ├── startups/               # Startups game implementation
│   ├── in_a_grove/             # In a Grove game implementation
│   │
│   │── # Chinese Card Games
│   ├── doudizhu/               # Doudizhu (斗地主) implementation
│   ├── guandan/                # Guandan (掼蛋) implementation
│   └── mahjong/                # Mahjong (麻将) implementation
│
├── tests/                      # Pytest test suite (mirrors games/ structure)
├── .github/                    # GitHub Actions CI configuration
├── pyproject.toml              # Project configuration & dependencies
├── uv.lock                     # Dependency lock file (DO NOT EDIT MANUALLY)
├── .pre-commit-config.yaml     # Git hooks configuration
└── README.md                   # This file
```

## Quick Start

### Using the Game Registry

```python
from games.registry import make_env, list_games

# List all available games
print(list_games())
# ['scout', 'kobayakawa', 'maskmen', 'startups', 'in_a_grove', 'doudizhu', 'guandan', 'mahjong']

# Create any game environment
env = make_env("doudizhu", render_mode="ansi")
obs, info = env.reset(seed=42)
```

### Playing a Game

```python
from games.doudizhu.doudizhu_game_env import DoudizhuGameEnv

# Create environment
env = DoudizhuGameEnv(render_mode="human")

# Reset and play
obs, info = env.reset(seed=42)
action_mask = info["action_mask"]

# Take a valid action
valid_actions = [i for i, v in enumerate(action_mask) if v == 1]
obs, reward, terminated, truncated, info = env.step(valid_actions[0])

env.render()
```

### Environment Interface

All game environments inherit from `BoardGameEnv` and provide:

```python
# Properties
env.num_players      # Number of players
env.current_player_idx  # Current player's turn
env.max_steps        # Maximum steps before truncation (optional)
env.current_step     # Current step count

# Methods
obs, info = env.reset(seed=42)  # Reset game
obs, reward, done, truncated, info = env.step(action)  # Take action
env.render()  # Display game state

# Info dict contains:
info["action_mask"]   # Valid actions for current player
info["global_state"]  # Full game state (for debugging/analysis)
```

### RL Training Utilities

The `games.core` module provides utilities for professional RL training:

#### Truncation Support (Preventing Infinite Loops)

```python
# Create environment with max_steps to prevent infinite episodes
env = make_env("scout", max_steps=1000)
obs, info = env.reset()

# Game will truncate after 1000 steps if not terminated naturally
obs, reward, terminated, truncated, info = env.step(action)
if truncated:
    print("Episode truncated due to max_steps")
```

#### Reward Shaping

```python
from games.core import RewardShaping

# Ranking-based rewards (zero-sum, good for self-play)
rewards = RewardShaping.ranking_reward(num_players=4, winner_idx=0)
# [0.75, -0.25, -0.25, -0.25]

# Relative score rewards (encourages maximizing gap vs opponents)
scores = [100, 80, 60, 40]
rewards = RewardShaping.relative_score_reward(scores, normalize=True)

# Simple win/lose rewards
rewards = RewardShaping.win_lose_reward(num_players=3, winner_idx=1)
# [-1.0, 1.0, -1.0]
```

#### Observation Space Builder

```python
from games.core import ObservationSpaceBuilder

# Build structured observation space (no magic numbers!)
builder = (
    ObservationSpaceBuilder()
    .add_box("hand", shape=(MAX_HAND_SIZE, 2), low=0, high=1)
    .add_box("board", shape=(MAX_BOARD_SIZE, 2), low=0, high=1)
    .add_discrete("current_player", n=NUM_PLAYERS)
)

# Get flattened space for RL algorithms
observation_space = builder.get_flat_space()

# Flatten/unflatten observations
flat_obs = builder.flatten(obs_dict)
obs_dict = builder.unflatten(flat_obs)
```

---

## Troubleshooting

**Q: `pre-commit` failed with formatting errors.**
A: Ruff likely auto-fixed your files. Just `git add` the modified files and try `git commit` again.

**Q: CI failed on GitHub but passes locally.**
A: Ensure you have run `uv sync` locally to match the lock file. Check if you forgot to add new files to git.

**Q: How do I add a new library?**
A: Use `uv add <package_name>`. For dev tools (like testing libraries), use `uv add --dev <package_name>`.
