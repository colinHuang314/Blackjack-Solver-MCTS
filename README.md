# Blackjack-Solver-MCTS

## Overview
This project is a Blackjack decision solver built using a customized Monte Carlo Tree Search (MCTS) framework. Given any player hand, dealer upcard, game rules, and deck state (including card count), the solver computes the optimal action and expected value (EV). When run at sufficient accuracy, results match industry-standard optimal-play tables and counting deviations. 

The solver has been validated against the Blackjack Apprenticeship strategy and deviation tables. See: https://www.blackjackapprenticeship.com/blackjack-strategy-charts/ and https://www.blackjackapprenticeship.com/wp-content/uploads/2019/07/BJA_H17.pdf

Unlike classic MCTS implementations, this solver is adapted for Blackjack’s stochastic nature, modeling card draws as chance events.

---

<details>
  <summary><strong>⚙️ How to Use</strong></summary>
  
## How to Use
Change lines **19-20** to pick a player hand and dealer card. All ten value cards are represented as 10, and aces are 11. This can run as is, but below are more settings you can change:
- line **30**:     runtime of algorithm
- lines **36-40**: game rules
- line **31**:     Feature to let the solver run until it is "confident" in its answer
- line **44**:     setting at which the confidence is measured
- line **29**:     if you want more details printed
- line **22**:     the true count at which to simulate the hand (a card counting metric)
- lines **53-68**: are not recommended to be changed

</details>

---

<details>
  <summary><strong>🔑 Key Features</strong></summary>
  
## Key Features
- Monte Carlo Tree Search–based decision engine
  - Computes optimal actions and EV for hit, stand, double, and split
  - 
- Explicit Chance Node Modeling
  - Separate nodes for:
    - Player actions
    - Random card outcomes (chance nodes)
      
- Card Counting & Deviations
  - Integrated Hi-Lo card counting system
  - Can simulate decisions at a specific running or true count
  - Includes a deviation finder to identify count thresholds where optimal decisions change
  - Minor discrepancies may arise depending on deck-remaining estimation and penetration assumptions

- Configurable Accuracy–Speed Tradeoffs
  - Adjustable runtime and iteration limits
  - Optional confidence-based early termination
  - Preset accuracy levels:
    - Low: Fast, correct for most hands
    - Medium: Correct for almost all non-splits and most splits
    - Tuned: Designed for ~99%+ accuracy across all hands

</details>

---

<details>
  <summary><strong>⚡️ Performance Characteristics</strong></summary>
  
## Performance Characteristics
- For my computer, I got:
  - ~10 million iterations in ~5 minutes = ~33,000 iterations/sec

- Typical convergence (Medium settings):
  - Most hands: 300k – 1M iterations
  - Splits: ~6× longer
  - Extremely close decisions: up to 30M iterations

- Accuracy notes:
  - Using basic strategy rollouts improves convergence speed
  - Non-tuned settings may produce incorrect decisions or incorrect EV estimates for splits

</details>

---

<details>
  <summary><strong>🤔 Example Deviations to Try</strong></summary>
  
## Example Deviations to Try
- Pair 10s vs 6
- Hard 8 vs 6
- Hard 16 vs 10
- Hard 12 vs 2
- Soft 19 vs 6

*Also you can try Soft 19 vs 6 under both H17 and S17 rules to observe rule-dependent decision changes.*

</details>

---

<details>
  <summary><strong>🧱 MCTS Design (Blackjack-Specific)</strong></summary>
  
## MCTS Design (Blackjack-Specific)
This solver doesn't use classic MCTS:
- Action nodes represent player decisions
- Chance nodes represent possible card draws
- During selection:
  - The best action node is chosen
  - A resulting state is sampled via the card probability distribution (simulateChanceNode())

- During expansion:
  - A chance node is created
  - One child node is added for each possible card outcome from the shoe

*There are likely other ways to do it, but this is how I chose to do it*

</details>

---

<details>
  <summary><strong>📒 Notes</strong></summary>
  
## Notes
Some configuration parameters have subtle interactions—tweaking settings may have non-obvious effects on convergence speed and accuracy.

Yes, the code is messy in certain parts. I'll use better Python conventions and write cleaner code in the future.

</details>
