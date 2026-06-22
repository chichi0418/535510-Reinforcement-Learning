# 535510 Reinforcement Learning
## Pre-Lecture Assignment 4

**Instructor:** Ping-Chun Hsieh
**Email:** pinghsieh@nycu.edu.tw
Department of CS, NYCU

---

## Problem: Value-Based RL

In Lectures 17-19, we proceed to introduce value-based RL methods. Let's try to get familiar with these methods and get prepared for Lectures 18-19. You can refer to the (1) attached reference slides on Value-based RL and the (2) Rainbow paper (https://arxiv.org/abs/1710.02298) as a starting point when answering the questions below.

**1) Q-learning:** Can you explain how the Q-learning update is derived? Please provide at least 2 different interpretations.

$$Q(s_t, a_t) \leftarrow Q(s_t, a_t) + \alpha(s_t, a_t) \left[ r_t + \gamma \max_{a'} Q(s_{t+1}, a') - Q(s_t, a_t) \right]$$

Can you explain why Q-learning is "off-policy"?

**2) Deep Q Network (DQN):** The standard DQN loss function is as follows. Explain the meanings of each component in the loss. Can you explain the principle behind this design?

$$L(w) := \mathbb{E}_{(s,a,r,s') \sim \rho} \left[ \frac{1}{2} \left( r + \gamma \max_{a' \in \mathcal{A}} \hat{Q}(s', a'; \hat{w}) - \hat{Q}(s, a; w) \right)^2 \right]$$

**3) Practical Implementation of DQN:** What is "Noisy Net" in the Rainbow paper?

---

## Submission Guidelines and Remarks

- Please write all your responses within 1 page (A4 paper, either hand-written or typeset). The responses that go beyond 1 page will not be graded.
- Your deliverable shall be a PDF file. Please put all your write-ups into one PDF file (photos/scanned copies are acceptable) and submit your deliverable via E3.
- Please submit it by 9pm, 5/12 (Tuesday).
