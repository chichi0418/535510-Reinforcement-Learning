# Pre-Lecture Assignment 3 — Spec

**Course:** 535510 Reinforcement Learning  
**Instructor:** Ping-Chun Hsieh (pinghsieh@nycu.edu.tw), Dept. of CS, NYCU  
**Due:** 9pm, 4/14 (Tuesday)

---

## Topic: DPG, DDPG, and Off-Policy Learning

**Background:** Lecture 11 covers the motivation for deterministic policies. Lecture 12 covers DPG in detail, including the concept of off-policy learning. This assignment prepares you for Lecture 12.

**References:**
- Slides of Lectures 11–12
- DPG paper: https://proceedings.mlr.press/v32/silver14.pdf

---

## Questions

### Q1 — Deterministic Policy Gradient (DPG)

Given the DPG expression:

$$\nabla_\theta V^{\pi_\theta}(\mu) = \frac{1}{1-\gamma} \mathbb{E}_{s \sim d^\pi_\mu} \left[ \nabla_\theta \pi_\theta(s) \, \nabla_a Q^{\pi_\theta}(s,a) \big|_{a=\pi_\theta(s)} \right]$$

- Describe each term in the expression and the intuition behind it.
- What is the difference between DPG and (P3) of Stochastic PG?

### Q2 — Off-Policy Learning

- Describe the concept of **off-policy learning** in 1–2 sentences.
- Provide a specific real-world example of off-policy learning and explain in detail what makes it off-policy.

### Q3 — Use Cases of DPG

- Why is DPG suitable for solving RL problems with **continuous action spaces**?

---

## Submission Guidelines

| Item | Requirement |
|------|-------------|
| Format | Single PDF file (handwritten or typeset; photos/scans acceptable) |
| Length | **1 page max** (A4). Responses beyond 1 page will **not** be graded. |
| Platform | Submit via **E3** |
| Deadline | **9pm, April 14 (Tuesday)** |
