"""Generate Pre-Lecture Assignment 4 answer PDF — fixed Unicode & layout."""
import io
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer,
                                 Image as RLImage, HRFlowable)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_LEFT, TA_CENTER
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

# ── Register Unicode-capable fonts ───────────────────────────────────────────
FONT_DIR = "/System/Library/Fonts/Supplemental/"
pdfmetrics.registerFont(TTFont("Arial",      FONT_DIR + "Arial.ttf"))
pdfmetrics.registerFont(TTFont("Arial-Bold", FONT_DIR + "Arial Bold.ttf"))
pdfmetrics.registerFont(TTFont("Arial-Ital", FONT_DIR + "Arial Italic.ttf"))

W, H = A4

# ── Helpers ──────────────────────────────────────────────────────────────────

def math_img(latex_str, fontsize=9.5, dpi=200, width_cm=15.5, height_cm=0.75):
    """Render LaTeX math via matplotlib mathtext → PNG → RLImage."""
    fig, ax = plt.subplots(figsize=(width_cm / 2.54, height_cm / 2.54))
    fig.patch.set_alpha(0)
    ax.set_axis_off()
    ax.text(0.5, 0.5, latex_str, transform=ax.transAxes,
            ha="center", va="center", fontsize=fontsize, color="black",
            usetex=False)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight",
                pad_inches=0.03, transparent=True)
    plt.close(fig)
    buf.seek(0)
    return RLImage(buf, width=width_cm * cm, height=height_cm * cm)


def build_pdf(path):
    doc = SimpleDocTemplate(
        path, pagesize=A4,
        leftMargin=1.8*cm, rightMargin=1.8*cm,
        topMargin=1.4*cm, bottomMargin=1.4*cm,
    )

    # ── Styles ────────────────────────────────────────────────────────────────
    title_s = ParagraphStyle("T", fontName="Arial-Bold",  fontSize=11.5,
                              spaceBefore=0, spaceAfter=2,  alignment=TA_CENTER)
    sub_s   = ParagraphStyle("S", fontName="Arial",        fontSize=9,
                              spaceBefore=0, spaceAfter=3,  alignment=TA_CENTER)
    q_s     = ParagraphStyle("Q", fontName="Arial-Bold",   fontSize=10,
                              spaceBefore=5, spaceAfter=2)
    body_s  = ParagraphStyle("B", fontName="Arial",        fontSize=9,
                              leading=13.5, spaceAfter=3)
    bul_s   = ParagraphStyle("BU", fontName="Arial",       fontSize=9,
                              leading=13, leftIndent=12, spaceAfter=2)

    def bold(t):  return f'<font name="Arial-Bold">{t}</font>'
    def ital(t):  return f'<font name="Arial-Ital">{t}</font>'

    story = []

    # ── Title ─────────────────────────────────────────────────────────────────
    story.append(Paragraph("Pre-Lecture Assignment 4 — Value-Based RL", title_s))
    story.append(Paragraph("535510 Reinforcement Learning  |  NYCU CS", sub_s))
    story.append(HRFlowable(width="100%", thickness=0.7, color=colors.grey, spaceAfter=4))

    # ══════════════════════════════════════════════════════════════════════════
    # Q1
    # ══════════════════════════════════════════════════════════════════════════
    story.append(Paragraph("1)  Q-learning: Derivation and Interpretations", q_s))

    story.append(Paragraph("The Q-learning update rule is:", body_s))
    story.append(math_img(
        r"$Q(s_t,a_t)\;\leftarrow\;Q(s_t,a_t)"
        r"+\alpha(s_t,a_t)\!\left[r_t+\gamma\max_{a'}Q(s_{t+1},a')-Q(s_t,a_t)\right]$",
        fontsize=10))

    story.append(Paragraph(
        bold("Interpretation 1 — Approximate Q-Value Iteration (Q-VI). ") +
        "In the model-known setting, Q-VI iterates "
        "Q(s,a) ← R + γ Σ P(s'|s,a) max Q(s',a'). "
        "Since P and R are unknown, we approximate: the immediate reward R by the "
        "sampled r_t, and the expectation over transitions by the single observed s_{t+1}. "
        "A step-size α then interpolates between the old estimate and the new sample target, "
        "yielding the Q-learning update exactly.",
        body_s))

    story.append(Paragraph(
        bold("Interpretation 2 — Stochastic correction of the Bellman (TD) error. ") +
        "Rewrite the update as Q(s_t,a_t) ← Q(s_t,a_t) + α·δ_t, where "
        "δ_t = r_t + γ max Q(s_{t+1},a') − Q(s_t,a_t) is the "
        + ital("temporal-difference error") + " — "
        "how far the current Q deviates from the Bellman optimality equation. "
        "Q-learning performs one stochastic gradient step to reduce this error, "
        "analogous to stochastic gradient descent on the squared Bellman error.",
        body_s))

    story.append(Paragraph(
        bold("Why is Q-learning off-policy? ") +
        "The " + ital("target policy") + " being learned is always greedy: "
        "π(s) = argmax_a Q(s,a) — reflected by the max operator in the update — "
        "while the " + ital("behaviour policy") + " collecting data can be "
        "any exploratory policy (e.g. ε-greedy). Because the policy being evaluated "
        "differs from the policy generating experience, Q-learning is " + ital("off-policy") + ".",
        body_s))

    story.append(HRFlowable(width="100%", thickness=0.4, color=colors.lightgrey, spaceAfter=3))

    # ══════════════════════════════════════════════════════════════════════════
    # Q2
    # ══════════════════════════════════════════════════════════════════════════
    story.append(Paragraph("2)  DQN Loss Function: Components and Design Principle", q_s))

    story.append(math_img(
        r"$L(w)\;:=\;\mathbb{E}_{(s,a,r,s')\sim\rho}"
        r"\!\left[\,\frac{1}{2}"
        r"\!\left(\,r+\gamma\max_{a'\in\mathcal{A}}\hat{Q}(s',a';\hat{w})"
        r"\;-\;\hat{Q}(s,a;\,w)\,\right)^{2}\right]$",
        fontsize=10))

    story.append(Paragraph(bold("Component meanings:"), body_s))
    story.append(Paragraph(
        "• " + bold("Q̂(s,a; w)") + " — the " + ital("online Q-network") +
        " with trainable weights w; predicts the current action-value for (s, a).",
        bul_s))
    story.append(Paragraph(
        "• " + bold("r + γ max Q̂(s',a'; ŵ)") + " — the " + ital("TD target") +
        ", a one-step Bellman backup used as the regression label for the online network.",
        bul_s))
    story.append(Paragraph(
        "• " + bold("ŵ (target network)") +
        " — a periodically-copied, frozen snapshot of w. "
        "Holding ŵ fixed between updates prevents the regression target from moving "
        "every step, which would destabilise training.",
        bul_s))
    story.append(Paragraph(
        "• " + bold("(s,a,r,s') ~ ρ (experience replay)") +
        " — transitions sampled i.i.d. from a replay buffer ρ, "
        "breaking temporal correlations and allowing data reuse.",
        bul_s))
    story.append(Paragraph(
        "• " + bold("½(·)²") +
        " — squared TD error; minimising it drives Q̂(s,a;w) toward the TD target.",
        bul_s))

    story.append(Paragraph(
        bold("Design principle. ") +
        "The loss treats Q-learning as " + ital("bootstrapped regression") + ": "
        "the TD target acts as a (temporarily fixed) label and the online network is "
        "fitted to it. Two stabilisation mechanisms address the divergence problem of "
        "vanilla Q-learning with function approximation: "
        "(1) " + bold("experience replay") + " decorrelates samples and reuses data efficiently; "
        "(2) the " + bold("target network") + " provides stable regression targets, "
        "preventing the feedback loop that causes divergence.",
        body_s))

    story.append(HRFlowable(width="100%", thickness=0.4, color=colors.lightgrey, spaceAfter=3))

    # ══════════════════════════════════════════════════════════════════════════
    # Q3
    # ══════════════════════════════════════════════════════════════════════════
    story.append(Paragraph("3)  Noisy Net (Rainbow Paper)", q_s))

    story.append(Paragraph(
        "Standard DQN explores with ε-greedy, which applies uniform random perturbations "
        "and is difficult to tune, especially in sparse-reward settings "
        "(e.g. Montezuma's Revenge). "
        + bold("Noisy Nets") + " (Fortunato et al., 2017) replace each linear layer "
        "y = b + Wx with a " + ital("noisy linear layer") + ":",
        body_s))

    story.append(math_img(
        r"$y\;=\;(b+Wx)"
        r"\;+\;"
        r"(b_{\mathrm{noisy}}\odot\varepsilon^b"
        r"+(W_{\mathrm{noisy}}\odot\varepsilon^w)\,x)$",
        fontsize=10))

    story.append(Paragraph(
        "where ε^b and ε^w are random noise variables sampled per forward pass, "
        "and ⊙ is the element-wise product. "
        "The noise magnitude parameters (W_noisy, b_noisy) are " + ital("learnable") + ", "
        "so the network adaptively reduces noise where it is already confident — "
        "a form of " + bold("self-annealing") + ". "
        "This enables " + bold("state-conditional, parametric exploration") + " "
        "without a manually scheduled ε. "
        "In Rainbow, all linear layers are replaced by noisy layers and ε is set to 0, "
        "with factorised Gaussian noise used to reduce the number of independent noise variables.",
        body_s))

    doc.build(story)
    print(f"PDF written to {path}")


if __name__ == "__main__":
    build_pdf("/Users/jacky/nycu/senior/RL/pre-lecture/hw4/hw4_answers.pdf")
