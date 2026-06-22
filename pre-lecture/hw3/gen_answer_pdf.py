import argparse
import os
import tempfile

# Keep matplotlib/font cache in a writable location to avoid runtime warnings.
if "MPLCONFIGDIR" not in os.environ:
    mpl_cfg = os.path.join(tempfile.gettempdir(), "mplconfig")
    os.makedirs(mpl_cfg, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = mpl_cfg
if "XDG_CACHE_HOME" not in os.environ:
    xdg_cache = os.path.join(tempfile.gettempdir(), "xdg-cache")
    os.makedirs(xdg_cache, exist_ok=True)
    os.environ["XDG_CACHE_HOME"] = xdg_cache

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, HRFlowable
)
from reportlab.lib.enums import TA_LEFT, TA_CENTER

W, H = A4

def math_img(latex, fontsize=11, dpi=180, scale=1):
    fig, ax = plt.subplots(figsize=(0.01, 0.01))
    fig.patch.set_alpha(0)
    ax.set_axis_off()
    t = ax.text(1, 1, f"${latex}$", fontsize=fontsize,
                ha='center', va='center', transform=ax.transAxes)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    bb = t.get_window_extent(renderer=renderer)
    pad = 4
    fig.set_size_inches((bb.width + pad) / dpi, (bb.height + pad) / dpi)
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=dpi, bbox_inches='tight',
                transparent=True, pad_inches=0.02)
    plt.close(fig)
    buf.seek(0)
    img = Image(buf)
    img.drawWidth  = img.imageWidth  * scale * (72 / dpi)
    img.drawHeight = img.imageHeight * scale * (72 / dpi)
    return img

def build(out):
    doc = SimpleDocTemplate(
        out, pagesize=A4,
        leftMargin=18*mm, rightMargin=18*mm,
        topMargin=13*mm, bottomMargin=11*mm,
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle('T', parent=styles['Normal'],
                                 fontSize=11, leading=14,
                                 fontName='Helvetica-Bold', alignment=TA_CENTER)
    hd = ParagraphStyle('H', parent=styles['Normal'],
                        fontSize=9.5, leading=12, fontName='Helvetica-Bold',
                        spaceBefore=2, spaceAfter=1)
    body = ParagraphStyle('B', parent=styles['Normal'],
                          fontSize=8.2, leading=11.5, fontName='Helvetica',
                          spaceAfter=2)
    small = ParagraphStyle('S', parent=styles['Normal'],
                           fontSize=7.8, leading=10.8, fontName='Helvetica',
                           spaceAfter=1)

    story = []

    # ── Title ──
    story.append(Paragraph("Pre-Lecture Assignment 3 — DPG, DDPG, and Off-Policy Learning", title_style))
    story.append(HRFlowable(width="100%", thickness=0.8, color=colors.black, spaceAfter=3))

    # ── Q1 ──
    story.append(Paragraph("Q1: Deterministic Policy Gradient (DPG)", hd))

    dpg_latex = (r"\nabla_\theta V^{\pi_\theta}(\mu)=\frac{1}{1-\gamma}"
                 r"\mathbb{E}_{s\sim d^{\pi_\theta}_\mu}"
                 r"\left[\nabla_\theta\pi_\theta(s)\,\nabla_a Q^{\pi_\theta}(s,a)|_{a=\pi_\theta(s)}\right]")
    story.append(math_img(dpg_latex, fontsize=10))
    story.append(Spacer(1, 1*mm))

    story.append(Paragraph("<b>Term-by-term description:</b>", body))
    terms = [
        (r"\nabla_\theta V^{\pi_\theta}(\mu)",
         "Gradient of the expected total discounted return w.r.t. policy parameters θ, averaged over the initial state distribution μ. This is what we want to maximize."),
        (r"\frac{1}{1-\gamma}",
         "Normalization constant arising from summing γ^t over t=0,1,2,… It converts the unnormalized discounted state visitation measure into a proper probability distribution."),
        (r"d^{\pi_\theta}_\mu",
         "Normalized discounted state visitation distribution under the current deterministic policy π_θ — how often (in discounted time) each state s is visited when starting from μ."),
        (r"\nabla_\theta\pi_\theta(s)",
         "Jacobian of the deterministic policy output w.r.t. θ. Since the policy maps s→a directly (no sampling), this tells us how the action changes as we adjust each parameter."),
        (r"\nabla_a Q^{\pi_\theta}(s,a)|_{a=\pi_\theta(s)}",
         "Gradient of the action-value function Q w.r.t. the action a, evaluated at the action currently chosen by the policy. Points in the direction in action space that most increases long-run value."),
    ]
    for tex, desc in terms:
        img = math_img(tex, fontsize=9.5)
        row = Table([[img, Paragraph(desc, small)]], colWidths=[None, 118*mm])
        row.setStyle(TableStyle([
            ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
            ('LEFTPADDING', (0,0), (-1,-1), 0),
            ('RIGHTPADDING', (0,0), (-1,-1), 2),
            ('TOPPADDING', (0,0), (-1,-1), 0),
            ('BOTTOMPADDING', (0,0), (-1,-1), 2),
        ]))
        story.append(row)

    story.append(Spacer(1, 1*mm))
    story.append(Paragraph(
        "<b>Intuition:</b> To improve the policy, at every visited state we apply the chain rule "
        "across two gradients: first, ∇<sub rise='2' size='7'>a</sub>Q identifies which direction "
        "in action space increases the Q-value; second, ∇<sub rise='2' size='7'>θ</sub>π<sub rise='2' size='7'>θ</sub> "
        "tells us how to shift the parameters θ so that the policy outputs an action closer to that "
        "better direction. Crucially, because the policy is deterministic, there is no integral over "
        "actions — the update is a pure closed-form chain rule, not a Monte Carlo estimate over actions.",
        body))

    story.append(Paragraph("<b>Difference from Stochastic PG (P3):</b>", body))

    spg_latex = r"\nabla_\theta J(\pi_\theta)=\mathbb{E}_{s\sim\rho^\pi,\,a\sim\pi_\theta}\left[\nabla_\theta\log\pi_\theta(a|s)\,Q^\pi(s,a)\right]"
    story.append(math_img(spg_latex, fontsize=9.5))
    story.append(Spacer(1, 1*mm))

    tdata = [
        [Paragraph("<b></b>", small), Paragraph("<b>Stochastic PG (P3)</b>", small), Paragraph("<b>DPG</b>", small)],
        [Paragraph("Expectation over", small), Paragraph("States <b>and actions</b>", small), Paragraph("States only", small)],
        [Paragraph("Gradient form", small), Paragraph("Score function  ∇log π(a|s)", small), Paragraph("Chain rule ∇<sub rise='2' size='6'>θ</sub>π · ∇<sub rise='2' size='6'>a</sub>Q", small)],
        [Paragraph("Action sampling", small), Paragraph("Required (must sample a~π)", small), Paragraph("Not needed (closed form)", small)],
        [Paragraph("Exploration", small), Paragraph("Inherent via stochastic policy", small), Paragraph("Requires separate behaviour policy β", small)],
        [Paragraph("Scalability", small), Paragraph("Degrades in high-dim action spaces", small), Paragraph("Efficient regardless of action dim", small)],
    ]
    col_w = [33*mm, 60*mm, 60*mm]
    t = Table(tdata, colWidths=col_w)
    t.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#DDDDDD')),
        ('GRID', (0,0), (-1,-1), 1, colors.grey),
        ('FONTSIZE', (0,0), (-1,-1), 8),
        ('TOPPADDING', (0,0), (-1,-1), 2),
        ('BOTTOMPADDING', (0,0), (-1,-1), 2),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
    ]))
    story.append(t)

    story.append(Paragraph(
        "Key insight: Stochastic PG integrates over both state and action space — variance "
        "grows with action dimensionality. DPG sidesteps the action integral entirely, computing "
        "the gradient deterministically through a single policy output.", body))

    # ── Q2 ──
    story.append(HRFlowable(width="100%", thickness=1, color=colors.grey, spaceBefore=1, spaceAfter=1))
    story.append(Paragraph("Q2: Off-Policy Learning", hd))
    story.append(Paragraph(
        "<b>Definition:</b> Off-policy learning is a paradigm in which an agent learns about and "
        "optimizes a <b>target policy</b> π (the policy we ultimately want) while generating "
        "experience by following a different <b>behaviour policy</b> β. This decouples data "
        "collection from policy improvement, enabling the reuse of historical transitions, "
        "human demonstrations, or exploratory data that the target policy itself never produced.",
        body))

    story.append(Paragraph(
        "<b>Example — Robot arm with human teleoperator:</b> "
        "A robot needs to learn a precise, deterministic grasping policy μ<sub rise='2' size='6'>θ</sub>(s). "
        "Because a purely deterministic policy cannot explore on its own, a human teleoperator "
        "(the behaviour policy β) drives the arm across a wide range of configurations, "
        "generating a dataset of transitions (s, a, r, s′) with broad state-space coverage. "
        "An off-policy DPG algorithm (e.g., COPDAC-Q) then updates μ<sub rise='2' size='6'>θ</sub> "
        "using the off-policy gradient:",
        body))

    offpol_latex = (r"\nabla_\theta J_\beta(\mu_\theta)="
                    r"\mathbb{E}_{s\sim\rho^\beta}"
                    r"\left[\nabla_\theta\mu_\theta(s)\,\nabla_a Q^\mu(s,a)|_{a=\mu_\theta(s)}\right]")
    story.append(math_img(offpol_latex, fontsize=9.5))
    story.append(Spacer(1, 1*mm))

    story.append(Paragraph(
        "The transitions are drawn from ρ<sup rise='3' size='6'>β</sup> (the state distribution "
        "induced by the human), not from ρ<sup rise='3' size='6'>μ</sup> — that mismatch between "
        "the data-generating policy β and the policy being improved μ<sub rise='2' size='6'>θ</sub> "
        "is precisely what makes it off-policy. "
        "In the stochastic off-policy case, this mismatch requires importance-sampling corrections "
        "π(a|s)/β(a|s) to correct for the action distribution shift. In DPG, the actor gradient "
        "has no integral over actions, so no importance-sampling correction is needed in the "
        "actor update — a major practical advantage.", body))

    # ── Q3 ──
    story.append(HRFlowable(width="100%", thickness=1, color=colors.grey, spaceBefore=1, spaceAfter=1))
    story.append(Paragraph("Q3: Why DPG Suits Continuous Action Spaces", hd))

    story.append(Paragraph(
        "<b>1. Eliminates the inner arg-max optimization:</b> "
        "In Q-learning-style methods extended to continuous actions, every policy improvement step "
        "requires solving arg max<sub rise='2' size='6'>a</sub> Q(s,a) — a separate nonlinear "
        "optimization for each state. This is computationally prohibitive when the action space is "
        "high-dimensional or continuous. DPG avoids this entirely: the deterministic policy "
        "μ<sub rise='2' size='6'>θ</sub> is updated by gradient ascent along "
        "∇<sub rise='2' size='6'>a</sub>Q, turning the inner optimization into a single "
        "backpropagation step through the critic.", body))

    story.append(Paragraph(
        "<b>2. Superior sample efficiency as action dimensionality grows:</b> "
        "Stochastic PG estimates the policy gradient by sampling actions a ~ π(·|s) and computing "
        "the score-function estimator ∇log π · Q. The variance of this estimator grows with the "
        "dimension of the action space, so exponentially more samples are needed for accurate "
        "gradient estimates in high-dimensional continuous spaces. DPG computes the gradient "
        "analytically — integrating only over the state space — so its sample complexity is "
        "independent of action dimensionality. Silver et al. (2014) demonstrated this empirically "
        "on continuous bandit tasks: DPG achieved lower regret than stochastic PG even with the "
        "same number of samples, and the gap widened as action dimensionality increased "
        "(10 → 25 → 50 dimensions).", body))

    story.append(Paragraph(
        "<b>3. Natural compatibility with function approximation (DDPG):</b> "
        "Because μ<sub rise='2' size='6'>θ</sub> is differentiable and deterministic, the full "
        "gradient chain ∇<sub rise='2' size='6'>θ</sub>μ<sub rise='2' size='6'>θ</sub>(s) · "
        "∇<sub rise='2' size='6'>a</sub>Q<sup rise='3' size='6'>μ</sup>(s,a) can be computed "
        "end-to-end via automatic differentiation through a neural-network actor and critic. "
        "This enables deep extensions (DDPG) that scale to complex perception-to-action tasks "
        "such as continuous robotic control and physics simulations.", body))

    doc.build(story)
    print(f"Saved: {out}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate the assignment answer PDF.")
    parser.add_argument(
        "-o",
        "--output",
        default="answer_final.pdf",
        help="Output PDF path (default: answer_final.pdf)",
    )
    args = parser.parse_args()
    build(os.path.abspath(args.output))
