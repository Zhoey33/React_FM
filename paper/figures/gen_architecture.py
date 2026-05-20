"""Generate architecture diagram for React_FM paper."""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

fig, ax = plt.subplots(1, 1, figsize=(7.5, 4.5))
ax.set_xlim(0, 10)
ax.set_ylim(0, 6)
ax.axis('off')

# Colors
C_AGENT = '#4472C4'
C_ENV = '#70AD47'
C_DETECT = '#ED7D31'
C_MEMORY = '#7030A0'
C_EXTRACT = '#BF8F00'
C_BG = '#F2F2F2'
ALPHA = 0.15

def box(x, y, w, h, text, color, fontsize=9, bold=False):
    rect = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1",
                          facecolor=color, edgecolor='white', alpha=0.85, linewidth=1.5)
    ax.add_patch(rect)
    weight = 'bold' if bold else 'normal'
    ax.text(x + w/2, y + h/2, text, ha='center', va='center',
            fontsize=fontsize, color='white', fontweight=weight)

def arrow(x1, y1, x2, y2, color='#333333', style='->', lw=1.5):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle=style, color=color, lw=lw))

def label(x, y, text, fontsize=7.5, color='#333333', ha='center'):
    ax.text(x, y, text, ha=ha, va='center', fontsize=fontsize, color=color, style='italic')

# === Main loop (top row) ===
box(0.3, 4.2, 1.6, 0.9, 'LLM Agent\n(ReAct)', C_AGENT, fontsize=9, bold=True)
box(2.8, 4.2, 1.6, 0.9, 'Environment\n(ALFWorld/\nWebShop/SW)', C_ENV, fontsize=7.5)

arrow(1.9, 4.65, 2.8, 4.65, C_AGENT)
label(2.35, 4.85, 'action $a_t$', fontsize=7)
arrow(2.8, 4.55, 1.9, 4.55, C_ENV)
label(2.35, 4.35, 'obs $o_t$', fontsize=7)

# === Failure Detector (middle) ===
box(5.0, 4.2, 2.0, 0.9, 'Failure Detector', C_DETECT, fontsize=9, bold=True)
# Sub-labels
ax.text(6.0, 3.95, 'Explicit (rules)  |  Implicit (LLM judge)', ha='center', va='center',
        fontsize=6.5, color='#666666')

arrow(4.4, 4.65, 5.0, 4.65, '#333333')
label(4.7, 4.85, '$(a_t, o_t)$', fontsize=7)

# === Memory Store (bottom center) ===
box(3.5, 1.8, 3.0, 1.0, 'Memory Store $\\mathcal{M}$\n(failure, obs, solution)\nper-env isolation', C_MEMORY, fontsize=8, bold=True)

# Failure → Memory retrieval
arrow(6.0, 4.2, 6.0, 2.8, C_DETECT)
label(6.35, 3.5, 'if failure\ndetected', fontsize=6.5)

# Memory → Agent (inject)
arrow(3.5, 2.3, 1.1, 4.2, C_MEMORY, lw=2)
label(1.8, 3.3, 'inject\nhints', fontsize=7, color=C_MEMORY)

# === Hybrid Retrieval (right side) ===
box(7.5, 2.5, 2.0, 0.7, 'Hybrid Retrieval\nBM25 + Emb + RRF', '#5B9BD5', fontsize=7.5)
arrow(6.5, 2.3, 7.5, 2.85, '#5B9BD5')
arrow(7.5, 2.85, 6.5, 2.3, '#5B9BD5')
label(7.3, 2.15, 'query / top-K', fontsize=6.5)

# === Post-Episode Extractor (bottom) ===
box(0.3, 0.5, 2.5, 0.8, 'Post-Episode Extractor\n(LLM)', C_EXTRACT, fontsize=8, bold=True)

arrow(1.55, 1.3, 3.5, 1.8, C_EXTRACT)
label(2.8, 1.4, 'store new\nmemories', fontsize=7, color=C_EXTRACT)

# Trajectory → Extractor
arrow(2.35, 4.2, 1.55, 1.3, '#999999', style='->', lw=1)
label(1.5, 2.8, 'trajectory\n(post-episode)', fontsize=6.5, color='#999999')

# === Title ===
ax.text(5, 5.7, 'React_FM: In-Loop Failure Memory Pipeline', ha='center', va='center',
        fontsize=11, fontweight='bold', color='#333333')

# === Legend for failure types ===
ax.text(7.8, 1.5, 'Explicit: "Nothing happens",\n  action loops, empty obs',
        fontsize=6, color=C_DETECT, ha='left')
ax.text(7.8, 0.8, 'Implicit: unproductive actions\n  (LLM judge)',
        fontsize=6, color=C_DETECT, ha='left')

fig.tight_layout()
fig.savefig('/Users/zhoey/React_FM/paper/figures/architecture.pdf', dpi=300, bbox_inches='tight')
fig.savefig('/Users/zhoey/React_FM/paper/figures/architecture.png', dpi=300, bbox_inches='tight')
plt.close()
print('Generated: architecture.pdf')
