import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle
import matplotlib.patheffects as path_effects

plt.rcParams['font.family'] = 'Arial'
plt.rcParams['axes.unicode_minus'] = False

# 画布背景设为极浅的薄荷绿
fig, ax = plt.subplots(figsize=(20, 10))
fig.patch.set_facecolor('#F4F9F4')
ax.set_facecolor('#F4F9F4')
ax.set_xlim(0, 20)
ax.set_ylim(0, 10)
ax.axis('off')

# ================= 颜色定义（浅绿色高级感） =================
C_CARD = '#FFFFFF'  # 卡片纯白底
E_CARD = '#81C784'  # 浅绿边框
C_PHASE1 = '#E8F5E9'  # Phase 1 极浅绿背景
C_PHASE2 = '#E8F5E9'  # Phase 2 极浅绿背景
C_LINE = '#555555'  # 连线深灰
C_TEXT = '#2E7D32'  # 强调文字深绿


# ================= 绘图辅助函数 =================
def draw_box(x, y, w, h, title, content, fc, ec, title_color='#333333'):
    # 阴影
    shadow = FancyBboxPatch((x + 0.08, y - 0.08), w, h, boxstyle="round,pad=0.1,rounding_size=0.3",
                            facecolor='#BDBDBD', edgecolor='none', alpha=0.3, zorder=1)
    ax.add_patch(shadow)
    # 主框
    box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1,rounding_size=0.3",
                         facecolor=fc, edgecolor=ec, linewidth=2.5, zorder=2)
    ax.add_patch(box)

    # 🔧 关键修复：强制处理标题换行，避免重叠
    if '\n' in title:
        ax.text(x + w / 2, y + h - 0.25, title, ha='center', va='top', fontsize=14, fontweight='bold',
                color=title_color, zorder=3, linespacing=1.2)
        ax.text(x + w / 2, y + h / 2 - 0.4, content, ha='center', va='center', fontsize=11, color='#555555', zorder=3,
                linespacing=1.6)
    else:
        ax.text(x + w / 2, y + h - 0.45, title, ha='center', va='center', fontsize=14, fontweight='bold',
                color=title_color, zorder=3)
        ax.text(x + w / 2, y + h / 2 - 0.2, content, ha='center', va='center', fontsize=11, color='#555555', zorder=3,
                linespacing=1.6)


def draw_arrow(x1, y1, x2, y2, color=C_LINE, ls='-', lw=2):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="->,head_width=0.6,head_length=0.8",
                                lw=lw, color=color, linestyle=ls, connectionstyle="arc3,rad=0"), zorder=1)


# ================= 1. 绘制两个大 Phase 的背景 =================
# Phase 1 (上半部分)
ax.add_patch(Rectangle((0.5, 5.8), 19.0, 3.5, fill=True, facecolor=C_PHASE1, edgecolor='#A5D6A7', linestyle='--', lw=2,
                       zorder=0))
ax.text(0.8, 9.1, "Phase 1: Spatial Feature Extraction & Temporal Modeling", fontsize=13, color=C_TEXT,
        fontweight='bold')

# Phase 2 (下半部分)
ax.add_patch(Rectangle((0.5, 0.8), 19.0, 4.2, fill=True, facecolor=C_PHASE2, edgecolor='#A5D6A7', linestyle='--', lw=2,
                       zorder=0))
ax.text(0.8, 4.7, "Phase 2: Spatial Reconstruction & Constraint Routing", fontsize=13, color=C_TEXT, fontweight='bold')

# ================= 2. 绘制各个模块卡片 =================
# 🔧 关键修复：所有多词标题都加上了 \n 换行

# [上排 1] Input
draw_box(1.0, 6.2, 2.5, 2.3, "Input\nSequence",
         "Multi-variate Time Series\n(B, T, C, H, W)\n\n30 × 8 × 80 × 140", C_CARD, E_CARD)

# [上排 2] Spatial Encoder
draw_box(4.5, 6.2, 3.2, 2.3, "Spatial\nEncoder",
         "3 × [Conv2d(3×3) + BN\n+ ReLU + MaxPool]\n\nChannels: 8 → 32 → 64 → 128", C_CARD, E_CARD)

# [上排 3] Temporal Transformer
draw_box(8.5, 6.2, 3.2, 2.3, "Temporal\nTransformer",
         "3 Encoder Layers\n4 Attention Heads\nFFN (d_ff=512)\nd_model = 128", C_CARD, E_CARD)

# [上排 4] Main Output
draw_box(13.5, 6.2, 2.5, 2.3, "Main\nOutput",
         "6 Extreme Indices\n(TXx, TNn, CDD...)\n\nShape: 6 × 80 × 140", C_CARD, E_CARD)

# [下排 1] Temporal Average
draw_box(4.5, 1.2, 3.2, 2.0, "Temporal\nAverage",
         "Global Feature Pooling\n\nOutput: (B, 128)", C_CARD, E_CARD)

# [下排 2] Deconv Decoder
draw_box(8.5, 1.2, 3.2, 2.0, "Deconv\nDecoder",
         "Linear + 3 × ConvTranspose2d\n\n(8×8) → (80×140)", C_CARD, E_CARD)

# [下排 3] Auxiliary Output
draw_box(13.5, 1.2, 2.5, 2.0, "Auxiliary\nOutput",
         "8 Meteorological Vars\n(tmax, prec...)\n\nShape: 8 × 80 × 140", C_CARD, E_CARD)

# [右侧大框] Physics Loss
draw_box(17.0, 1.2, 2.2, 7.3, "Physics\nLoss",
         "6 Constraints:\n\n• Continuity\n• Thermodynamics\n• Momentum\n• Energy\n• Phase\n• Covariance", C_CARD,
         E_CARD, title_color='#C62828')

# ================= 3. 绘制数据连线 =================

# 上排水平连线
draw_arrow(3.5, 7.35, 4.5, 7.35)  # Input -> Spatial
ax.text(4.0, 7.6, "(B, T, C, H, W)", ha='center', fontsize=9, style='italic', color='#666666')

draw_arrow(7.7, 7.35, 8.5, 7.35)  # Spatial -> Transformer
ax.text(8.1, 7.6, "Sequence", ha='center', fontsize=9, style='italic', color='#666666')

draw_arrow(11.7, 7.35, 13.5, 7.35)  # Transformer -> Main Output

# 下排水平连线
draw_arrow(7.7, 2.2, 8.5, 2.2)  # Temporal Avg -> Deconv
draw_arrow(11.7, 2.2, 13.5, 2.2)  # Deconv -> Auxiliary

# 跨排的折线（Transformer 下连到 Temporal Average）
ax.plot([10.1, 10.1], [6.2, 4.9], color=C_LINE, lw=2, zorder=1)
ax.plot([10.1, 6.1], [4.9, 4.9], color=C_LINE, lw=2, zorder=1)
draw_arrow(6.1, 4.9, 6.1, 3.2)  # 箭头进 Temporal Avg

# 跨排的折线（Deconv Decoder 上连到 Main Output）
ax.plot([10.1, 10.1], [3.2, 5.4], color=C_LINE, lw=2, zorder=1)
ax.plot([10.1, 14.75], [5.4, 5.4], color=C_LINE, lw=2, zorder=1)
draw_arrow(14.75, 5.4, 14.75, 6.2)  # 箭头进 Main Output

# 物理约束路由线（Auxiliary Output 连向 Physics Loss）
draw_arrow(16.0, 2.2, 17.0, 2.2, color='#C62828', ls='--', lw=2.5)
ax.text(16.5, 2.6, "Constraint\nRouting", ha='center', va='center', fontsize=10, color='#C62828', fontweight='bold',
        style='italic')

plt.tight_layout()
plt.savefig('fig1_architecture_green.png', dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor())
print("✅ 浅绿色高级架构图已重新生成（修复了文字重叠）: fig1_architecture_green.png")