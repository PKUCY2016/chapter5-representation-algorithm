from pathlib import Path
from xml.sax.saxutils import escape
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import BaseDocTemplate, Frame, PageTemplate, Paragraph, Table, TableStyle, PageBreak

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / 'chapter5_algorithm_report.pdf'
pdfmetrics.registerFont(TTFont('CN', '/System/Library/Fonts/STHeiti Medium.ttc'))
styles = getSampleStyleSheet()
styles.add(ParagraphStyle(name='TitleCN', parent=styles['Title'], fontName='CN', fontSize=19, leading=25, alignment=TA_CENTER, textColor=colors.HexColor('#17365D'), spaceAfter=12))
styles.add(ParagraphStyle(name='H1CN', parent=styles['Heading1'], fontName='CN', fontSize=14, leading=20, textColor=colors.HexColor('#17365D'), spaceBefore=10, spaceAfter=7))
styles.add(ParagraphStyle(name='BodyCN', parent=styles['BodyText'], fontName='CN', fontSize=9.2, leading=14, spaceAfter=5))
styles.add(ParagraphStyle(name='SmallCN', parent=styles['BodyText'], fontName='CN', fontSize=7.8, leading=11, textColor=colors.HexColor('#555555')))

def P(text, style='BodyCN'):
    return Paragraph(escape(text).replace('\n', '<br/>'), styles[style])

def table(data, widths):
    converted = [[Paragraph(escape(str(cell)), styles['SmallCN']) for cell in row] for row in data]
    t = Table(converted, colWidths=widths, repeatRows=1, hAlign='LEFT')
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#D9EAF7')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.HexColor('#17365D')),
        ('GRID', (0, 0), (-1, -1), 0.35, colors.HexColor('#A6A6A6')),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('LEFTPADDING', (0, 0), (-1, -1), 4), ('RIGHTPADDING', (0, 0), (-1, -1), 4),
        ('TOPPADDING', (0, 0), (-1, -1), 4), ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
    ]))
    return t

def footer(canvas, doc):
    canvas.saveState(); canvas.setFont('CN', 7.5); canvas.setFillColor(colors.HexColor('#666666'))
    canvas.drawString(18*mm, 10*mm, 'Chapter 5 文献转算法验收报告 | reference-only boundary')
    canvas.drawRightString(192*mm, 10*mm, f'第 {doc.page} 页'); canvas.restoreState()

doc = BaseDocTemplate(str(OUTPUT), pagesize=A4, rightMargin=18*mm, leftMargin=18*mm, topMargin=16*mm, bottomMargin=17*mm)
doc.addPageTemplates([PageTemplate(id='main', frames=[Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height, id='normal')], onPage=footer)])
story = [
    P('Chapter 5 文献转算法验收报告', 'TitleCN'), P('版本：2026-09-30 | 交付对象：算法专家代码验收', 'SmallCN'),
    P('1. 交付范围', 'H1CN'),
    P('本目录把 Yi Ma 等人的 Deep Representation Learning Book Chapter 5 中与 CRATE、MSSA/ISTA、TSSA/ToST 和 causal CRATE 相关的明确公式，转成 NumPy/float64 reference。它是可审计的算法原型，不是训练好的 CRATE 模型，也不是随机分析或全局收敛证明。随机分析主线 01_geometry、02_spectral、03_generative 没有被修改。'),
    P('2. 原始文献与逐项对应', 'H1CN'),
    P('主要来源：Chapter 5 https://ma-lab-berkeley.github.io/deep-representation-learning-book/Ch5.html；Chapter 5 TeX https://raw.githubusercontent.com/Ma-Lab-Berkeley/deep-representation-learning-book/main/chapters/chapter5/deep-networks.tex；官方 CRATE https://github.com/Ma-Lab-Berkeley/CRATE；White-Box Transformers https://arxiv.org/abs/2311.13110。'),
    table([
        ['文献对象', '公式/思想', '当前实现', '边界'],
        ['coding rate', '(5.2.2), (5.2.3)', 'coding_rate / subspace_coding_rate', '数值计算'],
        ['exact gradient', '(5.2.9)', 'exact_coding_rate_gradient', '不声称可扩展训练'],
        ['Neumann', '(5.2.10)-(5.2.12)', 'neumann_gradient + 余项', '要求 ||A||<1'],
        ['SSA/MSSA', '(5.2.14)-(5.2.15)', 'mssa', 'softmax kernel 不等同 Neumann 截断'],
        ['CRATE layer', '(5.2.20)-(5.2.22)', 'crate_layer', 'architecture-level Z+MSSA'],
        ['ISTA', '(5.2.21)', 'ista', 'nonnegative=True 对应 ReLU'],
        ['TSSA', '(5.3.16)-(5.3.18)', 'tssa', '未做 ToST 训练'],
        ['causal CRATE', '(5.3.19)-(5.3.30)', 'mask + projected cache', 'NumPy correctness reference'],
        ['Example 5.4', 'loop 研究方向', 'looped_crate', '无新增收敛定理'],
    ], [26*mm, 40*mm, 54*mm, 48*mm]),
    P('3. 核心接口与状态', 'H1CN'),
    P('接口：mssa(Z,U,epsilon,kappa,mask=None)、ista(H,D,eta,lambda_,nonnegative=False)、crate_layer(Z,layer_config,mask=None)、looped_crate(X,run_spec)。receipt 包含 run_id、source/config hash、seed、task、R、dtype/device、预算、状态、停止原因、coding-rate、sparsity、SNR、范数、operator residual、wall time、memory、FLOPs、compile time 和 diagnostics。'),
    P('状态固定为 SOURCE_READ、SOURCE_SPECIFIED、OPERATOR_IMPL、SYNTHETIC_CONDITIONAL、CONDITIONAL_APPROX、DERIVED_PROTOTYPE、PASS、FAIL_CLOSED、UPSTREAM_BLOCKED、UNKNOWN、NOT_RUN、NOT_EVALUATED、STALE。'),
    PageBreak(), P('4. R0-R3 验证', 'H1CN'),
    P('R0：来源已登记；Example 5.4 保留为研究方向。官方 CRATE 远程 commit 当前因本机网络代理不可达而无法 pin，记录为 REMOTE_COMMIT_UNAVAILABLE_NETWORK_BLOCKED。'),
    P('R1：float64，rtol=1e-8、atol=1e-10。覆盖 slogdet、exact gradient、Neumann 余项、MSSA softmax、ISTA、CRATE、TSSA/dense、causal cache、非法输入和 receipt schema。源码目录外运行时 14 项测试通过。'),
    P('R2：d=128、K=4、p=32、N=128、tau=0.75、eta=0.1，delta={0.05,0.10,0.20}，开发 seed={0,1,2}，审计 seed={10,11,12}。数据为低秩子空间联合空间中的 Gaussian toy，包含非正交、重尾和错误 U 负对照；状态为 SYNTHETIC_CONDITIONAL。'),
    P('R3：causal full-mask 与 projected-cache incremental reference 逐元素一致；complexity 覆盖 N={128,256,512,1024,2048} 的 MSSA/TSSA/dense wall time、tracemalloc 峰值和 analytic FLOPs estimate；loop 覆盖 6 seed、R={1,2,4,8}、parameter-matched no-loop 和 equal-FLOPs untied baseline。'),
    table([
        ['观测对象', 'log-log wall-time 斜率', '解释'],
        ['MSSA', '约 2.30', '本机 float64 reference 观察'],
        ['TSSA', '约 0.63', '本机 float64 reference 观察'],
        ['dense attention', '约 1.97', '本机 float64 reference 观察'],
    ], [45*mm, 45*mm, 78*mm]),
    P('5. 代码审计重点', 'H1CN'),
    P('1) mssa 的 softmax 是行归一化后转置聚合，测试验证行和为 1。2) ista(nonnegative=True) 才对应文献 ReLU/nonnegative LASSO；默认 False 是显式的一般 soft-threshold 选项。3) causal_mssa_incremental 使用 projected cache，但仍是 NumPy correctness reference。4) tssa 不构造 N×N token similarity 矩阵，但未完成 ToST 训练或真实任务评估。5) looped_crate 是 Example 5.4 的 derived prototype；局部 coding-rate、ISTA surrogate 或 SNR 变化不等于任务误差下降、全局收敛或随机分析定理。'),
    P('6. 验收结论', 'H1CN'),
    P('代码和证据可交给算法专家复核：算子实现为 OPERATOR_IMPL；定理 toy 为 SYNTHETIC_CONDITIONAL；causal/复杂度为 CONDITIONAL_APPROX；loop 为 DERIVED_PROTOTYPE；真实任务收益、生产级 KV cache 和远程 source commit pin 为 NOT_EVALUATED 或 UPSTREAM_BLOCKED。新的原始论文或差异应先更新 source_manifest.json，再修改代码、测试、hash 和 round evidence。'),
]
doc.build(story)
print(OUTPUT)
