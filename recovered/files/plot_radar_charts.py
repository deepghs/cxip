#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
雷达图绘制脚本
根据CCIP Zero Metrics数据生成雷达图，按N值分组
同一个N的所有A值和模型类型都在一个雷达图中
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from math import pi
import os
import seaborn as sns
from matplotlib.patches import Circle
import warnings

warnings.filterwarnings('ignore')

# 设置matplotlib后端为非交互模式，适合服务器环境
mpl.use('Agg')

# 设置Material Design风格
plt.style.use('default')  # 使用默认风格作为基础

# Material Design配色和字体设置
plt.rcParams['font.sans-serif'] = ['Roboto', 'Arial', 'DejaVu Sans', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.facecolor'] = '#FAFAFA'  # Material Design背景色
plt.rcParams['axes.facecolor'] = '#FFFFFF'
plt.rcParams['axes.edgecolor'] = '#E0E0E0'
plt.rcParams['axes.linewidth'] = 0.8
plt.rcParams['xtick.color'] = '#424242'
plt.rcParams['ytick.color'] = '#424242'
plt.rcParams['text.color'] = '#212121'

def create_modern_radar_chart(data, n_value, save_path):
    """
    为指定的N值创建Material Design风格的雷达图
    同一个图中包含所有A值和模型类型的组合
    
    Args:
        data: 过滤后的数据
        n_value: N值
        save_path: 保存路径
    """
    # 定义要显示的指标
    metrics = ['top1_accuracy', 'top5_accuracy', 'precision', 'recall', 'f1']
    metric_labels = ['Top-1\nAccuracy', 'Top-5\nAccuracy', 'Precision', 'Recall', 'F1 Score']
    
    # 计算角度
    angles = [n / float(len(metrics)) * 2 * pi for n in range(len(metrics))]
    angles += angles[:1]  # 闭合图形
    
    # 创建图形 - 优美的现代比例
    fig, ax = plt.subplots(figsize=(13, 11), subplot_kw=dict(projection='polar'))
    fig.patch.set_facecolor('#F8F9FD')
    
    # 优美柔和配色方案 - 基于用户提供的颜色
    colors = {
        1: '#FFAFCC',  # 柔美粉色
        3: '#B8E0D2',  # 清新薄荷绿  
        5: '#A2D2FF'   # 梦幻天蓝
    }
    
    # 更浅的辅助色调用于填充
    light_colors = {
        1: '#FFE1EC',  # 极浅粉色
        3: '#D4F1E8',  # 极浅绿色
        5: '#D1E7FF'   # 极浅蓝色
    }
    
    # 深色调用于强调和边框
    dark_colors = {
        1: '#FF8FAB',  # 深粉色
        3: '#95D3C0',  # 深绿色  
        5: '#809BCE'   # 蓝紫色 (用户提供的颜色)
    }
    
    # 线条样式定义
    linestyles = {'old': '-', 'new': '--'}
    linewidths = {'old': 2.8, 'new': 3.2}
    alphas = {'old': 0.85, 'new': 0.95}
    
    # 设置极坐标背景
    ax.set_facecolor('#FFFFFF')
    
    # 绘制优美的同心圆网格线
    grid_values = np.linspace(0.84, 1.0, 5)
    
    for i, r in enumerate(grid_values):
        # 绘制同心圆网格线 - 使用极坐标系绘制
        theta_grid = np.linspace(0, 2*pi, 100)
        r_grid = np.full_like(theta_grid, r)
        
        # 主网格圆圈 - 使用淡紫色
        ax.plot(theta_grid, r_grid, color='#CDB4DB', 
               alpha=0.4, linewidth=1.5, linestyle='-', zorder=1)
        
        # 添加微妙的阴影效果
        if i < len(grid_values) - 1:
            ax.plot(theta_grid, r_grid, color='#E8E2F0', 
                   alpha=0.3, linewidth=2.5, linestyle='-', zorder=0)
        
        # 在每条网格线上添加优美的百分比标签
        ax.text(pi/2 + 0.05, r, f'{int(r*100)}%', 
                ha='center', va='center', 
                fontsize=10, color='#5D4E75', 
                fontweight='600',
                bbox=dict(boxstyle='round,pad=0.4', 
                         facecolor='white', 
                         edgecolor='#CDB4DB',
                         linewidth=1,
                         alpha=0.95))
    
    # 绘制优美的径向网格线
    for angle in angles[:-1]:
        ax.plot([angle, angle], [0.82, 1.0], color='#CDB4DB', 
                alpha=0.3, linewidth=1.0, linestyle='-', zorder=1)
    
    # 收集所有数据组合并计算面积
    plot_data = []
    for a_val in [1, 3, 5]:
        for model_type in ['old', 'new']:
            subset = data[(data['A'] == a_val) & (data['model_type'] == model_type)]
            
            if len(subset) > 0:
                values = subset[metrics].iloc[0].values.tolist()
                # 计算面积（使用平均值作为面积的近似）
                area = np.mean(values)
                
                plot_data.append({
                    'a_val': a_val,
                    'model_type': model_type,
                    'values': values,
                    'area': area,
                    'label': f'{model_type.upper()} (A={a_val})'
                })
    
    # 按面积从大到小排序，确保小的图形在上层
    plot_data.sort(key=lambda x: x['area'], reverse=True)
    
    # 打印绘制顺序信息（调试用）
    if len(plot_data) > 0:
        print(f"    🎯 绘制顺序优化 (按面积从大到小):")
        for i, item in enumerate(plot_data):
            print(f"      {i+1}. {item['label']}: 面积={item['area']:.4f}")
    
    # 按排序后的顺序绘制数据线条，但不添加label（稍后手动设置图例顺序）
    legend_handles = []
    legend_labels = []
    
    for i, item in enumerate(plot_data):
        values = item['values'] + item['values'][:1]  # 闭合图形
        a_val = item['a_val']
        model_type = item['model_type']
        label = item['label']
        
        # 根据绘制顺序调整zorder，面积小的有更高的zorder
        base_zorder = 10 + (len(plot_data) - i)
        
        # 绘制优美的数据线条（不设置label）
        edge_color = dark_colors[a_val] if model_type == 'new' else 'white'
        line = ax.plot(angles, values, 
                      linestyle=linestyles[model_type],
                      linewidth=linewidths[model_type],
                      color=colors[a_val],
                      marker='o' if model_type == 'old' else 'D',
                      markersize=8 if model_type == 'old' else 7,
                      markerfacecolor=colors[a_val],
                      markeredgecolor=edge_color,
                      markeredgewidth=2.5,
                      alpha=alphas[model_type],
                      zorder=base_zorder)
        
        # 添加优美的渐变填充效果
        if model_type == 'new':
            ax.fill(angles, values, color=light_colors[a_val], 
                   alpha=0.25, zorder=base_zorder-5)
            # 添加额外的高亮边缘
            ax.plot(angles, values, color=dark_colors[a_val], 
                   linewidth=1, alpha=0.3, zorder=base_zorder-3)
    
    # 创建图例项，按原始顺序（A=1,3,5 x old,new）
    for model_type in ['old', 'new']:
        for a_val in [1, 3, 5]:
            # 创建图例用的线条样式
            legend_line = plt.Line2D([0], [0], 
                                   linestyle=linestyles[model_type],
                                   linewidth=linewidths[model_type],
                                   color=colors[a_val],
                                   marker='o' if model_type == 'old' else 'D',
                                   markersize=6,
                                   markerfacecolor=colors[a_val],
                                   markeredgecolor=dark_colors[a_val] if model_type == 'new' else 'white',
                                   markeredgewidth=2)
            legend_handles.append(legend_line)
            legend_labels.append(f'{model_type.upper()} (A={a_val})')
    
    # 隐藏默认轴标签，使用自定义位置避免遮挡
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([])  # 隐藏默认标签
    
    # 手动设置每个标签的位置，使其离外圈更远
    for angle, label in zip(angles[:-1], metric_labels):
        # 设置标签位置在1.12倍半径处，确保不被外圈遮挡
        ax.text(angle, 1.015, label, ha='center', va='center', 
               fontsize=13, fontweight='500', color='#212121',
               bbox=dict(boxstyle='round,pad=0.3', 
                        facecolor='white', 
                        edgecolor='none',
                        alpha=0.8))
    
    # 设置y轴 - 扩展范围以容纳外移的标签
    ax.set_ylim(0.82, 1.0)
    ax.set_yticks([])  # 隐藏默认的y轴标签，使用我们自定义的
    ax.set_theta_zero_location('N')  # 从顶部开始
    ax.set_theta_direction(-1)  # 顺时针方向
    
    # 移除默认网格
    ax.grid(False)
    
    # 优美的现代标题
    title_text = f'Model Performance Analysis (N={n_value})'
    ax.set_title(title_text, fontsize=19, fontweight='700', 
                color='#2D3748', pad=60)
    
    # 优美的现代图例，使用固定顺序，增加线条长度
    legend = ax.legend(legend_handles, legend_labels, 
                      loc='center left', bbox_to_anchor=(0.85, 0.97), 
                      fontsize=12, frameon=True, fancybox=True, shadow=True,
                      handlelength=3.5, handletextpad=1.0)  # 增加线条长度和间距
    legend.get_frame().set_facecolor('#FFFFFF')
    legend.get_frame().set_edgecolor('#CDB4DB')
    legend.get_frame().set_linewidth(1.5)
    legend.get_frame().set_alpha(0.98)
    
    # 添加优美的性能等级指示
    performance_text = 'Excellent Performance Zone'
    ax.text(0, 1.08, performance_text, ha='center', va='center',
           transform=ax.transAxes, fontsize=11, color='#718096', 
           fontweight='500', style='italic')
    
    # 添加优美的外圈装饰
    theta_outer = np.linspace(0, 2*pi, 200)
    r_outer = np.full_like(theta_outer, 1.0)
    ax.plot(theta_outer, r_outer, color='#CDB4DB', alpha=0.6, linewidth=3, zorder=15)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', 
                facecolor='#F8F9FD', edgecolor='none')
    plt.close()
    print(f"✨ 优美现代风格雷达图已保存: {save_path}")


def create_summary_table(data, save_path):
    """
    创建Material Design风格的性能汇总表
    
    Args:
        data: 完整数据
        save_path: 保存路径
    """
    fig, ax = plt.subplots(figsize=(15, 9))
    fig.patch.set_facecolor('#FAFAFA')
    ax.axis('tight')
    ax.axis('off')
    
    # 创建数据透视表
    pivot_data = []
    for n_val in sorted(data['N'].unique()):
        for a_val in sorted(data['A'].unique()):
            subset = data[(data['N'] == n_val) & (data['A'] == a_val)]
            old_data = subset[subset['model_type'] == 'old']
            new_data = subset[subset['model_type'] == 'new']
            
            if len(old_data) > 0 and len(new_data) > 0:
                old_f1 = old_data['f1'].iloc[0]
                new_f1 = new_data['f1'].iloc[0]
                improvement = (new_f1 - old_f1) / old_f1 * 100
                
                pivot_data.append([
                    f'N={n_val}, A={a_val}',
                    f'{old_f1:.4f}',
                    f'{new_f1:.4f}',
                    f'+{improvement:.1f}%'
                ])
    
    # 创建Material Design风格表格
    table = ax.table(cellText=pivot_data,
                    colLabels=['Configuration', 'OLD Model F1', 'NEW Model F1', 'Improvement'],
                    cellLoc='center',
                    loc='center',
                    colWidths=[0.28, 0.24, 0.24, 0.24])
    
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 2.2)
    
    # Material Design风格美化表格
    for i in range(len(pivot_data) + 1):
        for j in range(4):
            cell = table[i, j]
            if i == 0:  # 头部行 - Material Design Primary色调
                cell.set_facecolor('#2196F3')  # Material Blue 500
                cell.set_text_props(weight='600', color='white', fontsize=12)
                cell.set_edgecolor('#1976D2')  # Material Blue 700
            else:
                improvement_val = float(pivot_data[i-1][3][1:-1])  # 去掉+和%
                if improvement_val > 8:  # 高改进 - 绿色系
                    cell.set_facecolor('#E8F5E9')  # Material Green 50
                    cell.set_edgecolor('#C8E6C9')  # Material Green 100
                elif improvement_val > 4:  # 中等改进 - 橙色系
                    cell.set_facecolor('#FFF3E0')  # Material Orange 50
                    cell.set_edgecolor('#FFCC80')  # Material Orange 200
                else:  # 一般改进 - 灰色系
                    cell.set_facecolor('#FAFAFA')  # Material Grey 50
                    cell.set_edgecolor('#E0E0E0')  # Material Grey 300
                
                # 设置文字样式
                if j == 3:  # 改进列使用特殊颜色
                    if improvement_val > 8:
                        cell.set_text_props(weight='600', color='#2E7D32')  # Green 800
                    elif improvement_val > 4:
                        cell.set_text_props(weight='600', color='#F57C00')  # Orange 800
                    else:
                        cell.set_text_props(weight='500', color='#424242')  # Grey 800
                else:
                    cell.set_text_props(weight='400', color='#212121')  # Grey 900
                
                cell.set_linewidth(1)
    
    # Material Design风格标题
    ax.set_title('Model Performance Summary', 
                fontsize=20, fontweight='600', color='#212121', pad=30)
    
    # 添加副标题
    ax.text(0.5, 0.95, 'F1 Score Comparison and Performance Improvement Analysis', 
           ha='center', va='center', transform=ax.transAxes,
           fontsize=13, color='#757575', fontweight='400', style='italic')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='#FAFAFA')
    plt.close()
    print(f"✨ Material Design性能汇总表已保存: {save_path}")

def main():
    """主函数"""
    # 读取CSV文件
    csv_file = '/data/cxip/ccip_zero_metrics.csv'
    
    if not os.path.exists(csv_file):
        print(f"错误: 找不到文件 {csv_file}")
        return
    
    # 读取数据
    df = pd.read_csv(csv_file)
    print(f"✅ 成功读取数据，共 {len(df)} 行")
    print(f"📊 N值: {sorted(df['N'].unique())}")
    print(f"🔢 A值: {sorted(df['A'].unique())}")
    print(f"🤖 模型类型: {list(df['model_type'].unique())}")
    
    # 创建输出目录
    output_dir = '/data/cxip/beautiful_radar_charts'
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n🎨 正在生成优美现代风格雷达图...")
    
    # 为每个N值创建优美现代风格雷达图
    n_values = sorted(df['N'].unique())
    
    for n_val in n_values:
        print(f"  📊 正在生成 N={n_val} 的优美雷达图...")
        n_data = df[df['N'] == n_val]
        save_path = os.path.join(output_dir, f'beautiful_radar_N{n_val}.png')
        create_modern_radar_chart(n_data, n_val, save_path)
    
    # 创建性能汇总表
    print(f"  📋 正在生成性能汇总表...")
    summary_path = os.path.join(output_dir, 'performance_summary.png')
    create_summary_table(df, summary_path)
    
    print(f"\n🎉 所有优美现代风格图表已保存到: {output_dir}")
    print("📁 生成的文件:")
    for file in sorted(os.listdir(output_dir)):
        if file.endswith('.png'):
            print(f"  ✨ {file}")
    
    print(f"\n💡 优美现代风格图表说明:")
    print(f"  • 实线 = OLD模型，虚线 = NEW模型")
    print(f"  • 圆形 = OLD模型，菱形 = NEW模型") 
    print(f"  • 柔美粉色 = A=1，清新薄荷绿 = A=3，梦幻天蓝 = A=5")
    print(f"  • 采用用户指定的优美柔和配色方案")
    print(f"  • 淡紫色网格线，优美的百分比标签")
    print(f"  • 🎯 按面积排序绘制，防止小图形被大图形遮挡")
    print(f"  • 📝 指标标签外移，带白色背景框，避免被遮挡")
    print(f"  • 🔄 图例保持逻辑顺序(A=1→3→5, OLD→NEW)，不受绘制顺序影响")

if __name__ == "__main__":
    main()
