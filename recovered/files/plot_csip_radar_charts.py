#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CSIP雷达图绘制脚本
根据CSIP Zero Metrics数据生成雷达图，按N值分组
每个雷达图显示不同A值的性能对比
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

def create_csip_radar_chart(data, n_value, save_path):
    """
    为指定的N值创建CSIP专用的Material Design风格雷达图
    显示不同A值的性能对比
    
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
    
    # CSIP专用优美柔和配色方案
    colors = {
        1: '#FF6B6B',  # 珊瑚红
        3: '#4ECDC4',  # 青色
        5: '#45B7D1'   # 天蓝色
    }
    
    # 更浅的辅助色调用于填充
    light_colors = {
        1: '#FFE5E5',  # 极浅红色
        3: '#E5F9F7',  # 极浅青色
        5: '#E5F3FB'   # 极浅蓝色
    }
    
    # 深色调用于强调和边框
    dark_colors = {
        1: '#E55555',  # 深红色
        3: '#3AB8B1',  # 深青色
        5: '#3A9BC1'   # 深蓝色
    }
    
    # 设置极坐标背景
    ax.set_facecolor('#FFFFFF')
    
    # 绘制优美的同心圆网格线
    grid_values = np.linspace(0.5, 1.0, 10)
    
    for i, r in enumerate(grid_values):
        # 绘制同心圆网格线 - 使用极坐标系绘制
        theta_grid = np.linspace(0, 2*pi, 100)
        r_grid = np.full_like(theta_grid, r)
        
        # 主网格圆圈 - 使用淡紫色
        ax.plot(theta_grid, r_grid, color='#CDB4DB', 
               alpha=0.4, linewidth=1.2, linestyle='-', zorder=1)
        
        # 添加微妙的阴影效果
        if i < len(grid_values) - 1:
            ax.plot(theta_grid, r_grid, color='#E8E2F0', 
                   alpha=0.3, linewidth=2.0, linestyle='-', zorder=0)
        
        # 在每条网格线上添加优美的百分比标签
        if i % 2 == 0:  # 只在偶数索引显示标签，避免过于拥挤
            ax.text(pi/2 + 0.05, r, f'{int(r*100)}%', 
                    ha='center', va='center', 
                    fontsize=9, color='#5D4E75', 
                    fontweight='600',
                    bbox=dict(boxstyle='round,pad=0.3', 
                             facecolor='white', 
                             edgecolor='#CDB4DB',
                             linewidth=1,
                             alpha=0.95))
    
    # 绘制优美的径向网格线
    for angle in angles[:-1]:
        ax.plot([angle, angle], [0.45, 1.0], color='#CDB4DB', 
                alpha=0.3, linewidth=1.0, linestyle='-', zorder=1)
    
    # 收集所有数据组合并计算面积
    plot_data = []
    for a_val in [1, 3, 5]:
        subset = data[data['A'] == a_val]
        
        if len(subset) > 0:
            values = subset[metrics].iloc[0].values.tolist()
            # 计算面积（使用平均值作为面积的近似）
            area = np.mean(values)
            
            plot_data.append({
                'a_val': a_val,
                'values': values,
                'area': area,
                'label': f'CSIP (A={a_val})'
            })
    
    # 按面积从大到小排序，确保小的图形在上层
    plot_data.sort(key=lambda x: x['area'], reverse=True)
    
    # 打印绘制顺序信息（调试用）
    if len(plot_data) > 0:
        print(f"    🎯 绘制顺序优化 (按面积从大到小):")
        for i, item in enumerate(plot_data):
            print(f"      {i+1}. {item['label']}: 面积={item['area']:.4f}")
    
    # 按排序后的顺序绘制数据线条
    legend_handles = []
    legend_labels = []
    
    for i, item in enumerate(plot_data):
        values = item['values'] + item['values'][:1]  # 闭合图形
        a_val = item['a_val']
        label = item['label']
        
        # 根据绘制顺序调整zorder，面积小的有更高的zorder
        base_zorder = 10 + (len(plot_data) - i)
        
        # 绘制优美的数据线条
        line = ax.plot(angles, values, 
                      linestyle='-',
                      linewidth=3.0,
                      color=colors[a_val],
                      marker='o',
                      markersize=8,
                      markerfacecolor=colors[a_val],
                      markeredgecolor=dark_colors[a_val],
                      markeredgewidth=2.5,
                      alpha=0.9,
                      zorder=base_zorder,
                      label=label)
        
        # 添加优美的渐变填充效果
        ax.fill(angles, values, color=light_colors[a_val], 
               alpha=0.3, zorder=base_zorder-5)
        
        # 添加额外的高亮边缘
        ax.plot(angles, values, color=dark_colors[a_val], 
               linewidth=1.5, alpha=0.4, zorder=base_zorder-3)
    
    # 创建图例项，按A值顺序排列
    for a_val in [1, 3, 5]:
        # 创建图例用的线条样式
        legend_line = plt.Line2D([0], [0], 
                               linestyle='-',
                               linewidth=3.0,
                               color=colors[a_val],
                               marker='o',
                               markersize=7,
                               markerfacecolor=colors[a_val],
                               markeredgecolor=dark_colors[a_val],
                               markeredgewidth=2)
        legend_handles.append(legend_line)
        legend_labels.append(f'CSIP (A={a_val})')
    
    # 隐藏默认轴标签，使用自定义位置避免遮挡
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([])  # 隐藏默认标签
    
    # 手动设置每个标签的位置，使其离外圈更远
    for angle, label in zip(angles[:-1], metric_labels):
        # 设置标签位置在1.12倍半径处，确保不被外圈遮挡
        ax.text(angle, 1.08, label, ha='center', va='center', 
               fontsize=13, fontweight='500', color='#212121',
               bbox=dict(boxstyle='round,pad=0.3', 
                        facecolor='white', 
                        edgecolor='none',
                        alpha=0.8))
    
    # 设置y轴 - 扩展范围以容纳外移的标签
    ax.set_ylim(0.45, 1.0)
    ax.set_yticks([])  # 隐藏默认的y轴标签，使用我们自定义的
    ax.set_theta_zero_location('N')  # 从顶部开始
    ax.set_theta_direction(-1)  # 顺时针方向
    
    # 移除默认网格
    ax.grid(False)
    
    # 优美的现代标题
    title_text = f'CSIP Model Performance Analysis (N={n_value})'
    ax.set_title(title_text, fontsize=19, fontweight='700', 
                color='#2D3748', pad=60)
    
    # 优美的现代图例
    legend = ax.legend(legend_handles, legend_labels, 
                      loc='center left', bbox_to_anchor=(0.85, 0.97), 
                      fontsize=12, frameon=True, fancybox=True, shadow=True,
                      handlelength=3.5, handletextpad=1.0)
    legend.get_frame().set_facecolor('#FFFFFF')
    legend.get_frame().set_edgecolor('#CDB4DB')
    legend.get_frame().set_linewidth(1.5)
    legend.get_frame().set_alpha(0.98)
    
    # 添加优美的性能等级指示
    performance_text = 'Performance Comparison by A Value'
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
    print(f"✨ CSIP优美现代风格雷达图已保存: {save_path}")


def create_csip_summary_table(data, save_path):
    """
    创建CSIP专用的Material Design风格性能汇总表
    
    Args:
        data: 完整数据
        save_path: 保存路径
    """
    fig, ax = plt.subplots(figsize=(14, 8))
    fig.patch.set_facecolor('#FAFAFA')
    ax.axis('tight')
    ax.axis('off')
    
    # 创建数据透视表
    pivot_data = []
    for n_val in sorted(data['N'].unique()):
        for a_val in sorted(data['A'].unique()):
            subset = data[(data['N'] == n_val) & (data['A'] == a_val)]
            
            if len(subset) > 0:
                row_data = subset.iloc[0]
                pivot_data.append([
                    f'N={n_val}, A={a_val}',
                    f'{row_data["top1_accuracy"]:.3f}',
                    f'{row_data["top5_accuracy"]:.3f}',
                    f'{row_data["precision"]:.3f}',
                    f'{row_data["recall"]:.3f}',
                    f'{row_data["f1"]:.3f}'
                ])
    
    # 创建Material Design风格表格
    table = ax.table(cellText=pivot_data,
                    colLabels=['Configuration', 'Top-1 Acc', 'Top-5 Acc', 'Precision', 'Recall', 'F1 Score'],
                    cellLoc='center',
                    loc='center',
                    colWidths=[0.2, 0.16, 0.16, 0.16, 0.16, 0.16])
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 2.0)
    
    # Material Design风格美化表格
    for i in range(len(pivot_data) + 1):
        for j in range(6):
            cell = table[i, j]
            if i == 0:  # 头部行 - Material Design Primary色调
                cell.set_facecolor('#2196F3')  # Material Blue 500
                cell.set_text_props(weight='600', color='white', fontsize=11)
                cell.set_edgecolor('#1976D2')  # Material Blue 700
            else:
                # 根据F1分数设置背景颜色
                f1_val = float(pivot_data[i-1][5])  # F1 Score列
                if f1_val > 0.85:  # 高性能 - 绿色系
                    cell.set_facecolor('#E8F5E9')  # Material Green 50
                    cell.set_edgecolor('#C8E6C9')  # Material Green 100
                elif f1_val > 0.75:  # 中等性能 - 橙色系
                    cell.set_facecolor('#FFF3E0')  # Material Orange 50
                    cell.set_edgecolor('#FFCC80')  # Material Orange 200
                else:  # 一般性能 - 灰色系
                    cell.set_facecolor('#FAFAFA')  # Material Grey 50
                    cell.set_edgecolor('#E0E0E0')  # Material Grey 300
                
                # 设置文字样式
                if j == 5:  # F1列使用特殊颜色
                    if f1_val > 0.85:
                        cell.set_text_props(weight='600', color='#2E7D32')  # Green 800
                    elif f1_val > 0.75:
                        cell.set_text_props(weight='600', color='#F57C00')  # Orange 800
                    else:
                        cell.set_text_props(weight='500', color='#424242')  # Grey 800
                else:
                    cell.set_text_props(weight='400', color='#212121')  # Grey 900
                
                cell.set_linewidth(1)
    
    # Material Design风格标题
    ax.set_title('CSIP Model Performance Summary', 
                fontsize=20, fontweight='600', color='#212121', pad=30)
    
    # 添加副标题
    ax.text(0.5, 0.95, 'Comprehensive Performance Metrics Across Different Configurations', 
           ha='center', va='center', transform=ax.transAxes,
           fontsize=13, color='#757575', fontweight='400', style='italic')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='#FAFAFA')
    plt.close()
    print(f"✨ CSIP Material Design性能汇总表已保存: {save_path}")

def main():
    """主函数"""
    # 读取CSV文件
    csv_file = '/data/cxip/csip_zero_metrics.csv'
    
    if not os.path.exists(csv_file):
        print(f"❌ 错误: 找不到文件 {csv_file}")
        return
    
    # 读取数据
    df = pd.read_csv(csv_file)
    print(f"✅ 成功读取CSIP数据，共 {len(df)} 行")
    print(f"📊 N值: {sorted(df['N'].unique())}")
    print(f"🔢 A值: {sorted(df['A'].unique())}")
    print(f"🤖 模型类型: {list(df['model_type'].unique())}")
    
    # 创建输出目录
    output_dir = '/data/cxip/beautiful_radar_charts-csip'
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n🎨 正在生成CSIP优美现代风格雷达图...")
    
    # 为每个N值创建优美现代风格雷达图
    n_values = sorted(df['N'].unique())
    
    for n_val in n_values:
        print(f"  📊 正在生成 N={n_val} 的CSIP优美雷达图...")
        n_data = df[df['N'] == n_val]
        save_path = os.path.join(output_dir, f'beautiful_radar_N{n_val}.png')
        create_csip_radar_chart(n_data, n_val, save_path)
    
    # 创建性能汇总表
    print(f"  📋 正在生成CSIP性能汇总表...")
    summary_path = os.path.join(output_dir, 'performance_summary.png')
    create_csip_summary_table(df, summary_path)
    
    print(f"\n🎉 所有CSIP优美现代风格图表已保存到: {output_dir}")
    print("📁 生成的文件:")
    for file in sorted(os.listdir(output_dir)):
        if file.endswith('.png'):
            print(f"  ✨ {file}")
    
    print(f"\n💡 CSIP优美现代风格图表说明:")
    print(f"  • 每个雷达图显示同一N值下不同A值的性能对比")
    print(f"  • 珊瑚红 = A=1，青色 = A=3，天蓝色 = A=5")
    print(f"  • 实心圆点标记，带有优美的渐变填充效果")
    print(f"  • 淡紫色网格线，优美的百分比标签")
    print(f"  • 🎯 按面积排序绘制，防止小图形被大图形遮挡")
    print(f"  • 📝 指标标签外移，带白色背景框，避免被遮挡")
    print(f"  • 📊 汇总表根据F1分数使用不同颜色编码")

if __name__ == "__main__":
    main()