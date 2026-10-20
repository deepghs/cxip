#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Zerochan数据集整理脚本
按角色名将图片和标注文件分类到不同的文件夹
"""

import os
import json
import shutil
from pathlib import Path
from tqdm import tqdm
import argparse
from collections import defaultdict


def load_json_annotation(json_path):
    """加载JSON标注文件"""
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    except (json.JSONDecodeError, FileNotFoundError) as e:
        print(f"警告: 无法读取JSON文件 {json_path}: {e}")
        return None


def clean_character_name(character_name):
    """清理角色名，移除不适合作为文件夹名的字符"""
    if not character_name:
        return "unknown"
    
    # 移除或替换不适合作为文件夹名的字符
    invalid_chars = '<>:"/\\|?*'
    for char in invalid_chars:
        character_name = character_name.replace(char, '_')
    
    # 移除首尾空格并限制长度
    character_name = character_name.strip()[:100]
    
    return character_name if character_name else "unknown"


def organize_dataset(source_dir, output_dir, copy_files=True):
    """
    整理数据集
    
    Args:
        source_dir (str): 源数据目录路径
        output_dir (str): 输出目录路径
        copy_files (bool): 是否复制文件（False则移动文件）
    """
    source_path = Path(source_dir)
    output_path = Path(output_dir)
    
    if not source_path.exists():
        raise FileNotFoundError(f"源目录不存在: {source_dir}")
    
    # 创建输出目录
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 找到所有的webp文件
    webp_files = list(source_path.glob("*.webp"))
    
    if not webp_files:
        print(f"在目录 {source_dir} 中没有找到.webp文件")
        return
    
    print(f"找到 {len(webp_files)} 个图片文件")
    
    # 统计角色分布
    character_count = defaultdict(int)
    successful_pairs = 0
    failed_pairs = 0
    
    print("正在分析数据集...")
    for webp_file in tqdm(webp_files, desc="分析文件"):
        # 获取对应的JSON文件
        json_file = webp_file.with_suffix('.json')
        
        if not json_file.exists():
            print(f"警告: 找不到对应的JSON文件: {json_file}")
            failed_pairs += 1
            continue
        
        # 加载JSON标注
        annotation = load_json_annotation(json_file)
        if annotation is None:
            failed_pairs += 1
            continue
        
        # 获取角色名
        character = annotation.get('character', '')
        if isinstance(character, list):
            # 如果character是列表，取第一个或合并
            character = character[0] if character else ''
        
        character = clean_character_name(str(character))
        character_count[character] += 1
        successful_pairs += 1
    
    print(f"\n数据集统计:")
    print(f"成功处理的文件对: {successful_pairs}")
    print(f"失败的文件对: {failed_pairs}")
    print(f"发现角色数量: {len(character_count)}")
    
    # 显示角色分布
    print(f"\n角色分布 (前10个):")
    for char, count in sorted(character_count.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"  {char}: {count} 个文件")
    
    # 开始整理文件
    print(f"\n开始整理文件...")
    action = "复制" if copy_files else "移动"
    
    for webp_file in tqdm(webp_files, desc=f"{action}文件"):
        json_file = webp_file.with_suffix('.json')
        
        if not json_file.exists():
            continue
        
        # 加载JSON标注
        annotation = load_json_annotation(json_file)
        if annotation is None:
            continue
        
        # 获取角色名
        character = annotation.get('character', '')
        if isinstance(character, list):
            character = character[0] if character else ''
        
        character = clean_character_name(str(character))
        
        # 创建角色文件夹
        character_dir = output_path / character
        character_dir.mkdir(exist_ok=True)
        
        # 复制或移动文件
        try:
            if copy_files:
                shutil.copy2(webp_file, character_dir / webp_file.name)
                shutil.copy2(json_file, character_dir / json_file.name)
            else:
                shutil.move(str(webp_file), str(character_dir / webp_file.name))
                shutil.move(str(json_file), str(character_dir / json_file.name))
        except Exception as e:
            print(f"错误: 无法{action}文件 {webp_file.name}: {e}")
    
    print(f"\n整理完成!")
    print(f"输出目录: {output_dir}")
    
    # 最终统计
    created_dirs = list(output_path.iterdir())
    print(f"创建了 {len(created_dirs)} 个角色文件夹")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='整理Zerochan数据集按角色名分类')
    parser.add_argument('--source', '-s', 
                       default='/nfs/zerochan-character-w640-ws-m50-full/small_val',
                       help='源数据目录路径')
    parser.add_argument('--output', '-o', 
                       default='./zerochan_organized',
                       help='输出目录路径')
    parser.add_argument('--move', action='store_true',
                       help='移动文件而不是复制文件')
    parser.add_argument('--dry-run', action='store_true',
                       help='仅分析数据集而不实际移动/复制文件')
    
    args = parser.parse_args()
    
    if args.dry_run:
        print("--- 干运行模式 ---")
        source_path = Path(args.source)
        
        if not source_path.exists():
            print(f"错误: 源目录不存在: {args.source}")
            return
        
        webp_files = list(source_path.glob("*.webp"))
        character_count = defaultdict(int)
        
        print(f"分析源目录: {args.source}")
        print(f"找到 {len(webp_files)} 个图片文件")
        
        for webp_file in tqdm(webp_files, desc="分析文件"):
            json_file = webp_file.with_suffix('.json')
            if not json_file.exists():
                continue
            
            annotation = load_json_annotation(json_file)
            if annotation is None:
                continue
            
            character = annotation.get('character', '')
            if isinstance(character, list):
                character = character[0] if character else ''
            
            character = clean_character_name(str(character))
            character_count[character] += 1
        
        print(f"\n角色分布:")
        for char, count in sorted(character_count.items(), key=lambda x: x[1], reverse=True):
            print(f"  {char}: {count} 个文件")
        
    else:
        try:
            organize_dataset(
                source_dir=args.source,
                output_dir=args.output,
                copy_files=not args.move
            )
        except Exception as e:
            print(f"错误: {e}")
            return 1
    
    return 0


if __name__ == "__main__":
    exit(main())