#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
随机选择文件夹脚本
从指定目录中随机选择N个文件夹保留，其他文件夹可以删除或移动到备份目录
"""

import os
import random
import shutil
from pathlib import Path
from tqdm import tqdm
import argparse
from datetime import datetime


def get_folder_stats(folder_path):
    """获取文件夹统计信息"""
    folder_path = Path(folder_path)
    if not folder_path.is_dir():
        return 0, 0
    
    file_count = 0
    total_size = 0
    
    try:
        for file_path in folder_path.iterdir():
            if file_path.is_file():
                file_count += 1
                total_size += file_path.stat().st_size
    except PermissionError:
        pass
    
    return file_count, total_size


def format_size(size_bytes):
    """格式化文件大小"""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    elif size_bytes < 1024 * 1024 * 1024:
        return f"{size_bytes / (1024 * 1024):.1f} MB"
    else:
        return f"{size_bytes / (1024 * 1024 * 1024):.1f} GB"


def random_select_folders(source_dir, num_keep=100, backup_dir=None, delete_mode=False, seed=None):
    """
    随机选择文件夹保留
    
    Args:
        source_dir (str): 源目录路径
        num_keep (int): 要保留的文件夹数量
        backup_dir (str): 备份目录路径（如果不删除的话）
        delete_mode (bool): 是否直接删除不保留的文件夹
        seed (int): 随机种子
    """
    source_path = Path(source_dir)
    
    if not source_path.exists():
        raise FileNotFoundError(f"源目录不存在: {source_dir}")
    
    if not source_path.is_dir():
        raise ValueError(f"路径不是目录: {source_dir}")
    
    # 设置随机种子
    if seed is not None:
        random.seed(seed)
        print(f"使用随机种子: {seed}")
    
    # 获取所有子文件夹
    all_folders = [f for f in source_path.iterdir() if f.is_dir()]
    
    if not all_folders:
        print(f"在目录 {source_dir} 中没有找到子文件夹")
        return
    
    total_folders = len(all_folders)
    print(f"找到 {total_folders} 个文件夹")
    
    if num_keep >= total_folders:
        print(f"要保留的文件夹数量 ({num_keep}) 大于或等于总文件夹数量 ({total_folders})")
        print("所有文件夹都将被保留")
        return
    
    # 分析每个文件夹的内容
    print("正在分析文件夹内容...")
    folder_stats = {}
    total_files = 0
    total_size = 0
    
    for folder in tqdm(all_folders, desc="分析文件夹"):
        file_count, folder_size = get_folder_stats(folder)
        folder_stats[folder] = (file_count, folder_size)
        total_files += file_count
        total_size += folder_size
    
    print(f"\n数据集统计:")
    print(f"总文件夹数: {total_folders}")
    print(f"总文件数: {total_files}")
    print(f"总大小: {format_size(total_size)}")
    
    # 显示文件夹大小分布 (前10个最大的)
    print(f"\n文件夹大小排名 (前10个):")
    sorted_folders = sorted(folder_stats.items(), key=lambda x: x[1][0], reverse=True)
    for i, (folder, (file_count, folder_size)) in enumerate(sorted_folders[:10]):
        print(f"  {i+1:2d}. {folder.name}: {file_count} 个文件, {format_size(folder_size)}")
    
    # 随机选择要保留的文件夹
    folders_to_keep = random.sample(all_folders, num_keep)
    folders_to_remove = [f for f in all_folders if f not in folders_to_keep]
    
    print(f"\n随机选择结果:")
    print(f"保留文件夹数: {len(folders_to_keep)}")
    print(f"{'删除' if delete_mode else '移动'}文件夹数: {len(folders_to_remove)}")
    
    # 统计保留和移除的数据量
    keep_files = sum(folder_stats[f][0] for f in folders_to_keep)
    keep_size = sum(folder_stats[f][1] for f in folders_to_keep)
    remove_files = sum(folder_stats[f][0] for f in folders_to_remove)
    remove_size = sum(folder_stats[f][1] for f in folders_to_remove)
    
    print(f"\n保留数据统计:")
    print(f"  文件数: {keep_files} ({keep_files/total_files*100:.1f}%)")
    print(f"  大小: {format_size(keep_size)} ({keep_size/total_size*100:.1f}%)")
    
    print(f"\n{'删除' if delete_mode else '移动'}数据统计:")
    print(f"  文件数: {remove_files} ({remove_files/total_files*100:.1f}%)")
    print(f"  大小: {format_size(remove_size)} ({remove_size/total_size*100:.1f}%)")
    
    # 显示保留的文件夹名称
    print(f"\n保留的文件夹 (前20个):")
    keep_names = sorted([f.name for f in folders_to_keep])
    for name in keep_names[:20]:
        print(f"  {name}")
    if len(keep_names) > 20:
        print(f"  ... 还有 {len(keep_names) - 20} 个文件夹")
    
    return folders_to_keep, folders_to_remove


def process_folders(folders_to_remove, backup_dir=None, delete_mode=False):
    """处理要移除的文件夹"""
    if not folders_to_remove:
        print("没有需要处理的文件夹")
        return
    
    if delete_mode:
        print(f"\n开始删除 {len(folders_to_remove)} 个文件夹...")
        
        for folder in tqdm(folders_to_remove, desc="删除文件夹"):
            try:
                shutil.rmtree(folder)
            except Exception as e:
                print(f"错误: 无法删除文件夹 {folder.name}: {e}")
        
        print("删除完成!")
    
    else:
        if not backup_dir:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_dir = f"./backup_folders_{timestamp}"
        
        backup_path = Path(backup_dir)
        backup_path.mkdir(parents=True, exist_ok=True)
        
        print(f"\n开始移动 {len(folders_to_remove)} 个文件夹到 {backup_dir}...")
        
        for folder in tqdm(folders_to_remove, desc="移动文件夹"):
            try:
                dest_path = backup_path / folder.name
                shutil.move(str(folder), str(dest_path))
            except Exception as e:
                print(f"错误: 无法移动文件夹 {folder.name}: {e}")
        
        print(f"移动完成! 备份目录: {backup_dir}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='从目录中随机选择指定数量的文件夹保留')
    parser.add_argument('--source', '-s',
                       default='/nfs/zerochan-character-w640-ws-m50-full/small_val_folder',
                       help='源数据目录路径')
    parser.add_argument('--keep', '-k', type=int, default=100,
                       help='要保留的文件夹数量')
    parser.add_argument('--backup', '-b',
                       help='备份目录路径（如果不删除的话）')
    parser.add_argument('--delete', action='store_true',
                       help='直接删除不保留的文件夹，而不是移动到备份目录')
    parser.add_argument('--seed', type=int,
                       help='随机种子，用于可重复的选择')
    parser.add_argument('--dry-run', action='store_true',
                       help='仅分析和显示选择结果，不实际处理文件夹')
    
    args = parser.parse_args()
    
    try:
        result = random_select_folders(
            source_dir=args.source,
            num_keep=args.keep,
            backup_dir=args.backup,
            delete_mode=args.delete,
            seed=args.seed
        )
        
        if result is None:
            return 0
        
        folders_to_keep, folders_to_remove = result
        
        if args.dry_run:
            print("\n--- 干运行模式，不会实际处理文件夹 ---")
            return 0
        
        # 确认操作
        action = "删除" if args.delete else "移动到备份目录"
        response = input(f"\n确定要{action} {len(folders_to_remove)} 个文件夹吗? (输入 'yes' 确认): ")
        
        if response.lower() != 'yes':
            print("操作已取消")
            return 0
        
        # 执行处理
        process_folders(
            folders_to_remove=folders_to_remove,
            backup_dir=args.backup,
            delete_mode=args.delete
        )
        
        print(f"\n操作完成! 保留了 {len(folders_to_keep)} 个文件夹")
        
    except Exception as e:
        print(f"错误: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())