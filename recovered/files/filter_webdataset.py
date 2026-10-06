#!/usr/bin/env python3
"""
简单的WebDataset过滤脚本

使用webdataset的ShardWriter API自动创建多个分片文件
每个分片最多包含50000个样本（可配置）
"""

import webdataset as wds
import json
from pathlib import Path
from tqdm import tqdm


def filter_webdataset(input_shards, output_pattern, filter_func, max_samples=None, max_samples_per_shard=50000):
    """
    过滤webdataset并创建新的分片tar文件
    
    Args:
        input_shards: 输入分片文件路径列表或单个路径
        output_pattern: 输出文件模式，如 "./filtered/shard-%06d.tar"
        filter_func: 过滤函数，接收json metadata，返回True/False
        max_samples: 最大样本数量，None表示不限制
        max_samples_per_shard: 每个分片的最大样本数
    """
    
    if isinstance(input_shards, str):
        input_shards = [input_shards]
    
    # 创建输出目录
    output_dir = Path(output_pattern).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 创建数据管道
    dataset = wds.DataPipeline(
        wds.SimpleShardList(input_shards),
        wds.tarfile_to_samples(),
    )
    
    filtered_count = 0
    total_count = 0
    
    print(f"开始过滤数据集...")
    print(f"输入: {input_shards}")
    print(f"输出模式: {output_pattern}")
    print(f"每个分片最大样本数: {max_samples_per_shard}")
    
    # 使用ShardWriter写入分片
    with wds.ShardWriter(output_pattern, maxcount=max_samples_per_shard) as sink:
        for sample in tqdm(dataset, desc="过滤样本"):
            total_count += 1
            
            # 解析metadata
            try:
                if 'json' in sample:
                    json_data = sample['json']
                    if isinstance(json_data, bytes):
                        json_data = json_data.decode('utf-8')
                    metadata = json.loads(json_data)
                else:
                    metadata = {}
            except:
                print(f"跳过样本 {sample.get('__key__', 'unknown')}: JSON解析失败")
                continue
            
            # 应用过滤条件
            if filter_func(metadata):
                # 保留这个样本，准备写入数据
                key = sample.get('__key__', f'sample_{filtered_count}')
                
                # 构建输出样本字典
                output_sample = {"__key__": key}
                
                for ext, data in sample.items():
                    if ext.startswith('__'):
                        continue
                    output_sample[ext] = data
                
                # 写入样本
                sink.write(output_sample)
                
                filtered_count += 1
                
                # 检查是否达到最大样本数
                if max_samples and filtered_count >= max_samples:
                    break
    
    print(f"过滤完成!")
    print(f"总样本数: {total_count}")
    print(f"保留样本数: {filtered_count}")
    if total_count > 0:
        print(f"保留率: {filtered_count/total_count*100:.2f}%")


# 常用过滤函数
def filter_by_character(characters):
    """根据角色过滤"""
    def func(metadata):
        character = metadata.get('character', '')
        return character in characters
    return func


def filter_by_tags(required_tags=None, forbidden_tags=None):
    """根据标签过滤"""
    def func(metadata):
        tags = metadata.get('general_tags', [])
        
        if required_tags:
            if not all(tag in tags for tag in required_tags):
                return False
        
        if forbidden_tags:
            if any(tag in tags for tag in forbidden_tags):
                return False
        
        return True
    return func


def filter_by_artist(artists):
    """根据艺术家过滤"""
    def func(metadata):
        artist_tags = metadata.get('artist_tags', [])
        return any(artist in artist_tags for artist in artists)
    return func


if __name__ == "__main__":
    # 使用示例
    
    # 输入文件
    input_files = [
        "/nfs/danbooru-wdtagger-v4-w640-ws-full-witha/test/00001.tar"
    ]
    
    # 示例1: 过滤特定角色
    filter_webdataset(
        input_shards=input_files,
        output_pattern="./filtered_character/shard-%06d.tar",
        filter_func=filter_by_character(['hatsune_miku', 'kagamine_rin']),
        max_samples=100000,
        max_samples_per_shard=50000
    )
    
    # 示例2: 过滤特定标签
    filter_webdataset(
        input_shards=input_files,
        output_pattern="./filtered_tags/shard-%06d.tar",
        filter_func=filter_by_tags(required_tags=['1girl'], forbidden_tags=['nsfw']),
        max_samples=100000,
        max_samples_per_shard=50000
    )
    
    # 示例3: 自定义过滤条件
    def custom_filter(metadata):
        # 只要有艺术家标签且角色不为空
        return (metadata.get('artist_tags', []) and 
                metadata.get('character', ''))
    
    filter_webdataset(
        input_shards=input_files,
        output_pattern="./filtered_custom/shard-%06d.tar",
        filter_func=custom_filter,
        max_samples=200000,
        max_samples_per_shard=50000
    )