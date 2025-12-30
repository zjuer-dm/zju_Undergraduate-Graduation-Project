#!/usr/bin/env python3
"""
比较两个HDF5特征文件是否一致。
用于验证特征提取代码是否正确。
"""

import h5py
import numpy as np
import os


def compare_hdf5_files(file1_path, file2_path, name="", max_samples=10, atol=1e-5, rtol=1e-5):
    """
    比较两个HDF5文件的内容。
    
    Args:
        file1_path: 第一个HDF5文件路径 (生成的)
        file2_path: 第二个HDF5文件路径 (原始下载的)
        name: 用于日志的名称
        max_samples: 打印详细对比的最大样本数
        atol: 绝对误差容忍度
        rtol: 相对误差容忍度
    """
    print(f"\n{'='*80}")
    print(f"比较: {name}")
    print(f"{'='*80}")
    print(f"文件1 (生成的): {file1_path}")
    print(f"文件2 (原始的): {file2_path}")
    print(f"{'='*80}\n")
    
    # 检查文件是否存在
    if not os.path.exists(file1_path):
        print(f"❌ 错误: 文件1不存在: {file1_path}")
        return False
    if not os.path.exists(file2_path):
        print(f"❌ 错误: 文件2不存在: {file2_path}")
        return False
    
    # 获取文件大小
    size1 = os.path.getsize(file1_path) / (1024 * 1024)  # MB
    size2 = os.path.getsize(file2_path) / (1024 * 1024)  # MB
    print(f"文件1大小: {size1:.2f} MB")
    print(f"文件2大小: {size2:.2f} MB")
    print()
    
    with h5py.File(file1_path, 'r') as f1, h5py.File(file2_path, 'r') as f2:
        keys1 = set(f1.keys())
        keys2 = set(f2.keys())
        
        print(f"文件1 keys数量: {len(keys1)}")
        print(f"文件2 keys数量: {len(keys2)}")
        
        # 检查keys是否一致
        only_in_1 = keys1 - keys2
        only_in_2 = keys2 - keys1
        common_keys = keys1 & keys2
        
        if only_in_1:
            print(f"\n⚠️  只在文件1中存在的keys ({len(only_in_1)}个):")
            for k in list(only_in_1)[:5]:
                print(f"    {k}")
            if len(only_in_1) > 5:
                print(f"    ... 还有 {len(only_in_1) - 5} 个")
        
        if only_in_2:
            print(f"\n⚠️  只在文件2中存在的keys ({len(only_in_2)}个):")
            for k in list(only_in_2)[:5]:
                print(f"    {k}")
            if len(only_in_2) > 5:
                print(f"    ... 还有 {len(only_in_2) - 5} 个")
        
        print(f"\n共有 {len(common_keys)} 个相同的keys")
        
        if not common_keys:
            print("❌ 没有共同的keys可以比较!")
            return False
        
        # 检查第一个key的形状
        sample_key = list(common_keys)[0]
        shape1 = f1[sample_key].shape
        shape2 = f2[sample_key].shape
        dtype1 = f1[sample_key].dtype
        dtype2 = f2[sample_key].dtype
        
        print(f"\n样本数据形状:")
        print(f"  文件1 '{sample_key}': shape={shape1}, dtype={dtype1}")
        print(f"  文件2 '{sample_key}': shape={shape2}, dtype={dtype2}")
        
        # 检查所有数据是否一致
        print(f"\n{'='*60}")
        print("开始逐个比较特征值...")
        print(f"{'='*60}")
        
        total_checked = 0
        shape_mismatch = 0
        value_mismatch = 0
        exact_match = 0
        close_match = 0  # 在容忍度内匹配
        
        mismatch_details = []
        
        for i, key in enumerate(sorted(common_keys)):
            data1 = f1[key][...]
            data2 = f2[key][...]
            
            total_checked += 1
            
            # 检查形状
            if data1.shape != data2.shape:
                shape_mismatch += 1
                if len(mismatch_details) < max_samples:
                    mismatch_details.append({
                        'key': key,
                        'type': 'shape',
                        'shape1': data1.shape,
                        'shape2': data2.shape
                    })
                continue
            
            # 检查值是否完全相同
            if np.array_equal(data1, data2):
                exact_match += 1
            elif np.allclose(data1, data2, atol=atol, rtol=rtol):
                close_match += 1
            else:
                value_mismatch += 1
                if len(mismatch_details) < max_samples:
                    max_diff = np.max(np.abs(data1 - data2))
                    mean_diff = np.mean(np.abs(data1 - data2))
                    mismatch_details.append({
                        'key': key,
                        'type': 'value',
                        'max_diff': max_diff,
                        'mean_diff': mean_diff,
                        'sample1': data1.flatten()[:5],
                        'sample2': data2.flatten()[:5]
                    })
            
            # 进度显示
            if (i + 1) % 1000 == 0:
                print(f"  已检查 {i + 1}/{len(common_keys)} 个viewpoints...")
        
        # 打印结果统计
        print(f"\n{'='*60}")
        print("比较结果统计:")
        print(f"{'='*60}")
        print(f"总共检查: {total_checked} 个viewpoints")
        print(f"  ✅ 完全匹配: {exact_match}")
        print(f"  ✅ 近似匹配 (atol={atol}, rtol={rtol}): {close_match}")
        print(f"  ⚠️  形状不匹配: {shape_mismatch}")
        print(f"  ❌ 数值不匹配: {value_mismatch}")
        
        # 打印不匹配的详细信息
        if mismatch_details:
            print(f"\n{'='*60}")
            print(f"不匹配详情 (最多显示{max_samples}个):")
            print(f"{'='*60}")
            for detail in mismatch_details:
                if detail['type'] == 'shape':
                    print(f"\n  Key: {detail['key']}")
                    print(f"    类型: 形状不匹配")
                    print(f"    文件1 shape: {detail['shape1']}")
                    print(f"    文件2 shape: {detail['shape2']}")
                else:
                    print(f"\n  Key: {detail['key']}")
                    print(f"    类型: 数值不匹配")
                    print(f"    最大差异: {detail['max_diff']:.6e}")
                    print(f"    平均差异: {detail['mean_diff']:.6e}")
                    print(f"    文件1 前5个值: {detail['sample1']}")
                    print(f"    文件2 前5个值: {detail['sample2']}")
        
        # 总结
        print(f"\n{'='*60}")
        if shape_mismatch == 0 and value_mismatch == 0:
            print("✅ 结论: 两个文件完全相同或在容忍度内一致!")
            return True
        else:
            print("❌ 结论: 两个文件存在差异!")
            return False


def main():
    # 定义要比较的文件路径（相对于项目根目录）
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 比较对列表: (生成的文件, 原始文件, 名称)
    comparison_pairs = [
        (
            os.path.join(base_dir, "pretrain_src/image_feature/CLIP-ViT-B-32-views-habitat.hdf5"),
            os.path.join(base_dir, "pretrain_src/img_features/CLIP-ViT-B-32-views-habitat.hdf5"),
            "RGB Features (CLIP-ViT-B-32)"
        ),
        (
            os.path.join(base_dir, "pretrain_src/image_feature/ddppo_resnet50_depth_features.hdf5"),
            os.path.join(base_dir, "pretrain_src/img_features/ddppo_resnet50_depth_features.hdf5"),
            "Depth Features (ddppo_resnet50)"
        ),
    ]
    
    print("\n" + "=" * 80)
    print("HDF5 特征文件对比工具")
    print("=" * 80)
    print("\n这个脚本将比较生成的特征文件与原始下载的特征文件")
    print("以验证特征提取代码是否正确。\n")
    
    results = []
    for file1, file2, name in comparison_pairs:
        result = compare_hdf5_files(file1, file2, name)
        results.append((name, result))
    
    # 打印最终总结
    print("\n" + "=" * 80)
    print("最终总结")
    print("=" * 80)
    for name, result in results:
        status = "✅ 一致" if result else "❌ 不一致"
        print(f"  {name}: {status}")
    
    all_passed = all(r[1] for r in results)
    print()
    if all_passed:
        print("🎉 所有特征文件都一致! 特征提取代码应该是正确的。")
    else:
        print("⚠️  存在不一致的特征文件，需要进一步检查特征提取代码。")
    print("=" * 80)


if __name__ == "__main__":
    main()
