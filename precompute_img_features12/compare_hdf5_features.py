#!/usr/bin/env python3
"""
比较两个HDF5特征文件是否一致。
使用多种方法验证特征提取代码是否正确。
"""

import h5py
import numpy as np
import os


def compare_hdf5_files(file1_path, file2_path, name="", num_samples=20):
    """
    使用多种方法比较两个HDF5文件的内容。
    
    Args:
        file1_path: 第一个HDF5文件路径 (生成的)
        file2_path: 第二个HDF5文件路径 (原始下载的)
        name: 用于日志的名称
        num_samples: 详细分析的样本数量
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
        return
    if not os.path.exists(file2_path):
        print(f"❌ 错误: 文件2不存在: {file2_path}")
        return
    
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
        
        common_keys = sorted(keys1 & keys2)
        print(f"共有 {len(common_keys)} 个相同的keys")
        
        if not common_keys:
            print("❌ 没有共同的keys可以比较!")
            return
        
        # 检查数据形状
        sample_key = common_keys[0]
        shape1 = f1[sample_key].shape
        shape2 = f2[sample_key].shape
        print(f"\n样本数据信息:")
        print(f"  Key: '{sample_key}'")
        print(f"  文件1 shape: {shape1}, dtype: {f1[sample_key].dtype}")
        print(f"  文件2 shape: {shape2}, dtype: {f2[sample_key].dtype}")
        
        # ============================================================
        # 详细分析：随机抽取样本进行多种比较
        # ============================================================
        print(f"\n{'='*80}")
        print(f"详细分析: 随机抽取 {num_samples} 个样本")
        print(f"{'='*80}")
        
        # 随机抽取样本
        np.random.seed(42)
        sample_indices = np.random.choice(len(common_keys), min(num_samples, len(common_keys)), replace=False)
        sample_keys = [common_keys[i] for i in sample_indices]
        
        # 收集统计数据
        all_l2_diffs = []        # L2范数差异
        all_cosine_sims = []     # 余弦相似度
        all_rel_diffs = []       # 相对差异
        all_max_abs_diffs = []   # 最大绝对差异
        
        for idx, key in enumerate(sample_keys):
            data1 = f1[key][...].astype(np.float32)
            data2 = f2[key][...].astype(np.float32)
            
            if data1.shape != data2.shape:
                print(f"  ⚠️ {key}: 形状不匹配 {data1.shape} vs {data2.shape}")
                continue
            
            # 展平为1D进行比较
            flat1 = data1.flatten()
            flat2 = data2.flatten()
            
            # 1. L2范数 (欧氏距离)
            l2_diff = np.linalg.norm(flat1 - flat2)
            l2_norm1 = np.linalg.norm(flat1)
            l2_norm2 = np.linalg.norm(flat2)
            
            # 2. 余弦相似度
            cos_sim = np.dot(flat1, flat2) / (l2_norm1 * l2_norm2 + 1e-8)
            
            # 3. 相对差异 (相对于L2范数)
            rel_diff = l2_diff / (l2_norm2 + 1e-8)
            
            # 4. 最大绝对差异
            max_abs_diff = np.max(np.abs(flat1 - flat2))
            
            # 5. 平均绝对差异
            mean_abs_diff = np.mean(np.abs(flat1 - flat2))
            
            all_l2_diffs.append(l2_diff)
            all_cosine_sims.append(cos_sim)
            all_rel_diffs.append(rel_diff)
            all_max_abs_diffs.append(max_abs_diff)
            
            # 打印前几个样本的详细信息
            if idx < 5:
                print(f"\n  样本 {idx+1}: {key}")
                print(f"    形状: {data1.shape}")
                print(f"    文件1 L2范数: {l2_norm1:.4f}")
                print(f"    文件2 L2范数: {l2_norm2:.4f}")
                print(f"    L2距离 (差异向量范数): {l2_diff:.6f}")
                print(f"    相对差异 (L2距离/L2范数): {rel_diff:.6f} ({rel_diff*100:.4f}%)")
                print(f"    余弦相似度: {cos_sim:.6f}")
                print(f"    最大绝对差异: {max_abs_diff:.6f}")
                print(f"    平均绝对差异: {mean_abs_diff:.6f}")
                print(f"    文件1 前10个值: {flat1[:10]}")
                print(f"    文件2 前10个值: {flat2[:10]}")
                print(f"    差异 前10个值: {(flat1[:10] - flat2[:10])}")
        
        # ============================================================
        # 全局统计
        # ============================================================
        print(f"\n{'='*80}")
        print("全局统计 (基于抽样)")
        print(f"{'='*80}")
        
        print(f"\n  L2距离 (差异向量的范数):")
        print(f"    最小: {np.min(all_l2_diffs):.6f}")
        print(f"    最大: {np.max(all_l2_diffs):.6f}")
        print(f"    平均: {np.mean(all_l2_diffs):.6f}")
        print(f"    中位数: {np.median(all_l2_diffs):.6f}")
        
        print(f"\n  相对差异 (L2距离 / 文件2的L2范数):")
        print(f"    最小: {np.min(all_rel_diffs)*100:.4f}%")
        print(f"    最大: {np.max(all_rel_diffs)*100:.4f}%")
        print(f"    平均: {np.mean(all_rel_diffs)*100:.4f}%")
        print(f"    中位数: {np.median(all_rel_diffs)*100:.4f}%")
        
        print(f"\n  余弦相似度 (1.0表示完全相同方向):")
        print(f"    最小: {np.min(all_cosine_sims):.6f}")
        print(f"    最大: {np.max(all_cosine_sims):.6f}")
        print(f"    平均: {np.mean(all_cosine_sims):.6f}")
        
        print(f"\n  最大元素级绝对差异:")
        print(f"    最小: {np.min(all_max_abs_diffs):.6f}")
        print(f"    最大: {np.max(all_max_abs_diffs):.6f}")
        print(f"    平均: {np.mean(all_max_abs_diffs):.6f}")
        
        # ============================================================
        # 结论判断
        # ============================================================
        print(f"\n{'='*80}")
        print("结论")
        print(f"{'='*80}")
        
        avg_cos_sim = np.mean(all_cosine_sims)
        avg_rel_diff = np.mean(all_rel_diffs)
        
        if avg_cos_sim > 0.9999 and avg_rel_diff < 0.001:
            print("✅ 特征几乎完全相同 (余弦相似度 > 0.9999, 相对差异 < 0.1%)")
            print("   可能只是浮点精度差异，特征提取代码应该是正确的。")
        elif avg_cos_sim > 0.999 and avg_rel_diff < 0.01:
            print("✅ 特征非常接近 (余弦相似度 > 0.999, 相对差异 < 1%)")
            print("   差异很小，可能是随机性或浮点精度导致，特征提取代码很可能是正确的。")
        elif avg_cos_sim > 0.99 and avg_rel_diff < 0.05:
            print("⚠️ 特征比较接近 (余弦相似度 > 0.99, 相对差异 < 5%)")
            print("   存在一定差异，需要进一步检查是否有问题。")
        elif avg_cos_sim > 0.9:
            print("⚠️ 特征有明显差异 (余弦相似度在0.9-0.99之间)")
            print("   建议检查特征提取的预处理步骤或模型权重。")
        else:
            print("❌ 特征差异很大 (余弦相似度 < 0.9)")
            print("   特征提取代码可能存在问题，需要仔细检查。")


def main():
    # 假设脚本在 precompute_img_features12 目录下
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(script_dir)  # 项目根目录
    
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
    print("HDF5 特征文件对比工具 (改进版)")
    print("=" * 80)
    print("\n使用多种方法比较特征:")
    print("  1. L2范数 (欧氏距离)")
    print("  2. 余弦相似度")
    print("  3. 相对差异")
    print("  4. 最大/平均绝对差异")
    print()
    
    for file1, file2, name in comparison_pairs:
        try:
            compare_hdf5_files(file1, file2, name)
        except Exception as e:
            print(f"\n❌ 处理 {name} 时发生错误: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()
