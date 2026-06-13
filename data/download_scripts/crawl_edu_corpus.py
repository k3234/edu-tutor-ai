"""
教育语料采集脚本

采集中文教育领域的文本数据，用于训练 LMM 语言模型。
支持多种数据源，输出为纯文本文件。

数据源:
    - Wikipedia 中文 dump（百科知识）
    - 教材文本（需要手动准备）
    - 开源中文语料（如 WuDaoCorpora 子集）

使用:
    python data/download_scripts/crawl_edu_corpus.py --output_dir data/raw --max_size_mb 300
"""

import argparse
import os
import re
from pathlib import Path


# 教育领域关键词（用于筛选/过滤文本）
EDU_KEYWORDS = [
    "数学", "物理", "化学", "生物", "语文", "英语", "历史", "地理",
    "定理", "公式", "方程", "函数", "几何", "代数", "微积分",
    "牛顿", "爱因斯坦", "达尔文", "元素周期表",
    "光合作用", "细胞", "基因", "DNA",
    "中国古代", "朝代", "诗人", "文学作品",
]


def clean_text(text: str) -> str:
    """
    清洗文本

    去除多余空白、特殊字符、HTML 标签等。

    Args:
        text: 原始文本

    Returns:
        清洗后的文本
    """
    # 去除 HTML 标签
    text = re.sub(r'<[^>]+>', '', text)

    # 去除 URL
    text = re.sub(r'http[s]?://\S+', '', text)

    # 去除多余空白
    text = re.sub(r'\s+', ' ', text)

    # 去除特殊字符（保留中文、英文、数字、基本标点）
    text = re.sub(r'[^\u4e00-\u9fff\u3000-\u303f\uff00-\uffefa-zA-Z0-9.,;:!?\s]', '', text)

    # 去除过长数字（可能是乱码）
    text = re.sub(r'\d{10,}', '', text)

    return text.strip()


def is_edu_relevant(text: str, min_keywords: int = 1) -> bool:
    """
    判断文本是否与教育领域相关

    Args:
        text: 文本内容
        min_keywords: 最少需要匹配的关键词数量

    Returns:
        是否相关
    """
    matches = sum(1 for kw in EDU_KEYWORDS if kw in text)
    return matches >= min_keywords


def process_wikipedia_dump(input_file: str, output_file: str, max_articles: int = 10000):
    """
    处理 Wikipedia XML dump 文件

    提取纯文本内容，过滤教育相关文章。

    Args:
        input_file: Wikipedia dump 文件路径
        output_file: 输出文本文件路径
        max_articles: 最大处理文章数
    """
    print(f"处理 Wikipedia dump: {input_file}")

    try:
        import xml.etree.ElementTree as ET
    except ImportError:
        print("警告: 无法导入 xml 模块，跳过 Wikipedia 处理")
        return

    count = 0
    edu_count = 0

    with open(output_file, 'w', encoding='utf-8') as out:
        # 简化处理: 逐行读取，提取 <text> 标签内容
        # 实际生产环境建议使用 wikiextractor 工具
        for line in open(input_file, 'r', encoding='utf-8'):
            if '<text' in line and '</text>' in line:
                # 提取文本内容
                start = line.find('>') + 1
                end = line.find('</text>')
                text = line[start:end]

                # 清洗
                text = clean_text(text)

                if len(text) > 100:  # 过滤过短文本
                    count += 1
                    if is_edu_relevant(text):
                        out.write(text + '\n')
                        edu_count += 1

                if count >= max_articles:
                    break

    print(f"  处理完成: {count} 篇文章, {edu_count} 篇教育相关")


def generate_sample_corpus(output_dir: str, size_mb: int = 10):
    """
    生成示例语料（用于快速测试）

    当没有真实数据时，生成一些教育领域的示例文本。

    Args:
        output_dir: 输出目录
        size_mb: 目标文件大小（MB）
    """
    print(f"生成示例语料到: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)

    sample_texts = [
        # 数学
        "勾股定理是指直角三角形两直角边的平方和等于斜边的平方。设直角三角形的两直角边长度分别为a和b，斜边长度为c，则有a² + b² = c²。",
        "一元二次方程的一般形式为ax² + bx + c = 0，其中a、b、c为常数，且a≠0。求解公式为x = (-b ± √(b² - 4ac)) / 2a。",
        "函数的导数表示函数在某一点的变化率。若f(x) = x²，则f'(x) = 2x。",

        # 物理
        "牛顿第一定律，又称惯性定律，指出：任何物体都要保持匀速直线运动或静止状态，直到外力迫使它改变运动状态为止。",
        "能量守恒定律：能量既不会凭空产生，也不会凭空消失，它只会从一种形式转化为另一种形式，或者从一个物体转移到另一个物体，而能量的总量保持不变。",
        "欧姆定律：在同一电路中，通过某段导体的电流跟这段导体两端的电压成正比，跟这段导体的电阻成反比。公式为I = U/R。",

        # 化学
        "元素周期表是按照原子序数从小到大排列的化学元素列表。目前已发现118种元素。",
        "光合作用是绿色植物利用光能，将二氧化碳和水转化为有机物并释放氧气的过程。化学方程式为：6CO₂ + 6H₂O → C₆H₁₂O₆ + 6O₂。",
        "酸碱中和反应是指酸和碱互相交换成分，生成盐和水的反应。例如：HCl + NaOH → NaCl + H₂O。",

        # 生物
        "细胞是生物体结构和功能的基本单位。所有生物都由细胞组成（病毒除外）。",
        "DNA是脱氧核糖核酸的缩写，是生物体内储存遗传信息的分子。DNA由四种碱基组成：腺嘌呤(A)、胸腺嘧啶(T)、鸟嘌呤(G)、胞嘧啶(C)。",
        "自然选择是达尔文进化论的核心概念，指环境对生物变异进行选择，适应环境的个体更容易生存和繁殖。",

        # 语文/历史
        "《诗经》是中国最早的诗歌总集，收录了西周初年至春秋中叶的诗歌305篇，分为风、雅、颂三部分。",
        "唐朝是中国历史上最强盛的朝代之一，共历二十一帝，享国二百八十九年。",
        "李白是唐代伟大的浪漫主义诗人，被后人誉为诗仙，代表作有《静夜思》《将进酒》等。",

        # 英语
        "English is a West Germanic language that was first spoken in early medieval England. It is the third most spoken native language in the world.",
        "William Shakespeare was an English playwright, poet, and actor, widely regarded as the greatest writer in the English language.",
    ]

    output_file = os.path.join(output_dir, "sample_corpus.txt")

    with open(output_file, 'w', encoding='utf-8') as f:
        # 重复写入直到达到目标大小
        target_bytes = size_mb * 1024 * 1024
        current_bytes = 0

        while current_bytes < target_bytes:
            for text in sample_texts:
                line = text + '\n\n'
                f.write(line)
                current_bytes += len(line.encode('utf-8'))

                if current_bytes >= target_bytes:
                    break

    actual_size = os.path.getsize(output_file) / (1024 * 1024)
    print(f"  示例语料生成完成: {actual_size:.2f} MB")
    print(f"  文件路径: {output_file}")


def main():
    parser = argparse.ArgumentParser(description="教育语料采集")
    parser.add_argument("--output_dir", type=str, default="data/raw", help="输出目录")
    parser.add_argument("--max_size_mb", type=int, default=50, help="目标语料大小（MB）")
    parser.add_argument("--sample", action="store_true", help="生成示例语料（用于测试）")
    parser.add_argument("--wiki_file", type=str, default=None, help="Wikipedia dump 文件路径")

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if args.sample:
        # 生成示例语料
        generate_sample_corpus(args.output_dir, args.max_size_mb)
    elif args.wiki_file:
        # 处理 Wikipedia dump
        output_file = os.path.join(args.output_dir, "wikipedia_edu.txt")
        process_wikipedia_dump(args.wiki_file, output_file)
    else:
        print("请指定数据源:")
        print("  --sample      生成示例语料")
        print("  --wiki_file   处理 Wikipedia dump")
        print("\n示例:")
        print("  python crawl_edu_corpus.py --sample --max_size_mb 100")


if __name__ == "__main__":
    main()