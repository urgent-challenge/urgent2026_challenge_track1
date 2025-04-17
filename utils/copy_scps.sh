#!/bin/bash

input_scp=$1
output_dir=$2

# 参数检查
if [[ -z "$input_scp" || -z "$output_dir" ]]; then
    echo "用法: $0 <输入.scp文件> <输出目录>"
    exit 1
fi

# 创建输出目录
mkdir -p "$output_dir"

# 新scp文件路径（自动生成）
output_scp="${input_scp%.*}_relative.scp"

# 清空或创建新scp文件
> "$output_scp"

# 逐行处理
while IFS=" " read -r uid rate path || [ -n "$path" ]; do
    # 提取原文件扩展名（兼容带空格路径）
    filename=$(basename "$path")
    extension="${filename##*.}"

    # 生成唯一文件名：ID.扩展名
    new_file="${uid}.${extension}"
        
    # 复制文件到目标目录（保留元数据）
    cp "$path" "${output_dir}/${new_file}"
    
    # 写入新.scp文件
    echo "${uid} ${rate} ${output_dir}/${new_file}" >> "$output_scp"
done < "$input_scp"

echo "处理完成。新文件: $output_scp"