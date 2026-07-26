import os
import re
from pathlib import Path

def fix_merged_lines(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    lines = content.split('\n')
    fixed_lines = []
    
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        
        if not stripped:
            fixed_lines.append(line)
            i += 1
            continue
        
        # 查找字符串后面紧跟变量赋值的情况
        # 例如: model_path = ""     Init_Epoch = 0
        # 或者: ENDC = '\033[0m'      BOLD = '\033[1m'
        
        # 匹配模式: 字符串结尾 + 空格 + 新的变量名 = 
        pattern = r"^(.*?['\"].*?['\"])\s{2,}([a-zA-Z_]\w*\s*=.*)$"
        match = re.match(pattern, line)
        
        if match:
            part1 = match.group(1).rstrip()
            part2 = match.group(2)
            indent = len(line) - len(line.lstrip())
            fixed_lines.append(part1)
            fixed_lines.append(' ' * indent + part2)
        else:
            fixed_lines.append(line)
        
        i += 1
    
    return '\n'.join(fixed_lines)

def main():
    project_dir = Path(r'e:\BaiduSyncdisk\研究文件夹\#研究生学习阶段\小论文\基于多尺度增强与特征交互分组注意力的服装语义分割\代码文件')
    
    py_files = list(project_dir.rglob('*.py'))
    
    fixed_count = 0
    for py_file in py_files:
        if py_file.name in ['remove_comments.py', 'fix_newlines.py']:
            continue
        
        print(f'Fixing: {py_file}')
        try:
            fixed_content = fix_merged_lines(py_file)
            
            with open(py_file, 'w', encoding='utf-8', newline='\r\n') as f:
                f.write(fixed_content)
            
            fixed_count += 1
        except Exception as e:
            print(f'Error fixing {py_file}: {e}')
    
    print(f'\nDone! Fixed {fixed_count} files.')

if __name__ == '__main__':
    main()