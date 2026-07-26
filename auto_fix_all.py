import re
from pathlib import Path

def fix_merged_statements(content):
    lines = content.split('\n')
    fixed_lines = []
    
    for line in lines:
        while True:
            match = re.search(r"(['\"].*?['\"])\s{2,}([a-zA-Z_]\w*\s*=)", line)
            if not match:
                match2 = re.search(r"(\))\s{2,}([a-zA-Z_]\w*\s*=)", line)
                if not match2:
                    break
                else:
                    match = match2
            
            indent = len(line) - len(line.lstrip())
            split_pos = match.start(2)
            
            part1 = line[:split_pos].rstrip()
            part2 = ' ' * indent + line[split_pos:]
            
            fixed_lines.append(part1)
            line = part2
        
        fixed_lines.append(line)
    
    return '\n'.join(fixed_lines)

def main():
    project_dir = Path(r'e:\BaiduSyncdisk\研究文件夹\#研究生学习阶段\小论文\基于多尺度增强与特征交互分组注意力的服装语义分割\代码文件')
    
    py_files = list(project_dir.rglob('*.py'))
    
    fixed_count = 0
    for py_file in py_files:
        if py_file.name in ['remove_comments.py', 'fix_newlines.py', 'auto_fix_all.py']:
            continue
        
        try:
            with open(py_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            fixed_content = fix_merged_statements(content)
            
            if content != fixed_content:
                with open(py_file, 'w', encoding='utf-8', newline='\r\n') as f:
                    f.write(fixed_content)
                print(f'Fixed: {py_file}')
                fixed_count += 1
        except Exception as e:
            print(f'Error processing {py_file}: {e}')
    
    print(f'\nDone! Fixed {fixed_count} files.')

if __name__ == '__main__':
    main()