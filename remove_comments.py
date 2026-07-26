import tokenize
import io
import os
from pathlib import Path

def remove_comments_from_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        source = f.read()
    
    try:
        tokens = tokenize.generate_tokens(io.StringIO(source).readline)
        result = []
        prev_token = None
        
        for token_type, token_string, start, end, line in tokens:
            if token_type == tokenize.COMMENT:
                continue
            if token_type == tokenize.STRING:
                if (prev_token and prev_token[0] == tokenize.NEWLINE or 
                    prev_token is None or 
                    prev_token[0] == tokenize.INDENT or
                    prev_token[0] == tokenize.NL):
                    continue
            
            result.append(token_string)
            prev_token = (token_type, token_string, start, end, line)
        
        return tokenize.untokenize(result)
    except tokenize.TokenError:
        return source

def remove_comments_simple(file_path):
    import re
    
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    result = []
    in_multiline_string = False
    multiline_char = None
    
    for line in lines:
        stripped = line.strip()
        
        if stripped.startswith('#'):
            continue
        
        if '#' in line:
            in_string = False
            string_char = None
            new_line = []
            i = 0
            while i < len(line):
                char = line[i]
                
                if in_string:
                    new_line.append(char)
                    if char == string_char and (i == 0 or line[i-1] != '\\'):
                        in_string = False
                elif char in ['"', "'"]:
                    if i + 2 < len(line) and line[i:i+3] in ['"""', "'''"]:
                        quote = line[i:i+3]
                        end_pos = line.find(quote, i+3)
                        if end_pos == -1:
                            i += 3
                            continue
                        new_line.append(quote)
                        i = end_pos + 3
                        continue
                    else:
                        in_string = True
                        string_char = char
                        new_line.append(char)
                elif char == '#':
                    break
                else:
                    new_line.append(char)
                
                i += 1
            
            result.append(''.join(new_line))
        else:
            result.append(line)
    
    return ''.join(result)

def main():
    project_dir = Path(r'e:\BaiduSyncdisk\研究文件夹\#研究生学习阶段\小论文\基于多尺度增强与特征交互分组注意力的服装语义分割\代码文件')
    
    py_files = list(project_dir.rglob('*.py'))
    
    processed = 0
    for py_file in py_files:
        if py_file.name == 'remove_comments.py':
            continue
        
        print(f'Processing: {py_file}')
        try:
            cleaned_code = remove_comments_simple(py_file)
            
            with open(py_file, 'w', encoding='utf-8') as f:
                f.write(cleaned_code)
            
            processed += 1
        except Exception as e:
            print(f'Error processing {py_file}: {e}')
    
    print(f'\nDone! Processed {processed} files.')

if __name__ == '__main__':
    main()