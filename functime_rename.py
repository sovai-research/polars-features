import os
import re

def replace_imports(root_dir, old_module, new_module):
    pattern_import = re.compile(rf'(^|\n)\s*import\s+{re.escape(old_module)}(\b|\.| as )')
    pattern_from = re.compile(rf'(^|\n)\s*from\s+{re.escape(old_module)}(\b|\.| )')

    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith('.py'):
                file_path = os.path.join(dirpath, filename)
                with open(file_path, 'r', encoding='utf-8') as file:
                    content = file.read()

                new_content = pattern_import.sub(lambda m: m.group(0).replace(old_module, new_module), content)
                new_content = pattern_from.sub(lambda m: m.group(0).replace(old_module, new_module), new_content)

                if new_content != content:
                    with open(file_path, 'w', encoding='utf-8') as file:
                        file.write(new_content)
                    print(f'Modified: {file_path}')

dir_name=os.path.dirname(os.path.abspath(__file__))
print(dir_name)

replace_imports(dir_name, 'functime', 'polars_features')
