import os
import re

root = '.'  # Set this to your project root
imports = set()

for dirpath, _, filenames in os.walk(root):
    for f in filenames:
        if f.endswith('.py'):
            with open(os.path.join(dirpath, f), 'r', encoding='utf-8', errors='ignore') as file:
                for line in file:
                    if line.startswith('import ') or line.startswith('from '):
                        match = re.match(r'(?:from|import)\s+([\w\.]+)', line)
                        if match:
                            imports.add(match.group(1).split('.')[0])

print("Unique top-level imported modules:")
for imp in sorted(imports):
    print(imp)
