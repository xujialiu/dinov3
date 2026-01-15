import sys
import runpy
import os
from pathlib import Path
import shlex
import subprocess
import re


def expand_variables(text, shell_vars):
    """Expand shell variables like $var and ${var}"""
    # Expand from shell_vars first
    for var_name, var_value in shell_vars.items():
        text = re.sub(rf'\$\{{{var_name}\}}', var_value, text)
        text = re.sub(rf'\${var_name}(?=\W|$)', var_value, text)
    
    # Then expand from os.environ
    def replace_env_var(match):
        var_name = match.group(1) or match.group(2)
        return os.environ.get(var_name, match.group(0))
    
    text = re.sub(r'\$\{(\w+)\}|\$(\w+)', replace_env_var, text)
    return text


def replace_backticks(cmd):
    """Replace `command` with its output"""
    def replacer(match):
        command = match.group(1)
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        return result.stdout.strip()
    return re.sub(r'`([^`]+)`', replacer, cmd)


def parse_shell_script(cmd):
    """Parse shell script and execute commands"""
    shell_vars = {}
    
    # Handle line continuations (backslash + newline)
    cmd = re.sub(r'\\\n\s*', ' ', cmd)
    
    lines = cmd.strip().split('\n')
    
    for line in lines:
        line = line.strip()
        
        # Skip empty lines and comments
        if not line or line.startswith('#'):
            continue
        
        # Handle backticks
        line = replace_backticks(line)
        
        # Handle export statements
        if line.startswith('export '):
            rest = line[7:].strip()
            match = re.match(r'(\w+)=(.+)', rest)
            if match:
                var_name = match.group(1)
                var_value = match.group(2).strip('"\'')
                var_value = expand_variables(var_value, shell_vars)
                os.environ[var_name] = var_value
                shell_vars[var_name] = var_value
                print(f"[ENV] {var_name}={var_value}")
            continue
        
        # Handle simple variable assignments (VAR=value with no command after)
        match = re.match(r'^(\w+)=("[^"]*"|\'[^\']*\'|\S*)$', line)
        if match:
            var_name = match.group(1)
            var_value = match.group(2).strip('"\'')
            var_value = expand_variables(var_value, shell_vars)
            shell_vars[var_name] = var_value
            print(f"[VAR] {var_name}={var_value}")
            continue
        
        # This is an actual command - expand variables first
        line = expand_variables(line, shell_vars)
        print(f"[CMD] {line}")
        
        # Parse inline environment variables (e.g., CUDA_VISIBLE_DEVICES=1 command)
        tokens = shlex.split(line)
        cmd_start_idx = 0
        
        for i, token in enumerate(tokens):
            if '=' in token and not token.startswith('-'):
                eq_pos = token.index('=')
                potential_var = token[:eq_pos]
                # Check if it looks like an env var (typically uppercase)
                if potential_var.replace('_', '').isalnum() and potential_var[0].isalpha():
                    os.environ[potential_var] = token[eq_pos + 1:]
                    cmd_start_idx = i + 1
                else:
                    break
            else:
                break
        
        if cmd_start_idx >= len(tokens):
            continue
        
        actual_cmd = tokens[cmd_start_idx:]
        
        # Check if it's a Python command
        if actual_cmd[0] in ('python', 'python3'):
            actual_cmd.pop(0)
            if actual_cmd and actual_cmd[0] == '-m':
                actual_cmd.pop(0)
                sys.argv = [actual_cmd[0]] + actual_cmd[1:]
                runpy.run_module(actual_cmd[0], run_name="__main__")
            else:
                sys.argv = actual_cmd
                runpy.run_path(actual_cmd[0], run_name="__main__")
        else:
            # Run as external command via subprocess
            subprocess.run(actual_cmd)


def main(cmd):
    current_dir = Path(__file__).parent.resolve()
    os.chdir(current_dir)
    parse_shell_script(cmd)


if __name__ == "__main__":

    cmd = """
PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 python dinov3/eval/segmentation/run.py \
      config=dinov3/eval/segmentation/configs/config-retinal-artery-vein-seg-m2f-training.yaml \
      output_dir=./output/test \
      model.dino_hub=dinov3_vitl16
    """
    
    main(cmd)