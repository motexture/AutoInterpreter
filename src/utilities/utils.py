import subprocess
import re
import threading

def extract_code_blocks(text: str) -> str:
    code_blocks = re.findall(r'```(?:[a-zA-Z]*\n)?(.*?)(?:```|$)', text, re.DOTALL)
    combined_code = '\n'.join(block.strip() for block in code_blocks)
    filtered_code = '\n'.join(
        line for line in combined_code.split('\n') if not re.match(r'^\s*$', line)
    ).strip()
    return filtered_code

def execute_code(code, python_env, timeout=360):
    def read_output(process, output_list):
        for line in iter(process.stdout.readline, ''):
            output_list.append(line)
            print(line, end='')

    def read_errors(process, error_list):
        for line in iter(process.stderr.readline, ''):
            error_list.append(line)
            print(line, end='')

    try:
        process = subprocess.Popen(
            [python_env, "-c", code],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1
        )

        stdout_lines = []
        stderr_lines = []

        stdout_thread = threading.Thread(target=read_output, args=(process, stdout_lines))
        stderr_thread = threading.Thread(target=read_errors, args=(process, stderr_lines))

        stdout_thread.start()
        stderr_thread.start()

        stdout_thread.join(timeout)
        stderr_thread.join(timeout)

        process.wait(timeout=timeout)
        
        if process.poll() is None:
            process.kill()
            stdout_thread.join()
            stderr_thread.join()
            return '*** Code execution output: \n\n' + 'Process terminated due to timeout\n\n'
        
        if stdout_lines:
            return '*** Code execution output: \n\n' + ''.join(stdout_lines).strip() + '\n\n'
        elif stderr_lines:
            return '*** Code execution output: \n\n' + ''.join(stderr_lines).strip() + '\n\n'
        else:
            return '*** Code execution output: \n\n' + 'Empty output from code execution\n\n'
    except subprocess.TimeoutExpired:
        process.kill()
        stdout_thread.join()
        stderr_thread.join()
        return '*** Code execution output: \n\n' + 'Process terminated due to timeout\n\n'
    except Exception as e:
        process.kill()
        return f'*** Code execution output: \n\n' + str(e) + '\n\n'
