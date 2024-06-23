import subprocess
import re
import threading

def extract_code_blocks(text: str) -> str:
    code_blocks = re.findall(r'```[a-zA-Z]*\n(.*?)(```|$)', text, re.DOTALL)
    combined_code = '\n'.join(block[0].strip() for block in code_blocks)
    return combined_code

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
            return "The code execution exceeded the timeout period."
        
        if stdout_lines:
            return ''.join(stdout_lines).strip()
        elif stderr_lines:
            return ''.join(stderr_lines).strip()
        else:
            return "The code executed successfully but produced no output."
    except subprocess.TimeoutExpired:
        process.kill()
        stdout_thread.join()
        stderr_thread.join()
        return "The code execution exceeded the timeout period."
    except Exception as e:
        process.kill()
        return f"An error occurred: {e}"