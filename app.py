from flask import Flask, request, jsonify
from flask_cors import CORS
import subprocess
import threading
import queue
import time
import re
import uuid
import os
import tempfile
import signal
import sys
import platform
import select
import errno
from groq import Groq
from dotenv import load_dotenv
import json

# Load environment variables
load_dotenv()

app = Flask(__name__)
CORS(app)

# Store running processes with their state
active_processes = {}

class ComplexityAnalyzer:
    def __init__(self):
        # Initialize Groq client
        self.groq_client = None
        api_key = os.getenv('GROQ_API_KEY')
        if api_key:
            self.groq_client = Groq(api_key=api_key)
        else:
            print("⚠️  Warning: GROQ_API_KEY not found in environment variables")
    
    def analyze_complexity(self, code, language):
        """Analyze time and space complexity using Groq AI"""
        if not self.groq_client:
            return {
                'error': 'Groq API key not configured',
                'time_complexity': 'Unable to analyze',
                'space_complexity': 'Unable to analyze'
            }
        
        try:
            # Create a detailed prompt for complexity analysis
            prompt = self._create_complexity_prompt(code, language)
            
            # Call Groq API
            response = self.groq_client.chat.completions.create(
                model="llama-3.1-8b-instant",  # Using a powerful model for accurate analysis
                messages=[
                    {
                        "role": "system",
                        "content": self._get_system_prompt()
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.1,  # Low temperature for consistent, accurate results
                max_tokens=1000,
                response_format={"type": "json_object"}
            )
            
            # Parse the response
            result = json.loads(response.choices[0].message.content)
            
            # Validate and structure the response
            return self._validate_and_structure_response(result)
            
        except Exception as e:
            print(f"❌ Error analyzing complexity: {str(e)}")
            return {
                'error': f'Complexity analysis failed: {str(e)}',
                'time_complexity': 'Unable to analyze',
                'space_complexity': 'Unable to analyze'
            }
    
    def _get_system_prompt(self):
        """Get the system prompt for Groq to ensure accurate complexity analysis"""
        return """You are an expert computer science professor specializing in algorithm analysis and computational complexity. Your task is to analyze code and provide precise time and space complexity analysis.

CRITICAL REQUIREMENTS:
1. Always respond in valid JSON format with the exact structure specified
2. Use Big O notation (e.g., O(1), O(n), O(n²), O(log n), O(n log n))
3. Consider ALL operations including loops, recursion, data structure operations
4. For nested loops, multiply complexities (O(n) * O(m) = O(n*m))
5. For sequential operations, take the dominant complexity
6. Consider space used by variables, data structures, and recursion stack
7. Provide clear, educational explanations
8. If multiple algorithms exist in code, analyze the overall dominant complexity

RESPONSE FORMAT (must be valid JSON):
{
  "time_complexity": "O(notation)",
  "space_complexity": "O(notation)",
  "time_explanation": "Detailed explanation of time complexity analysis",
  "space_explanation": "Detailed explanation of space complexity analysis",
  "dominant_operations": ["list of key operations affecting complexity"],
  "complexity_breakdown": {
    "best_case": "O(notation)",
    "average_case": "O(notation)", 
    "worst_case": "O(notation)"
  }
}

Be extremely careful and accurate. Consider:
- Loop structures and nesting
- Recursive calls and their depth
- Data structure operations (array access, hash table operations, etc.)
- Memory allocation for variables and data structures
- Function call overhead and recursion stack space"""

    def _create_complexity_prompt(self, code, language):
        """Create a detailed prompt for complexity analysis"""
        return f"""Analyze the time and space complexity of the following {language} code. Provide a thorough analysis considering all algorithmic aspects:

LANGUAGE: {language}

CODE:
```{language.lower()}
{code}
```

ANALYSIS REQUIREMENTS:
1. Identify all loops, recursive calls, and data structure operations
2. Calculate time complexity by considering:
   - Loop iterations and nesting levels
   - Recursive call patterns and depth
   - Built-in function complexities (sort, search, etc.)
   - Data structure operation costs

3. Calculate space complexity by considering:
   - Variables and arrays declared
   - Recursion stack space
   - Additional data structures used
   - Input space vs auxiliary space

4. Provide best, average, and worst-case analysis where applicable
5. Explain your reasoning step by step

Respond in the exact JSON format specified in the system prompt."""

    def _validate_and_structure_response(self, result):
        """Validate and structure the Groq response"""
        try:
            # Ensure required fields exist with defaults
            structured_result = {
                'time_complexity': result.get('time_complexity', 'O(?)'),
                'space_complexity': result.get('space_complexity', 'O(?)'),
                'time_explanation': result.get('time_explanation', 'Analysis not available'),
                'space_explanation': result.get('space_explanation', 'Analysis not available'),
                'dominant_operations': result.get('dominant_operations', []),
                'complexity_breakdown': result.get('complexity_breakdown', {
                    'best_case': result.get('time_complexity', 'O(?)'),
                    'average_case': result.get('time_complexity', 'O(?)'),
                    'worst_case': result.get('time_complexity', 'O(?)')
                }),
                'analysis_confidence': 'high',
                'error': None
            }
            
            # Validate Big O notation format
            complexities = [
                structured_result['time_complexity'],
                structured_result['space_complexity']
            ]
            
            for complexity in complexities:
                if not self._is_valid_big_o(complexity):
                    print(f"⚠️  Warning: Invalid Big O notation detected: {complexity}")
            
            return structured_result
            
        except Exception as e:
            return {
                'error': f'Failed to parse complexity analysis: {str(e)}',
                'time_complexity': 'Unable to analyze',
                'space_complexity': 'Unable to analyze',
                'time_explanation': 'Analysis parsing failed',
                'space_explanation': 'Analysis parsing failed',
                'dominant_operations': [],
                'complexity_breakdown': {
                    'best_case': 'Unable to analyze',
                    'average_case': 'Unable to analyze',
                    'worst_case': 'Unable to analyze'
                },
                'analysis_confidence': 'low'
            }
    
    def _is_valid_big_o(self, notation):
        """Validate Big O notation format"""
        if not notation or not isinstance(notation, str):
            return False
        
        # Common valid Big O patterns
        valid_patterns = [
            r'^O\(1\)$',           # O(1)
            r'^O\(log\s*n\)$',     # O(log n)
            r'^O\(n\)$',           # O(n)
            r'^O\(n\s*log\s*n\)$', # O(n log n)
            r'^O\(n\²\)$',         # O(n²)
            r'^O\(n\^2\)$',        # O(n^2)
            r'^O\(2\^n\)$',        # O(2^n)
            r'^O\(n!\)$',          # O(n!)
            r'^O\([nm]\*[nm]\)$',  # O(n*m)
            r'^O\(\?\)$',          # O(?) for unknown
        ]
        
        for pattern in valid_patterns:
            if re.match(pattern, notation, re.IGNORECASE):
                return True
        
        return False

class DynamicCodeExecutor:
    def __init__(self):
        self.language_patterns = {
            'C': [
                r'#include\s*<[^>]+>',
                r'int\s+main\s*\(',
                r'printf\s*\(',
                r'scanf\s*\(',
                r'/\*.*?\*/',
                r'//.*$'
            ],
            'C++': [
                r'#include\s*<iostream>',
                r'using\s+namespace\s+std',
                r'cout\s*<<|cin\s*>>',
                r'std::',
                r'class\s+\w+',
                r'int\s+main\s*\(',
                r'#include\s*<[^>]+>'
            ],
            'Python': [
                r'def\s+\w+\s*\(',
                r'import\s+\w+|from\s+\w+\s+import',
                r'print\s*\(',
                r'input\s*\(',
                r'if\s+__name__\s*==\s*[\'"]__main__[\'"]',
                r':\s*$',
                r'^\s*(def|class|if|for|while|try|with)\s'
            ],
            'Java': [
                r'public\s+class\s+\w+',
                r'public\s+static\s+void\s+main',
                r'System\.out\.print',
                r'Scanner\s+\w+',
                r'import\s+java\.',
                r'\{\s*$'
            ]
        }

    def detect_language(self, code):
        """Detect programming language from code content"""
        scores = {}
        
        for language, patterns in self.language_patterns.items():
            score = 0
            for pattern in patterns:
                matches = len(re.findall(pattern, code, re.MULTILINE | re.IGNORECASE))
                score += matches
            scores[language] = score
        
        if not scores or max(scores.values()) == 0:
            return None
            
        detected_lang = max(scores, key=scores.get)
        
        # Additional validation for better accuracy
        if detected_lang == 'C++' and 'iostream' in code:
            return 'C++'
        elif detected_lang == 'C' and any(keyword in code for keyword in ['#include <stdio.h>', 'printf', 'scanf']):
            return 'C'
        elif detected_lang == 'Python' and any(keyword in code for keyword in ['def ', 'print(', 'import ']):
            return 'Python'
        elif detected_lang == 'Java' and 'public class' in code:
            return 'Java'
        
        return detected_lang

    def needs_input(self, code, language):
        """Check if code requires user input"""
        input_patterns = {
            'C': [r'scanf\s*\(', r'getchar\s*\(', r'gets\s*\(', r'fgets\s*\('],
            'C++': [r'cin\s*>>', r'getline\s*\('],
            'Python': [r'input\s*\(', r'raw_input\s*\('],
            'Java': [r'Scanner\s+\w+', r'\.next\w*\s*\(', r'\.read\w*\s*\(']
        }
        
        if language in input_patterns:
            for pattern in input_patterns[language]:
                if re.search(pattern, code, re.IGNORECASE):
                    return True
        return False

    def create_temp_file(self, code, language):
        """Create temporary file for code execution"""
        extensions = {
            'C': '.c',
            'C++': '.cpp',
            'Python': '.py',
            'Java': '.java'
        }
        
        # Create unique filename
        unique_id = str(uuid.uuid4())[:8]
        filename = f"code_{unique_id}{extensions.get(language, '.txt')}"
        temp_path = os.path.join(tempfile.gettempdir(), filename)
        
        with open(temp_path, 'w') as f:
            f.write(code)
        return temp_path

    def execute_code(self, code, language, input_data=None):
        """Execute code and return result"""
        try:
            temp_file = self.create_temp_file(code, language)
            execution_id = str(uuid.uuid4())
            
            # Prepare execution commands
            python_cmd = 'python3' if platform.system() != 'Windows' else 'python'
            commands = {
                'Python': [python_cmd, '-u', temp_file],  # -u for unbuffered output
                'C': self._compile_and_run_c(temp_file),
                'C++': self._compile_and_run_cpp(temp_file),
                'Java': self._compile_and_run_java(temp_file, code)
            }
            
            if language not in commands:
                return {'error': f'Language {language} not supported'}
            
            cmd = commands[language]
            if not cmd:
                return {'error': f'Failed to prepare {language} execution'}
            
            # Execute with proper environment
            env = os.environ.copy()
            env['PYTHONUNBUFFERED'] = '1'  # Ensure Python output is unbuffered
            
            process = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=0,  # Unbuffered
                universal_newlines=True,
                env=env,
                preexec_fn=os.setsid if platform.system() != 'Windows' else None
            )
            
            # Make stdout non-blocking on Unix systems
            # if platform.system() != 'Windows':
            #     fd = process.stdout.fileno()
            #     fl = fcntl.fcntl(fd, fcntl.F_GETFL)
            #     fcntl.fcntl(fd, fcntl.F_SETFL, fl | os.O_NONBLOCK)
            
            # Store process info
            active_processes[execution_id] = {
                'process': process,
                'temp_file': temp_file,
                'language': language,
                'needs_input': self.needs_input(code, language),
                'output_buffer': [],
                'completed': False,
                'error_buffer': []
            }
            
            # Start output reader thread
            output_thread = threading.Thread(
                target=self._read_output_continuously,
                args=(execution_id,),
                daemon=True
            )
            output_thread.start()
            
            # Wait a bit for initial output
            time.sleep(0.1)
            
            # Get initial output
            initial_output = self._get_current_output(execution_id)
            
            # Check if process has completed
            if process.poll() is not None:
                # Process completed
                self._mark_process_completed(execution_id)
                return {
                    'execution_id': execution_id,
                    'output': initial_output,
                    'error': self._get_current_error(execution_id),
                    'needs_input': False,
                    'completed': True
                }
            
            # Check if it needs input
            if self.needs_input(code, language) and not initial_output:
                # Wait a bit more for prompts
                time.sleep(0.2)
                initial_output = self._get_current_output(execution_id)
            
            return {
                'execution_id': execution_id,
                'output': initial_output,
                'error': self._get_current_error(execution_id),
                'needs_input': self.needs_input(code, language) and process.poll() is None,
                'completed': process.poll() is not None
            }
                    
        except Exception as e:
            return {'error': f'Execution error: {str(e)}'}

    def _read_output_continuously(self, execution_id):
        """Continuously read output from process"""
        if execution_id not in active_processes:
            return
            
        process_info = active_processes[execution_id]
        process = process_info['process']
        
        while process.poll() is None:
            try:
                if platform.system() == 'Windows':
                    # Windows doesn't support select on pipes
                    line = process.stdout.readline()
                    if line:
                        process_info['output_buffer'].append(line)
                else:
                    # Unix systems
                    ready, _, _ = select.select([process.stdout], [], [], 0.1)
                    if ready:
                        try:
                            line = process.stdout.readline()
                            if line:
                                process_info['output_buffer'].append(line)
                        except IOError as e:
                            if e.errno != errno.EAGAIN:
                                break
                
                # Read stderr
                try:
                    if platform.system() != 'Windows':
                        stderr_ready, _, _ = select.select([process.stderr], [], [], 0.1)
                        if stderr_ready:
                            err_line = process.stderr.readline()
                            if err_line:
                                process_info['error_buffer'].append(err_line)
                    else:
                        # For Windows, we'll read stderr after process completion
                        pass
                except:
                    pass
                    
                time.sleep(0.01)  # Small delay to prevent busy waiting
                
            except Exception:
                break
        
        # Read any remaining output
        try:
            remaining_output = process.stdout.read()
            if remaining_output:
                process_info['output_buffer'].append(remaining_output)
                
            remaining_error = process.stderr.read()
            if remaining_error:
                process_info['error_buffer'].append(remaining_error)
        except:
            pass
        
        process_info['completed'] = True

    def _get_current_output(self, execution_id):
        """Get accumulated output from process"""
        if execution_id not in active_processes:
            return ""
        
        process_info = active_processes[execution_id]
        return ''.join(process_info['output_buffer'])

    def _get_current_error(self, execution_id):
        """Get accumulated error from process"""
        if execution_id not in active_processes:
            return ""
        
        process_info = active_processes[execution_id]
        return ''.join(process_info['error_buffer'])

    def _mark_process_completed(self, execution_id):
        """Mark process as completed"""
        if execution_id in active_processes:
            active_processes[execution_id]['completed'] = True

    def send_input_to_process(self, execution_id, input_data):
        """Send input to running process"""
        if execution_id not in active_processes:
            return {'error': 'Process not found or already completed'}
        
        process_info = active_processes[execution_id]
        process = process_info['process']
        
        if process.poll() is not None:
            # Process already completed
            final_output = self._get_current_output(execution_id)
            final_error = self._get_current_error(execution_id)
            self._cleanup_process(execution_id)
            
            return {
                'execution_id': execution_id,
                'output': final_output,
                'error': final_error,
                'needs_input': False,
                'completed': True
            }
        
        try:
            # Send input
            process.stdin.write(input_data + '\n')
            process.stdin.flush()
            
            # Wait for new output
            time.sleep(0.2)  # Give time for output to appear
            
            current_output = self._get_current_output(execution_id)
            current_error = self._get_current_error(execution_id)
            
            # Check if process completed
            if process.poll() is not None:
                # Process completed
                self._cleanup_process(execution_id)
                return {
                    'execution_id': execution_id,
                    'output': current_output,
                    'error': current_error,
                    'needs_input': False,
                    'completed': True
                }
            
            # Still running, might need more input
            return {
                'execution_id': execution_id,
                'output': current_output,
                'error': current_error,
                'needs_input': True,
                'completed': False
            }
            
        except Exception as e:
            self._cleanup_process(execution_id)
            return {'error': f'Input error: {str(e)}'}

    def get_process_status(self, execution_id):
        """Get current status of a process"""
        if execution_id not in active_processes:
            return {'error': 'Process not found'}
        
        process_info = active_processes[execution_id]
        process = process_info['process']
        
        current_output = self._get_current_output(execution_id)
        current_error = self._get_current_error(execution_id)
        
        if process.poll() is not None and not process_info.get('completed_reported'):
            # Process completed
            process_info['completed_reported'] = True
            self._cleanup_process(execution_id)
            
            return {
                'execution_id': execution_id,
                'output': current_output,
                'error': current_error,
                'needs_input': False,
                'completed': True
            }
        
        return {
            'execution_id': execution_id,
            'output': current_output,
            'error': current_error,
            'needs_input': process.poll() is None,
            'completed': process.poll() is not None
        }

    def _compile_and_run_c(self, source_file):
        """Compile and prepare C code execution"""
        try:
            executable = source_file.replace('.c', '')
            if platform.system() == 'Windows':
                executable += '.exe'
            
            compile_result = subprocess.run(
                ['gcc', source_file, '-o', executable],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if compile_result.returncode != 0:
                return None
            
            return [executable]
        except:
            return None

    def _compile_and_run_cpp(self, source_file):
        """Compile and prepare C++ code execution"""
        try:
            executable = source_file.replace('.cpp', '')
            if platform.system() == 'Windows':
                executable += '.exe'
            
            compile_result = subprocess.run(
                ['g++', source_file, '-o', executable],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if compile_result.returncode != 0:
                return None
            
            return [executable]
        except:
            return None

    def _compile_and_run_java(self, source_file, code):
        """Compile and prepare Java code execution"""
        try:
            # Extract class name
            class_match = re.search(r'public\s+class\s+(\w+)', code)
            if not class_match:
                return None
            
            class_name = class_match.group(1)
            
            # Compile
            compile_result = subprocess.run(
                ['javac', source_file],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if compile_result.returncode != 0:
                return None
            
            # Get directory and run
            file_dir = os.path.dirname(source_file)
            return ['java', '-cp', file_dir, class_name]
        except:
            return None

    def _cleanup_process(self, execution_id):
        """Clean up process and temporary files"""
        if execution_id not in active_processes:
            return
            
        process_info = active_processes[execution_id]
        
        # Kill process if still running
        try:
            if process_info['process'].poll() is None:
                if platform.system() != 'Windows':
                    os.killpg(os.getpgid(process_info['process'].pid), signal.SIGTERM)
                else:
                    process_info['process'].terminate()
                time.sleep(0.1)
                if process_info['process'].poll() is None:
                    if platform.system() != 'Windows':
                        os.killpg(os.getpgid(process_info['process'].pid), signal.SIGKILL)
                    else:
                        process_info['process'].kill()
        except:
            pass
        
        # Remove temporary files
        try:
            temp_file = process_info['temp_file']
            if os.path.exists(temp_file):
                os.remove(temp_file)
            
            # Remove compiled files
            if process_info['language'] in ['C', 'C++']:
                executable = temp_file.replace(f".{process_info['language'].lower()}", '')
                if platform.system() == 'Windows':
                    executable += '.exe'
                if os.path.exists(executable):
                    os.remove(executable)
            elif process_info['language'] == 'Java':
                class_file = temp_file.replace('.java', '.class')
                if os.path.exists(class_file):
                    os.remove(class_file)
        except:
            pass
        
        del active_processes[execution_id]

# Initialize executor and complexity analyzer
executor = DynamicCodeExecutor()
complexity_analyzer = ComplexityAnalyzer()

# ========== NEW ENDPOINT FOR COMPLEXITY ANALYSIS ==========
@app.route('/analyze-complexity', methods=['POST'])
def analyze_complexity():
    """Endpoint to analyze time and space complexity of code"""
    try:
        data = request.get_json()
        code = data.get('code', '')
        language = data.get('language')
        
        if not code.strip():
            return jsonify({'error': 'No code provided'}), 400
        
        # Detect language if not provided
        if not language:
            language = executor.detect_language(code)
        
        if not language:
            return jsonify({'error': 'Could not detect programming language'}), 400
        
        print(f"🔍 Analyzing complexity for {language} code...")
        
        # Analyze complexity using Groq AI
        complexity_result = complexity_analyzer.analyze_complexity(code, language)
        
        # Add metadata
        complexity_result['language'] = language
        complexity_result['timestamp'] = time.time()
        
        print(f"✅ Complexity analysis completed: {complexity_result.get('time_complexity', 'N/A')}")
        
        return jsonify(complexity_result)
        
    except Exception as e:
        print(f"❌ Error in complexity analysis: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/detect-language', methods=['POST'])
def detect_language():
    """Endpoint to detect programming language"""
    try:
        data = request.get_json()
        code = data.get('code', '')
        
        if not code.strip():
            return jsonify({'error': 'No code provided'}), 400
        
        detected_lang = executor.detect_language(code)
        
        return jsonify({
            'language': detected_lang,
            'confidence': 'high' if detected_lang else 'low'
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/execute-code', methods=['POST'])
def execute_code():
    """Endpoint to execute code"""
    try:
        data = request.get_json()
        code = data.get('code', '')
        language = data.get('language')
        input_data = data.get('input_data', '')

        if not code.strip():
            return jsonify({'error': 'No code provided'}), 400

        # Detect language if not provided
        if not language:
            language = executor.detect_language(code)
        if not language:
            return jsonify({'error': 'Could not detect programming language'}), 400

        result = executor.execute_code(code, language, input_data)
        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/send-input', methods=['POST'])
def send_input():
    """Endpoint to send input to running process"""
    try:
        data = request.get_json()
        execution_id = data.get('execution_id', '')
        input_data = data.get('input', '')
        
        if not execution_id:
            return jsonify({'error': 'No execution ID provided'}), 400
        
        result = executor.send_input_to_process(execution_id, input_data)
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/get-status', methods=['POST'])
def get_status():
    """Endpoint to get current status of a process"""
    try:
        data = request.get_json()
        execution_id = data.get('execution_id', '')
        
        if not execution_id:
            return jsonify({'error': 'No execution ID provided'}), 400
        
        result = executor.get_process_status(execution_id)
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/stop-process', methods=['POST'])
def stop_process():
    """Endpoint to stop a running process"""
    try:
        data = request.get_json()
        execution_id = data.get('execution_id', '')
        
        if not execution_id:
            return jsonify({'error': 'No execution ID provided'}), 400
        
        if execution_id in active_processes:
            executor._cleanup_process(execution_id)
            return jsonify({'message': 'Process stopped successfully'})
        else:
            return jsonify({'error': 'Process not found'}), 404
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    groq_status = "configured" if complexity_analyzer.groq_client else "not configured"
    
    return jsonify({
        'status': 'healthy', 
        'active_processes': len(active_processes),
        'process_ids': list(active_processes.keys()),
        'groq_status': groq_status,
        'features': [
            'Code execution',
            'Language detection', 
            'Interactive input handling',
            'Complexity analysis (AI-powered)'
        ]
    })

# Cleanup on shutdown
def cleanup_all_processes():
    """Clean up all active processes on shutdown"""
    for execution_id in list(active_processes.keys()):
        executor._cleanup_process(execution_id)

import atexit
atexit.register(cleanup_all_processes)

# Handle signals for graceful shutdown
def signal_handler(signum, frame):
    print(f"\n🛑 Received signal {signum}, cleaning up...")
    cleanup_all_processes()
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

if __name__ == '__main__':
    print("🚀 AI Code Editor Backend Starting...")
    print("📍 Server will be available at: http://localhost:5000")
    
    # Check Groq API configuration
    if os.getenv('GROQ_API_KEY'):
        print("✅ Groq API configured for complexity analysis")
    else:
        print("⚠️  Groq API not configured - complexity analysis disabled")
        print("   Add GROQ_API_KEY to your .env file to enable AI features")
    
    app.run(debug=True, host='0.0.0.0', port=5000, threaded=True)