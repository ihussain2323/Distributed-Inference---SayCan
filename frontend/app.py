"""
Flask backend for SayCan frontend
Connects the web interface to the actual SayCan system
"""

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import sys
import os
import subprocess
import json
import re
from dataclasses import dataclass, asdict

# Add the parent directory to path to import SayCan modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

app = Flask(__name__)
CORS(app)

@dataclass
class RobotAction:
    name: str
    coordinates: list
    affordance: float = 0.0

def calculate_affordance(coords):
    """Calculate affordance based on distance from robot."""
    distance = sum(x ** 2 for x in coords) ** 0.5
    return max(0.0, 1.0 - distance / 5.0)

def parse_saycan_output(output_text):
    """Parse the output from SayCan clean script."""
    try:
        lines = output_text.split('\n')
        
        # Extract plan
        plan = []
        in_plan = False
        for line in lines:
            if '📋 5-Step Plan:' in line:
                in_plan = True
                continue
            elif '📊 Action Analysis:' in line:
                break
            elif in_plan and line.strip():
                # Clean up the plan step
                step = line.strip()
                if step and not step.startswith('system') and not step.startswith('user') and not step.startswith('assistant'):
                    plan.append(step)
        
        # Extract actions (expecting ActionName | Affordance | [x, y, z])
        actions = []
        in_actions = False
        for line in lines:
            if '📊 Action Analysis:' in line:
                in_actions = True
                continue
            elif '🏆 Best Action:' in line:
                break
            elif in_actions and '|' in line and not line.startswith('Action') and not line.startswith('---'):
                parts = line.split('|')
                if len(parts) >= 3:
                    try:
                        name = parts[0].strip()
                        affordance_str = parts[1].strip()
                        coords_str = parts[2].strip()
                        
                        # Parse affordance
                        affordance = float(affordance_str)
                        
                        # Parse coordinates
                        coords_match = re.search(r'\[([^\]]+)\]', coords_str)
                        if coords_match:
                            coords = [float(x.strip()) for x in coords_match.group(1).split(',')]
                            if len(coords) == 3:
                                actions.append(RobotAction(name, coords, affordance))
                    except (ValueError, IndexError) as e:
                        print(f"[WARN] Skipping action due to parse error: {e} | line: {line}")
                        continue
        
        # Find best action
        best_action = None
        for line in lines:
            if '🏆 Best Action:' in line:
                best_action_name = line.split('Best Action:')[1].strip()
                for action in actions:
                    if action.name == best_action_name:
                        best_action = action
                        break
                break
        
        # If no valid actions, provide fallback
        if not actions:
            fallback_action = RobotAction("return to base", [0.0, 0.0, 0.0], 1.0)
            actions = [fallback_action]
            best_action = fallback_action
        elif not best_action and actions:
            best_action = max(actions, key=lambda a: a.affordance)
        
        # Ensure we have exactly 5 plan steps
        if len(plan) < 5:
            plan = plan + ["5. Complete the task."] * (5 - len(plan))
        plan = plan[:5]
        
        return {
            'plan': plan,
            'actions': [asdict(action) for action in actions],
            'best_action': asdict(best_action) if best_action else None
        }
    except Exception as e:
        print(f"Error parsing output: {e}")
        # Return fallback data
        fallback_action = RobotAction("return to base", [0.0, 0.0, 0.0], 1.0)
        return {
            'plan': [
                "1. Move to the target location.",
                "2. Locate the target object.",
                "3. Approach the object safely.",
                "4. Perform the required action.",
                "5. Return to starting position."
            ],
            'actions': [asdict(fallback_action)],
            'best_action': asdict(fallback_action)
        }

@app.route('/')
def index():
    """Serve the main page."""
    return render_template('index.html')

@app.route('/api/saycan', methods=['POST'])
def run_saycan():
    """Run SayCan with the provided goal."""
    try:
        data = request.get_json()
        goal = data.get('goal', '')
        
        if not goal:
            return jsonify({'error': 'No goal provided'}), 400
        
        # Run the SayCan clean script
        script_path = os.path.join(os.path.dirname(__file__), '..', 'saycan-clean.py')
        
        # Create a temporary script to run with the specific goal
        temp_script = f"""
import sys, os
sys.path.append(os.path.dirname(__file__))
sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

try:
    from saycan_clean import saycan_clean_planner
    import ray
    from llama_client import wait_for_workers

    # Initialize Ray
    ray.init(namespace="llama_inference", ignore_reinit_error=True)

    # Wait for workers
    if wait_for_workers():
        # Run planning
        saycan_clean_planner("{goal}")
        
        # Cleanup
        ray.shutdown()
    else:
        print("❌ Workers not available")
        ray.shutdown()
except Exception as e:
    print(f"Error: {{e}}")
    ray.shutdown()
"""
        
        # Write temporary script
        temp_script_path = '/tmp/saycan_temp.py'
        with open(temp_script_path, 'w') as f:
            f.write(temp_script)
        
        # Run the script
        result = subprocess.run(
            ['python', temp_script_path],
            capture_output=True,
            text=True,
            cwd=os.path.join(os.path.dirname(__file__), '..')
        )
        
        # Clean up
        os.remove(temp_script_path)
        
        if result.returncode != 0:
            return jsonify({'error': f'SayCan execution failed: {result.stderr}'}), 500
        
        # Parse the output
        parsed_result = parse_saycan_output(result.stdout)
        
        if not parsed_result:
            return jsonify({'error': 'Failed to parse SayCan output'}), 500
        
        return jsonify(parsed_result)
        
    except Exception as e:
        return jsonify({'error': f'Server error: {str(e)}'}), 500

@app.route('/api/status')
def status():
    """Check if Ray and workers are available."""
    try:
        import ray
        ray.init(namespace="llama_inference")
        
        from llama_client import wait_for_workers
        workers_available = wait_for_workers()
        
        ray.shutdown()
        
        return jsonify({
            'ray_available': True,
            'workers_available': workers_available
        })
    except Exception as e:
        return jsonify({
            'ray_available': False,
            'workers_available': False,
            'error': str(e)
        })

if __name__ == '__main__':
    import sys
    port = 5001
    if "--port" in sys.argv:
        port_index = sys.argv.index("--port") + 1
        if port_index < len(sys.argv):
            port = int(sys.argv[port_index])
    print("🚀 Starting SayCan Frontend Server")
    print("📁 Make sure Ray cluster and workers are running!")
    print(f"🌐 Open http://localhost:{port} in your browser")
    app.run(debug=True, host='0.0.0.0', port=port) 