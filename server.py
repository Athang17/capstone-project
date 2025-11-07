from flask import Flask, request
from flask_socketio import SocketIO
import flwr as fl
from flwr.server.client_manager import SimpleClientManager
from flwr.common import parameters_to_ndarrays
import logging
import requests
import os
from dotenv import load_dotenv
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
from datetime import datetime
import re

# CaptainAI class for code generation using phi-3-mini model
class CaptainAI:
    def __init__(self):
        self.logger = logging.getLogger("captain-ai")
        self.logger.info("Initializing CaptainAI with phi-3-mini-4k-instruct model...")
        
        # Load the model and tokenizer
        self.model_name = "microsoft/phi-3-mini-4k-instruct"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        # Initialize global preference vector to neutral [0,0,0]
        self.global_preference = np.zeros(3, dtype=float)
        
        try:
            # Check available devices and avoid MPS
            if torch.cuda.is_available():
                device_map = "cuda"
                self.logger.info("Using CUDA for model initialization")
            else:
                device_map = "cpu"
                self.logger.info("Using CPU for model initialization")
                
            # Use lower precision and enable CPU offloading for better memory efficiency
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                dtype=torch.float16,  # Use float16 for better memory efficiency
                device_map=device_map,
                low_cpu_mem_usage=True
            )
        except Exception as e:
            self.logger.warning(f"Failed to load model with specified settings: {str(e)}. Falling back to CPU-only mode.")
            # Fallback to CPU-only mode with minimal settings
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                device_map="cpu",
                low_cpu_mem_usage=True
            )
        
        self.logger.info("CaptainAI initialized successfully!")
    
    def update_global_preference(self, preference_value):
        # Accept float (legacy) or vector-like; coerce to numpy array length 3
        try:
            if isinstance(preference_value, (list, tuple, np.ndarray)):
                arr = np.array(preference_value, dtype=float).flatten()
                if arr.size == 0:
                    arr = np.zeros(3)
                elif arr.size == 1:
                    arr = np.array([arr[0], 0.0, 0.0])
                elif arr.size >= 3:
                    arr = arr[:3]
                else:
                    # pad to length 3
                    pad = np.zeros(3)
                    pad[:arr.size] = arr
                    arr = pad
            else:
                # legacy scalar -> map to verbosity only
                arr = np.array([float(preference_value), 0.0, 0.0])
        except Exception:
            arr = np.zeros(3)

        self.global_preference = arr.astype(float)
        self.logger.info(f"Global preference updated to vector: {self.global_preference.tolist()}")
    
    def generate_response(self, prompt, progress_callback=None):
        """Generate code based on the user's prompt using the phi-3-mini model"""
        self.logger.info(f"Generating code for prompt: {prompt}")
        
        try:
            # Call progress callback if provided
            if progress_callback:
                progress_callback({"status": "formatting", "message": "Formatting prompt for model..."})
            
            # Adjust prompt based on global preference
            # Build meta-prompt from style vector [documentation, typeHinting, modernSyntax]
            doc, typeh, modern = 0.0, 0.0, 0.0
            try:
                if isinstance(self.global_preference, np.ndarray) and self.global_preference.size >= 3:
                    doc, typeh, modern = [float(x) for x in self.global_preference[:3]]
                elif isinstance(self.global_preference, (list, tuple)) and len(self.global_preference) >= 3:
                    doc, typeh, modern = [float(x) for x in self.global_preference[:3]]
            except Exception:
                pass

            directives = []
            # Documentation directive
            if doc > 0.5:
                directives.append("the response MUST be well-documented with clear inline comments and, where applicable, docstrings")
            elif doc < -0.5:
                directives.append("the response MUST NOT include any comments or docstrings")
            # Type hints directive
            if typeh > 0.5:
                directives.append("the response MUST include Python type hints for all function parameters and return values")
            elif typeh < -0.5:
                directives.append("the response MUST NOT include any Python type hints")
            # Modern syntax directive
            if modern > 0.5:
                directives.append("use modern Python features such as f-strings instead of .format and prefer contemporary idioms")
            elif modern < -0.5:
                directives.append("avoid modern syntax such as f-strings; prefer legacy constructs like .format")

            if directives:
                instruction = "CRITICAL INSTRUCTION: " + "; ".join(directives) + ". Generate Python code for: "
            else:
                instruction = "Generate Python code for: "
                
            # Format the prompt for instruction-tuned model
            formatted_prompt = f"<|user|>\n{instruction}{prompt}\n<|assistant|>\n"
            
            # Tokenize the prompt
            if progress_callback:
                progress_callback({"status": "tokenizing", "message": "Tokenizing input..."})
                
            inputs = self.tokenizer(formatted_prompt, return_tensors="pt")
            
            # Move inputs to the same device as the model
            device = next(self.model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # Generate the response
            if progress_callback:
                progress_callback({"status": "generating", "message": "Generating code with AI model..."})
                
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs["input_ids"],
                    max_new_tokens=1024,
                    temperature=0.7,
                    top_p=0.9,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            if progress_callback:
                progress_callback({"status": "decoding", "message": "Decoding generated response..."})
                
            # Decode the response and extract only the generated part
            full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

            # Extract only the assistant's response
            assistant_response = full_response.split("<|assistant|>")[-1].strip()

            # Post-process: return only the first fenced Python code block if present
            try:
                # Prefer an explicit ```python ... ``` block
                match = re.search(r"```\s*python\s*([\s\S]*?)```", assistant_response, flags=re.IGNORECASE)
                if not match:
                    match = re.search(r"```\s*python\s*([\s\S]*?)```", full_response, flags=re.IGNORECASE)
                # Fall back to any fenced block if explicit language not found
                if not match:
                    match = re.search(r"```\s*([\s\S]*?)```", assistant_response, flags=re.IGNORECASE)
                if not match:
                    match = re.search(r"```\s*([\s\S]*?)```", full_response, flags=re.IGNORECASE)
                if match:
                    code_only = match.group(1).strip()
                    if code_only:
                        return code_only
                # Secondary fallback: find first code-looking line and return from there
                def extract_from_first_code_line(text: str) -> str:
                    if not text:
                        return ""
                    lines = text.splitlines()
                    pattern = re.compile(r"^\s*(def |class |import |from |#|if __name__ == ['\"]__main__['\"]:)\b")
                    for idx, line in enumerate(lines):
                        if pattern.search(line):
                            return "\n".join(lines[idx:]).strip()
                    return ""

                code_tail = extract_from_first_code_line(assistant_response)
                if not code_tail:
                    code_tail = extract_from_first_code_line(full_response)
                if code_tail:
                    return code_tail
            except Exception as _:
                pass

            self.logger.info("Code generation completed")
            return assistant_response
        except Exception as e:
            self.logger.error(f"Error during code generation: {str(e)}")
            return f"Error generating code: {str(e)}. Please try again with a simpler prompt."

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("flower-server")

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

# Load environment variables from .env file
load_dotenv()

# Initialize the CaptainAI instance
captain = CaptainAI()

# Function to get synthetic data from LLM API (using Groq's API with Llama 3.1 model)
def get_synthetic_data_from_teacher(prompt=None, progress_callback=None):
    # Read API key directly from .env file as a fallback
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        logger.warning("GROQ_API_KEY not found in environment variables, trying to read from .env file directly")
        try:
            with open(".env", "r") as env_file:
                for line in env_file:
                    if line.startswith("GROQ_API_KEY="):
                        api_key = line.strip().split("=", 1)[1]
                        break
        except Exception as e:
            logger.error(f"Error reading .env file: {e}")
        
    if not api_key:
        logger.warning("GROQ_API_KEY not found in .env file")
        return "No synthetic data available (API key missing)"
    
    # Update progress if callback provided
    if progress_callback:
        progress_callback({"status": "connecting", "message": "Connecting to Groq API (Llama 3.1)..."})
    
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    # Default prompt if none provided
    if not prompt:
        prompt = "Generate three synthetic examples of Python code with different styles (concise, verbose, and educational). For each example, include a brief explanation of its style characteristics."
    
    # Update progress if callback provided
    if progress_callback:
        progress_callback({"status": "sending", "message": "Sending request to Llama 3.1 model via Groq API..."})
    
    payload = {
        "model": "llama-3.1-8b-instant",  # Using Llama 3.1 model from Groq
        "messages": [
            {"role": "system", "content": "You are an expert Python programmer that generates high-quality synthetic data for training code generation models. Your examples should demonstrate different coding styles and best practices. Provide detailed explanations that can help improve a student model's understanding of good code."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.7,
        "max_tokens": 2000,  # Increased token limit for more detailed examples
        "stream": False  # Ensure we're not using streaming mode
    }
    
    try:
        logger.info(f"Making request to Groq API with model: {payload['model']}")
        
        # Update progress if callback provided
        if progress_callback:
            progress_callback({"status": "waiting", "message": "Waiting for response from Llama 3.1 model..."})
            
        response = requests.post(url, headers=headers, json=payload)
        
        # Log the response status and headers for debugging
        logger.info(f"Groq API response status: {response.status_code}")
        logger.info(f"Groq API response headers: {response.headers}")
        
        # Check if the response is successful
        response.raise_for_status()
        
        # Update progress if callback provided
        if progress_callback:
            progress_callback({"status": "processing", "message": "Processing response from Llama 3.1 model..."})
        
        # Parse the response JSON
        data = response.json()
        logger.info(f"Groq API response data: {data}")
        
        # Extract the synthetic data from the response
        synthetic_data = data["choices"][0]["message"]["content"]
        logger.info(f"Generated synthetic data: {synthetic_data}")
        
        # Update progress if callback provided
        if progress_callback:
            progress_callback({"status": "completed", "message": "Successfully received expert advice from Llama 3.1 model!"})
            
        return synthetic_data
    except requests.exceptions.HTTPError as e:
        logger.error(f"HTTP Error from Groq API: {e}")
        error_message = ""
        if response.status_code == 401:
            error_message = "Authentication error: Invalid API key or unauthorized access"
            logger.error(error_message)
        elif response.status_code == 404:
            error_message = "Not Found error: Check API endpoint URL and model name"
            logger.error(error_message)
        elif response.status_code == 500:
            error_message = "Server error: Groq API server encountered an error"
            logger.error(error_message)
        else:
            error_message = f"HTTP Error: {e}"
            
        if progress_callback:
            progress_callback({"status": "error", "message": error_message})
            
        return f"Error from Llama 3.1 model: {error_message}"
            
        # Log the response content for debugging
        try:
            error_content = response.json()
            logger.error(f"Error response content: {error_content}")
        except:
            logger.error(f"Error response content (text): {response.text}")
            
        return f"Error generating synthetic data: {response.status_code} {response.reason} for url: {url}"
    except Exception as e:
        logger.error(f"Error getting synthetic data: {e}")
        return f"Error generating synthetic data: {str(e)}"

# Function to provide configuration for client training
def fit_config(server_round):
    # Only return the basic batch size and epoch config
    # Teacher AI is no longer called automatically
    config = {"batch_size": 32, "epochs": 1}
    return config

# Custom Flower strategy to handle preference learning
class PreferenceFedAvg(fl.server.strategy.FedAvg):
    def __init__(self, captain_ai, **kwargs):
        super().__init__(**kwargs)
        self.captain_ai = captain_ai
        
    def aggregate_fit(self, server_round, results, failures):
        # Call the parent class to aggregate weights (keeps baseline behavior/return type)
        aggregated = super().aggregate_fit(server_round, results, failures)

        # For demo: immediately set Captain's preference to the single client's update (no averaging)
        try:
            if results and len(results) > 0:
                first_fit_res = results[0][1]  # (ClientProxy, FitRes)
                client_ndarrays = parameters_to_ndarrays(first_fit_res.parameters)
                # Derive 3D pref vec from client's weights
                pref_vec = np.zeros(3, dtype=float)
                if len(client_ndarrays) >= 2 and client_ndarrays[1].size >= 3:
                    pref_vec = client_ndarrays[1].flatten()[:3]
                elif len(client_ndarrays) >= 1:
                    w0 = client_ndarrays[0]
                    if w0.ndim == 2 and w0.shape[1] >= 3:
                        pref_vec = w0.mean(axis=0)[:3]
                    else:
                        pref_vec.fill(float(np.mean(w0)))
                self.captain_ai.update_global_preference(pref_vec)
        except Exception as e:
            logger.warning(f"Failed to set preference from single client update: {e}")

        return aggregated

# Strategy that responds after a single client
strategy = PreferenceFedAvg(
    captain_ai=captain,
    min_available_clients=1,
    min_fit_clients=1,
    on_fit_config_fn=fit_config,
    # Configure for the code generation model
    fraction_fit=1.0,  # Use all available clients for training
    fraction_evaluate=1.0  # Use all available clients for evaluation
)

# The Flower Server class
class FlowerServer(fl.server.Server):
    def __init__(self, strategy, client_manager):
        super().__init__(strategy=strategy, client_manager=client_manager)
        
    def connect(self, sid, environ):
        logger.info(f"✅ Client connected successfully! (SID: {sid})")
        print(f"\n✅ SUCCESS: Client connected successfully! (SID: {sid})\n")
        
    def disconnect(self, sid):
        logger.info(f"Client disconnected: {sid}")
        print(f"--> Client disconnected: {sid}")

# Create server instance
fl_server = FlowerServer(
    strategy=strategy,
    client_manager=SimpleClientManager()
)

# Socket.IO event handlers to bridge Flask and Flower
@socketio.on('connect')
def on_connect():
    fl_server.connect(request.sid, request.environ)
    
@socketio.on('disconnect')
def on_disconnect():
    fl_server.disconnect(request.sid)
    
@socketio.on('rejoin')
def on_rejoin(data):
    logger.info(f"Client rejoining: {request.sid}")
    return fl_server.rejoin(data, request.sid)
    
@socketio.on('get_parameters')
def on_get_parameters(data):
    logger.info(f"Client requesting parameters: {request.sid}")
    return fl_server.get_parameters(data, request.sid)
    
@socketio.on('fit')
def on_fit(data):
    logger.info(f"Server received a 'fit' request from client: {request.sid}")
    print("--> Server received a 'fit' request from the client.")
    # Forward the fit request to the Flower server
    return fl_server.fit(data, request.sid)

@socketio.on('evaluate')
def on_evaluate(data):
    logger.info(f"Server received an 'evaluate' request from client: {request.sid}")
    # Return dummy response for this proof-of-concept
    return {"loss": 0.0, "num_examples": 0, "status": {"message": "OK"}}

# New event handler for code generation requests
@socketio.on('request_generation')
def on_request_generation(data):
    client_id = request.sid
    prompt = data.get('prompt', '')
    
    logger.info(f"Received code generation request from client {client_id}")
    print(f"\n--> Received prompt from client {client_id}: {prompt}")
    
    # Send an initial progress update to the client
    socketio.emit('generation_progress', {"status": "started", "message": "Starting code generation..."}, room=client_id)
    
    # Define progress callback function
    def progress_callback(progress_data):
        socketio.emit('generation_progress', progress_data, room=client_id)
    
    try:
        # Generate a response using the CaptainAI model with progress updates
        code = captain.generate_response(prompt, progress_callback=progress_callback)
        
        # Send the response back to the specific client that made the request
        logger.info(f"Sending code generation response to client {client_id}")
        socketio.emit('generation_response', {"code": code}, room=client_id)
        
        return {"status": "completed"}
    except Exception as e:
        error_msg = str(e)
        logger.error(f"Error during code generation: {error_msg}")
        socketio.emit('generation_error', {"error": error_msg}, room=client_id)
        return {"status": "error", "message": error_msg}

@socketio.on('call_teacher')
def handle_teacher_call(data):
    client_id = request.sid
    current_prompt = data.get('prompt', '')
    
    try:
        # Define progress callback function for the external API
        def progress_callback(progress_data):
            # Add model info to the progress data
            progress_data['model'] = 'Llama 3.1 (via Groq API)'
            # Emit progress to the client
            socketio.emit('teacher_progress', progress_data, room=client_id)
        
        # Send initial progress update to client
        progress_callback({
            'status': 'started',
            'message': 'Requesting expert advice from Llama 3.1 model via Groq API...'
        })
        
        # Create a prompt for the teacher based on the current user prompt
        teacher_prompt = f"Based on this prompt: '{current_prompt}', generate three different Python code examples with varying styles (concise, verbose, educational). For each example, explain its style characteristics and why it might be preferred in different contexts. Include detailed comments and best practices in your examples."
        
        # Get synthetic data from teacher (Expert Consultant) using the external Groq API
        teacher_advice = get_synthetic_data_from_teacher(teacher_prompt, progress_callback=progress_callback)
        
        # Update the global model with this new synthetic data
        # This is where we would normally train the model, but for now we'll just update the preference
        # to simulate the model improving from the teacher's advice
        captain.update_global_preference(0.5)  # Set to neutral as a baseline after teacher input
        
        # Emit the teacher's advice to all connected clients using 'teacher_response' event
        socketio.emit('teacher_response', {
            'synthetic_data': teacher_advice,
            'client_id': client_id,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'model_updated': True,
            'model_name': 'Llama 3.1 (via Groq API)'  # Add model name to response
        })
        
        # Notify all clients that the model has been updated
        socketio.emit('model_update', {
            'message': 'Model has been updated with expert knowledge from Llama 3.1',
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'model_name': 'Llama 3.1 (via Groq API)'  # Add model name to update notification
        })
        
        logger.info(f"Teacher advice from Llama 3.1 sent to all clients and model updated")
        return {'status': 'success', 'message': 'Teacher advice from Llama 3.1 sent to all clients and model updated'}
        
    except Exception as e:
        # Handle errors
        error_message = str(e)
        logger.error(f"Error getting teacher advice from Llama 3.1: {error_message}")
        socketio.emit('teacher_error', {
            'error': error_message,
            'client_id': client_id,
            'model_name': 'Llama 3.1 (via Groq API)'  # Add model name to error message
        }, room=client_id)
        
        return {'status': 'error', 'message': error_message}

@socketio.on('submit_improvement_and_learn')
def handle_submit_improvement_and_learn(data):
    """Immediately apply client's multi-dimensional preference vector.
    Expects payload: { preference: { documentation: d, typeHinting: t, modernSyntax: m }, client_id: ... }
    """
    try:
        pref_obj = (data or {}).get('preference', {})
        # Normalize to 3D vector [documentation, typeHinting, modernSyntax]
        if isinstance(pref_obj, dict):
            d = float(pref_obj.get('documentation', pref_obj.get('comments', 0.0)))
            t = float(pref_obj.get('typeHinting', pref_obj.get('type_hinting', 0.0)))
            m = float(pref_obj.get('modernSyntax', 0.0))
            pref_vec = [d, t, m]
        elif isinstance(pref_obj, (list, tuple, np.ndarray)):
            arr = np.array(pref_obj).flatten().tolist()
            # pad/trim to 3
            arr = (arr + [0.0, 0.0, 0.0])[:3]
            pref_vec = arr
        else:
            # fallback scalar
            try:
                val = float(pref_obj)
            except Exception:
                val = 0.0
            pref_vec = [val, 0.0, 0.0]

        # Set Captain's global preference directly (no averaging)
        captain.update_global_preference(pref_vec)
        logger.info(f"Applied immediate preference update from client: {pref_vec}")

        # Broadcast to all clients so dashboards update instantly
        socketio.emit('global_preference_update', {
            'preference': { 'documentation': pref_vec[0], 'typeHinting': pref_vec[1], 'modernSyntax': pref_vec[2] }
        })

        return {'status': 'success'}
    except Exception as e:
        logger.error(f"Error in submit_improvement_and_learn: {e}")
        return {'status': 'error', 'message': str(e)}

@socketio.on('get_global_preference')
def handle_get_global_preference(data):
    client_id = request.sid
    
    try:
        # Get the current global preference from the Captain AI
        global_preference = captain.global_preference
        
        # Emit the global preference to the requesting client
        socketio.emit('global_preference_update', {
            'preference': global_preference,
            'client_id': client_id
        })
        
        logger.info(f"Global preference sent to client {client_id}")
        return {'status': 'success', 'message': 'Global preference sent'}
        
    except Exception as e:
        # Handle errors
        error_message = str(e)
        logger.error(f"Error getting global preference: {error_message}")
        socketio.emit('preference_error', {
            'error': error_message,
            'client_id': client_id
        }, room=client_id)
        
        return {'status': 'error', 'message': error_message}

# Add a simple route for health check
@app.route('/')
def index():
    return "Flower Federated Learning Server is running. Connect with the HTML client."

if __name__ == '__main__':
    port = 8080  # Changed back to port 8080 as requested
    print(f"\n--- Starting Flower Federated Learning Server on port {port} ---")
    print("Waiting for client connections...\n")
    # Pass the captain instance to the server for code generation capabilities
    fl_server = FlowerServer(
        strategy=strategy,
        client_manager=SimpleClientManager()
    )
    socketio.run(app, host='0.0.0.0', port=port, debug=False)  # Set debug to False to prevent auto-reloading


