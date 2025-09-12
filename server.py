from flask import Flask, request
from flask_socketio import SocketIO
import flwr as fl
from flwr.server.client_manager import SimpleClientManager
import logging
import requests
import os
from dotenv import load_dotenv
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
from datetime import datetime

# CaptainAI class for code generation using phi-3-mini model
class CaptainAI:
    def __init__(self):
        self.logger = logging.getLogger("captain-ai")
        self.logger.info("Initializing CaptainAI with phi-3-mini-4k-instruct model...")
        
        # Load the model and tokenizer
        self.model_name = "microsoft/phi-3-mini-4k-instruct"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        # Initialize global preference to neutral (0)
        self.global_preference = 0.0
        
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
        # Update the global preference with the new value
        self.global_preference = preference_value
        self.logger.info(f"Global preference updated to: {self.global_preference}")
    
    def generate_response(self, prompt, progress_callback=None):
        """Generate code based on the user's prompt using the phi-3-mini model"""
        self.logger.info(f"Generating code for prompt: {prompt}")
        
        try:
            # Call progress callback if provided
            if progress_callback:
                progress_callback({"status": "formatting", "message": "Formatting prompt for model..."})
            
            # Adjust prompt based on global preference
            instruction = "Write Python code for: "
            if self.global_preference > 0.5:
                instruction = "Write detailed, well-documented Python code with explanations for: "
            elif self.global_preference < -0.5:
                instruction = "Write minimal, concise Python code for: "
                
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
        # Call the parent class to aggregate weights
        aggregated_weights = super().aggregate_fit(server_round, results, failures)
        
        if aggregated_weights is not None:
            # Extract the preference value from the aggregated weights
            # The preference is stored in the first weight of the model
            preference_tensor = aggregated_weights[0]
            preference_value = float(np.mean(preference_tensor))
            
            # Update the global preference in CaptainAI
            self.captain_ai.update_global_preference(preference_value)
            
        return aggregated_weights

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

@socketio.on('submit_improvement')
def handle_submit_improvement(data):
    client_id = request.sid
    original_code = data.get('original_code', '')
    edited_code = data.get('edited_code', '')
    prompt = data.get('prompt', '')
    
    logger.info(f"Received code improvement from client {client_id}")
    
    try:
        # Compare the line count of original vs edited code
        original_line_count = len(original_code.split('\n'))
        edited_line_count = len(edited_code.split('\n'))
        
        # Determine preference: 1 for longer code, -1 for shorter code
        preference_value = 0
        if edited_line_count > original_line_count:
            preference_value = 1  # User prefers longer, more verbose code
            logger.info(f"User made code longer - preference for verbose code")
        elif edited_line_count < original_line_count:
            preference_value = -1  # User prefers shorter, more concise code
            logger.info(f"User made code shorter - preference for concise code")
        else:
            # Code length unchanged, but content might have changed
            # Check if comments were removed
            original_comment_count = original_code.count('#')
            edited_comment_count = edited_code.count('#')
            
            if edited_comment_count < original_comment_count:
                preference_value = -0.7  # User prefers less comments
                logger.info(f"User removed comments - preference for less commented code")
            elif edited_comment_count > original_comment_count:
                preference_value = 0.7  # User prefers more comments
                logger.info(f"User added comments - preference for more commented code")
            else:
                preference_value = 0.1  # Small change
                logger.info(f"Code structure unchanged - small preference update")
        
        # Update the global preference directly
        captain.update_global_preference(preference_value)
        
        # Store the prompt for future reference
        # This allows the model to remember user preferences for specific prompts
        # In a real implementation, you would store this in a database
        
        # Notify the client that the preference has been updated
        socketio.emit('preference_updated', {
            'preference_value': preference_value,
            'message': 'Your code preference has been learned',
            'client_id': client_id
        }, room=client_id)
        
        # Notify all clients that the model has been updated
        socketio.emit('model_updated', {
            'message': 'Model has been updated with new preferences',
            'updated_by': client_id
        })
        
        logger.info(f"Preference updated for client {client_id}: {preference_value}")
        return {'status': 'success', 'message': 'Preference updated'}
        
    except Exception as e:
        # Handle errors
        error_message = str(e)
        logger.error(f"Error updating preference: {error_message}")
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


