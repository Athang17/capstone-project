// Local implementation of the Flower client browser library
// This is a simplified version for demonstration purposes

// Create a global fl object
window.fl = {
    client: {
        // Client class that can be extended
        Client: class {
            constructor() {
                console.log("Flower Client initialized");
            }
            
            // Default methods that can be overridden
            getWeights() { return []; }
            setWeights(weights) {}
            async fit(parameters, config) {
                return { parameters: [], status: { message: "OK" }, num_examples: 0 };
            }
            async evaluate(parameters, config) {
                return { loss: 0.0, status: { message: "OK" }, num_examples: 0 };
            }
        },
        
        // Function to start the client and connect to the server
        startClient: function(serverURL, client) {
            console.log(`Connecting to Flower server at ${serverURL}`);
            
            // Create a socket connection
            const socket = io(serverURL);
            
            return new Promise((resolve, reject) => {
                // Handle connection events
                socket.on('connect', () => {
                    console.log('Socket connected to server');
                    
                    // Handle various server events
                    socket.on('get_parameters', async (data, callback) => {
                        console.log('Server requested parameters');
                        try {
                            const weights = client.getWeights();
                            callback({ parameters: weights, status: { message: "OK" } });
                        } catch (e) {
                            console.error('Error getting parameters:', e);
                            callback({ status: { message: "Error" } });
                        }
                    });
                    
                    socket.on('fit', async (data, callback) => {
                        console.log('Server requested fit');
                        try {
                            const result = await client.fit(data.parameters, data.config);
                            callback(result);
                        } catch (e) {
                            console.error('Error during fit:', e);
                            callback({ status: { message: "Error" } });
                        }
                    });
                    
                    socket.on('evaluate', async (data, callback) => {
                        console.log('Server requested evaluate');
                        try {
                            const result = await client.evaluate(data.parameters, data.config);
                            callback(result);
                        } catch (e) {
                            console.error('Error during evaluate:', e);
                            callback({ status: { message: "Error" } });
                        }
                    });
                    
                    // Resolve the promise when connected
                    resolve();
                });
                
                socket.on('connect_error', (error) => {
                    console.error('Connection error:', error);
                    reject(error);
                });
                
                socket.on('disconnect', () => {
                    console.log('Disconnected from server');
                });
            });
        }
    }
};

console.log("Flower client browser library loaded");