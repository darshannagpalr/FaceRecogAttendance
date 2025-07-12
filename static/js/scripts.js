document.addEventListener('DOMContentLoaded', function () {
    // Handle Train Model Button
    const trainModelBtn = document.getElementById('trainModelBtn');
    const trainMessage = document.getElementById('trainMessage');

    // ... (your existing JavaScript code) ...

    // Set current year for copyright
    const currentYearSpan = document.getElementById('currentYear');
    if (currentYearSpan) {
        currentYearSpan.textContent = new Date().getFullYear();
    }

    if (trainModelBtn) {
        trainModelBtn.addEventListener('click', function () {
            trainMessage.textContent = "Training model... please wait.";
            trainModelBtn.disabled = true;

            fetch('/train_model')
                .then(response => response.json())
                .then(data => {
                    trainMessage.textContent = data.message;
                    if (data.status === 'success') {
                        trainMessage.style.color = 'green';
                    } else {
                        trainMessage.style.color = 'red';
                    }
                })
                .catch(error => {
                    console.error('Error:', error);
                    trainMessage.textContent = 'An error occurred during training.';
                    trainMessage.style.color = 'red';
                })
                .finally(() => {
                    trainModelBtn.disabled = false;
                });
        });
    }

    // Handle Live Recognition Stream Buttons
    const startRecognitionBtn = document.getElementById('startRecognitionBtn');
    const stopRecognitionBtn = document.getElementById('stopRecognitionBtn');
    const videoFeed = document.getElementById('videoFeed');
    const recognitionStatus = document.getElementById('recognitionStatus');
    const loadingMessage = document.getElementById('loadingMessage');

    function updateRecognitionStatus(message, color = 'black') {
        if (recognitionStatus) {
            recognitionStatus.textContent = message;
            recognitionStatus.style.color = color;
        }
    }

    if (startRecognitionBtn && stopRecognitionBtn && videoFeed) {
        startRecognitionBtn.addEventListener('click', function () {
            updateRecognitionStatus('Starting recognition...', 'orange');
            startRecognitionBtn.disabled = true;
            stopRecognitionBtn.disabled = false;
            loadingMessage.style.display = 'block'; // Show loading message

            fetch('/start_recognition')
                .then(response => response.json())
                .then(data => {
                    updateRecognitionStatus(data.message, data.status === 'started' ? 'green' : 'red');
                    if (data.status === 'started' || data.status === 'already_running') {
                        videoFeed.src = "/video_feed?" + new Date().getTime(); // Add timestamp to force reload
                        videoFeed.style.display = 'block'; // Show video
                        loadingMessage.style.display = 'none'; // Hide loading message
                    } else {
                        startRecognitionBtn.disabled = false; // Re-enable if failed to start
                        stopRecognitionBtn.disabled = true;
                        videoFeed.style.display = 'none'; // Hide video
                    }
                })
                .catch(error => {
                    console.error('Error starting recognition:', error);
                    updateRecognitionStatus('Error starting recognition. Check console.', 'red');
                    startRecognitionBtn.disabled = false;
                    stopRecognitionBtn.disabled = true;
                    videoFeed.style.display = 'none'; // Hide video
                });
        });

        stopRecognitionBtn.addEventListener('click', function () {
            updateRecognitionStatus('Stopping recognition...', 'orange');
            stopRecognitionBtn.disabled = true;
            startRecognitionBtn.disabled = false;
            videoFeed.style.display = 'none'; // Hide video immediately
            loadingMessage.style.display = 'block'; // Show loading message

            fetch('/stop_recognition')
                .then(response => response.json())
                .then(data => {
                    updateRecognitionStatus(data.message, data.status === 'stopped' ? 'green' : 'red');
                    if (data.status === 'stopped' || data.status === 'not_running') {
                        videoFeed.src = ""; // Clear video source
                    }
                })
                .catch(error => {
                    console.error('Error stopping recognition:', error);
                    updateRecognitionStatus('Error stopping recognition. Check console.', 'red');
                });
        });

        // Initial state for buttons on page load
        if (window.location.pathname === '/recognition_stream') {
            stopRecognitionBtn.disabled = true; // Disable stop button initially
            // You might want to check the actual recognition status from backend here if needed
            // For now, assume it's off by default
        }
    }
});

