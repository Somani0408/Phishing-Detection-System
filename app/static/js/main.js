// Main JavaScript for Phishing Detection System

document.addEventListener('DOMContentLoaded', function() {
    // Tab switching functionality
    const tabButtons = document.querySelectorAll('.tab-button');
    const tabContents = document.querySelectorAll('.tab-content');
    
    tabButtons.forEach(button => {
        button.addEventListener('click', () => {
            const targetTab = button.getAttribute('data-tab');
            
            // Update active tab button
            tabButtons.forEach(btn => btn.classList.remove('active'));
            button.classList.add('active');
            
            // Update active tab content
            tabContents.forEach(content => {
                content.classList.remove('active');
                if (content.id === `${targetTab}-tab`) {
                    content.classList.add('active');
                }
            });
            
            // Clear previous results
            clearResults();
        });
    });
    
    // URL submission
    const urlSubmit = document.getElementById('url-submit');
    const urlInput = document.getElementById('url-input');
    
    urlSubmit.addEventListener('click', () => {
        const url = urlInput.value.trim();
        if (url) {
            makePrediction(url, 'url');
        } else {
            showError('Please enter a URL');
        }
    });
    
    // Allow Enter key for URL input
    urlInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') {
            urlSubmit.click();
        }
    });
    
    // Email submission
    const emailSubmit = document.getElementById('email-submit');
    const emailInput = document.getElementById('email-input');
    
    emailSubmit.addEventListener('click', () => {
        const email = emailInput.value.trim();
        if (email) {
            makePrediction(email, 'email');
        } else {
            showError('Please enter email text');
        }
    });
    
    // Make prediction request
    function makePrediction(value, type) {
        // Clear previous results and errors
        clearResults();
        hideError();
        
        // Show loading
        showLoading();
        
        // Disable submit buttons
        urlSubmit.disabled = true;
        emailSubmit.disabled = true;
        
        // Make API request
        fetch('/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                type: type,
                value: value
            })
        })
        .then(response => response.json())
        .then(data => {
            hideLoading();
            urlSubmit.disabled = false;
            emailSubmit.disabled = false;
            
            if (data.status === 'success') {
                displayResult(data);
            } else {
                showError(data.error || 'An error occurred');
            }
        })
        .catch(error => {
            hideLoading();
            urlSubmit.disabled = false;
            emailSubmit.disabled = false;
            showError('Network error: ' + error.message);
        });
    }
    
    // Display prediction result
    function displayResult(data) {
        const resultContainer = document.getElementById('result-container');
        const resultContent = document.getElementById('result-content');
        
        const isPhishing = data.label === 'Phishing';
        const confidence = data.confidence || 0;
        
        resultContent.innerHTML = `
            <div class="result-label ${isPhishing ? 'phishing' : 'legitimate'}">
                ${isPhishing ? '⚠️ PHISHING DETECTED' : '✅ LEGITIMATE'}
            </div>
            <div class="confidence-score">
                <p>Confidence: ${confidence}%</p>
                <div class="confidence-bar">
                    <div class="confidence-fill ${isPhishing ? 'phishing' : 'legitimate'}" 
                         style="width: ${confidence}%">
                        ${confidence}%
                    </div>
                </div>
            </div>
            <div class="result-details">
                <strong>Input Type:</strong> ${data.input_type.toUpperCase()}<br>
                <strong>Analyzed:</strong> ${data.input_value}
            </div>
        `;
        
        resultContainer.classList.remove('hidden');
    }
    
    // Show loading indicator
    function showLoading() {
        document.getElementById('loading').classList.remove('hidden');
    }
    
    // Hide loading indicator
    function hideLoading() {
        document.getElementById('loading').classList.add('hidden');
    }
    
    // Show error message
    function showError(message) {
        const errorContainer = document.getElementById('error-container');
        const errorMessage = document.getElementById('error-message');
        errorMessage.textContent = message;
        errorContainer.classList.remove('hidden');
    }
    
    // Hide error message
    function hideError() {
        document.getElementById('error-container').classList.add('hidden');
    }
    
    // Clear results
    function clearResults() {
        document.getElementById('result-container').classList.add('hidden');
        hideError();
    }
});

