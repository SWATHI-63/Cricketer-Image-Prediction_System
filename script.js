/* ===================================
   CRICKAI - CRICKETER PREDICTION SYSTEM
   Enhanced JavaScript with Real ML Backend
   =================================== */

// ===================================
// CONFIGURATION
// ===================================

const API_URL = 'http://localhost:5000'; // Flask backend URL
let useRealAPI = false; // Will be set to true if API is available

// ===================================
// CRICKETER DATABASE (Fallback)
// ===================================

// Dataset of cricketers with their info
// In production, this will come from the ML model's label_mapping.json
const cricketersDatabase = {
    'virat kohli': {
        name: 'Virat Kohli',
        role: 'Batsman',
        country: 'India',
        images: 38,
        accuracy: '97.2%'
    },
    'ms dhoni': {
        name: 'MS Dhoni',
        role: 'Wicket-keeper Batsman',
        country: 'India',
        images: 42,
        accuracy: '96.8%'
    },
    'rohit sharma': {
        name: 'Rohit Sharma',
        role: 'Batsman',
        country: 'India',
        images: 35,
        accuracy: '95.5%'
    },
    'sachin tendulkar': {
        name: 'Sachin Tendulkar',
        role: 'Batsman',
        country: 'India',
        images: 40,
        accuracy: '98.1%'
    },
    'jasprit bumrah': {
        name: 'Jasprit Bumrah',
        role: 'Fast Bowler',
        country: 'India',
        images: 36,
        accuracy: '94.3%'
    },
    'hardik pandya': {
        name: 'Hardik Pandya',
        role: 'All-rounder',
        country: 'India',
        images: 34,
        accuracy: '93.7%'
    },
    'ravindra jadeja': {
        name: 'Ravindra Jadeja',
        role: 'All-rounder',
        country: 'India',
        images: 38,
        accuracy: '95.2%'
    },
    'kl rahul': {
        name: 'KL Rahul',
        role: 'Wicket-keeper Batsman',
        country: 'India',
        images: 32,
        accuracy: '94.8%'
    },
    'shubman gill': {
        name: 'Shubman Gill',
        role: 'Batsman',
        country: 'India',
        images: 30,
        accuracy: '92.4%'
    },
    'rishabh pant': {
        name: 'Rishabh Pant',
        role: 'Wicket-keeper Batsman',
        country: 'India',
        images: 37,
        accuracy: '93.9%'
    },
    'mohammed shami': {
        name: 'Mohammed Shami',
        role: 'Fast Bowler',
        country: 'India',
        images: 35,
        accuracy: '94.1%'
    },
    'suryakumar yadav': {
        name: 'Suryakumar Yadav',
        role: 'Batsman',
        country: 'India',
        images: 33,
        accuracy: '93.5%'
    },
    'yuzvendra chahal': {
        name: 'Yuzvendra Chahal',
        role: 'Leg Spinner',
        country: 'India',
        images: 36,
        accuracy: '95.0%'
    },
    'kuldeep yadav': {
        name: 'Kuldeep Yadav',
        role: 'Chinaman Bowler',
        country: 'India',
        images: 34,
        accuracy: '94.6%'
    },
    'shreyas iyer': {
        name: 'Shreyas Iyer',
        role: 'Batsman',
        country: 'India',
        images: 36,
        accuracy: '93.2%'
    }
};

// Quick search tags (popular cricketers)
const quickSearchTags = [
    'Virat Kohli',
    'MS Dhoni',
    'Rohit Sharma',
    'Sachin Tendulkar',
    'Jasprit Bumrah'
];

// ===================================
// DOM ELEMENTS
// ===================================

// Navigation
const mobileMenuBtn = document.getElementById('mobileMenuBtn');
const mobileMenu = document.getElementById('mobileMenu');
const closeMenu = document.getElementById('closeMenu');
const overlay = document.getElementById('overlay');

// Upload Section
const uploadZone = document.getElementById('uploadZone');
const imageInput = document.getElementById('imageInput');
const previewArea = document.getElementById('previewArea');
const imagePreview = document.getElementById('imagePreview');
const removeBtn = document.getElementById('removeBtn');
const predictBtn = document.getElementById('predictBtn');
const predictLoading = document.getElementById('predictLoading');

// Result Section
const resultBox = document.getElementById('resultBox');
const predictedAvatar = document.getElementById('predictedAvatar');
const predictedName = document.getElementById('predictedName');
const predictedRole = document.getElementById('predictedRole');
const confidenceValue = document.getElementById('confidenceValue');
const confidenceFill = document.getElementById('confidenceFill');

// Search Section
const nameInput = document.getElementById('nameInput');
const clearInput = document.getElementById('clearInput');
const searchBtn = document.getElementById('searchBtn');
const cricketerList = document.getElementById('cricketerList');
const searchResultBox = document.getElementById('searchResultBox');
const searchImage = document.getElementById('searchImage');
const searchResultName = document.getElementById('searchResultName');
const searchResultRole = document.getElementById('searchResultRole');
const imageCount = document.getElementById('imageCount');
const accuracy = document.getElementById('accuracy');
const quickTags = document.getElementById('quickTags');

// State
let uploadedFile = null;

// ===================================
// INITIALIZATION
// ===================================

document.addEventListener('DOMContentLoaded', () => {
    console.log('🏏 CrickAI Prediction System Initialized');
    
    // Check if API is available
    checkAPIStatus();
    
    initializeNavigation();
    initializeUpload();
    initializeSearch();
    populateDatalist();
    populateQuickTags();
});

// Check if Flask API is running
async function checkAPIStatus() {
    try {
        const response = await fetch(`${API_URL}/api/status`, { 
            method: 'GET',
            mode: 'cors'
        });
        if (response.ok) {
            const data = await response.json();
            useRealAPI = data.model_loaded;
            if (useRealAPI) {
                console.log('✅ Connected to ML Backend!');
                showNotification('Connected to AI Model - Real predictions enabled!', 'success');
            } else {
                console.log('⚠️ API running but model not loaded');
                showNotification('Model not trained yet. Using demo mode.', 'warning');
            }
        }
    } catch (error) {
        console.log('ℹ️ API not available - using demo mode');
        useRealAPI = false;
    }
}

// ===================================
// NAVIGATION
// ===================================

function initializeNavigation() {
    if (mobileMenuBtn) {
        mobileMenuBtn.addEventListener('click', openMobileMenu);
    }
    
    if (closeMenu) {
        closeMenu.addEventListener('click', closeMobileMenu);
    }
    
    if (overlay) {
        overlay.addEventListener('click', closeMobileMenu);
    }
}

function openMobileMenu() {
    mobileMenu.classList.add('active');
    overlay.classList.add('show');
    document.body.style.overflow = 'hidden';
}

function closeMobileMenu() {
    mobileMenu.classList.remove('active');
    overlay.classList.remove('show');
    document.body.style.overflow = '';
}

// ===================================
// UPLOAD FUNCTIONALITY
// ===================================

function initializeUpload() {
    if (!uploadZone) return;
    
    // Click to upload
    uploadZone.addEventListener('click', () => imageInput.click());
    
    // File input change
    imageInput.addEventListener('change', handleFileSelect);
    
    // Drag and drop
    uploadZone.addEventListener('dragover', handleDragOver);
    uploadZone.addEventListener('dragleave', handleDragLeave);
    uploadZone.addEventListener('drop', handleDrop);
    
    // Remove button
    if (removeBtn) {
        removeBtn.addEventListener('click', (e) => {
            e.stopPropagation();
            resetUpload();
        });
    }
    
    // Predict button
    if (predictBtn) {
        predictBtn.addEventListener('click', handlePredict);
    }
}

function handleDragOver(e) {
    e.preventDefault();
    uploadZone.classList.add('dragover');
}

function handleDragLeave(e) {
    e.preventDefault();
    uploadZone.classList.remove('dragover');
}

function handleDrop(e) {
    e.preventDefault();
    uploadZone.classList.remove('dragover');
    
    const file = e.dataTransfer.files[0];
    if (validateFile(file)) {
        processFile(file);
    }
}

function handleFileSelect(e) {
    const file = e.target.files[0];
    if (validateFile(file)) {
        processFile(file);
    }
}

function validateFile(file) {
    if (!file) {
        showNotification('Please select a file', 'error');
        return false;
    }
    
    const validTypes = ['image/jpeg', 'image/png', 'image/jpg'];
    if (!validTypes.includes(file.type)) {
        showNotification('Please upload a JPG or PNG image', 'error');
        return false;
    }
    
    const maxSize = 5 * 1024 * 1024; // 5MB
    if (file.size > maxSize) {
        showNotification('File size must be less than 5MB', 'error');
        return false;
    }
    
    return true;
}

function processFile(file) {
    uploadedFile = file;
    
    const reader = new FileReader();
    reader.onload = (e) => {
        imagePreview.src = e.target.result;
        uploadZone.style.display = 'none';
        previewArea.classList.add('show');
        resultBox.classList.remove('show');
    };
    reader.readAsDataURL(file);
}

function resetUpload() {
    uploadedFile = null;
    imageInput.value = '';
    imagePreview.src = '';
    previewArea.classList.remove('show');
    uploadZone.style.display = 'block';
    resultBox.classList.remove('show');
    confidenceFill.style.width = '0%';
}

async function handlePredict() {
    if (!uploadedFile) {
        showNotification('Please upload an image first', 'error');
        return;
    }
    
    // Show loading state
    predictBtn.querySelector('.btn-text').textContent = 'Analyzing...';
    predictLoading.classList.add('active');
    predictBtn.disabled = true;
    
    try {
        if (useRealAPI) {
            // Use real ML model prediction
            await realPrediction();
        } else {
            // Fallback to demo mode
            await simulatePrediction();
            showNotification('Demo mode - Train model for accurate predictions', 'warning');
        }
    } catch (error) {
        console.error('Prediction error:', error);
        showNotification('Prediction failed. Please try again.', 'error');
    } finally {
        // Reset button state
        predictBtn.querySelector('.btn-text').textContent = 'Predict Cricketer';
        predictLoading.classList.remove('active');
        predictBtn.disabled = false;
    }
}

async function realPrediction() {
    // Create form data with the image
    const formData = new FormData();
    formData.append('image', uploadedFile);
    
    // Call the Flask API
    const response = await fetch(`${API_URL}/api/predict`, {
        method: 'POST',
        body: formData
    });
    
    const data = await response.json();
    
    if (data.success) {
        // Display real prediction
        resultBox.classList.add('show');
        predictedName.textContent = data.prediction;
        predictedRole.textContent = 'Cricketer • Predicted by AI';
        confidenceValue.textContent = `${data.confidence}%`;
        
        setTimeout(() => {
            confidenceFill.style.width = `${data.confidence}%`;
        }, 100);
        
        resultBox.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    } else {
        throw new Error(data.error || 'Prediction failed');
    }
}

async function simulatePrediction() {
    // Simulate network delay
    await new Promise(resolve => setTimeout(resolve, 1800));
    
    // Get random cricketer from database
    const keys = Object.keys(cricketersDatabase);
    const randomKey = keys[Math.floor(Math.random() * keys.length)];
    const cricketer = cricketersDatabase[randomKey];
    
    // Generate random confidence between 85-99%
    const confidence = (85 + Math.random() * 14).toFixed(1);
    
    // Display result
    displayPredictionResult(cricketer, confidence);
}

function displayPredictionResult(cricketer, confidence) {
    resultBox.classList.add('show');
    
    // Update player info
    predictedName.textContent = cricketer.name;
    predictedRole.textContent = `${cricketer.role} • ${cricketer.country}`;
    
    // Update confidence meter with animation
    confidenceValue.textContent = `${confidence}%`;
    
    // Animate confidence bar
    setTimeout(() => {
        confidenceFill.style.width = `${confidence}%`;
    }, 100);
    
    // Scroll to result
    resultBox.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

// ===================================
// SEARCH FUNCTIONALITY
// ===================================

function initializeSearch() {
    if (!nameInput) return;
    
    // Input events
    nameInput.addEventListener('input', handleInputChange);
    nameInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') handleSearch();
    });
    
    // Clear button
    if (clearInput) {
        clearInput.addEventListener('click', () => {
            nameInput.value = '';
            clearInput.classList.remove('show');
            searchResultBox.classList.remove('show');
        });
    }
    
    // Search button
    if (searchBtn) {
        searchBtn.addEventListener('click', handleSearch);
    }
}

function handleInputChange() {
    if (nameInput.value.trim()) {
        clearInput.classList.add('show');
    } else {
        clearInput.classList.remove('show');
    }
}

function populateDatalist() {
    if (!cricketerList) return;
    
    Object.values(cricketersDatabase).forEach(cricketer => {
        const option = document.createElement('option');
        option.value = cricketer.name;
        cricketerList.appendChild(option);
    });
}

function populateQuickTags() {
    if (!quickTags) return;
    
    quickSearchTags.forEach(name => {
        const tag = document.createElement('button');
        tag.className = 'quick-tag';
        tag.textContent = name;
        tag.addEventListener('click', () => {
            nameInput.value = name;
            handleSearch();
        });
        quickTags.appendChild(tag);
    });
}

function handleSearch() {
    const query = nameInput.value.trim().toLowerCase();
    
    if (!query) {
        showNotification('Please enter a cricketer name', 'error');
        return;
    }
    
    // Find matching cricketer
    const cricketer = cricketersDatabase[query];
    
    if (cricketer) {
        displaySearchResult(cricketer);
    } else {
        // Try partial match
        const partialMatch = Object.values(cricketersDatabase).find(c => 
            c.name.toLowerCase().includes(query) ||
            query.includes(c.name.toLowerCase().split(' ')[0])
        );
        
        if (partialMatch) {
            displaySearchResult(partialMatch);
        } else {
            showNotification(`"${nameInput.value}" not found. Try: ${quickSearchTags.slice(0, 3).join(', ')}`, 'error');
            searchResultBox.classList.remove('show');
        }
    }
}

function displaySearchResult(cricketer) {
    searchResultBox.classList.add('show');
    
    // Update result card
    searchResultName.textContent = cricketer.name;
    searchResultRole.textContent = `${cricketer.role} • ${cricketer.country}`;
    imageCount.textContent = cricketer.images;
    accuracy.textContent = cricketer.accuracy;
    
    // Update image (placeholder - in production, load actual image)
    const initials = cricketer.name.split(' ').map(n => n[0]).join('');
    searchImage.alt = cricketer.name;
    
    // Scroll to result
    searchResultBox.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

// ===================================
// UTILITY FUNCTIONS
// ===================================

function showNotification(message, type = 'info') {
    // Create notification element
    const notification = document.createElement('div');
    notification.className = `notification ${type}`;
    notification.innerHTML = `
        <span>${type === 'error' ? '⚠️' : 'ℹ️'}</span>
        <span>${message}</span>
    `;
    
    // Style the notification
    Object.assign(notification.style, {
        position: 'fixed',
        bottom: '20px',
        right: '20px',
        padding: '15px 25px',
        background: type === 'error' ? '#fee2e2' : '#e6fffa',
        color: type === 'error' ? '#dc2626' : '#0d9488',
        borderRadius: '12px',
        boxShadow: '0 5px 20px rgba(0, 0, 0, 0.1)',
        display: 'flex',
        alignItems: 'center',
        gap: '10px',
        fontWeight: '500',
        zIndex: '9999',
        animation: 'slideIn 0.3s ease'
    });
    
    document.body.appendChild(notification);
    
    // Remove after 3 seconds
    setTimeout(() => {
        notification.style.animation = 'slideOut 0.3s ease';
        setTimeout(() => notification.remove(), 300);
    }, 3000);
}

// Add CSS animation for notifications
const style = document.createElement('style');
style.textContent = `
    @keyframes slideIn {
        from { transform: translateX(100%); opacity: 0; }
        to { transform: translateX(0); opacity: 1; }
    }
    @keyframes slideOut {
        from { transform: translateX(0); opacity: 1; }
        to { transform: translateX(100%); opacity: 0; }
    }
`;
document.head.appendChild(style);

// ===================================
// BACKEND INTEGRATION NOTES
// ===================================

/*
 * PRODUCTION BACKEND INTEGRATION:
 * 
 * 1. Replace simulatePrediction() with actual API call:
 * 
 *    async function predictWithModel(file) {
 *        const formData = new FormData();
 *        formData.append('image', file);
 *        
 *        const response = await fetch('/api/predict', {
 *            method: 'POST',
 *            body: formData
 *        });
 *        
 *        if (!response.ok) throw new Error('Prediction failed');
 *        return await response.json();
 *    }
 * 
 * 2. Load cricketersDatabase from label_mapping.json:
 * 
 *    async function loadCricketerData() {
 *        const response = await fetch('/models/label_mapping.json');
 *        const data = await response.json();
 *        // Process and populate cricketersDatabase
 *    }
 * 
 * 3. Load actual images from data/images folder:
 * 
 *    function getPlayerImage(name) {
 *        const filename = name.toLowerCase().replace(' ', '_');
 *        return `/data/images/${filename}_1.jpg`;
 *    }
 */

console.log('🏏 CrickAI Ready for Predictions!');
