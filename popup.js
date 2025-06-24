document.getElementById('startDetection').addEventListener('click', () => {
    chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
        chrome.scripting.executeScript({
            target: { tabId: tabs[0].id },
            func: startPhishingDetection
        });
    });
});

function startPhishingDetection() {
    // Logic to start phishing detection (e.g., loading the model and making predictions)
    console.log('Phishing detection started!');
}