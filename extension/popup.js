document.addEventListener('DOMContentLoaded', () => {
    chrome.tabs.query({ active: true, currentWindow: true }, function (tabs) {
        const currentTab = tabs[0];
        const url = currentTab.url;

        chrome.runtime.sendMessage({ action: "checkUrl", url: url }, (response) => {
            document.getElementById('loading').style.display = 'none';
            const statusBox = document.getElementById('status-box');
            const details = document.getElementById('details');

            if (!response || response.error) {
                statusBox.textContent = "Error connecting to API";
                statusBox.style.backgroundColor = "#999";
                statusBox.style.display = 'block';
                return;
            }

            if (response.is_phishing) {
                statusBox.textContent = "PHISHING DETECTED";
                statusBox.className = "status phishing";
            } else {
                statusBox.textContent = "SAFE";
                statusBox.className = "status safe";
            }
            statusBox.style.display = 'block';

            // Show details
            document.getElementById('confidence').textContent = (response.confidence * 100).toFixed(1);
            const featuresList = document.getElementById('features-list');
            featuresList.innerHTML = '';

            // Show top relevant features (example subset)
            const relevantKeys = ['entropy', 'suspicious_tld', 'domain_length', 'ratio_special'];
            for (const key of relevantKeys) {
                const li = document.createElement('li');
                li.textContent = `${key}: ${response.features[key]}`;
                featuresList.appendChild(li);
            }
            details.style.display = 'block';
        });
    });
});
