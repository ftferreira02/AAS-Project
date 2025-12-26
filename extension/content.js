// Content Script
// Runs on every page load

const url = window.location.href;

chrome.runtime.sendMessage({ action: "checkUrl", url: url }, (response) => {
    if (response && response.is_phishing) {
        showWarning(response);
    }
});

function showWarning(data) {
    // Create overlay
    const overlay = document.createElement('div');
    overlay.id = 'phish-detector-overlay';
    overlay.style.position = 'fixed';
    overlay.style.top = '0';
    overlay.style.left = '0';
    overlay.style.width = '100%';
    overlay.style.height = '100%';
    overlay.style.backgroundColor = 'rgba(200, 0, 0, 0.9)';
    overlay.style.zIndex = '999999';
    overlay.style.display = 'flex';
    overlay.style.flexDirection = 'column';
    overlay.style.justifyContent = 'center';
    overlay.style.alignItems = 'center';
    overlay.style.color = 'white';
    overlay.style.fontFamily = 'Arial, sans-serif';
    overlay.style.textAlign = 'center';

    const content = `
        <div style="background: white; color: black; padding: 40px; border-radius: 10px; max-width: 600px; box-shadow: 0 0 20px rgba(0,0,0,0.5);">
            <h1 style="color: #d32f2f; margin-top: 0;">⚠️ Phishing Detected</h1>
            <p style="font-size: 18px;">The URL <strong>${data.url}</strong> has been flagged as potential phishing.</p>
            <p><strong>Confidence:</strong> ${(data.confidence * 100).toFixed(1)}%</p>
            <div style="margin-top: 30px;">
                <button id="phish-proceed-btn" style="background: #ef5350; color: white; border: none; padding: 10px 20px; font-size: 16px; border-radius: 5px; cursor: pointer;">Proceed Anyway (Unsafe)</button>
                <button id="phish-back-btn" style="background: #4caf50; color: white; border: none; padding: 10px 20px; font-size: 16px; border-radius: 5px; cursor: pointer; margin-left: 10px;">Go Back</button>
            </div>
        </div>
    `;

    overlay.innerHTML = content;
    document.body.appendChild(overlay);

    // Event listeners
    document.getElementById('phish-proceed-btn').addEventListener('click', () => {
        overlay.remove();
    });

    document.getElementById('phish-back-btn').addEventListener('click', () => {
        window.history.back();
    });
}
