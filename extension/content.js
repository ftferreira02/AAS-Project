// Content Script
// Runs on every page load

const currentUrl = window.location.href;

chrome.runtime.sendMessage({ action: "checkUrl", url: currentUrl }, (response) => {
    if (response && response.is_phishing) {
        if (document.body) {
            showWarning(response);
        } else {
            document.addEventListener('DOMContentLoaded', () => showWarning(response));
        }
    }
});

function showWarning(data) {
    // Prevent duplicate overlays
    if (document.getElementById('phish-detector-overlay')) return;

    // Create overlay container
    const overlay = document.createElement('div');
    overlay.id = 'phish-detector-overlay';
    Object.assign(overlay.style, {
        position: 'fixed',
        top: '0',
        left: '0',
        width: '100%',
        height: '100%',
        backgroundColor: 'rgba(200, 0, 0, 0.9)',
        zIndex: '999999',
        display: 'flex',
        flexDirection: 'column',
        justifyContent: 'center',
        alignItems: 'center',
        color: 'white',
        fontFamily: 'Arial, sans-serif',
        textAlign: 'center'
    });

    // Create Content Box
    const box = document.createElement('div');
    Object.assign(box.style, {
        background: 'white',
        color: 'black',
        padding: '40px',
        borderRadius: '10px',
        maxWidth: '600px',
        boxShadow: '0 0 20px rgba(0,0,0,0.5)'
    });

    // Header
    const h1 = document.createElement('h1');
    h1.textContent = '⚠️ Phishing Detected';
    h1.style.color = '#d32f2f';
    h1.style.marginTop = '0';
    box.appendChild(h1);

    // Message
    const p1 = document.createElement('p');
    p1.style.fontSize = '18px';
    const strongUrl = document.createElement('strong');
    strongUrl.textContent = data.url; // Safe text insertion
    p1.appendChild(document.createTextNode('The URL '));
    p1.appendChild(strongUrl);
    p1.appendChild(document.createTextNode(' has been flagged as potential phishing.'));
    box.appendChild(p1);

    // Confidence
    const p2 = document.createElement('p');
    const strongConf = document.createElement('strong');
    strongConf.textContent = 'Confidence: ';
    p2.appendChild(strongConf);
    p2.appendChild(document.createTextNode((data.confidence * 100).toFixed(1) + '%'));
    box.appendChild(p2);

    // Buttons Container
    const btnContainer = document.createElement('div');
    btnContainer.style.marginTop = '30px';

    // Proceed Button
    const proceedBtn = document.createElement('button');
    proceedBtn.textContent = 'Proceed Anyway (Unsafe)';
    Object.assign(proceedBtn.style, {
        background: '#ef5350',
        color: 'white',
        border: 'none',
        padding: '10px 20px',
        fontSize: '16px',
        borderRadius: '5px',
        cursor: 'pointer'
    });
    proceedBtn.onclick = () => overlay.remove();
    btnContainer.appendChild(proceedBtn);

    // Trust Button
    const trustBtn = document.createElement('button');
    trustBtn.textContent = 'Trust this Domain';
    Object.assign(trustBtn.style, {
        background: '#ffa000',
        color: 'white',
        border: 'none',
        padding: '10px 20px',
        fontSize: '16px',
        borderRadius: '5px',
        cursor: 'pointer',
        marginLeft: '10px'
    });
    trustBtn.onclick = () => {
        const domain = new URL(window.location.href).hostname;
        chrome.runtime.sendMessage({ action: "addToAllowlist", domain: domain }, () => {
            alert("Domain added to allowlist. Please reload.");
            overlay.remove();
        });
    };
    btnContainer.appendChild(trustBtn);

    // Back Button (Smarter)
    const backBtn = document.createElement('button');
    backBtn.textContent = 'Go Back to Safety';
    Object.assign(backBtn.style, {
        background: '#4caf50',
        color: 'white',
        border: 'none',
        padding: '10px 20px',
        fontSize: '16px',
        borderRadius: '5px',
        cursor: 'pointer',
        marginLeft: '10px'
    });
    backBtn.onclick = () => {
        if (window.history.length > 1) {
            window.history.back();
        } else {
            window.location.href = "https://www.google.com";
        }
    };
    btnContainer.appendChild(backBtn);

    box.appendChild(btnContainer);
    overlay.appendChild(box);
    document.body.appendChild(overlay);
}
