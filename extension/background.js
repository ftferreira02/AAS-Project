// Background Service Worker

// Listen for messages from content script
chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
    if (request.action === "checkUrl") {
        checkUrl(request.url).then(result => {
            sendResponse(result);
        });
        return true; // Will respond asynchronously
    }
});

async function checkUrl(url) {
    try {
        const response = await fetch("http://127.0.0.1:5000/predict", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({ url: url })
        });

        if (!response.ok) {
            return { error: "API Error" };
        }

        const data = await response.json();
        return data; // { is_phishing: bool, confidence: float, ... }

    } catch (error) {
        console.error("Error checking URL:", error);
        return { error: "Network Error" };
    }
}
