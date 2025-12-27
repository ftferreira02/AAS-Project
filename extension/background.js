// Background Service Worker
// Cache: { url: { result: Object, timestamp: Number } }
const cache = new Map();
const CACHE_TTL_MS = 5 * 60 * 1000; // 5 minutes

// Prune cache periodically (every 10 min)
setInterval(() => {
    const now = Date.now();
    for (const [url, entry] of cache.entries()) {
        if (now - entry.timestamp > CACHE_TTL_MS) {
            cache.delete(url);
        }
    }
}, 10 * 60 * 1000);

// Listen for messages from content script or popup
chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
    if (request.action === "checkUrl") {
        handleCheckUrl(request.url).then(sendResponse);
        return true; // Will respond asynchronously
    }
});

async function handleCheckUrl(url) {
    // 1. Check Cache
    const now = Date.now();
    if (cache.has(url)) {
        const entry = cache.get(url);
        if (now - entry.timestamp < CACHE_TTL_MS) {
            console.log("Returning cached result for:", url);
            return entry.result;
        } else {
            cache.delete(url);
        }
    }

    // 2. Call API
    console.log("Calling API for:", url);
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

        // 3. Update Cache
        cache.set(url, {
            result: data,
            timestamp: now
        });

        return data; // { is_phishing: bool, confidence: float, ... }

    } catch (error) {
        console.error("Error checking URL:", error);
        return { error: "Network Error" };
    }
}
