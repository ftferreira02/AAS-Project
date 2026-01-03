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

// Listen for messages including allowlist updates
chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
    if (request.action === "checkUrl") {
        const tabId = request.tabId || (sender.tab ? sender.tab.id : undefined);
        handleCheckUrl(request.url, tabId).then(sendResponse);
        return true;
    } else if (request.action === "addToAllowlist") {
        addToAllowlist(request.domain).then(() => sendResponse({ success: true }));
        return true;
    }
});

// Auto-analyze when tab updates
chrome.tabs.onUpdated.addListener((tabId, changeInfo, tab) => {
    if (changeInfo.status === 'complete' && tab.url && tab.url.startsWith('http')) {
        handleCheckUrl(tab.url, tabId);
    }
});

async function addToAllowlist(domain) {
    const { allowlist = [] } = await chrome.storage.local.get("allowlist");
    if (!allowlist.includes(domain)) {
        allowlist.push(domain);
        await chrome.storage.local.set({ allowlist });
    }
}

async function handleCheckUrl(url, tabId) {
    // 0. Check Allowlist
    try {
        const { allowlist = [] } = await chrome.storage.local.get("allowlist");
        const domain = new URL(url).hostname;
        if (allowlist.includes(domain)) {
            if (tabId) {
                chrome.action.setBadgeText({ text: "✓", tabId: tabId });
                chrome.action.setBadgeBackgroundColor({ color: "#4caf50", tabId: tabId });
            }
            return { is_phishing: false, feature_override: "User Allowed" };
        }
    } catch (e) { console.error(e); }

    // 1. Check Cache
    const now = Date.now();
    if (cache.has(url)) {
        const entry = cache.get(url);
        if (now - entry.timestamp < CACHE_TTL_MS) {
            console.log("Returning cached result for:", url);
            updateBadge(entry.result, tabId);
            return entry.result;
        } else {
            cache.delete(url);
        }
    }

    // 2. Call API
    console.log("Calling API for:", url);
    if (tabId) {
        chrome.action.setBadgeText({ text: "...", tabId: tabId });
    }

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

        // 3. Update Cache & Badge
        cache.set(url, {
            result: data,
            timestamp: now
        });
        updateBadge(data, tabId);

        return data; // { is_phishing: bool, confidence: float, ... }

    } catch (error) {
        console.error("Error:", error);
        return { error: "Network Error" };
    }
}

function updateBadge(data, tabId) {
    if (!tabId) return; // Cannot update badge without tabId

    if (data.level === "warning") {
        chrome.action.setBadgeText({ text: "?", tabId: tabId });
        chrome.action.setBadgeBackgroundColor({ color: "#ff9800", tabId: tabId }); // Orange
    } else if (data.level === "unsafe" || (data.is_phishing && !data.level)) {
        chrome.action.setBadgeText({ text: "!", tabId: tabId });
        chrome.action.setBadgeBackgroundColor({ color: "#d32f2f", tabId: tabId }); // Red
    } else {
        chrome.action.setBadgeText({ text: "✓", tabId: tabId });
        chrome.action.setBadgeBackgroundColor({ color: "#4caf50", tabId: tabId }); // Green
    }
}
