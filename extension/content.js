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
    if (document.getElementById('phish-detector-host')) return;

    // --- Configuration ---
    const isWarning = data.level === 'warning';

    // Modern Color Palette (Tailwind-inspired)
    const colors = {
        overlay: 'rgba(0, 0, 0, 0.6)', // Darker, blurred overlay
        cardBg: '#ffffff',
        textPrimary: '#111827',
        textSecondary: '#4b5563',

        // Semantic Colors
        danger: '#ef4444',     // Red-500
        warning: '#f59e0b',    // Amber-500
        success: '#10b981',    // Emerald-500

        // Gradients
        dangerGrad: 'linear-gradient(135deg, #ef4444 0%, #dc2626 100%)',
        warningGrad: 'linear-gradient(135deg, #f59e0b 0%, #d97706 100%)',
        successGrad: 'linear-gradient(135deg, #10b981 0%, #059669 100%)',
    };

    const config = {
        icon: isWarning ? '⚠️' : '🚨',
        title: isWarning ? 'Suspicious Site Detected' : 'Phishing Threat Detected',
        message: `This site checks out as <strong>${isWarning ? 'suspicious' : 'unsafe'}</strong>. Use caution.`,
        themeColor: isWarning ? colors.warning : colors.danger,
        themeGrad: isWarning ? colors.warningGrad : colors.dangerGrad
    };

    // --- DOM Construction ---
    const host = document.createElement('div');
    host.id = 'phish-detector-host';
    document.body.appendChild(host);

    const shadow = host.attachShadow({ mode: 'closed' });

    // 1. Backdrop Overlay (Glassmorphism)
    const overlay = document.createElement('div');
    Object.assign(overlay.style, {
        position: 'fixed',
        top: '0', left: '0', width: '100vw', height: '100vh',
        backgroundColor: colors.overlay,
        backdropFilter: 'blur(8px)', // Glass effect
        zIndex: '2147483647',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif, "Apple Color Emoji", "Segoe UI Emoji"',
        opacity: '0',
        transition: 'opacity 0.3s ease-out'
    });

    // 2. Main Card
    const card = document.createElement('div');
    Object.assign(card.style, {
        background: colors.cardBg,
        width: '90%',
        maxWidth: '480px',
        borderRadius: '16px',
        padding: '32px',
        boxShadow: '0 20px 25px -5px rgba(0, 0, 0, 0.1), 0 10px 10px -5px rgba(0, 0, 0, 0.04)',
        textAlign: 'center',
        transform: 'scale(0.95)',
        transition: 'transform 0.3s cubic-bezier(0.34, 1.56, 0.64, 1)'
    });

    // Animate In
    requestAnimationFrame(() => {
        overlay.style.opacity = '1';
        card.style.transform = 'scale(1)';
    });

    // --- Content ---

    // Icon Wrapper
    const iconWrapper = document.createElement('div');
    Object.assign(iconWrapper.style, {
        fontSize: '48px',
        marginBottom: '16px',
        animation: 'bounce 1s infinite alternate' // Subtle bounce
    });
    iconWrapper.textContent = config.icon;
    card.appendChild(iconWrapper);

    // Title
    const h1 = document.createElement('h1');
    h1.textContent = config.title;
    Object.assign(h1.style, {
        color: colors.textPrimary,
        fontSize: '24px',
        fontWeight: '700',
        margin: '0 0 12px 0',
        lineHeight: '1.2'
    });
    card.appendChild(h1);

    // Description
    const p = document.createElement('p');
    p.innerHTML = `The URL <code style="background:#f3f4f6; padding:2px 4px; border-radius:4px; color:${colors.textPrimary}; font-size: 0.9em; word-break: break-all;">${data.url}</code> has been flagged. Confidence: <strong>${(data.confidence * 100).toFixed(1)}%</strong>`;
    Object.assign(p.style, {
        color: colors.textSecondary,
        fontSize: '16px',
        lineHeight: '1.6',
        margin: '0 0 32px 0'
    });
    card.appendChild(p);

    // --- Actions ---
    const btnGroup = document.createElement('div');
    Object.assign(btnGroup.style, {
        display: 'flex',
        flexDirection: 'column',
        gap: '12px'
    });

    // Helper: Button Creator
    const createButton = (text, type, handler) => {
        const btn = document.createElement('button');
        btn.textContent = text;
        const baseStyle = {
            width: '100%',
            padding: '12px 20px',
            borderRadius: '8px',
            fontSize: '16px',
            fontWeight: '600',
            cursor: 'pointer',
            border: 'none',
            transition: 'transform 0.1s, opacity 0.2s',
            outline: 'none'
        };

        if (type === 'primary') {
            Object.assign(btn.style, baseStyle, {
                background: colors.successGrad,
                color: 'white',
                boxShadow: '0 4px 6px -1px rgba(16, 185, 129, 0.4)'
            });
        } else if (type === 'danger') {
            Object.assign(btn.style, baseStyle, {
                background: 'transparent',
                border: `1px solid ${colors.danger}`,
                color: colors.danger
            });
        } else if (type === 'warning') {
            Object.assign(btn.style, baseStyle, {
                background: 'transparent',
                border: `1px solid ${colors.warning}`,
                color: colors.warning
            });
        } else { // Ghost / Link
            Object.assign(btn.style, baseStyle, {
                background: 'transparent',
                color: colors.textSecondary,
                fontSize: '14px',
                fontWeight: '500'
            });
        }

        btn.onmousedown = () => btn.style.transform = 'scale(0.98)';
        btn.onmouseup = () => btn.style.transform = 'scale(1)';
        btn.onmouseenter = () => btn.style.opacity = '0.9';
        btn.onmouseleave = () => btn.style.opacity = '1';
        btn.onclick = handler;
        return btn;
    };

    // 1. Primary Action: Go Back (Safest)
    const btnBack = createButton('Go Back to Safety', 'primary', () => {
        if (window.history.length > 1) window.history.back();
        else window.location.href = "https://www.google.com";
    });
    btnGroup.appendChild(btnBack);

    // 2. Secondary Action: Proceed (Dangerous)
    const btnProceed = createButton(
        isWarning ? 'Proceed with Caution' : 'I understand the risks, proceed',
        isWarning ? 'warning' : 'danger',
        () => host.remove()
    );
    btnGroup.appendChild(btnProceed);

    // 3. Tertiary Action: Trust
    const btnTrust = createButton('Don\'t ask again for this site', 'ghost', () => {
        const domain = new URL(data.url).hostname;
        // eslint-disable-next-line no-undef
        chrome.runtime.sendMessage({ action: "addToAllowlist", domain: domain }, () => {
            alert(`"${domain}" added to allowlist.`);
            host.remove();
        });
    });
    btnGroup.appendChild(btnTrust);

    card.appendChild(btnGroup);
    overlay.appendChild(card);
    shadow.appendChild(overlay);
}
