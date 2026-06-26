/* ═══════════════════════════════════════════════════════════════
   Motion & Balance Detection — Frontend Application
   ═══════════════════════════════════════════════════════════════ */

(function () {
    'use strict';

    // ─── Configuration ───
    const CONFIG = {
        wsUrl: `ws://${location.host}/ws/stream`,
        apiBase: `http://${location.host}/api`,
        reconnectBaseMs: 1000,
        reconnectMaxMs: 30000,
        maxAlerts: 50,
    };

    // ─── DOM References ───
    const $ = (sel) => document.querySelector(sel);
    const dom = {
        // Header
        statusDot:       $('#statusDot'),
        statusText:      $('#statusText'),
        clock:           $('#clock'),
        // Video
        canvas:          $('#videoCanvas'),
        placeholder:     $('#videoPlaceholder'),
        overlayState:    $('#videoOverlayState'),
        overlayStateText:$('#overlayStateText'),
        overlayFps:      $('#overlayFps'),
        videoContainer:  $('#videoContainer'),
        btnFullscreen:   $('#btnFullscreen'),
        // Status card
        cardStatus:      $('#cardStatus'),
        statusIcon:      $('#statusIcon'),
        statusValue:     $('#statusValue'),
        statusSubtext:   $('#statusSubtext'),
        // Gauges
        gaugeTrunk:      $('#gaugeTrunkAngle'),
        gaugeCoM:        $('#gaugeComDeviation'),
        gaugeVel:        $('#gaugeVelocity'),
        valueTrunk:      $('#valueTrunkAngle'),
        valueCoM:        $('#valueComDeviation'),
        valueVel:        $('#valueVelocity'),
        // FPS
        fpsValue:        $('#fpsValue'),
        fpsBar:          $('#fpsBar'),
        // Alerts
        alertsList:      $('#alertsList'),
        alertsEmpty:     $('#alertsEmpty'),
        btnClearAlerts:  $('#btnClearAlerts'),
        // Controls
        selectSource:    $('#selectVideoSource'),
        btnRefresh:      $('#btnRefreshVideos'),
        sliderTrunk:     $('#sliderTrunk'),
        sliderVelocity:  $('#sliderVelocity'),
        sliderCom:       $('#sliderCom'),
        sliderTrunkVal:  $('#sliderTrunkVal'),
        sliderVelocityVal:$('#sliderVelocityVal'),
        sliderComVal:    $('#sliderComVal'),
        btnStartStop:    $('#btnStartStop'),
        inputUpload:     $('#inputUpload'),
        btnSoundToggle:  $('#btnSoundToggle'),
        soundIcon:       $('#soundIcon'),
    };

    const ctx = dom.canvas.getContext('2d');

    // ─── State ───
    let ws = null;
    let reconnectAttempts = 0;
    let reconnectTimer = null;
    let isRunning = false;
    let previousState = 'Normal';
    let alertCount = 0;
    let soundEnabled = true;
    let audioCtx = null;

    // Animation targets (for smooth gauge interpolation)
    let animTargets = { trunk: 0, com: 0, vel: 0 };
    let animCurrent = { trunk: 0, com: 0, vel: 0 };
    let currentState = 'Normal';

    // ═══════════════════════ CLOCK ═══════════════════════
    function updateClock() {
        const now = new Date();
        dom.clock.textContent = now.toLocaleTimeString('en-US', { hour12: false });
    }
    setInterval(updateClock, 1000);
    updateClock();

    // ═══════════════════════ WEBSOCKET MANAGER ═══════════════════════
    function connect() {
        if (ws && (ws.readyState === WebSocket.OPEN || ws.readyState === WebSocket.CONNECTING)) return;

        ws = new WebSocket(CONFIG.wsUrl);
        ws.binaryType = 'blob';

        ws.onopen = () => {
            reconnectAttempts = 0;
            setConnectionStatus(true);
            console.log('[WS] Connected');
        };

        ws.onmessage = (event) => {
            try {
                const data = JSON.parse(event.data);
                handleFrame(data);
            } catch (err) {
                console.error('[WS] Parse error:', err);
            }
        };

        ws.onclose = (e) => {
            setConnectionStatus(false);
            console.log('[WS] Closed', e.code, e.reason);
            scheduleReconnect();
        };

        ws.onerror = (err) => {
            console.error('[WS] Error:', err);
            ws.close();
        };
    }

    function scheduleReconnect() {
        if (reconnectTimer) clearTimeout(reconnectTimer);
        const delay = Math.min(
            CONFIG.reconnectBaseMs * Math.pow(2, reconnectAttempts),
            CONFIG.reconnectMaxMs
        );
        reconnectAttempts++;
        console.log(`[WS] Reconnecting in ${delay}ms (attempt ${reconnectAttempts})`);
        reconnectTimer = setTimeout(connect, delay);
    }

    function setConnectionStatus(connected) {
        dom.statusDot.classList.toggle('connected', connected);
        dom.statusText.textContent = connected ? 'Connected' : 'Disconnected';
    }

    // ═══════════════════════ FRAME HANDLER ═══════════════════════
    const frameImg = new Image();
    let pendingFrame = null;

    function handleFrame(data) {
        // Update frame
        if (data.frame) {
            pendingFrame = 'data:image/jpeg;base64,' + data.frame;
            dom.placeholder.classList.add('hidden');
        }

        // Update state
        if (data.state) {
            currentState = data.state;
            updateStatusDisplay(data.state);
        }

        // Update metric targets
        if (data.trunk_angle != null) animTargets.trunk = data.trunk_angle;
        if (data.com_deviation != null) animTargets.com = data.com_deviation;
        if (data.velocity != null) animTargets.vel = data.velocity;

        // FPS
        if (data.fps != null) {
            const fps = Math.round(data.fps);
            dom.fpsValue.textContent = fps;
            dom.overlayFps.textContent = fps + ' FPS';
            dom.fpsBar.style.width = Math.min(fps / 60 * 100, 100) + '%';
        }

        // Alert on state change
        if (data.state && data.state !== previousState) {
            if (data.state === 'Imbalanced' || data.state === 'Fall_Detected') {
                addAlert(data);
            }
            previousState = data.state;
        }
    }

    // ═══════════════════════ RENDER LOOP ═══════════════════════
    function renderLoop() {
        // Draw frame if available
        if (pendingFrame) {
            frameImg.onload = () => {
                // Resize canvas to match frame aspect ratio within container
                const containerW = dom.canvas.parentElement.clientWidth;
                const containerH = dom.canvas.parentElement.clientHeight;
                const imgAspect = frameImg.width / frameImg.height;
                const containerAspect = containerW / containerH;

                let drawW, drawH;
                if (imgAspect > containerAspect) {
                    drawW = containerW;
                    drawH = containerW / imgAspect;
                } else {
                    drawH = containerH;
                    drawW = containerH * imgAspect;
                }

                dom.canvas.width = drawW;
                dom.canvas.height = drawH;
                ctx.drawImage(frameImg, 0, 0, drawW, drawH);
            };
            frameImg.src = pendingFrame;
            pendingFrame = null;
        }

        // Animate gauges (lerp)
        const lerpFactor = 0.15;
        animCurrent.trunk += (animTargets.trunk - animCurrent.trunk) * lerpFactor;
        animCurrent.com   += (animTargets.com   - animCurrent.com)   * lerpFactor;
        animCurrent.vel   += (animTargets.vel   - animCurrent.vel)   * lerpFactor;

        updateGauge(dom.gaugeTrunk, dom.valueTrunk, animCurrent.trunk, 90, '°', 1);
        updateGauge(dom.gaugeCoM,   dom.valueCoM,   animCurrent.com,   1,  '',  3);
        updateGauge(dom.gaugeVel,   dom.gaugeVel,    animCurrent.vel,   1,  '',  3);

        // Value text for velocity (gauge element is fill, need to update text separately)
        dom.valueVel.textContent = animCurrent.vel.toFixed(3);

        requestAnimationFrame(renderLoop);
    }

    function updateGauge(fillEl, valueEl, value, maxVal, suffix, decimals) {
        const pct = Math.min(Math.max(value / maxVal * 100, 0), 100);
        fillEl.style.width = pct + '%';

        // Color gauge based on percentage and current state
        fillEl.classList.remove('amber', 'red');
        if (currentState === 'Fall_Detected') {
            fillEl.classList.add('red');
        } else if (currentState === 'Imbalanced') {
            fillEl.classList.add('amber');
        }

        if (valueEl !== fillEl) {
            valueEl.textContent = value.toFixed(decimals);
        }
    }

    // ═══════════════════════ STATUS DISPLAY ═══════════════════════
    function updateStatusDisplay(state) {
        const overlay = dom.overlayState;
        overlay.classList.remove('imbalanced', 'fall');

        let dataState, icon, label, subtext;

        switch (state) {
            case 'Fall_Detected':
                dataState = 'fall';
                icon = '⚠';
                label = 'FALL DETECTED';
                subtext = 'Immediate attention required';
                overlay.classList.add('fall');
                if (soundEnabled) playAlertSound();
                break;
            case 'Imbalanced':
                dataState = 'imbalanced';
                icon = '◈';
                label = 'IMBALANCED';
                subtext = 'Balance deviation detected';
                overlay.classList.add('imbalanced');
                break;
            default:
                dataState = 'normal';
                icon = '✓';
                label = 'NORMAL';
                subtext = 'All metrics within safe range';
                break;
        }

        dom.cardStatus.dataset.state = dataState;
        dom.statusIcon.textContent = icon;
        dom.statusValue.textContent = label;
        dom.statusSubtext.textContent = subtext;
        dom.overlayStateText.textContent = label;
    }

    // ═══════════════════════ ALERT LOG ═══════════════════════
    function addAlert(data) {
        dom.alertsEmpty.classList.add('hidden');

        const entry = document.createElement('div');
        entry.className = 'alert-entry ' + (data.state === 'Fall_Detected' ? 'fall' : 'imbalanced');

        const time = data.timestamp
            ? new Date(data.timestamp).toLocaleTimeString('en-US', { hour12: false })
            : new Date().toLocaleTimeString('en-US', { hour12: false });

        const stateLabel = data.state === 'Fall_Detected' ? 'FALL DETECTED' : 'IMBALANCED';
        const details = [
            data.trunk_angle != null ? `Trunk: ${data.trunk_angle.toFixed(1)}°` : null,
            data.com_deviation != null ? `CoM: ${data.com_deviation.toFixed(3)}` : null,
            data.velocity != null ? `Vel: ${data.velocity.toFixed(3)}` : null,
        ].filter(Boolean).join(' · ');

        entry.innerHTML = `
            <span class="alert-time">${time}</span>
            <div class="alert-content">
                <span class="alert-state">${stateLabel}</span>
                <span class="alert-detail">${details}</span>
            </div>
        `;

        dom.alertsList.insertBefore(entry, dom.alertsList.firstChild);
        alertCount++;

        // Trim old alerts
        while (alertCount > CONFIG.maxAlerts) {
            dom.alertsList.removeChild(dom.alertsList.lastChild);
            alertCount--;
        }
    }

    function clearAlerts() {
        dom.alertsList.innerHTML = '';
        alertCount = 0;
        dom.alertsEmpty.classList.remove('hidden');
        dom.alertsList.appendChild(dom.alertsEmpty);
    }

    // ═══════════════════════ SOUND ALERT ═══════════════════════
    function initAudio() {
        if (!audioCtx) {
            audioCtx = new (window.AudioContext || window.webkitAudioContext)();
        }
    }

    function playAlertSound() {
        try {
            initAudio();
            if (audioCtx.state === 'suspended') audioCtx.resume();

            const osc = audioCtx.createOscillator();
            const gain = audioCtx.createGain();
            osc.connect(gain);
            gain.connect(audioCtx.destination);

            osc.type = 'sine';
            osc.frequency.setValueAtTime(880, audioCtx.currentTime);
            osc.frequency.setValueAtTime(660, audioCtx.currentTime + 0.15);
            osc.frequency.setValueAtTime(880, audioCtx.currentTime + 0.3);

            gain.gain.setValueAtTime(0.3, audioCtx.currentTime);
            gain.gain.exponentialRampToValueAtTime(0.01, audioCtx.currentTime + 0.5);

            osc.start(audioCtx.currentTime);
            osc.stop(audioCtx.currentTime + 0.5);
        } catch (e) {
            console.warn('[Audio] Alert sound failed:', e);
        }
    }

    function toggleSound() {
        soundEnabled = !soundEnabled;
        dom.btnSoundToggle.classList.toggle('muted', !soundEnabled);
        if (soundEnabled) {
            dom.soundIcon.innerHTML = '<path d="M11 5L6 9H2v6h4l5 4V5z"/><path d="M19.07 4.93a10 10 0 0 1 0 14.14M15.54 8.46a5 5 0 0 1 0 7.07"/>';
        } else {
            dom.soundIcon.innerHTML = '<path d="M11 5L6 9H2v6h4l5 4V5z"/><line x1="23" y1="9" x2="17" y2="15"/><line x1="17" y1="9" x2="23" y2="15"/>';
        }
    }

    // ═══════════════════════ REST API CONTROLS ═══════════════════════
    async function apiFetch(path, options = {}) {
        try {
            const res = await fetch(CONFIG.apiBase + path, {
                ...options,
                headers: {
                    'Content-Type': 'application/json',
                    ...(options.headers || {}),
                },
            });
            if (!res.ok) throw new Error(`HTTP ${res.status}`);
            return res.json();
        } catch (err) {
            console.error(`[API] ${path}:`, err);
            return null;
        }
    }

    async function loadVideoList() {
        const data = await apiFetch('/videos');
        dom.selectSource.innerHTML = '<option value="">— Select source —</option>';
        if (data && Array.isArray(data)) {
            data.forEach((name) => {
                const opt = document.createElement('option');
                opt.value = name;
                opt.textContent = name;
                dom.selectSource.appendChild(opt);
            });
        } else if (data && data.videos) {
            data.videos.forEach((name) => {
                const opt = document.createElement('option');
                opt.value = name;
                opt.textContent = name;
                dom.selectSource.appendChild(opt);
            });
        }
    }

    async function selectVideoSource(filename) {
        if (!filename) return;
        await apiFetch(`/video/select/${encodeURIComponent(filename)}`, { method: 'POST' });
    }

    async function updateConfig() {
        const payload = {
            trunk_angle_threshold: parseFloat(dom.sliderTrunk.value),
            velocity_threshold: parseFloat(dom.sliderVelocity.value),
            com_margin: parseFloat(dom.sliderCom.value),
        };
        await apiFetch('/config', {
            method: 'PUT',
            body: JSON.stringify(payload),
        });
    }

    async function uploadVideo(file) {
        if (!file) return;
        const formData = new FormData();
        formData.append('file', file);
        try {
            const res = await fetch(CONFIG.apiBase + '/video/upload', {
                method: 'POST',
                body: formData,
            });
            if (res.ok) {
                console.log('[Upload] Success');
                await loadVideoList();
            } else {
                console.error('[Upload] Failed', res.status);
            }
        } catch (err) {
            console.error('[Upload] Error:', err);
        }
    }

    async function toggleStartStop() {
        if (!isRunning) {
            await apiFetch('/start', { method: 'POST' });
            isRunning = true;
            dom.btnStartStop.classList.add('active');
            dom.btnStartStop.querySelector('span').textContent = 'Stop';
            dom.btnStartStop.querySelector('svg').innerHTML =
                '<rect x="6" y="4" width="4" height="16"/><rect x="14" y="4" width="4" height="16"/>';
        } else {
            await apiFetch('/stop', { method: 'POST' });
            isRunning = false;
            dom.btnStartStop.classList.remove('active');
            dom.btnStartStop.querySelector('span').textContent = 'Start';
            dom.btnStartStop.querySelector('svg').innerHTML =
                '<polygon points="5 3 19 12 5 21 5 3"/>';
        }
    }

    // ═══════════════════════ FULLSCREEN ═══════════════════════
    function toggleFullscreen() {
        if (!document.fullscreenElement) {
            dom.videoContainer.requestFullscreen().catch(console.error);
        } else {
            document.exitFullscreen().catch(console.error);
        }
    }

    // ═══════════════════════ SLIDER UPDATES ═══════════════════════
    let configDebounce = null;

    function onSliderInput() {
        dom.sliderTrunkVal.textContent = dom.sliderTrunk.value + '°';
        dom.sliderVelocityVal.textContent = parseFloat(dom.sliderVelocity.value).toFixed(2);
        dom.sliderComVal.textContent = parseFloat(dom.sliderCom.value).toFixed(2);

        if (configDebounce) clearTimeout(configDebounce);
        configDebounce = setTimeout(updateConfig, 400);
    }

    // ═══════════════════════ EVENT BINDINGS ═══════════════════════
    dom.btnFullscreen.addEventListener('click', toggleFullscreen);
    dom.btnClearAlerts.addEventListener('click', clearAlerts);
    dom.btnRefresh.addEventListener('click', loadVideoList);
    dom.btnStartStop.addEventListener('click', toggleStartStop);
    dom.btnSoundToggle.addEventListener('click', toggleSound);

    dom.selectSource.addEventListener('change', (e) => selectVideoSource(e.target.value));

    dom.sliderTrunk.addEventListener('input', onSliderInput);
    dom.sliderVelocity.addEventListener('input', onSliderInput);
    dom.sliderCom.addEventListener('input', onSliderInput);

    dom.inputUpload.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            uploadVideo(e.target.files[0]);
            e.target.value = '';
        }
    });

    // Initialize audio context on first user interaction
    document.addEventListener('click', () => initAudio(), { once: true });

    // ═══════════════════════ BOOTSTRAP ═══════════════════════
    function init() {
        connect();
        loadVideoList();
        requestAnimationFrame(renderLoop);
    }

    init();
})();
