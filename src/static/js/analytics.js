// Analytics Dashboard Real-Time Monitor
class AnalyticsMonitor {
    constructor() {
        this.refreshInterval = 2000; // 2 seconds
        this.charts = {};
        this.init();
    }

    init() {
        console.log('[Analytics] Initializing real-time monitor...');
        this.initializeCharts();
        this.startPolling();
    }

    // Fetch data from endpoints
    async fetchStatus() {
        try {
            const response = await fetch('/status');
            if (!response.ok) throw new Error(`HTTP ${response.status}`);
            return await response.json();
        } catch (error) {
            console.error('[Analytics] Error fetching status:', error);
            return null;
        }
    }

    async fetchSystemHealth() {
        try {
            const response = await fetch('/system_health');
            if (!response.ok) throw new Error(`HTTP ${response.status}`);
            return await response.json();
        } catch (error) {
            console.error('[Analytics] Error fetching system health:', error);
            return null;
        }
    }

    async fetchAnalyticsData() {
        try {
            const response = await fetch('/analytics_data');
            if (!response.ok) throw new Error(`HTTP ${response.status}`);
            return await response.json();
        } catch (error) {
            console.error('[Analytics] Error fetching analytics data:', error);
            return null;
        }
    }

    // Update UI elements
    updateSystemHealth(data) {
        // CPU Usage
        const cpuEl = document.getElementById('cpu-usage');
        if (cpuEl) {
            cpuEl.textContent = data.cpu_percent?.toFixed(1) + '%' || '0.0%';
            this.animateValue(cpuEl);
        }

        // Memory Usage
        const memoryEl = document.getElementById('memory-usage');
        if (memoryEl) {
            memoryEl.textContent = Math.round(data.memory_mb || 0) + ' MB';
            this.animateValue(memoryEl);
        }

        // Camera Status
        const cameraEl = document.getElementById('camera-status');
        if (cameraEl) {
            const status = data.camera_status || 'Inactive';
            cameraEl.textContent = status;
            cameraEl.className = 'badge rounded-pill ' + (status === 'Active' ? 'bg-success' : 'bg-danger');
        }

        // Detector Status
        const detectorEl = document.getElementById('detector-status');
        if (detectorEl) {
            const isRunning = data.detector_running || false;
            detectorEl.textContent = isRunning ? 'Running' : 'Stopped';
            detectorEl.className = 'badge rounded-pill ' + (isRunning ? 'bg-success' : 'bg-secondary');
        }
    }

    updateSessionStats(statusData, analyticsData) {
        // Current Fatigue Level
        const fatigueEl = document.getElementById('current-fatigue');
        if (fatigueEl && statusData) {
            const levelNames = ['Fully Alert', 'Mild Fatigue', 'Moderate Fatigue', 'Severe Drowsiness', 'Critical Microsleep'];
            const level = statusData.fatigue_level || 0;
            fatigueEl.textContent = levelNames[level];
            this.animateValue(fatigueEl);
        }

        // Total Alerts
        const alertsEl = document.getElementById('total-alerts');
        if (alertsEl && statusData) {
            alertsEl.textContent = statusData.total_alerts || 0;
            this.animateValue(alertsEl);
        }

        // Yawns Detected
        const yawnsEl = document.getElementById('yawns-detected');
        if (yawnsEl && statusData) {
            yawnsEl.textContent = statusData.yawn_count || 0;
            this.animateValue(yawnsEl);
        }

        // Blink Rate
        const blinkEl = document.getElementById('blink-rate');
        if (blinkEl && statusData) {
            const rate = statusData.blink_rate || 0;
            blinkEl.textContent = rate.toFixed(1) + ' bpm';
            this.animateValue(blinkEl);
        }

        // Session Duration
        const durationEl = document.getElementById('session-duration');
        if (durationEl && statusData) {
            const minutes = statusData.session_duration_min || 0;
            durationEl.textContent = this.formatDuration(minutes);
            this.animateValue(durationEl);
        }
    }

    formatDuration(minutes) {
        if (minutes < 1) {
            return Math.round(minutes * 60) + 's';
        } else if (minutes < 60) {
            return minutes.toFixed(1) + 'min';
        } else {
            const hours = Math.floor(minutes / 60);
            const mins = Math.round(minutes % 60);
            return `${hours}h ${mins}min`;
        }
    }

    updateConfidenceChart(analyticsData) {
        if (!analyticsData || !analyticsData.confidence_timeline) return;

        const chartEl = document.getElementById('confidenceChart');
        if (!chartEl) return;

        const ctx = chartEl.getContext('2d');
        const data = analyticsData.confidence_timeline.slice(-60); // Last 60 data points

        if (!this.charts.confidence) {
            // Initialize chart if it doesn't exist
            this.charts.confidence = new Chart(ctx, {
                type: 'line',
                data: {
                    labels: data.map((d, i) => i),
                    datasets: [
                        {
                            label: 'Confidence Score (%)',
                            data: data.map(d => d.value),
                            borderColor: 'rgb(59, 130, 246)',
                            backgroundColor: 'rgba(59, 130, 246, 0.1)',
                            tension: 0.4,
                            fill: true
                        },
                        {
                            label: 'Alert Threshold',
                            data: Array(data.length).fill(80), // Default threshold
                            borderColor: 'rgb(239, 68, 68)',
                            borderDash: [5, 5],
                            fill: false
                        }
                    ]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        legend: {
                            labels: { color: '#fff' }
                        }
                    },
                    scales: {
                        y: {
                            beginAtZero: true,
                            max: 100,
                            ticks: { color: '#fff' },
                            grid: { color: 'rgba(255, 255, 255, 0.1)' }
                        },
                        x: {
                            ticks: { color: '#fff' },
                            grid: { color: 'rgba(255, 255, 255, 0.1)' }
                        }
                    }
                }
            });
        } else {
            // Update existing chart
            this.charts.confidence.data.labels = data.map((d, i) => i);
            this.charts.confidence.data.datasets[0].data = data.map(d => d.value);
            this.charts.confidence.data.datasets[1].data = Array(data.length).fill(80);
            this.charts.confidence.update('none'); // Update without animation for smoother updates
        }
    }

    updateFatigueChart(analyticsData) {
        if (!analyticsData || !analyticsData.fatigue_timeline) return;

        const chartEl = document.getElementById('fatigueChart');
        if (!chartEl) return;

        const ctx = chartEl.getContext('2d');
        const data = analyticsData.fatigue_timeline.slice(-60); // Last 60 data points

        const levelColors = [
            'rgb(34, 197, 94)',  // Green - Alert
            'rgb(59, 130, 246)', // Blue - Mild
            'rgb(234, 179, 8)',  // Yellow - Moderate
            'rgb(249, 115, 22)', // Orange - Severe
            'rgb(239, 68, 68)'   // Red - Critical
        ];

        if (!this.charts.fatigue) {
            this.charts.fatigue = new Chart(ctx, {
                type: 'bar',
                data: {
                    labels: data.map((d, i) => i),
                    datasets: [{
                        label: 'Fatigue Level',
                        data: data.map(d => d.level),
                        backgroundColor: data.map(d => levelColors[d.level]),
                        borderColor: data.map(d => levelColors[d.level]),
                        borderWidth: 1
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        legend: {
                            labels: { color: '#fff' }
                        }
                    },
                    scales: {
                        y: {
                            beginAtZero: true,
                            max: 4,
                            ticks: {
                                color: '#fff',
                                stepSize: 1,
                                callback: function (value) {
                                    const labels = ['Alert', 'Mild', 'Moderate', 'Severe', 'Critical'];
                                    return labels[value] || '';
                                }
                            },
                            grid: { color: 'rgba(255, 255, 255, 0.1)' }
                        },
                        x: {
                            ticks: { color: '#fff' },
                            grid: { color: 'rgba(255, 255, 255, 0.1)' }
                        }
                    }
                }
            });
        } else {
            this.charts.fatigue.data.labels = data.map((d, i) => i);
            this.charts.fatigue.data.datasets[0].data = data.map(d => d.level);
            this.charts.fatigue.data.datasets[0].backgroundColor = data.map(d => levelColors[d.level]);
            this.charts.fatigue.update('none');
        }
    }

    initializeCharts() {
        // Initialize charts if Chart.js is available
        if (typeof Chart === 'undefined') {
            console.warn('[Analytics] Chart.js not loaded, charts will not be displayed');
            return;
        }

        console.log('[Analytics] Charts initialized');
    }

    animateValue(element) {
        if (!element) return;
        element.style.transform = 'scale(1.05)';
        element.style.transition = 'transform 0.2s ease';
        setTimeout(() => {
            element.style.transform = 'scale(1)';
        }, 200);
    }

    async updateAll() {
        const statusData = await this.fetchStatus();
        const analyticsData = await this.fetchAnalyticsData();

        if (statusData) {
            // Update system health from status endpoint (it now includes all health data)
            this.updateSystemHealth(statusData);
            this.updateSessionStats(statusData, analyticsData || {});
        }

        if (analyticsData) {
            this.updateConfidenceChart(analyticsData);
            this.updateFatigueChart(analyticsData);
        }
    }

    startPolling() {
        // Initial update
        this.updateAll();

        // Poll every 2 seconds
        setInterval(() => {
            this.updateAll();
        }, this.refreshInterval);

        console.log(`[Analytics] Polling started (every ${this.refreshInterval}ms)`);
    }
}

// Initialize when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    // Only initialize on analytics page
    if (document.getElementById('cpu-usage') || document.getElementById('confidenceChart')) {
        console.log('[Analytics] Page detected, initializing monitor');
        new AnalyticsMonitor();
    }
});
