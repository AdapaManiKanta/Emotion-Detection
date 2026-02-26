const ctx = document.getElementById('emotionChart').getContext('2d');
let emotionChart;

const emotions = ["angry", "disgust", "scared", "happy", "sad", "surprised", "neutral"];
const colors = {
    "angry": 'rgb(255, 99, 132)',
    "disgust": 'rgb(153, 102, 255)',
    "scared": 'rgb(255, 159, 64)',
    "happy": 'rgb(75, 192, 192)',
    "sad": 'rgb(54, 162, 235)',
    "surprised": 'rgb(255, 205, 86)',
    "neutral": 'rgb(201, 203, 207)'
};

// Initialize Chart
function initChart() {
    const datasets = emotions.map(emotion => ({
        label: emotion.charAt(0).toUpperCase() + emotion.slice(1),
        borderColor: colors[emotion],
        data: [],
        fill: false,
        tension: 0.4
    }));

    emotionChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: [],
            datasets: datasets
        },
        options: {
            responsive: true,
            animation: false, // Turn off animation for smoother live updates
            scales: {
                x: {
                    display: false // Hide x-axis labels to look cleaner
                },
                y: {
                    min: 0,
                    max: 100,
                    ticks: { color: '#ccc' }
                }
            },
            plugins: {
                legend: { labels: { color: '#fff' } }
            }
        }
    });
}

// Fetch stats every second
async function fetchStats() {
    try {
        const response = await fetch('/api/stats');
        const data = await response.json();

        if (Object.keys(data).length === 0) return;

        console.log("Live Stats Data:", data); // Check values in browser console

        updateChart(data);
        updateDominantEmotion(data);
    } catch (error) {
        console.error("Error fetching stats:", error);
    }
}

function updateChart(data) {
    const now = new Date().toLocaleTimeString();

    // Keep max 20 data points on screen
    if (emotionChart.data.labels.length > 20) {
        emotionChart.data.labels.shift();
        emotionChart.data.datasets.forEach(dataset => dataset.data.shift());
    }

    emotionChart.data.labels.push(now);

    emotionChart.data.datasets.forEach(dataset => {
        const emotionName = dataset.label.toLowerCase();
        // data[emotionName] should be a percentage
        if (data.hasOwnProperty(emotionName)) {
            dataset.data.push(data[emotionName]);
        } else {
            dataset.data.push(0);
        }
    });

    emotionChart.update();
}

function updateDominantEmotion(data) {
    let maxEmotion = "neutral";
    let maxVal = -1;

    console.log(`[Dashboard] Update Heartbeat: ${data.count} | Age: ${(Date.now() / 1000 - data.timestamp).toFixed(2)}s`);

    for (const [key, value] of Object.entries(data)) {
        if (key === 'count' || key === 'timestamp') continue; // Skip metadata
        if (value > maxVal) {
            maxVal = value;
            maxEmotion = key;
        }
    }

    const domEl = document.getElementById('domEmotion');
    domEl.innerText = maxEmotion.charAt(0).toUpperCase() + maxEmotion.slice(1);

    // Explicitly override Bootstrap text-success or other classes
    domEl.style.color = colors[maxEmotion];
    domEl.classList.remove('text-success', 'text-warning', 'text-danger', 'text-info', 'text-primary');
}

// AR Emoji Toggle
document.getElementById('toggleEmojiBtn').addEventListener('click', async () => {
    const res = await fetch('/api/toggle_emoji', { method: 'POST' });
    const data = await res.json();
    const btn = document.getElementById('toggleEmojiBtn');

    if (data.emoji_enabled) {
        btn.classList.replace('btn-outline-light', 'btn-light');
        btn.classList.add('text-dark');
    } else {
        btn.classList.replace('btn-light', 'btn-outline-light');
        btn.classList.remove('text-dark');
    }
});

// Start loop
initChart();
setInterval(fetchStats, 1000); // 1 second interval
