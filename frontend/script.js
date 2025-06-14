const form = document.getElementById('forecast-form');
const resultsDiv = document.getElementById('forecast-data');
const loadingDiv = document.getElementById('loading');
const errorMessageDiv = document.getElementById('error-message');
const forecastTypeSelect = document.getElementById('forecast_type');
const dayOfWeekGroup = document.getElementById('day-of-week-group');
const dayOfWeekSelect = document.getElementById('day_of_week');
const modelSelect = document.getElementById('model_name');
const prophetOptions = document.querySelectorAll('.prophet-options');
const metricsContainer = document.getElementById('model-metrics');
const metricsDataDiv = document.getElementById('metrics-data');
const metricsLoadingDiv = document.getElementById('metrics-loading');
const metricsErrorDiv = document.getElementById('metrics-error');

const API_BASE_URL = 'http://127.0.0.1:8000';

// Show/hide day-of-week selector based on forecast type
forecastTypeSelect.addEventListener('change', () => {
    if (forecastTypeSelect.value === '1week' || forecastTypeSelect.value === '2weeks') {
        dayOfWeekGroup.style.display = 'block';
    } else {
        dayOfWeekGroup.style.display = 'none';
        dayOfWeekSelect.value = '';
    }
});

// Show/hide Prophet-specific options
modelSelect.addEventListener('change', () => {
    if (modelSelect.value === 'Prophet') {
        prophetOptions.forEach(option => {
            option.style.display = 'block';
        });
    } else {
        prophetOptions.forEach(option => {
            option.style.display = 'none';
            // Reset values
            document.getElementById('prophet_extended').value = '';
            document.getElementById('include_bounds').checked = false;
        });
    }
});

// On page load, check if Prophet is selected
window.addEventListener('load', () => {
    if (modelSelect.value === 'Prophet') {
        prophetOptions.forEach(option => {
            option.style.display = 'block';
        });
    }
});

// Add event listeners for city and model selection to fetch metrics
document.getElementById('city').addEventListener('change', fetchModelMetrics);
modelSelect.addEventListener('change', fetchModelMetrics);

// Function to fetch model metrics
async function fetchModelMetrics() {
    const city = document.getElementById('city').value;
    const model = modelSelect.value;
    
    if (!city || !model) return;
    
    metricsContainer.style.display = 'block';
    metricsDataDiv.innerHTML = '';
    metricsErrorDiv.style.display = 'none';
    metricsLoadingDiv.style.display = 'block';
    
    try {
        const response = await fetch(`${API_BASE_URL}/model-metrics?city=${city}&model_name=${model}`);
        
        if (!response.ok) {
            let errorMsg = `Error: ${response.status} ${response.statusText}`;
            try {
                const errorData = await response.json();
                errorMsg = `Error ${response.status}: ${errorData.detail || response.statusText}`;
            } catch (e) {
                // If can't parse JSON error, use default message
            }
            throw new Error(errorMsg);
        }
        
        const metrics = await response.json();
        if (metrics && Object.keys(metrics).length > 0) {
            renderMetrics(metrics, city, model);
        } else {
            metricsDataDiv.innerHTML = '<p>No metrics available for this model-city combination.</p>';
        }
    } catch (error) {
        console.error('Metrics fetch error:', error);
        metricsErrorDiv.textContent = `Failed to fetch metrics: ${error.message}`;
        metricsErrorDiv.style.display = 'block';
    } finally {
        metricsLoadingDiv.style.display = 'none';
    }
}

function renderMetrics(metrics, city, model) {
    metricsDataDiv.innerHTML = '';
    
    // Create overall metrics card
    const overallCard = document.createElement('div');
    overallCard.className = 'metrics-card';
    
    const overallTitle = document.createElement('h3');
    overallTitle.textContent = `${model} Model Performance for ${city.charAt(0).toUpperCase() + city.slice(1)}`;
    overallCard.appendChild(overallTitle);
    
    if (metrics.overall) {
        const overallTable = createMetricsTable(metrics.overall, 'Overall');
        overallCard.appendChild(overallTable);
    }
    
    metricsDataDiv.appendChild(overallCard);
    
    // Create tabs for feature-specific metrics
    const tabs = document.createElement('div');
    tabs.className = 'metrics-tabs';
    
    const tabContents = document.createElement('div');
    tabContents.className = 'tab-contents';
    
    let firstTab = true;
    
    // Create a tab for each feature
    for (const feature of Object.keys(metrics)) {
        if (feature === 'overall') continue;
        
        const tab = document.createElement('div');
        tab.className = `metrics-tab${firstTab ? ' active' : ''}`;
        tab.textContent = feature;
        tab.dataset.feature = feature;
        tabs.appendChild(tab);
        
        // Create content for each tab
        const content = document.createElement('div');
        content.className = `tab-content${firstTab ? ' active' : ''}`;
        content.dataset.feature = feature;
        content.style.display = firstTab ? 'block' : 'none';
        
        const table = createMetricsTable(metrics[feature], feature);
        content.appendChild(table);
        tabContents.appendChild(content);
        
        firstTab = false;
        
        // Add click event to the tab
        tab.addEventListener('click', function() {
            // Hide all tab contents
            document.querySelectorAll('.tab-content').forEach(el => el.style.display = 'none');
            document.querySelectorAll('.metrics-tab').forEach(el => el.classList.remove('active'));
            
            // Show the clicked tab content
            document.querySelector(`.tab-content[data-feature="${this.dataset.feature}"]`).style.display = 'block';
            this.classList.add('active');
        });
    }
    
    if (Object.keys(metrics).length > 1) {
        metricsDataDiv.appendChild(tabs);
        metricsDataDiv.appendChild(tabContents);
    }
}

function createMetricsTable(metricData, title) {
    const table = document.createElement('table');
    table.className = 'metrics-table';
    
    const thead = document.createElement('thead');
    const headerRow = document.createElement('tr');
    
    ['Metric', 'Value', 'Interpretation'].forEach(header => {
        const th = document.createElement('th');
        th.textContent = header;
        headerRow.appendChild(th);
    });
    
    thead.appendChild(headerRow);
    table.appendChild(thead);
    
    const tbody = document.createElement('tbody');
    
    // Add rows for each metric
    const metricLabels = {
        'mae': 'Mean Absolute Error',
        'rmse': 'Root Mean Squared Error',
        'mape': 'Mean Absolute Percentage Error (%)',
        'r2': 'R² Score'
    };
    
    for (const [metric, value] of Object.entries(metricData)) {
        if (value === null) continue;
        
        const row = document.createElement('tr');
        
        // Metric name
        const metricCell = document.createElement('td');
        metricCell.textContent = metricLabels[metric] || metric;
        row.appendChild(metricCell);
        
        // Value
        const valueCell = document.createElement('td');
        const formatted = metric === 'mape' ? 
            `${value.toFixed(2)}%` : 
            metric === 'r2' ? 
                value.toFixed(3) : 
                value.toFixed(4);
        valueCell.textContent = formatted;
        row.appendChild(valueCell);
        
        // Interpretation
        const interpCell = document.createElement('td');
        let interpretation = '';
        let className = '';
        
        if (metric === 'r2') {
            if (value > 0.8) {
                interpretation = 'Excellent';
                className = 'metric-good';
            } else if (value > 0.6) {
                interpretation = 'Good';
                className = 'metric-good';
            } else if (value > 0.4) {
                interpretation = 'Moderate';
                className = 'metric-moderate';
            } else {
                interpretation = 'Poor';
                className = 'metric-poor';
            }
        } else if (metric === 'mape') {
            if (value < 10) {
                interpretation = 'Excellent';
                className = 'metric-good';
            } else if (value < 20) {
                interpretation = 'Good';
                className = 'metric-good';
            } else if (value < 50) {
                interpretation = 'Moderate';
                className = 'metric-moderate';
            } else {
                interpretation = 'Poor';
                className = 'metric-poor';
            }
        }
        
        interpCell.textContent = interpretation;
        interpCell.className = className;
        row.appendChild(interpCell);
        
        tbody.appendChild(row);
    }
    
    table.appendChild(tbody);
    return table;
}

// Initial metrics fetch on page load
window.addEventListener('load', fetchModelMetrics);

form.addEventListener('submit', async (event) => {
    event.preventDefault();

    resultsDiv.innerHTML = '';
    errorMessageDiv.style.display = 'none';
    errorMessageDiv.textContent = '';
    loadingDiv.style.display = 'block';

    const city = document.getElementById('city').value;
    const model_name = document.getElementById('model_name').value;
    const forecast_type = forecastTypeSelect.value;
    const day_of_week = dayOfWeekSelect.value;

    const params = new URLSearchParams({
        city: city,
        model_name: model_name,
        forecast_type: forecast_type,
    });

    // Add day of week if applicable
    if ((forecast_type === '1week' || forecast_type === '2weeks') && day_of_week !== '') {
        params.append('day_of_week', day_of_week);
    }

    // Add Prophet-specific parameters if applicable
    if (model_name === 'Prophet') {
        const prophetExtended = document.getElementById('prophet_extended').value;
        const includeBounds = document.getElementById('include_bounds').checked;
        
        if (prophetExtended) {
            params.append('prophet_extended', prophetExtended);
        }
        
        if (includeBounds) {
            params.append('include_bounds', 'true');
        }
    }

    const apiUrl = `${API_BASE_URL}/predict?${params.toString()}`;

    console.log(`Fetching: ${apiUrl}`);

    try {
        const response = await fetch(apiUrl);

        if (!response.ok) {
            let errorMsg = `Error: ${response.status} ${response.statusText}`;
            try {
                const errorData = await response.json();
                errorMsg = `Error ${response.status}: ${errorData.detail || response.statusText}`;
            } catch (e) {
                // If can't parse JSON error, use default message
            }
            throw new Error(errorMsg);
        }

        const data = await response.json();

        if (data && data.length > 0) {
            renderResults(data);
        } else {
            resultsDiv.innerHTML = '<p>No forecast data available for the selected criteria.</p>';
        }

    } catch (error) {
        console.error('Fetch error:', error);
        errorMessageDiv.textContent = `Failed to fetch forecast: ${error.message}`;
        errorMessageDiv.style.display = 'block';
    } finally {
        loadingDiv.style.display = 'none';
    }
});

function renderResults(data) {
    if (!data || data.length === 0) {
        resultsDiv.innerHTML = '<p>No forecast data received.</p>';
        return;
    }

    const table = document.createElement('table');
    const thead = document.createElement('thead');
    const tbody = document.createElement('tbody');
    const headerRow = document.createElement('tr');

    const headers = Object.keys(data[0]);
    headers.forEach(headerText => {
        const th = document.createElement('th');
        th.textContent = headerText;
        headerRow.appendChild(th);
    });
    thead.appendChild(headerRow);

    data.forEach(rowData => {
        const row = document.createElement('tr');
        headers.forEach(header => {
            const td = document.createElement('td');
            td.textContent = rowData[header];
            row.appendChild(td);
        });
        tbody.appendChild(row);
    });

    table.appendChild(thead);
    table.appendChild(tbody);
    resultsDiv.appendChild(table);
}