/**
 * Sentiment Pulse - API Client Service (api.js)
 * Centralized API client service for sentiment analysis and customer feedback intelligence.
 */

let lastColdStartNotice = 0;
function showColdStartNotice() {
  const now = Date.now();
  if (now - lastColdStartNotice > 5000) {
    lastColdStartNotice = now;
    if (typeof window !== 'undefined' && typeof window.showToast === 'function') {
      window.showToast('Server is waking up (~50s), retrying…', 'info');
    }
  }
}

const API = {
  /**
   * Core request dispatcher with automatic error parsing and cold-start retry handling
   */
  async request(endpoint, options = {}) {
    const url = endpoint.startsWith('/') ? endpoint : `/${endpoint}`;
    const retryCount = options._retryCount || 0;
    const maxRetries = options.maxRetries !== undefined ? options.maxRetries : 3;
    const retryDelay = options.retryDelayMs !== undefined ? options.retryDelayMs : 3000;
    const timeoutMs = options.timeoutMs !== undefined ? options.timeoutMs : 60000;

    const config = {
      ...options,
      headers: {
        Accept: 'application/json',
        ...(options.headers || {}),
      },
    };

    // If body is an object and not FormData, serialize to JSON
    if (config.body && !(config.body instanceof FormData) && typeof config.body === 'object') {
      config.headers['Content-Type'] = 'application/json';
      config.body = JSON.stringify(config.body);
    }

    const controller =
      typeof AbortController !== 'undefined' && !options.signal ? new AbortController() : null;
    let timerId = null;

    if (controller) {
      config.signal = controller.signal;
      timerId = setTimeout(() => {
        controller.abort(new Error('timeout'));
      }, timeoutMs);
    }

    try {
      const response = await fetch(url, config);
      if (timerId) clearTimeout(timerId);

      // Handle Render cold starts (502 / 503 while booting server)
      if (response.status === 502 || response.status === 503) {
        if (retryCount < maxRetries) {
          showColdStartNotice();
          await new Promise((resolve) => setTimeout(resolve, retryDelay));
          return this.request(endpoint, {
            ...options,
            _retryCount: retryCount + 1,
          });
        }
      }

      const requestId = response.headers.get('X-Request-ID');
      const processTime = response.headers.get('X-Process-Time');

      // Handle raw non-JSON responses (e.g. CSV exports, HTML fallbacks)
      const contentType = response.headers.get('content-type') || '';
      if (contentType.includes('text/csv') || contentType.includes('application/octet-stream')) {
        if (!response.ok) {
          throw new Error(`Export failed with status ${response.status}`);
        }
        return await response.blob();
      }

      const isJson = contentType.includes('application/json');
      const data = isJson ? await response.json() : await response.text();

      if (!response.ok) {
        let errorMsg = 'An unexpected server error occurred.';
        if (data && typeof data === 'object') {
          if (data.detail) {
            errorMsg = typeof data.detail === 'string' ? data.detail : JSON.stringify(data.detail);
          } else if (data.message) {
            errorMsg = data.message;
          }
        } else if (typeof data === 'string' && data.trim()) {
          errorMsg = data;
        }

        const error = new Error(errorMsg);
        error.status = response.status;
        error.requestId = requestId;
        error.processTime = processTime;
        throw error;
      }

      return data;
    } catch (err) {
      if (timerId) clearTimeout(timerId);

      const isTimeout =
        err.name === 'AbortError' ||
        err.name === 'TimeoutError' ||
        (err.message && (err.message.includes('timeout') || err.message.includes('timed out')));

      if (isTimeout && retryCount < maxRetries) {
        showColdStartNotice();
        await new Promise((resolve) => setTimeout(resolve, retryDelay));
        return this.request(endpoint, {
          ...options,
          _retryCount: retryCount + 1,
        });
      }

      if (
        (err.name === 'TypeError' && err.message.includes('fetch')) ||
        (typeof navigator !== 'undefined' && !navigator.onLine && !err.status)
      ) {
        if (typeof navigator !== 'undefined' && !navigator.onLine) {
          throw new Error('Network offline: Server connection is required for this action.');
        }
        throw new Error(
          'Network error: Unable to connect to server. Please check your internet connection.'
        );
      }
      throw err;
    }
  },

  // Prediction Endpoints
  async predict(text) {
    return this.request('/predict', {
      method: 'POST',
      body: { text },
    });
  },

  async predictBatch(texts) {
    return this.request('/predict/batch', {
      method: 'POST',
      body: { texts },
    });
  },

  // Customer Feedback Management Endpoints
  async createFeedback(payload) {
    return this.request('/feedback', {
      method: 'POST',
      body: payload,
    });
  },

  async listFeedback(params = {}) {
    const query = new URLSearchParams();
    Object.entries(params).forEach(([key, val]) => {
      if (val !== undefined && val !== null && val !== '') {
        query.append(key, val);
      }
    });
    const queryString = query.toString();
    return this.request(`/feedback${queryString ? `?${queryString}` : ''}`);
  },

  async getFeedback(id) {
    return this.request(`/feedback/${id}`);
  },

  async deleteFeedback(id) {
    return this.request(`/feedback/${id}`, {
      method: 'DELETE',
    });
  },

  async getStats() {
    return this.request('/feedback/stats');
  },

  async getAnalytics(params = {}) {
    const query = new URLSearchParams();
    if (params.start_date) query.append('start_date', params.start_date);
    if (params.end_date) query.append('end_date', params.end_date);
    const qs = query.toString();
    return this.request(`/feedback/analytics${qs ? `?${qs}` : ''}`);
  },

  // Import / Export Endpoints
  async uploadCSV(file) {
    const formData = new FormData();
    formData.append('file', file);
    return this.request('/feedback/import', {
      method: 'POST',
      body: formData,
      timeoutMs: 300000, // 5-minute timeout for large batch CSV uploads (up to 100 MB)
    });
  },

  async getImportJob(jobId) {
    return this.request(`/feedback/import/${jobId}`);
  },

  async exportCSV() {
    const blob = await this.request('/feedback/export');
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `sentiment_pulse_export_${new Date().toISOString().slice(0, 10)}.csv`;
    document.body.appendChild(a);
    a.click();
    window.URL.revokeObjectURL(url);
    document.body.removeChild(a);
  },

  // Health / Probes
  async checkHealth() {
    return this.request('/health');
  },

  async checkReadiness() {
    return this.request('/health/ready');
  },
};

window.api = API;
