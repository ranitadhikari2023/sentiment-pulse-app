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
      const cfRay = response.headers.get('cf-ray') || response.headers.get('CF-Ray');
      const cfMitigated = response.headers.get('cf-mitigated');
      const retryAfterHeader =
        response.headers.get('retry-after') || response.headers.get('Retry-After');
      const parsedRetryAfter = retryAfterHeader ? parseInt(retryAfterHeader, 10) : null;
      const retryAfter = !isNaN(parsedRetryAfter) && parsedRetryAfter > 0 ? parsedRetryAfter : null;

      // Handle raw non-JSON responses (e.g. CSV exports, binary streams)
      const contentType = response.headers.get('content-type') || '';
      if (contentType.includes('text/csv') || contentType.includes('application/octet-stream')) {
        if (!response.ok) {
          throw new Error(`Export failed with status ${response.status}`);
        }
        return await response.blob();
      }

      const isJson = contentType.includes('application/json');
      let data = null;
      let rawText = '';
      let isCloudflareChallenge = false;
      let isHtmlResponse = false;

      if (isJson) {
        try {
          data = await response.json();
        } catch {
          // Response declared application/json but contained malformed payload
          data = null;
          isHtmlResponse = false;
        }
      } else {
        // Read text payload safely to inspect for edge HTML challenges
        try {
          rawText = await response.text();
        } catch {
          rawText = '';
        }

        const trimmed = rawText.trim();
        if (
          contentType.includes('text/html') ||
          trimmed.startsWith('<!DOCTYPE') ||
          trimmed.startsWith('<html')
        ) {
          isHtmlResponse = true;

          // Detect Cloudflare Managed Challenge or Bot Fight Mode
          if (
            cfMitigated === 'challenge' ||
            rawText.includes('cf_chl_opt') ||
            rawText.includes('<title>Just a moment...</title>') ||
            rawText.includes('Checking your browser') ||
            rawText.includes('cloudflare-static') ||
            (rawText.includes('Cloudflare') && [403, 429, 503].includes(response.status))
          ) {
            isCloudflareChallenge = true;
          }
        } else {
          // Plain text fallback
          data = trimmed;
        }
      }

      if (!response.ok) {
        let errorMsg = 'An unexpected server error occurred.';
        let errorType = null;
        let jobId = null;

        if (isCloudflareChallenge) {
          errorMsg = 'Cloudflare security challenge encountered. Edge verification required.';
          errorType = 'cloudflare_challenge';
          // Never assign raw HTML challenge payload to detail
          data = null;
        } else if (isHtmlResponse) {
          // Provide clean, human-readable edge error instead of raw HTML
          if (response.status === 502) {
            errorMsg = 'Bad Gateway: The application server is temporarily unreachable (502).';
            errorType = 'edge_bad_gateway';
          } else if (response.status === 503) {
            errorMsg =
              'Service Unavailable: The server is starting or capacity is constrained (503).';
            errorType = 'edge_service_unavailable';
          } else if (response.status === 504) {
            errorMsg = 'Gateway Timeout: Request exceeded server response deadline (504).';
            errorType = 'edge_gateway_timeout';
          } else if (response.status === 403) {
            errorMsg = 'Access Denied: Request challenged or blocked by edge security rules (403).';
            errorType = 'edge_access_denied';
          } else {
            errorMsg = `Server or edge proxy returned an HTML error page (HTTP ${response.status}).`;
            errorType = 'edge_html_response';
          }
          data = null;
        } else if (data && typeof data === 'object') {
          if (data.detail) {
            if (typeof data.detail === 'string') {
              errorMsg = data.detail;
            } else if (typeof data.detail === 'object') {
              errorMsg = data.detail.detail || data.detail.message || JSON.stringify(data.detail);
              errorType = data.detail.error_type || data.detail.type || null;
              jobId = data.detail.job_id || null;
            }
          } else if (data.message) {
            errorMsg = data.message;
          }
          if (data.error_type) errorType = data.error_type;
          if (data.job_id) jobId = data.job_id;
        } else if (typeof data === 'string' && data.trim()) {
          errorMsg = data;
        } else if (!isJson && !data) {
          errorMsg = `Server returned invalid or unparseable response (HTTP ${response.status}).`;
          errorType = 'invalid_response';
        }

        const error = new Error(errorMsg);
        error.status = response.status;
        error.requestId = requestId;
        error.processTime = processTime;
        error.errorType = errorType;
        error.jobId = jobId;
        error.detail = data;
        error.retryAfter = retryAfter;
        error.cfRay = cfRay || null;
        error.isCloudflareChallenge = isCloudflareChallenge;
        error.isHtmlResponse = isHtmlResponse;
        throw error;
      }

      // If response is OK but data is null because of malformed JSON
      if (data === null && !isHtmlResponse) {
        throw new Error('Received invalid JSON response from server.');
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

  async getImportJob(jobId, options = {}) {
    return this.request(`/feedback/import/${jobId}`, {
      method: 'GET',
      headers: {
        'Cache-Control': 'no-cache, no-store',
        Pragma: 'no-cache',
        ...(options.headers || {}),
      },
      ...options,
    });
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
