/**
 * Sentiment Pulse - Import & Export Controller (imports.js)
 * Handles CSV drag-and-drop uploads, validation, chunked batch execution,
 * job status polling, and sanitized CSV exports.
 */

const MAX_UPLOAD_BYTES = 100 * 1024 * 1024; // 100 MB (104,857,600 bytes)
const MAX_IMPORT_ROWS = 1050000;
const MAX_IMPORT_ROWS_FORMATTED = '1,050,000';

const POLLING_CONFIG = {
  initialIntervalMs: 2000,
  retryScheduleMs: [2000, 4000, 8000, 15000],
  maxIntervalMs: 15000,
  jitterRatio: 0.15,
  backgroundIntervalMs: 30000,
  pollTimeoutMs: 15000,
  maxConsecutiveErrors: 20,
};

const ImportsController = {
  selectedFile: null,
  isUploading: false,
  MAX_UPLOAD_BYTES: MAX_UPLOAD_BYTES,
  MAX_IMPORT_ROWS: MAX_IMPORT_ROWS,
  MAX_IMPORT_ROWS_FORMATTED: MAX_IMPORT_ROWS_FORMATTED,
  POLLING_CONFIG: POLLING_CONFIG,
  pollingInterval: null,
  _pollTimeoutId: null,
  _pollAbortController: null,
  _visibilityListenerBound: false,
  _lastProgressPct: 0,
  _consecutivePollErrors: 0,
  _statusPulseTween: null,
  _activeImportTween: null,
  _activeJobId: null,
  _isTerminal: false,
  _pollSequence: 0,
  _pollInFlight: false,
  _ingestStartTime: null,
  _ingestStartProcessed: 0,

  formatFileSize(bytes) {
    if (bytes === 0) return '0 B';
    if (bytes >= 1024 * 1024) {
      return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
    }
    return `${(bytes / 1024).toFixed(1)} KB`;
  },

  init() {
    this.bindEvents();
    this.setupVisibilityListener();
    this.restoreActiveJob();
  },

  setupVisibilityListener() {
    if (
      this._visibilityListenerBound ||
      typeof document === 'undefined' ||
      typeof document.addEventListener !== 'function'
    ) {
      return;
    }
    document.addEventListener('visibilitychange', () => {
      if (!this._activeJobId || this._isTerminal) return;
      if (document.visibilityState === 'visible') {
        // Tab foregrounded: trigger an immediate status poll if no poll is in flight
        if (!this._pollInFlight && this._pollTimeoutId) {
          clearTimeout(this._pollTimeoutId);
          this._pollTimeoutId = null;
          this.executePollTick(this._activeJobId);
        }
      }
    });
    this._visibilityListenerBound = true;
  },

  restoreActiveJob() {
    try {
      const savedJobId = localStorage.getItem('sentiment_pulse_active_import_job');
      if (savedJobId) {
        const statusCard = document.getElementById('import-status-card');
        if (statusCard) statusCard.classList.remove('hidden');
        this.pollJob(savedJobId);
      }
    } catch {}
  },

  cleanup() {
    if (this._pollTimeoutId) {
      clearTimeout(this._pollTimeoutId);
      this._pollTimeoutId = null;
    }
    if (this.pollingInterval) {
      clearInterval(this.pollingInterval);
      this.pollingInterval = null;
    }
    if (this._pollAbortController) {
      try {
        this._pollAbortController.abort();
      } catch {}
      this._pollAbortController = null;
    }
    this._consecutivePollErrors = 0;
    this._isTerminal = false;
    this._activeJobId = null;
    this._pollInFlight = false;
    this._lastProgressPct = 0;
    this._ingestStartTime = null;
    this._ingestStartProcessed = 0;

    try {
      localStorage.removeItem('sentiment_pulse_active_import_job');
    } catch {}

    this.stopPollingPulse();

    if (this._activeImportTween) {
      try {
        if (typeof this._activeImportTween.kill === 'function') this._activeImportTween.kill();
      } catch {}
      this._activeImportTween = null;
    }

    if (window.AnimationManager) {
      window.AnimationManager.killViewAnimations('import-export');
    }
  },

  startPollingPulse() {
    const pulseDot = document.getElementById('import-pulse-dot');
    if (!pulseDot) return;

    pulseDot.classList.remove('hidden');

    const isReduced = window.AnimationManager && window.AnimationManager.isReducedMotion();
    if (isReduced) {
      pulseDot.classList.remove('animate-pulse');
      return;
    }

    pulseDot.classList.add('animate-pulse');
  },

  stopPollingPulse() {
    const pulseDot = document.getElementById('import-pulse-dot');
    if (pulseDot) {
      pulseDot.classList.add('hidden');
      pulseDot.classList.remove('animate-pulse');
    }
    if (this._statusPulseTween) {
      try {
        if (typeof this._statusPulseTween.kill === 'function') this._statusPulseTween.kill();
      } catch {}
      this._statusPulseTween = null;
    }
  },

  bindEvents() {
    const dropZone = document.getElementById('import-dropzone');
    const fileInput = document.getElementById('import-file-input');
    const selectFileBtn = document.getElementById('import-select-btn');
    const uploadBtn = document.getElementById('import-submit-btn');
    const cancelBtn = document.getElementById('import-cancel-btn');
    const exportBtn = document.getElementById('import-export-btn');
    const sampleTemplateBtn = document.getElementById('import-sample-template-btn');

    if (selectFileBtn && fileInput) {
      selectFileBtn.addEventListener('click', () => fileInput.click());
    }

    if (sampleTemplateBtn) {
      sampleTemplateBtn.addEventListener('click', () => this.downloadSampleTemplate());
    }

    if (fileInput) {
      fileInput.addEventListener('change', (e) => {
        if (e.target.files && e.target.files[0]) {
          this.handleFileSelected(e.target.files[0]);
        }
      });
    }

    if (dropZone) {
      ['dragenter', 'dragover'].forEach((eventName) => {
        dropZone.addEventListener(eventName, (e) => {
          e.preventDefault();
          e.stopPropagation();
          dropZone.classList.add('drag-active');
        });
      });

      ['dragleave', 'drop'].forEach((eventName) => {
        dropZone.addEventListener(eventName, (e) => {
          if (e && typeof e.preventDefault === 'function') e.preventDefault();
          if (e && typeof e.stopPropagation === 'function') e.stopPropagation();
          dropZone.classList.remove('drag-active');
        });
      });

      dropZone.addEventListener('drop', (e) => {
        if (e.dataTransfer && e.dataTransfer.files && e.dataTransfer.files[0]) {
          this.handleFileSelected(e.dataTransfer.files[0]);
        }
      });
    }

    if (cancelBtn) {
      cancelBtn.addEventListener('click', () => this.clearSelection());
    }

    if (uploadBtn) {
      uploadBtn.addEventListener('click', () => this.startUpload());
    }

    if (exportBtn) {
      exportBtn.addEventListener('click', async () => {
        try {
          window.showToast('Generating sanitized CSV export...', 'info');
          await window.api.exportCSV();
          window.showToast('CSV export downloaded successfully!', 'success');
        } catch (err) {
          window.showToast(err.message || 'Export failed.', 'error');
        }
      });
    }
  },

  handleFileSelected(file) {
    const dropZone = document.getElementById('import-dropzone');

    if (!file) {
      this.clearSelection();
      return;
    }

    // Verify CSV extension and acceptable MIME types (if provided by browser)
    const hasCsvExt = file.name && file.name.toLowerCase().endsWith('.csv');
    const validMimes = [
      'text/csv',
      'application/vnd.ms-excel',
      'text/plain',
      'application/csv',
      'text/x-csv',
      '',
    ];
    const hasValidMime = !file.type || validMimes.includes(file.type.toLowerCase());

    if (!hasCsvExt || !hasValidMime) {
      this.clearSelection();
      if (dropZone && window.AnimationManager && !window.AnimationManager.isReducedMotion()) {
        window.AnimationManager.play('errorShake', dropZone, {}, 'import-export');
      }
      window.showToast('Only CSV (.csv) files are supported.', 'error');
      return;
    }

    if (file.size === 0) {
      this.clearSelection();
      if (dropZone && window.AnimationManager && !window.AnimationManager.isReducedMotion()) {
        window.AnimationManager.play('errorShake', dropZone, {}, 'import-export');
      }
      window.showToast('Uploaded CSV file is empty.', 'error');
      return;
    }

    if (file.size > MAX_UPLOAD_BYTES) {
      this.clearSelection();
      if (dropZone && window.AnimationManager && !window.AnimationManager.isReducedMotion()) {
        window.AnimationManager.play('errorShake', dropZone, {}, 'import-export');
      }
      const actualSizeStr = this.formatFileSize(file.size);
      window.showToast(
        `File size (${actualSizeStr}) exceeds maximum allowed limit of 100 MB.`,
        'error'
      );
      return;
    }

    this.selectedFile = file;

    const fileMeta = document.getElementById('import-file-meta');
    const fileName = document.getElementById('import-file-name');
    const fileSize = document.getElementById('import-file-size');
    const promptZone = document.getElementById('import-prompt-zone');
    const uploadBtn = document.getElementById('import-submit-btn');

    if (fileName) fileName.textContent = file.name;
    if (fileSize) fileSize.textContent = this.formatFileSize(file.size);
    if (fileMeta) fileMeta.classList.remove('hidden');
    if (promptZone) promptZone.classList.add('hidden');
    if (uploadBtn) uploadBtn.disabled = false;
  },

  clearSelection() {
    this.selectedFile = null;
    this.isUploading = false;
    this._isTerminal = false;
    this._activeJobId = null;
    this._pollInFlight = false;
    const fileInput = document.getElementById('import-file-input');
    const fileMeta = document.getElementById('import-file-meta');
    const promptZone = document.getElementById('import-prompt-zone');
    const uploadBtn = document.getElementById('import-submit-btn');

    if (fileInput) fileInput.value = '';
    if (fileMeta) fileMeta.classList.add('hidden');
    if (promptZone) promptZone.classList.remove('hidden');
    if (uploadBtn) uploadBtn.disabled = true;
  },

  categorizeError(err) {
    const status = err.status || 0;
    const msg = (err.message || '').toLowerCase();
    const detail = (
      typeof err.detail === 'string' ? err.detail : JSON.stringify(err.detail || '')
    ).toLowerCase();
    const errorType = err.errorType || '';

    // 1. File size limit
    if (status === 413 || msg.includes('maximum allowed size') || msg.includes('100 mb')) {
      return {
        title: 'File Too Large',
        message: err.message || 'The uploaded file exceeds the 100 MB maximum size limit.',
        statusText: '413 Payload Too Large',
      };
    }

    // 2. Cloudflare challenge / edge security
    if (err.isCloudflareChallenge || msg.includes('cloudflare') || detail.includes('cf_chl_opt')) {
      return {
        title: 'Edge Security Challenge',
        message:
          err.message || 'Cloudflare security challenge encountered. Edge verification required.',
        statusText: '403 Forbidden',
      };
    }

    // 3. Unsupported encoding
    if (msg.includes('encoding') || detail.includes('encoding')) {
      return {
        title: 'Unsupported CSV Encoding',
        message:
          'Unsupported character encoding. Please save your CSV as standard UTF-8 or Latin-1.',
        statusText: status ? `${status} Bad Request` : 'Encoding Error',
      };
    }

    // 4. Missing required text column
    if (
      msg.includes("must contain a 'text'") ||
      msg.includes('missing text column') ||
      detail.includes("must contain a 'text'")
    ) {
      return {
        title: 'Missing Required Text Column',
        message:
          'CSV header must include a required feedback text column (e.g. "text", "review", or "feedback").',
        statusText: status ? `${status} Bad Request` : 'Header Validation Error',
      };
    }

    // 5. Empty CSV
    if (msg.includes('empty') || detail.includes('empty')) {
      return {
        title: 'Empty CSV File',
        message: 'The uploaded CSV file is empty or contains no data records.',
        statusText: status ? `${status} Bad Request` : 'Validation Error',
      };
    }

    // 6. Database Schema Error
    if (
      errorType === 'database_schema_error' ||
      msg.includes('database schema') ||
      (msg.includes('column') && msg.includes('does not exist')) ||
      detail.includes('database schema') ||
      detail.includes('undefinedcolumn')
    ) {
      return {
        title: 'Database Schema Error',
        message:
          'Database schema mismatch: required table or column does not exist. Database migrations must be applied.',
        statusText: `${status || 500} Database Schema Error`,
      };
    }

    // 7. Database Connection Failure
    if (
      errorType === 'database_connection_error' ||
      status === 503 ||
      msg.includes('database connection') ||
      detail.includes('database connection') ||
      msg.includes('operationalerror')
    ) {
      return {
        title: 'Database Connection Failure',
        message:
          'Unable to connect to the database. The database service may be restarting or unavailable.',
        statusText: `${status || 503} Service Unavailable`,
      };
    }

    // 8. Request Timeout
    if (
      msg.includes('timeout') ||
      msg.includes('timed out') ||
      err.name === 'AbortError' ||
      err.name === 'TimeoutError'
    ) {
      return {
        title: 'Request Timeout',
        message:
          err.message ||
          'Upload request timed out before the server completed receipt. Please retry.',
        statusText: '408 Request Timeout',
      };
    }

    // 9. Rate Limiting / Concurrent Limits
    if (status === 429 || msg.includes('concurrent import limit')) {
      return {
        title: 'Concurrent Import Limit Reached',
        message:
          'Too many batch imports are currently running. Please wait for active jobs to complete.',
        statusText: '429 Too Many Requests',
      };
    }

    // 10. Import Job Creation Failure
    if (
      errorType === 'import_job_creation_failed' ||
      msg.includes('failed to initialize import job')
    ) {
      return {
        title: 'Import Job Creation Failed',
        message: 'Server failed to initialize batch import job record.',
        statusText: `${status || 500} Internal Server Error`,
      };
    }

    // 11. General Server-Side Processing Failure
    return {
      title: 'Server-Side Processing Error',
      message:
        err.message ||
        'An unexpected internal error occurred on the server while processing the upload.',
      statusText: `${status || 500} Internal Server Error`,
    };
  },

  async startUpload() {
    if (!this.selectedFile || this.isUploading) return;

    if (this.selectedFile.size > MAX_UPLOAD_BYTES) {
      window.showToast('File size exceeds maximum allowed limit of 100 MB.', 'error');
      this.clearSelection();
      return;
    }

    this.isUploading = true;
    this._isTerminal = false;
    this._activeJobId = null;
    this._pollInFlight = false;
    this._pollSequence = 0;

    const uploadBtn = document.getElementById('import-submit-btn');
    const statusCard = document.getElementById('import-status-card');
    const progressText = document.getElementById('import-progress-status');
    const progressBar = document.getElementById('import-progress-bar');
    const progressPct = document.getElementById('import-progress-pct');
    const resultSummary = document.getElementById('import-result-summary');

    this._lastProgressPct = 20;

    if (uploadBtn) {
      uploadBtn.disabled = true;
      if (window.AnimationManager && !window.AnimationManager.isReducedMotion()) {
        window.AnimationManager.play('progressMorph', uploadBtn, {}, 'import-export');
      }
    }

    if (statusCard) statusCard.classList.remove('hidden');
    if (progressText) progressText.textContent = 'Uploading CSV file to server...';
    if (progressBar) {
      progressBar.style.width = '20%';
      progressBar.classList.remove('bg-rose-500', 'animate-pulse');
      progressBar.classList.add('bg-indigo-600');
    }
    if (progressPct) {
      progressPct.textContent = '20%';
      progressPct.classList.remove('hidden', 'text-rose-500');
      progressPct.classList.add('text-slate-500');
    }
    if (resultSummary) resultSummary.classList.add('hidden');

    this.startPollingPulse();

    try {
      const job = await window.api.uploadCSV(this.selectedFile);
      this.clearSelection();
      this._activeJobId = job.id;

      if (job.status === 'COMPLETED') {
        this.renderJobSuccess(job);
      } else if (job.status === 'FAILED') {
        this.renderJobFailure(job);
      } else {
        // PENDING or PROCESSING: transition UI to accepted state and poll until terminal state
        if (progressText) {
          progressText.textContent = `Upload accepted [Job ID: ${job.id}]. Validating and analyzing batch...`;
        }
        if (progressBar) progressBar.style.width = '25%';
        if (progressPct) progressPct.textContent = '25%';
        this._lastProgressPct = 25;
        this.pollJob(job.id);
      }
    } catch (err) {
      console.error('Import upload error:', err);
      this.isUploading = false;
      this._isTerminal = true;
      this.stopPollingPulse();

      const cat = this.categorizeError(err);
      const rawMsg = err.message || '';
      let errorDisplay = rawMsg || `Upload error [${cat.statusText}]: ${cat.message}`;
      if (err.jobId) {
        errorDisplay += ` (Job ID: ${err.jobId})`;
      } else if (err.requestId) {
        errorDisplay += ` (Request ID: ${err.requestId.slice(0, 8)})`;
      }

      if (progressText) progressText.textContent = errorDisplay;
      if (progressBar) {
        progressBar.style.width = '0%';
        progressBar.classList.remove('animate-pulse', 'bg-indigo-600');
        progressBar.classList.add('bg-rose-500');
      }
      if (progressPct) {
        progressPct.textContent = '0%';
        progressPct.classList.remove('text-slate-500');
        progressPct.classList.add('text-rose-500');
      }

      if (statusCard && window.AnimationManager && !window.AnimationManager.isReducedMotion()) {
        window.AnimationManager.play('errorShake', statusCard, {}, 'import-export');
      }

      window.showToast(`${cat.title}: ${cat.message}`, 'error');
      if (uploadBtn) uploadBtn.disabled = false;
    }
  },

  computeNextIntervalMs(consecutiveErrors, retryAfterSec) {
    // If the edge server supplied Retry-After, honor it directly
    if (retryAfterSec && typeof retryAfterSec === 'number' && retryAfterSec > 0) {
      return Math.max(retryAfterSec * 1000, 2000);
    }

    // When the browser tab is backgrounded, reduce polling frequency to save battery and edge quota
    if (typeof document !== 'undefined' && document.visibilityState === 'hidden') {
      return this.POLLING_CONFIG.backgroundIntervalMs;
    }

    // Progressive exponential backoff schedule: 2s -> 4s -> 8s -> 15s max
    let baseMs = this.POLLING_CONFIG.initialIntervalMs;
    if (consecutiveErrors > 0) {
      const idx = Math.min(consecutiveErrors, this.POLLING_CONFIG.retryScheduleMs.length - 1);
      baseMs = this.POLLING_CONFIG.retryScheduleMs[idx];
    }

    // Apply random retry jitter (±15%) to avoid client synchronization / thundering herds
    const jitter = (Math.random() * 2 - 1) * this.POLLING_CONFIG.jitterRatio;
    const intervalWithJitter = Math.round(baseMs * (1 + jitter));
    return Math.min(Math.max(intervalWithJitter, 1000), this.POLLING_CONFIG.maxIntervalMs);
  },

  scheduleNextPoll(jobId, delayMs) {
    if (this._isTerminal || this._activeJobId !== jobId) return;
    if (this._pollTimeoutId) {
      clearTimeout(this._pollTimeoutId);
      this._pollTimeoutId = null;
    }
    this._pollTimeoutId = setTimeout(() => {
      this._pollTimeoutId = null;
      this.executePollTick(jobId);
    }, delayMs);
    this.pollingInterval = this._pollTimeoutId;
  },

  pollJob(jobId) {
    try {
      localStorage.setItem('sentiment_pulse_active_import_job', jobId);
    } catch {}

    if (this._pollTimeoutId) {
      clearTimeout(this._pollTimeoutId);
      this._pollTimeoutId = null;
    }
    if (this.pollingInterval) {
      clearInterval(this.pollingInterval);
      this.pollingInterval = null;
    }
    if (this._pollAbortController) {
      try {
        this._pollAbortController.abort();
      } catch {}
      this._pollAbortController = null;
    }

    this._consecutivePollErrors = 0;
    this._activeJobId = jobId;
    this._isTerminal = false;
    this._pollInFlight = false;

    // Trigger first poll using configured initial interval
    this.scheduleNextPoll(jobId, this.POLLING_CONFIG.initialIntervalMs);
  },

  async executePollTick(jobId) {
    // 1. Guard against polling when job has reached terminal state or switched jobs
    if (this._isTerminal || this._activeJobId !== jobId) {
      return;
    }

    // 2. Prevent overlapping async fetch calls
    if (this._pollInFlight) {
      return;
    }

    this._pollInFlight = true;
    const seq = ++this._pollSequence;

    const controller = typeof AbortController !== 'undefined' ? new AbortController() : null;
    this._pollAbortController = controller;

    const progressText = document.getElementById('import-progress-status');
    const progressBar = document.getElementById('import-progress-bar');
    const progressPct = document.getElementById('import-progress-pct');
    const uploadBtn = document.getElementById('import-submit-btn');

    try {
      const job = await window.api.getImportJob(jobId, {
        signal: controller ? controller.signal : undefined,
        timeoutMs: this.POLLING_CONFIG.pollTimeoutMs,
      });

      this._pollInFlight = false;
      this._pollAbortController = null;
      this._consecutivePollErrors = 0;

      // 3. Drop response if terminal state was reached or new job began while in-flight
      if (this._isTerminal || this._activeJobId !== jobId || seq < this._pollSequence) {
        return;
      }

      // 4. Authoritative check: handle terminal states immediately before any progress rendering
      if (job.status === 'COMPLETED' || job.status === 'FAILED' || job.status === 'CANCELLED') {
        this._isTerminal = true;
        if (this._pollTimeoutId) {
          clearTimeout(this._pollTimeoutId);
          this._pollTimeoutId = null;
        }
        if (this.pollingInterval) {
          clearInterval(this.pollingInterval);
          this.pollingInterval = null;
        }
        this.stopPollingPulse();

        if (job.status === 'COMPLETED') {
          this.renderJobSuccess(job);
        } else if (job.status === 'CANCELLED') {
          this.renderJobCancelled(job);
        } else {
          this.renderJobFailure(job);
        }
        return;
      }

      // 5. Non-terminal states: PENDING
      if (job.status === 'PENDING') {
        if (progressText) {
          progressText.textContent = 'Import queued in background. Waiting for worker...';
        }
        if (progressBar) {
          progressBar.style.width = '15%';
          progressBar.classList.remove('animate-pulse');
        }
        if (progressPct) progressPct.textContent = '15%';
        this.scheduleNextPoll(jobId, this.computeNextIntervalMs(0));
        return;
      }

      // 6. Non-terminal states: PROCESSING
      const processed = job.processed_rows || 0;
      const total = job.total_rows || 0;
      const successful =
        job.successful_rows !== undefined ? job.successful_rows : job.inserted_rows || 0;
      const failed = job.failed_rows || 0;
      const stage = job.stage || 'PROCESSING';

      if (stage === 'FINALIZING') {
        this._lastProgressPct = Math.max(98, this._lastProgressPct);
        if (progressBar) {
          progressBar.style.width = '98%';
          progressBar.classList.add('animate-pulse');
        }
        if (progressPct) progressPct.textContent = '98%';
        if (progressText) {
          progressText.textContent = 'Finalizing batch and committing records to database...';
        }
        this.scheduleNextPoll(jobId, this.computeNextIntervalMs(0));
        return;
      }

      if (stage === 'PARSING') {
        this._lastProgressPct = Math.max(20, this._lastProgressPct);
        if (progressBar) {
          progressBar.style.width = '20%';
          progressBar.classList.add('animate-pulse');
        }
        if (progressPct) progressPct.textContent = '20%';
        if (progressText) {
          progressText.textContent = 'Parsing CSV file structure and detecting delimiters...';
        }
        this.scheduleNextPoll(jobId, this.computeNextIntervalMs(0));
        return;
      }

      if (stage === 'VALIDATING') {
        this._lastProgressPct = Math.max(25, this._lastProgressPct);
        if (progressBar) {
          progressBar.style.width = '25%';
          progressBar.classList.add('animate-pulse');
        }
        if (progressPct) progressPct.textContent = '25%';
        if (progressText) {
          progressText.textContent = 'Validating columns and calculating record boundaries...';
        }
        this.scheduleNextPoll(jobId, this.computeNextIntervalMs(0));
        return;
      }

      if (processed > 0 && total > 0) {
        const ratio = Math.min(processed / total, 1.0);
        const pct = Math.min(25 + Math.round(ratio * 70), 96);
        const displayPct = Math.max(pct, this._lastProgressPct);
        this._lastProgressPct = displayPct;

        if (!this._ingestStartTime) {
          this._ingestStartTime = Date.now();
          this._ingestStartProcessed = processed;
        }
        let etaStr = '';
        const elapsedSec = (Date.now() - this._ingestStartTime) / 1000;
        const processedDelta = processed - this._ingestStartProcessed;
        if (elapsedSec > 2 && processedDelta > 0 && total > processed) {
          const rps = processedDelta / elapsedSec;
          const remaining = total - processed;
          const etaSec = Math.round(remaining / Math.max(rps, 0.1));
          if (etaSec > 60) {
            etaStr = ` • ETA: ~${Math.floor(etaSec / 60)}m ${etaSec % 60}s`;
          } else if (etaSec > 0) {
            etaStr = ` • ETA: ~${etaSec}s`;
          }
        }

        if (progressBar) {
          progressBar.style.width = `${displayPct}%`;
          progressBar.classList.remove('animate-pulse');
        }
        if (progressPct) progressPct.textContent = `${displayPct}%`;
        if (progressText) {
          progressText.textContent = `Processing ${processed.toLocaleString()} / ${total.toLocaleString()} records${etaStr} (Inserted: ${successful.toLocaleString()} | Rejected: ${failed.toLocaleString()})`;
        }
      } else if (total > 0) {
        const displayPct = Math.max(25, this._lastProgressPct);
        this._lastProgressPct = displayPct;
        if (progressBar) {
          progressBar.style.width = `${displayPct}%`;
          progressBar.classList.add('animate-pulse');
        }
        if (progressPct) progressPct.textContent = `${displayPct}%`;
        if (progressText) {
          progressText.textContent = `Preparing ingestion of ${total.toLocaleString()} records...`;
        }
      } else {
        const displayPct = Math.max(20, this._lastProgressPct);
        this._lastProgressPct = displayPct;
        if (progressBar) {
          progressBar.style.width = `${displayPct}%`;
          progressBar.classList.add('animate-pulse');
        }
        if (progressPct) progressPct.textContent = `${displayPct}%`;
        if (progressText) {
          progressText.textContent = 'Analyzing and processing batch in background...';
        }
      }

      // Schedule next poll using adaptive schedule
      this.scheduleNextPoll(jobId, this.computeNextIntervalMs(0));
    } catch (err) {
      this._pollInFlight = false;
      this._pollAbortController = null;
      if (this._isTerminal || this._activeJobId !== jobId) {
        return;
      }

      this._consecutivePollErrors += 1;
      const retryAfterSec = err.retryAfter || null;
      const nextDelayMs = this.computeNextIntervalMs(this._consecutivePollErrors, retryAfterSec);
      const nextSec = Math.round(nextDelayMs / 1000);

      // Determine human-readable diagnosis without leaking raw HTML
      let issueDesc = 'Temporary connectivity interruption';
      if (err.isCloudflareChallenge) {
        issueDesc = 'Edge security challenge encountered';
      } else if (err.status === 429) {
        issueDesc = 'Rate limit backoff active';
      } else if (err.status === 502 || err.status === 503) {
        issueDesc = 'Server restarting or temporarily unavailable';
      }

      // Resilient error threshold: do NOT prematurely abort running background jobs
      if (this._consecutivePollErrors < this.POLLING_CONFIG.maxConsecutiveErrors) {
        if (progressText) {
          progressText.textContent = `Progress monitoring temporarily unavailable (${issueDesc}). Ingestion continues on server. Retrying in ${nextSec}s (${this._consecutivePollErrors}/${this.POLLING_CONFIG.maxConsecutiveErrors})...`;
        }
        // Preserve existing progress bar and percentage intact!
        this.scheduleNextPoll(jobId, nextDelayMs);
      } else {
        // Paused after repeated failures: do NOT mark the job as FAILED
        if (progressText) {
          progressText.textContent =
            'Progress monitoring paused after multiple network interruptions. The import may still be processing on the server.';
        }
        this.stopPollingPulse();
        if (uploadBtn) uploadBtn.disabled = false;
        window.showToast(
          'Progress monitoring paused due to network interruptions. Import continues on server.',
          'info'
        );
      }
    }
  },

  downloadSampleTemplate() {
    const csvContent =
      'text,category,source,rating\n' +
      '"The sentiment analysis platform is remarkably fast and intuitive.",UX,Web,5\n' +
      '"Encountered intermittent timeout during checkout payment processing.",Billing,Mobile App,2\n' +
      '"Customer service representative resolved my support ticket promptly.",Support,Email,4\n';

    const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'sample_feedback_template.csv';
    document.body.appendChild(a);
    a.click();
    window.URL.revokeObjectURL(url);
    document.body.removeChild(a);
    if (typeof window.showToast === 'function') {
      window.showToast('Sample CSV template downloaded!', 'success');
    }
  },

  async renderJobSuccess(job) {
    this._isTerminal = true;
    try {
      localStorage.removeItem('sentiment_pulse_active_import_job');
    } catch {}
    if (this.pollingInterval) {
      clearInterval(this.pollingInterval);
      this.pollingInterval = null;
    }
    this.stopPollingPulse();
    this.isUploading = false;
    this._lastProgressPct = 0;
    this._ingestStartTime = null;
    this._ingestStartProcessed = 0;

    const progressText = document.getElementById('import-progress-status');
    const progressBar = document.getElementById('import-progress-bar');
    const progressPct = document.getElementById('import-progress-pct');
    const resultSummary = document.getElementById('import-result-summary');
    const summaryHeading = document.getElementById('import-summary-heading');
    const successIcon = document.getElementById('import-success-icon');
    const totalCount = document.getElementById('import-stat-total');
    const successCount = document.getElementById('import-stat-success');
    const failCount = document.getElementById('import-stat-failed');
    const errorBox = document.getElementById('import-error-box');
    const uploadBtn = document.getElementById('import-submit-btn');

    if (progressBar) {
      progressBar.style.width = '100%';
      progressBar.classList.remove('animate-pulse', 'bg-rose-500');
      progressBar.classList.add('bg-indigo-600');
    }
    if (progressPct) {
      progressPct.textContent = '100%';
      progressPct.classList.remove('text-rose-500');
      progressPct.classList.add('text-slate-500');
    }
    if (progressText) progressText.textContent = 'Import Job Completed Successfully!';
    if (summaryHeading) summaryHeading.textContent = 'Batch Ingestion Results';
    if (totalCount) totalCount.textContent = job.total_rows;
    if (successCount) successCount.textContent = job.successful_rows;
    if (failCount) failCount.textContent = job.failed_rows;

    if (errorBox) {
      if (job.error_summary) {
        try {
          const errors = JSON.parse(job.error_summary);
          errorBox.innerHTML = errors.map((e) => `<li>${window.escapeHtml(e)}</li>`).join('');
          errorBox.classList.remove('hidden');
        } catch {
          errorBox.classList.add('hidden');
        }
      } else {
        errorBox.classList.add('hidden');
      }
    }

    if (resultSummary) {
      resultSummary.classList.remove('hidden');
      const isReduced = window.AnimationManager && window.AnimationManager.isReducedMotion();

      if (!isReduced && window.AnimationManager) {
        // Celebratory summary card entrance
        window.AnimationManager.play('celebrationBurst', resultSummary, {}, 'import-export');

        if (successIcon) {
          window.AnimationManager.play('successPop', successIcon, {}, 'import-export');
        }

        // Bounded, one-shot confetti celebration (lazy-loaded via VendorLoader)
        window.AnimationManager.celebrate();
      }
    }

    if (uploadBtn) uploadBtn.disabled = true;
    this.isUploading = false;
    window.showToast(`Imported ${job.successful_rows} feedback records!`, 'success');

    // Notify Explorer and Overview to refresh
    window.dispatchEvent(new CustomEvent('feedback-created'));
  },

  renderJobFailure(job) {
    this._isTerminal = true;
    try {
      localStorage.removeItem('sentiment_pulse_active_import_job');
    } catch {}
    if (this.pollingInterval) {
      clearInterval(this.pollingInterval);
      this.pollingInterval = null;
    }
    this.stopPollingPulse();
    this.isUploading = false;
    this._lastProgressPct = 0;
    this._ingestStartTime = null;
    this._ingestStartProcessed = 0;

    const progressText = document.getElementById('import-progress-status');
    const progressBar = document.getElementById('import-progress-bar');
    const progressPct = document.getElementById('import-progress-pct');
    const resultSummary = document.getElementById('import-result-summary');
    const summaryHeading = document.getElementById('import-summary-heading');
    const errorBox = document.getElementById('import-error-box');
    const statusCard = document.getElementById('import-status-card');
    const uploadBtn = document.getElementById('import-submit-btn');
    const totalCount = document.getElementById('import-stat-total');
    const successCount = document.getElementById('import-stat-success');
    const failCount = document.getElementById('import-stat-failed');

    if (progressBar) {
      progressBar.classList.remove('animate-pulse');
      progressBar.style.width = '100%';
      progressBar.classList.remove('bg-indigo-600');
      progressBar.classList.add('bg-rose-500');
    }
    if (progressPct) {
      progressPct.textContent = 'Failed';
      progressPct.classList.remove('hidden', 'text-slate-500');
      progressPct.classList.add('text-rose-500');
    }
    if (progressText) {
      progressText.textContent = 'Import Job Failed.';
    }

    const totalRows = job.total_rows !== undefined && job.total_rows !== null ? job.total_rows : 0;
    const successfulRows =
      job.successful_rows !== undefined && job.successful_rows !== null ? job.successful_rows : 0;
    const failedRows =
      job.failed_rows !== undefined && job.failed_rows !== null ? job.failed_rows : 0;

    if (totalCount) totalCount.textContent = totalRows;
    if (successCount) successCount.textContent = successfulRows;
    if (failCount) failCount.textContent = failedRows;

    let errors = ['The server was unable to parse or process the CSV file.'];
    if (job.error_summary) {
      try {
        const parsed = JSON.parse(job.error_summary);
        if (Array.isArray(parsed) && parsed.length > 0) {
          errors = parsed;
        }
      } catch {}
    }

    if (summaryHeading) {
      summaryHeading.textContent =
        successfulRows === 0
          ? 'File Validation Failed (Rejected Before Ingestion)'
          : 'Batch Ingestion Failed';
    }

    if (errorBox) {
      errorBox.innerHTML = errors.map((e) => `<li>${window.escapeHtml(e)}</li>`).join('');
      errorBox.classList.remove('hidden');
    }

    if (resultSummary) resultSummary.classList.remove('hidden');

    const isReduced = window.AnimationManager && window.AnimationManager.isReducedMotion();
    if (!isReduced && window.AnimationManager) {
      const shakeTarget = errorBox || statusCard;
      if (shakeTarget) {
        window.AnimationManager.play('errorShake', shakeTarget, {}, 'import-export');
      }
    }

    if (uploadBtn) uploadBtn.disabled = false;
    const primaryError = errors[0] || 'Import job failed.';
    window.showToast(primaryError, 'error');
  },

  renderJobCancelled(_job) {
    this._isTerminal = true;
    try {
      localStorage.removeItem('sentiment_pulse_active_import_job');
    } catch {}
    if (this._pollTimeoutId) {
      clearTimeout(this._pollTimeoutId);
      this._pollTimeoutId = null;
    }
    if (this.pollingInterval) {
      clearInterval(this.pollingInterval);
      this.pollingInterval = null;
    }
    this.stopPollingPulse();
    this.isUploading = false;

    const progressText = document.getElementById('import-progress-status');
    const progressBar = document.getElementById('import-progress-bar');
    const progressPct = document.getElementById('import-progress-pct');
    const uploadBtn = document.getElementById('import-submit-btn');

    if (progressBar) {
      progressBar.classList.remove('animate-pulse', 'bg-indigo-600', 'bg-rose-500');
      progressBar.classList.add('bg-amber-500');
      progressBar.style.width = '100%';
    }
    if (progressPct) {
      progressPct.textContent = 'Cancelled';
      progressPct.classList.remove('text-slate-500', 'text-rose-500');
      progressPct.classList.add('text-amber-500');
    }
    if (progressText) {
      progressText.textContent = 'Import Job Cancelled.';
    }
    if (uploadBtn) uploadBtn.disabled = false;
    window.showToast('Import job was cancelled.', 'info');
  },
};

if (typeof window !== 'undefined') {
  window.MAX_UPLOAD_BYTES = MAX_UPLOAD_BYTES;
  window.MAX_IMPORT_ROWS = MAX_IMPORT_ROWS;
  window.MAX_IMPORT_ROWS_FORMATTED = MAX_IMPORT_ROWS_FORMATTED;
  window.ImportsController = ImportsController;
}
if (typeof globalThis !== 'undefined') {
  globalThis.MAX_UPLOAD_BYTES = MAX_UPLOAD_BYTES;
  globalThis.MAX_IMPORT_ROWS = MAX_IMPORT_ROWS;
  globalThis.MAX_IMPORT_ROWS_FORMATTED = MAX_IMPORT_ROWS_FORMATTED;
  globalThis.ImportsController = ImportsController;
}
