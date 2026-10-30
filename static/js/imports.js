/**
 * Sentiment Pulse - Import & Export Controller (imports.js)
 * Handles CSV drag-and-drop uploads, validation, chunked batch execution,
 * job status polling, and sanitized CSV exports.
 */

const MAX_UPLOAD_BYTES = 100 * 1024 * 1024; // 100 MB (104,857,600 bytes)

const ImportsController = {
  selectedFile: null,
  isUploading: false,
  MAX_UPLOAD_BYTES: MAX_UPLOAD_BYTES,
  pollingInterval: null,
  _lastProgressPct: 0,
  _consecutivePollErrors: 0,
  _statusPulseTween: null,
  _activeImportTween: null,
  _activeJobId: null,
  _isTerminal: false,
  _pollSequence: 0,
  _pollInFlight: false,

  formatFileSize(bytes) {
    if (bytes === 0) return '0 B';
    if (bytes >= 1024 * 1024) {
      return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
    }
    return `${(bytes / 1024).toFixed(1)} KB`;
  },

  init() {
    this.bindEvents();
  },

  cleanup() {
    if (this.pollingInterval) {
      clearInterval(this.pollingInterval);
      this.pollingInterval = null;
    }
    this._consecutivePollErrors = 0;
    this._isTerminal = false;
    this._activeJobId = null;
    this._pollInFlight = false;
    this._lastProgressPct = 0;

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
          progressText.textContent = 'Upload accepted. Validating and analyzing batch...';
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

      const errorMessage =
        err.status === 413
          ? err.message || 'File exceeds maximum allowed size (100 MB).'
          : err.message || 'CSV upload failed.';

      if (progressText) progressText.textContent = `Upload error: ${errorMessage}`;
      if (progressBar) {
        progressBar.style.width = '0%';
        progressBar.classList.remove('animate-pulse');
      }
      if (progressPct) progressPct.textContent = '0%';

      if (statusCard && window.AnimationManager && !window.AnimationManager.isReducedMotion()) {
        window.AnimationManager.play('errorShake', statusCard, {}, 'import-export');
      }

      window.showToast(errorMessage, 'error');
      if (uploadBtn) uploadBtn.disabled = false;
    }
  },

  pollJob(jobId) {
    const progressText = document.getElementById('import-progress-status');
    const progressBar = document.getElementById('import-progress-bar');
    const progressPct = document.getElementById('import-progress-pct');
    const uploadBtn = document.getElementById('import-submit-btn');

    if (this.pollingInterval) {
      clearInterval(this.pollingInterval);
      this.pollingInterval = null;
    }
    this._consecutivePollErrors = 0;
    this._activeJobId = jobId;
    this._isTerminal = false;
    this._pollInFlight = false;

    this.pollingInterval = setInterval(async () => {
      // 1. Guard against stale polling when job has reached terminal state or switched jobs
      if (this._isTerminal || this._activeJobId !== jobId) {
        if (this.pollingInterval) {
          clearInterval(this.pollingInterval);
          this.pollingInterval = null;
        }
        return;
      }

      // 2. Prevent overlapping async fetch calls
      if (this._pollInFlight) {
        return;
      }

      this._pollInFlight = true;
      const seq = ++this._pollSequence;

      try {
        const job = await window.api.getImportJob(jobId);
        this._pollInFlight = false;
        this._consecutivePollErrors = 0;

        // 3. Drop response if terminal state was reached or new job began while in-flight
        if (this._isTerminal || this._activeJobId !== jobId || seq < this._pollSequence) {
          return;
        }

        // 4. Authoritative check: handle terminal states immediately before any progress rendering
        if (job.status === 'COMPLETED' || job.status === 'FAILED') {
          this._isTerminal = true;
          if (this.pollingInterval) {
            clearInterval(this.pollingInterval);
            this.pollingInterval = null;
          }
          this.stopPollingPulse();

          if (job.status === 'COMPLETED') {
            this.renderJobSuccess(job);
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
            progressBar.style.width = '25%';
            progressBar.classList.remove('animate-pulse');
          }
          if (progressPct) progressPct.textContent = '25%';
          return;
        }

        // 6. Non-terminal states: PROCESSING
        const processed = job.processed_rows || 0;
        const total = job.total_rows || 0;

        if (processed > 0) {
          const pct = total > 0 ? Math.min(Math.round((processed / total) * 100), 95) : 50;
          const displayPct = Math.max(pct, this._lastProgressPct);
          this._lastProgressPct = displayPct;

          if (progressBar) {
            progressBar.style.width = `${displayPct}%`;
            progressBar.classList.remove('animate-pulse');
          }
          if (progressPct) progressPct.textContent = `${displayPct}%`;
          if (progressText) {
            progressText.textContent = `Processing records (${processed.toLocaleString()} / ${total.toLocaleString()})...`;
          }
        } else {
          // Bounded atomic processing state: records are flushed without partial intermediate commits
          const displayPct = Math.max(50, this._lastProgressPct);
          this._lastProgressPct = displayPct;

          if (progressBar) {
            progressBar.style.width = `${displayPct}%`;
            progressBar.classList.add('animate-pulse');
          }
          if (progressPct) progressPct.textContent = 'Processing...';
          if (progressText) {
            progressText.textContent =
              total > 0
                ? `Ingesting and analyzing batch (${total.toLocaleString()} records in atomic transaction)...`
                : 'Validating and analyzing batch in background...';
          }
        }
      } catch (err) {
        this._pollInFlight = false;
        if (this._isTerminal || this._activeJobId !== jobId) {
          return;
        }

        this._consecutivePollErrors += 1;
        const MAX_CONSECUTIVE_POLL_ERRORS = 5;

        if (this._consecutivePollErrors < MAX_CONSECUTIVE_POLL_ERRORS) {
          if (progressText) {
            progressText.textContent = `Temporary network interruption. Retrying status check (${this._consecutivePollErrors}/${MAX_CONSECUTIVE_POLL_ERRORS})...`;
          }
        } else {
          this._isTerminal = true;
          if (this.pollingInterval) {
            clearInterval(this.pollingInterval);
            this.pollingInterval = null;
          }
          this.stopPollingPulse();
          this.isUploading = false;
          if (uploadBtn) uploadBtn.disabled = false;
          if (progressText) {
            progressText.textContent = `Polling error: ${err.message || 'Network connection lost.'}`;
          }
          window.showToast('Unable to check import progress due to network error.', 'error');
        }
      }
    }, 1500);
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
    if (this.pollingInterval) {
      clearInterval(this.pollingInterval);
      this.pollingInterval = null;
    }
    this.stopPollingPulse();
    this.isUploading = false;
    this._lastProgressPct = 0;

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
    if (this.pollingInterval) {
      clearInterval(this.pollingInterval);
      this.pollingInterval = null;
    }
    this.stopPollingPulse();
    this.isUploading = false;
    this._lastProgressPct = 0;

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
};

if (typeof window !== 'undefined') {
  window.MAX_UPLOAD_BYTES = MAX_UPLOAD_BYTES;
  window.ImportsController = ImportsController;
}
if (typeof globalThis !== 'undefined') {
  globalThis.MAX_UPLOAD_BYTES = MAX_UPLOAD_BYTES;
  globalThis.ImportsController = ImportsController;
}
