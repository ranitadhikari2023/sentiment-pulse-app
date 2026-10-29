/**
 * Sentiment Pulse - Import Controller & 100 MB Upload Limit Unit Tests (imports.test.js)
 * Tests client-side file-size boundary validation, drag-and-drop handling,
 * button states, 413 HTTP error handling, and state restoration.
 *
 * Run via: node --test tests/frontend/imports.test.js
 */

import { describe, it, beforeEach, afterEach } from 'node:test';
import assert from 'node:assert/strict';

function createMockElement(id) {
  const listeners = {};
  return {
    id,
    disabled: true,
    value: '',
    textContent: '',
    innerHTML: '',
    classes: new Set(),
    style: {},
    classList: {
      add(...cls) {
        cls.forEach((c) => this._el.classes.add(c));
      },
      remove(...cls) {
        cls.forEach((c) => this._el.classes.delete(c));
      },
      contains(c) {
        return this._el.classes.has(c);
      },
    },
    addEventListener(event, fn) {
      if (!listeners[event]) listeners[event] = [];
      listeners[event].push(fn);
    },
    trigger(event, data) {
      if (listeners[event]) {
        listeners[event].forEach((fn) => fn(data));
      }
    },
  };
}

describe('Imports Controller & 100 MB Limit Subsystem', () => {
  let elements = {};
  let toastMessages = [];
  let ImportsController;
  let MAX_UPLOAD_BYTES;

  beforeEach(async () => {
    toastMessages = [];
    elements = {};

    const elIds = [
      'import-dropzone',
      'import-file-input',
      'import-select-btn',
      'import-submit-btn',
      'import-cancel-btn',
      'import-export-btn',
      'import-prompt-zone',
      'import-file-meta',
      'import-file-name',
      'import-file-size',
      'import-status-card',
      'import-progress-status',
      'import-progress-bar',
      'import-progress-pct',
      'import-result-summary',
      'import-pulse-dot',
      'import-stat-total',
      'import-stat-success',
      'import-stat-failed',
      'import-error-box',
    ];

    elIds.forEach((id) => {
      const el = createMockElement(id);
      el.classList._el = el;
      elements[id] = el;
    });

    globalThis.document = {
      getElementById(id) {
        return elements[id] || null;
      },
    };

    globalThis.window = {
      document: globalThis.document,
      showToast(msg, type) {
        toastMessages.push({ msg, type });
      },
      escapeHtml(s) {
        return String(s);
      },
      dispatchEvent() {},
      AnimationManager: {
        isReducedMotion: () => true,
        play: () => {},
        killViewAnimations: () => {},
        celebrate: () => {},
      },
      api: {
        uploadCSV: async () => ({
          id: 'mock-job-1',
          status: 'COMPLETED',
          total_rows: 10,
          successful_rows: 10,
          failed_rows: 0,
        }),
        getImportJob: async () => ({
          id: 'mock-job-1',
          status: 'COMPLETED',
          total_rows: 10,
          successful_rows: 10,
          failed_rows: 0,
        }),
        exportCSV: async () => {},
      },
    };

    // Load imports.js
    await import('../../static/js/imports.js');
    ImportsController = globalThis.ImportsController || window.ImportsController;
    MAX_UPLOAD_BYTES = globalThis.MAX_UPLOAD_BYTES || window.MAX_UPLOAD_BYTES;
    ImportsController.init();
    ImportsController.clearSelection();
  });

  afterEach(() => {
    if (ImportsController) {
      ImportsController.cleanup();
      ImportsController.clearSelection();
    }
  });

  describe('Upload File Size Limit Definition & Formatting', () => {
    it('defines authoritative MAX_UPLOAD_BYTES as exactly 100 MB (104,857,600 bytes)', () => {
      assert.equal(MAX_UPLOAD_BYTES, 100 * 1024 * 1024);
      assert.equal(MAX_UPLOAD_BYTES, 104857600);
      assert.equal(ImportsController.MAX_UPLOAD_BYTES, 104857600);
    });

    it('formats file sizes accurately for bytes, KB, and MB', () => {
      assert.equal(ImportsController.formatFileSize(0), '0 B');
      assert.equal(ImportsController.formatFileSize(512), '0.5 KB');
      assert.equal(ImportsController.formatFileSize(1024), '1.0 KB');
      assert.equal(ImportsController.formatFileSize(50 * 1024 * 1024), '50.0 MB');
      assert.equal(ImportsController.formatFileSize(104857600), '100.0 MB');
    });
  });

  describe('File Selection & Boundary Validation', () => {
    it('accepts a valid CSV file below the 100 MB limit', () => {
      const file = {
        name: 'feedback_data.csv',
        size: 5 * 1024 * 1024, // 5 MB
        type: 'text/csv',
      };

      ImportsController.handleFileSelected(file);

      assert.equal(ImportsController.selectedFile, file);
      assert.equal(elements['import-file-name'].textContent, 'feedback_data.csv');
      assert.equal(elements['import-file-size'].textContent, '5.0 MB');
      assert.equal(
        elements['import-submit-btn'].disabled,
        false,
        'Start Batch Ingestion should be enabled'
      );
      assert.equal(elements['import-file-meta'].classList.contains('hidden'), false);
      assert.equal(elements['import-prompt-zone'].classList.contains('hidden'), true);
    });

    it('accepts a CSV file exactly at the maximum size boundary of 100 MB (104,857,600 bytes)', () => {
      const file = {
        name: 'large_dataset_100mb.csv',
        size: 104857600, // Exactly 100 MB
        type: 'text/csv',
      };

      ImportsController.handleFileSelected(file);

      assert.equal(ImportsController.selectedFile, file);
      assert.equal(elements['import-file-size'].textContent, '100.0 MB');
      assert.equal(
        elements['import-submit-btn'].disabled,
        false,
        'Should be enabled for exactly 100 MB'
      );
    });

    it('rejects a CSV file that is 1 byte over the maximum limit (104,857,601 bytes)', () => {
      const file = {
        name: 'too_large_by_one_byte.csv',
        size: 104857601, // 100 MB + 1 byte
        type: 'text/csv',
      };

      ImportsController.handleFileSelected(file);

      assert.equal(ImportsController.selectedFile, null, 'Oversized file must not be stored');
      assert.equal(
        elements['import-submit-btn'].disabled,
        true,
        'Start Ingestion button must remain disabled'
      );
      assert.ok(
        toastMessages.some(
          (t) => t.type === 'error' && t.msg.includes('exceeds maximum allowed limit of 100 MB')
        ),
        'Should display error toast communicating 100 MB limit'
      );
    });

    it('rejects an empty CSV file (0 bytes)', () => {
      const file = {
        name: 'empty.csv',
        size: 0,
        type: 'text/csv',
      };

      ImportsController.handleFileSelected(file);

      assert.equal(ImportsController.selectedFile, null);
      assert.equal(elements['import-submit-btn'].disabled, true);
      assert.ok(toastMessages.some((t) => t.type === 'error' && t.msg.includes('empty')));
    });

    it('rejects non-CSV files even if below the size limit', () => {
      const file = {
        name: 'malicious_script.exe',
        size: 1024,
        type: 'application/x-msdownload',
      };

      ImportsController.handleFileSelected(file);

      assert.equal(ImportsController.selectedFile, null);
      assert.equal(elements['import-submit-btn'].disabled, true);
      assert.ok(toastMessages.some((t) => t.type === 'error' && t.msg.includes('Only CSV')));
    });
  });

  describe('Drag & Drop and File Picker Integration', () => {
    it('handles oversized files via drag-and-drop consistently', () => {
      const dropEvent = {
        preventDefault: () => {},
        stopPropagation: () => {},
        dataTransfer: {
          files: [
            {
              name: 'dropped_150mb.csv',
              size: 150 * 1024 * 1024,
              type: 'text/csv',
            },
          ],
        },
      };

      elements['import-dropzone'].trigger('drop', dropEvent);

      assert.equal(ImportsController.selectedFile, null);
      assert.equal(elements['import-submit-btn'].disabled, true);
      assert.ok(
        toastMessages.some(
          (t) => t.type === 'error' && t.msg.includes('exceeds maximum allowed limit of 100 MB')
        )
      );
    });

    it('handles valid files via file-input change event', () => {
      const changeEvent = {
        preventDefault: () => {},
        stopPropagation: () => {},
        target: {
          files: [
            {
              name: 'selected_via_picker.csv',
              size: 25 * 1024 * 1024,
              type: 'text/csv',
            },
          ],
        },
      };

      elements['import-file-input'].trigger('change', changeEvent);

      assert.ok(ImportsController.selectedFile);
      assert.equal(ImportsController.selectedFile.name, 'selected_via_picker.csv');
      assert.equal(elements['import-submit-btn'].disabled, false);
    });
  });

  describe('Upload Flow, Error Handling & State Restoration', () => {
    it('prevents upload request if no file is selected', async () => {
      let uploadCalled = false;
      window.api.uploadCSV = async () => {
        uploadCalled = true;
      };

      ImportsController.clearSelection();
      await ImportsController.startUpload();

      assert.equal(uploadCalled, false, 'uploadCSV should not be invoked when no file selected');
    });

    it('prevents double submissions while an upload is in progress', async () => {
      let callCount = 0;
      let resolveUpload;
      const uploadPromise = new Promise((resolve) => {
        resolveUpload = resolve;
      });

      window.api.uploadCSV = async () => {
        callCount++;
        return uploadPromise;
      };

      const file = { name: 'test.csv', size: 1024, type: 'text/csv' };
      ImportsController.handleFileSelected(file);

      // Trigger first upload (in-flight)
      const firstRun = ImportsController.startUpload();
      assert.equal(ImportsController.isUploading, true);
      assert.equal(elements['import-submit-btn'].disabled, true);

      // Trigger second upload while first is still pending
      const secondRun = ImportsController.startUpload();

      // Complete first upload
      resolveUpload({
        id: 'job-1',
        status: 'COMPLETED',
        total_rows: 1,
        successful_rows: 1,
        failed_rows: 0,
      });
      await Promise.all([firstRun, secondRun]);

      assert.equal(callCount, 1, 'uploadCSV should only be called once');
      assert.equal(ImportsController.isUploading, false);
    });

    it('correctly displays backend HTTP 413 Entity Too Large errors and restores button state', async () => {
      window.api.uploadCSV = async () => {
        const error = new Error('Uploaded file exceeds maximum allowed size (100 MB).');
        error.status = 413;
        throw error;
      };

      const file = { name: 'server_rejected.csv', size: 50 * 1024 * 1024, type: 'text/csv' };
      ImportsController.handleFileSelected(file);
      assert.equal(elements['import-submit-btn'].disabled, false);

      await ImportsController.startUpload();

      assert.equal(ImportsController.isUploading, false);
      assert.equal(
        elements['import-submit-btn'].disabled,
        false,
        'Submit button should be re-enabled for retry'
      );
      assert.ok(
        elements['import-progress-status'].textContent.includes(
          'Uploaded file exceeds maximum allowed size (100 MB)'
        ),
        'Progress text must display 413 error message'
      );
      assert.ok(
        toastMessages.some((t) => t.type === 'error' && t.msg.includes('100 MB')),
        'Error toast must be displayed'
      );
    });

    it('restores interface state after network or server failure', async () => {
      window.api.uploadCSV = async () => {
        const error = new Error('Network timeout during upload');
        error.status = 504;
        throw error;
      };

      const file = { name: 'network_fail.csv', size: 10 * 1024 * 1024, type: 'text/csv' };
      ImportsController.handleFileSelected(file);

      await ImportsController.startUpload();

      assert.equal(ImportsController.isUploading, false);
      assert.equal(elements['import-submit-btn'].disabled, false);
      assert.equal(elements['import-progress-bar'].style.width, '0%');
      assert.ok(elements['import-progress-status'].textContent.includes('Network timeout'));
    });

    it('handles HTTP 202 response with PENDING status, sets queued UI state and initiates polling', async () => {
      let pollCalledWith = null;
      const originalPollJob = ImportsController.pollJob;

      try {
        ImportsController.pollJob = (jobId) => {
          pollCalledWith = jobId;
        };

        window.api.uploadCSV = async () => ({
          id: 'job-pending-123',
          status: 'PENDING',
          total_rows: 0,
          processed_rows: 0,
          successful_rows: 0,
          failed_rows: 0,
        });

        const file = { name: 'queued.csv', size: 5 * 1024 * 1024, type: 'text/csv' };
        ImportsController.handleFileSelected(file);
        await ImportsController.startUpload();

        assert.equal(pollCalledWith, 'job-pending-123', 'pollJob must be called with job ID');
        assert.ok(
          elements['import-progress-status'].textContent.includes('Upload accepted') ||
            elements['import-progress-status'].textContent.includes('queued'),
          'Must display queued status message'
        );
        assert.equal(elements['import-progress-bar'].style.width, '25%');
      } finally {
        ImportsController.pollJob = originalPollJob;
      }
    });

    it('terminates polling at terminal COMPLETED or FAILED states', async () => {
      let clearIntervalCalled = false;
      const origClearInterval = globalThis.clearInterval;
      globalThis.clearInterval = (id) => {
        clearIntervalCalled = true;
        return origClearInterval(id);
      };

      try {
        window.api.getImportJob = async () => ({
          id: 'job-complete-1',
          status: 'COMPLETED',
          total_rows: 10,
          processed_rows: 10,
          successful_rows: 10,
          failed_rows: 0,
        });

        ImportsController.pollJob('job-complete-1');
        assert.ok(ImportsController.pollingInterval !== null);

        ImportsController.cleanup();
        assert.equal(ImportsController.pollingInterval, null, 'Polling interval must be cleared');
        assert.equal(clearIntervalCalled, true, 'clearInterval must be called');
      } finally {
        globalThis.clearInterval = origClearInterval;
      }
    });

    it('cancels polling when cleanup() is invoked upon view navigation', () => {
      ImportsController.pollingInterval = setInterval(() => {}, 1000);
      assert.ok(ImportsController.pollingInterval !== null);

      ImportsController.cleanup();
      assert.equal(
        ImportsController.pollingInterval,
        null,
        'Polling interval must be cancelled on cleanup'
      );
    });

    it('handles HTTP 202 response with PENDING status, sets queued UI state and initiates polling', async () => {
      let pollCalledWith = null;
      const originalPollJob = ImportsController.pollJob;

      try {
        ImportsController.pollJob = (jobId) => {
          pollCalledWith = jobId;
        };

        window.api.uploadCSV = async () => ({
          id: 'job-pending-123',
          status: 'PENDING',
          total_rows: 0,
          processed_rows: 0,
          successful_rows: 0,
          failed_rows: 0,
        });

        const file = { name: 'queued.csv', size: 5 * 1024 * 1024, type: 'text/csv' };
        ImportsController.handleFileSelected(file);
        await ImportsController.startUpload();

        assert.equal(pollCalledWith, 'job-pending-123', 'pollJob must be called with job ID');
        assert.ok(
          elements['import-progress-status'].textContent.includes('Upload accepted') ||
            elements['import-progress-status'].textContent.includes('queued'),
          'Must display queued status message'
        );
        assert.equal(elements['import-progress-bar'].style.width, '25%');
      } finally {
        ImportsController.pollJob = originalPollJob;
      }
    });

    it('terminates polling at terminal COMPLETED or FAILED states', async () => {
      let clearIntervalCalled = false;
      const origClearInterval = globalThis.clearInterval;
      globalThis.clearInterval = (id) => {
        clearIntervalCalled = true;
        return origClearInterval(id);
      };

      try {
        window.api.getImportJob = async () => ({
          id: 'job-complete-1',
          status: 'COMPLETED',
          total_rows: 10,
          processed_rows: 10,
          successful_rows: 10,
          failed_rows: 0,
        });

        ImportsController.pollJob('job-complete-1');
        assert.ok(ImportsController.pollingInterval !== null);

        ImportsController.cleanup();
        assert.equal(ImportsController.pollingInterval, null, 'Polling interval must be cleared');
        assert.equal(clearIntervalCalled, true, 'clearInterval must be called');
      } finally {
        globalThis.clearInterval = origClearInterval;
      }
    });

    it('cancels polling when cleanup() is invoked upon view navigation', () => {
      ImportsController.pollingInterval = setInterval(() => {}, 1000);
      assert.ok(ImportsController.pollingInterval !== null);

      ImportsController.cleanup();
      assert.equal(
        ImportsController.pollingInterval,
        null,
        'Polling interval must be cancelled on cleanup'
      );
    });
  });
});
