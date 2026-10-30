/**
 * Sentiment Pulse - Service Worker (sw.js)
 * Native Service Worker implementing versioned offline application shell caching.
 *
 * Rules:
 * - Cache-first for static application shell assets
 * - API requests (/predict, /feedback, /api, etc.) are NEVER cached
 * - Authenticated requests and credentials are NEVER persisted
 * - Navigation requests fall back to cached shell base '/'
 * - Versioned cache cleanup during activation
 */

const CACHE_VERSION = 'v5';
const CACHE_NAME = `sentiment-pulse-shell-${CACHE_VERSION}`;

const SHELL_ASSETS = [
  '/',
  '/static/manifest.webmanifest',
  '/static/css/tailwind.min.css',
  '/static/css/app.css',
  '/static/css/motion-tokens.css',
  '/static/css/skeletons.css',
  '/static/theme.js',
  '/static/icons/icon-192x192.png',
  '/static/icons/icon-512x512.png',
  '/static/icons/icon-maskable-192x192.png',
  '/static/icons/icon-maskable-512x512.png',
  '/static/js/app.js',
  '/static/js/api.js',
  '/static/js/overview.js',
  '/static/js/analyze.js',
  '/static/js/explorer.js',
  '/static/js/analytics.js',
  '/static/js/imports.js',
  '/static/js/history.js',
  '/static/js/explain.js',
  '/static/js/lib/formatters.js',
  '/static/js/lib/chunker.js',
  '/static/js/lib/modal.js',
  '/static/js/lib/idb.js',
  '/static/js/lib/store.js',
  '/static/js/lib/bulk.js',
  '/static/js/lib/confidence.js',
  '/static/js/lib/pwa-helpers.js',
  '/static/js/lib/motion/animation.js',
  '/static/js/lib/motion/vendor-loader.js',
  '/static/js/lib/motion/presets.js',
  '/static/js/lib/motion/reveal.js',
  '/static/js/lib/motion/transitions.js',
  '/static/js/lib/motion/counters.js',
  '/static/js/lib/motion/skeletons.js',
  '/static/js/lib/motion/charts-motion.js',
  '/static/js/lib/motion/micro.js',
  '/static/js/lib/vendor/vendor.manifest.json',
  '/static/js/lib/vendor/gsap.min.js',
  '/static/js/lib/vendor/ScrollTrigger.min.js',
  '/static/js/lib/vendor/Flip.min.js',
  '/static/js/lib/vendor/countup.min.js',
  '/static/js/lib/vendor/split-type.min.js',
  '/static/js/lib/vendor/confetti.min.js',
  '/static/js/lib/vendor/chart.min.js',
  '/static/js/lib/vendor/three.module.js',
  '/static/js/lib/vendor/three.core.js',
  '/static/js/3d/core/fallback-controller.js',
  '/static/js/3d/core/performance-governor.js',
  '/static/js/3d/core/renderer-manager.js',
  '/static/js/3d/core/camera-rig.js',
  '/static/js/3d/core/scene-manager.js',
  '/static/js/3d/scenes/overview-scene.js',
  '/static/js/3d/scenes/analyze-scene.js',
  '/static/js/3d/three-runtime.js',
];

const API_PREFIXES = ['/predict', '/feedback', '/api', '/health', '/docs', '/openapi.json'];

const STATIC_EXTENSIONS = [
  '.css',
  '.js',
  '.mjs',
  '.png',
  '.jpg',
  '.jpeg',
  '.svg',
  '.ico',
  '.woff2',
  '.woff',
  '.ttf',
  '.webmanifest',
  '.json',
];

const STATIC_HOSTS = [
  'fonts.googleapis.com',
  'fonts.gstatic.com',
  'cdn.tailwindcss.com',
  'cdn.jsdelivr.net',
];

function isApiUrl(url) {
  const pathname = url.pathname;
  return API_PREFIXES.some((prefix) => pathname === prefix || pathname.startsWith(`${prefix}/`));
}

function isStaticAssetUrl(url) {
  if (isApiUrl(url)) return false;
  const pathname = url.pathname;
  const hostname = url.hostname;

  if (pathname === '/' || pathname === '/index.html') return true;
  if (pathname.startsWith('/static/')) return true;
  if (STATIC_HOSTS.some((h) => hostname === h || hostname.endsWith(`.${h}`))) return true;
  return STATIC_EXTENSIONS.some((ext) => pathname.endsWith(ext));
}

// -----------------------------------------------------------------------------
// INSTALL: Pre-cache the essential application shell
// -----------------------------------------------------------------------------
self.addEventListener('install', (event) => {
  event.waitUntil(
    caches
      .open(CACHE_NAME)
      .then((cache) => cache.addAll(SHELL_ASSETS))
      .then(() => self.skipWaiting())
      .catch((err) => {
        console.warn('[SW] Shell precache failed:', err);
      })
  );
});

// -----------------------------------------------------------------------------
// ACTIVATE: Clean up obsolete cache versions safely
// -----------------------------------------------------------------------------
self.addEventListener('activate', (event) => {
  event.waitUntil(
    caches
      .keys()
      .then((cacheNames) => {
        return Promise.all(
          cacheNames.map((name) => {
            if (name.startsWith('sentiment-pulse-shell-') && name !== CACHE_NAME) {
              return caches.delete(name);
            }
            return Promise.resolve();
          })
        );
      })
      .then(() => self.clients.claim())
  );
});

// -----------------------------------------------------------------------------
// FETCH: Static shell cache-first, navigation fallback, zero API caching
// -----------------------------------------------------------------------------
self.addEventListener('fetch', (event) => {
  const req = event.request;
  const url = new URL(req.url);

  // 1. Only process GET requests; never intercept POST/PUT/DELETE
  if (req.method !== 'GET') {
    return;
  }

  // 2. Never cache authenticated or private requests
  if (req.headers && (req.headers.has('Authorization') || req.headers.has('authorization'))) {
    return;
  }

  // 3. Never cache dynamic/private API responses (predict, feedback, analytics, etc.)
  if (isApiUrl(url)) {
    return;
  }

  // 4. Client Navigation requests: Network-first with cached shell fallback
  if (req.mode === 'navigate') {
    event.respondWith(
      fetch(req).catch(async () => {
        const cachedShell = await caches.match('/');
        if (cachedShell) return cachedShell;
        return caches.match('/static/manifest.webmanifest');
      })
    );
    return;
  }

  // 5. Static Shell Assets: Cache-first strategy with background cache-fill
  if (isStaticAssetUrl(url)) {
    event.respondWith(
      caches.match(req).then((cachedResponse) => {
        if (cachedResponse) {
          return cachedResponse;
        }

        return fetch(req).then((networkResponse) => {
          // Cache only valid 200 OK responses or opaque 0 responses from known CDNs
          if (
            networkResponse &&
            (networkResponse.status === 200 ||
              (networkResponse.status === 0 && STATIC_HOSTS.some((h) => url.hostname.includes(h))))
          ) {
            const clone = networkResponse.clone();
            caches.open(CACHE_NAME).then((cache) => {
              cache.put(req, clone);
            });
          }
          return networkResponse;
        });
      })
    );
  }
});
